"""
Tests for sps.core: AdversarialSPSEstimator and AdversarialSPSReport.

Covers:
  - AdversarialSPSReport: dataclass fields, adv_gap_ratio bounds, summary formatting.
  - AdversarialSPSEstimator.estimate(): gap_ratio in (0, 1], sps ordering,
    CI containment, LOO stability keys, constant-model degeneracy.
  - build_adversarial_sps_estimator(): factory correctness.
  - Monotonicity: SPS_adv <= SPS_emb (adversarial >= random sensitivity in expectation).
  - Summary labels: near-adversarial / moderate / large gap branches.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps.core import (
    AdversarialSPSEstimator,
    AdversarialSPSReport,
    SPSConfig,
    build_adversarial_sps_estimator,
    build_sps_estimator,
)
from sps.transformations import (
    AdversarialEmbeddingConfig,
    AdversarialEmbeddingFamily,
    EmbeddingPerturbationConfig,
    EmbeddingPerturbationFamily,
)
from sps.utils import cosine_divergence, set_seed


# ─────────────────────────────────────────────────────────────────────────────
# Constants and fixtures
# ─────────────────────────────────────────────────────────────────────────────

HIDDEN = 32
SEQ_LEN = 8
BATCH = 4
VOCAB = 100
N_BATCHES = 3


@pytest.fixture(autouse=True)
def fix_seed():
    set_seed(0)


@pytest.fixture
def embedding_layer() -> nn.Embedding:
    emb = nn.Embedding(VOCAB, HIDDEN)
    nn.init.normal_(emb.weight, std=0.1)
    return emb


@pytest.fixture
def t_emb(embedding_layer) -> EmbeddingPerturbationFamily:
    cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
    return EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)


@pytest.fixture
def batches(embedding_layer) -> list[dict]:
    result = []
    for i in range(N_BATCHES):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            emb = embedding_layer(ids)
        result.append({"input_ids": ids, "attention_mask": mask, "embeddings": emb})
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Dummy models
# ─────────────────────────────────────────────────────────────────────────────

class ConstantModel(nn.Module):
    """Always returns a fixed vector — zero sensitivity, SPS = 1."""
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.out = nn.Parameter(torch.randn(hidden), requires_grad=False)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        B = inputs_embeds.size(0)
        hidden_out = torch.zeros(B, inputs_embeds.size(1), self.out.size(0))
        hidden_out[:, 0, :] = self.out.unsqueeze(0).expand(B, -1)

        class _Out:
            last_hidden_state = hidden_out
        return _Out()


class LinearModel(nn.Module):
    """Linear projection on CLS token — known sensitivity."""
    def __init__(self, hidden: int, out_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        full = torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1), 16)
        cls = inputs_embeds[:, 0, :]
        full[:, 0, :] = self.proj(cls)

        class _Out:
            last_hidden_state = full
        return _Out()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: build adversarial estimator from a linear model
# ─────────────────────────────────────────────────────────────────────────────

def make_adversarial_estimator(
    embedding_layer: nn.Embedding,
    model: nn.Module,
    n_directions: int = 4,
    config: SPSConfig | None = None,
) -> AdversarialSPSEstimator:
    config = config or SPSConfig(epsilon=0.1, m_transforms=8, seed=0)

    emb_cfg = EmbeddingPerturbationConfig(n_directions=n_directions, use_synonym_directions=False)
    emb_family = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=emb_cfg)

    def forward_fn(x: torch.Tensor) -> torch.Tensor:
        attn = torch.ones(x.size(0), x.size(1), dtype=torch.long)
        out = model(inputs_embeds=x, attention_mask=attn)
        return out.last_hidden_state[:, 0, :]

    adv_cfg = AdversarialEmbeddingConfig(
        n_directions=n_directions,
        use_synonym_directions=False,
    )
    adv_family = AdversarialEmbeddingFamily(
        forward_fn=forward_fn,
        embedding_layer=embedding_layer,
        config=adv_cfg,
    )
    return build_adversarial_sps_estimator(model, emb_family, adv_family, config)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: AdversarialSPSReport dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialSPSReport:

    def _dummy_report(self, **kwargs) -> AdversarialSPSReport:
        defaults = dict(
            sps_emb=0.80, sps_adv=0.72, adv_gap_ratio=0.90,
            mean_sens_emb=0.22, std_sens_emb=0.05,
            mean_sens_adv=0.33, std_sens_adv=0.06,
            n_samples=48,
            emb_sens_ci=(0.22, 0.18, 0.26),
            adv_sens_ci=(0.33, 0.28, 0.38),
            loo_emb={"stable": True, "loo_range": 0.02},
            loo_adv={"stable": True, "loo_range": 0.03},
        )
        defaults.update(kwargs)
        return AdversarialSPSReport(**defaults)

    def test_fields_accessible(self) -> None:
        r = self._dummy_report()
        assert r.sps_emb == pytest.approx(0.80)
        assert r.sps_adv == pytest.approx(0.72)
        assert r.adv_gap_ratio == pytest.approx(0.90)
        assert r.n_samples == 48

    def test_config_optional(self) -> None:
        r = self._dummy_report()
        assert r.config is None

    def test_summary_contains_key_fields(self) -> None:
        r = self._dummy_report()
        s = r.summary()
        assert "0.8000" in s
        assert "0.7200" in s
        assert "0.9000" in s
        assert "48" in s

    def test_summary_near_adversarial_label(self) -> None:
        r = self._dummy_report(adv_gap_ratio=0.97)
        assert "near-adversarial" in r.summary()

    def test_summary_moderate_label(self) -> None:
        r = self._dummy_report(adv_gap_ratio=0.85)
        assert "moderate" in r.summary()

    def test_summary_large_gap_label(self) -> None:
        r = self._dummy_report(adv_gap_ratio=0.50)
        assert "LARGE" in r.summary()

    def test_summary_is_string(self) -> None:
        r = self._dummy_report()
        assert isinstance(r.summary(), str)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: AdversarialSPSEstimator.estimate()
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialSPSEstimator:

    def test_returns_adversarial_sps_report(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert isinstance(report, AdversarialSPSReport)

    def test_sps_emb_in_unit_interval(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert 0.0 < report.sps_emb <= 1.0 + 1e-9

    def test_sps_adv_in_unit_interval(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert 0.0 < report.sps_adv <= 1.0 + 1e-9

    def test_adv_gap_ratio_in_unit_interval(self, embedding_layer, batches) -> None:
        """adv_gap_ratio = SPS_adv / SPS_emb must be in (0, 1] since T_adv is worst-case."""
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert 0.0 < report.adv_gap_ratio <= 1.0 + 1e-6

    def test_sps_adv_leq_sps_emb(self, embedding_layer, batches) -> None:
        """T_adv finds worst-case direction → sensitivity_adv >= sensitivity_emb
        → SPS_adv <= SPS_emb (in expectation over the dataset)."""
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model, n_directions=8)
        report = est.estimate(batches)
        # SPS_adv may be slightly above SPS_emb for a small number of samples
        # due to randomness, but never by a large margin.
        assert report.sps_adv <= report.sps_emb + 0.05

    def test_n_samples_matches_batch_total(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert report.n_samples == BATCH * N_BATCHES

    def test_mean_sens_adv_geq_emb(self, embedding_layer, batches) -> None:
        """Adversarial sensitivity >= random sensitivity (in expectation)."""
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model, n_directions=8)
        report = est.estimate(batches)
        # Allow small tolerance for finite-sample fluctuations.
        assert report.mean_sens_adv >= report.mean_sens_emb - 0.05

    def test_ci_tuples_have_length_3(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert len(report.emb_sens_ci) == 3
        assert len(report.adv_sens_ci) == 3

    def test_ci_lower_leq_point_leq_upper(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        pt_e, lo_e, hi_e = report.emb_sens_ci
        pt_a, lo_a, hi_a = report.adv_sens_ci
        assert lo_e <= pt_e <= hi_e
        assert lo_a <= pt_a <= hi_a

    def test_loo_dicts_have_required_keys(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        # With N_BATCHES * BATCH = 12 samples, LOO is well-defined.
        for d in (report.loo_emb, report.loo_adv):
            assert "stable" in d

    def test_constant_model_gap_ratio_near_one(self, embedding_layer, batches) -> None:
        """Constant model: Sens=0 for all transforms → SPS_adv = SPS_emb = 1 → ratio = 1."""
        model = ConstantModel(hidden=HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        assert math.isclose(report.sps_emb, 1.0, abs_tol=1e-5)
        assert math.isclose(report.sps_adv, 1.0, abs_tol=1e-5)
        assert math.isclose(report.adv_gap_ratio, 1.0, abs_tol=1e-5)

    def test_config_stored_in_report(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        config = SPSConfig(epsilon=0.15, m_transforms=6, seed=7)
        est = make_adversarial_estimator(embedding_layer, model, config=config)
        report = est.estimate(batches)
        assert report.config is not None
        assert report.config.epsilon == pytest.approx(0.15)

    def test_summary_contains_sps_values(self, embedding_layer, batches) -> None:
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model)
        report = est.estimate(batches)
        s = report.summary()
        assert f"{report.sps_emb:.4f}" in s
        assert f"{report.sps_adv:.4f}" in s
        assert f"{report.adv_gap_ratio:.4f}" in s


# ─────────────────────────────────────────────────────────────────────────────
# Tests: build_adversarial_sps_estimator factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildAdversarialSPSEstimator:

    def test_returns_adversarial_sps_estimator(self, embedding_layer) -> None:
        model = LinearModel(HIDDEN)
        emb_cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
        emb_family = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=emb_cfg)

        def forward_fn(x):
            out = model(inputs_embeds=x, attention_mask=torch.ones(x.size(0), x.size(1), dtype=torch.long))
            return out.last_hidden_state[:, 0, :]

        adv_cfg = AdversarialEmbeddingConfig(n_directions=4, use_synonym_directions=False)
        adv_family = AdversarialEmbeddingFamily(
            forward_fn=forward_fn, embedding_layer=embedding_layer, config=adv_cfg
        )
        config = SPSConfig(epsilon=0.1, m_transforms=4)
        est = build_adversarial_sps_estimator(model, emb_family, adv_family, config)
        assert isinstance(est, AdversarialSPSEstimator)

    def test_default_config_applied_when_none(self, embedding_layer) -> None:
        model = LinearModel(HIDDEN)
        emb_cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
        emb_family = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=emb_cfg)

        def forward_fn(x):
            out = model(inputs_embeds=x, attention_mask=torch.ones(x.size(0), x.size(1), dtype=torch.long))
            return out.last_hidden_state[:, 0, :]

        adv_cfg = AdversarialEmbeddingConfig(n_directions=4, use_synonym_directions=False)
        adv_family = AdversarialEmbeddingFamily(
            forward_fn=forward_fn, embedding_layer=embedding_layer, config=adv_cfg
        )
        est = build_adversarial_sps_estimator(model, emb_family, adv_family, config=None)
        assert est.config is not None
        assert est.config.epsilon == pytest.approx(0.1)   # SPSConfig default


# ─────────────────────────────────────────────────────────────────────────────
# Tests: Monotonicity and edge properties
# ─────────────────────────────────────────────────────────────────────────────

class TestAdversarialMonotonicity:

    def test_gap_ratio_not_above_one_for_linear_model(self, embedding_layer, batches) -> None:
        """For a linear model the adversarial attack should never yield a higher SPS than
        random sampling by a significant margin."""
        model = LinearModel(HIDDEN)
        est = make_adversarial_estimator(embedding_layer, model, n_directions=8)
        report = est.estimate(batches)
        # Allow tiny numerical tolerance (< 2%) but not more.
        assert report.adv_gap_ratio <= 1.02

    def test_larger_direction_set_reduces_gap_ratio(self, embedding_layer, batches) -> None:
        """More adversarial directions → closer to true worst case → smaller or equal gap ratio."""
        model = LinearModel(HIDDEN)
        set_seed(3)
        est_small = make_adversarial_estimator(embedding_layer, model, n_directions=2)
        set_seed(3)
        est_large = make_adversarial_estimator(embedding_layer, model, n_directions=16)
        report_small = est_small.estimate(batches)
        report_large = est_large.estimate(batches)
        # More directions → adversarial effect >= small direction case (in expectation)
        # gap_ratio_large <= gap_ratio_small (or at most slightly above due to finite samples)
        assert report_large.adv_gap_ratio <= report_small.adv_gap_ratio + 0.10
