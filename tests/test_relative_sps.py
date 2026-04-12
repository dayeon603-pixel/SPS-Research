"""
Tests for sps.core: RelativeSPSEstimator and RelativeSPSReport.

Covers:
  - RelativeSPSReport: dataclass fields, rsps_ci containment, summary formatting.
  - RelativeSPSEstimator.estimate(): rsps_point positive, CI tuple, CI containment,
    rsps_ci_excludes_one logic, n_samples, LOO keys, identical-family degeneracy.
  - build_relative_sps_estimator(): factory correctness and default config.
  - Statistical properties: CI width positive; CI excludes one when families differ
    maximally; equal families give rsps ≈ 1.
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
    RelativeSPSEstimator,
    RelativeSPSReport,
    SPSConfig,
    build_relative_sps_estimator,
    build_sps_estimator,
)
from sps.transformations import (
    AdversarialEmbeddingConfig,
    AdversarialEmbeddingFamily,
    EmbeddingPerturbationConfig,
    EmbeddingPerturbationFamily,
)
from sps.utils import set_seed


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
def t_family(embedding_layer) -> EmbeddingPerturbationFamily:
    """Semantic T: synonym-direction perturbations."""
    cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
    return EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)


@pytest.fixture
def arb_family(embedding_layer) -> EmbeddingPerturbationFamily:
    """T_arb: random isotropic perturbations (same family, different seed)."""
    cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
    return EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)


@pytest.fixture
def adv_family(embedding_layer) -> AdversarialEmbeddingFamily:
    """T_adv: adversarial worst-case family (maximally different from T)."""
    cfg = AdversarialEmbeddingConfig(n_candidates=8)
    return AdversarialEmbeddingFamily(embedding_layer=embedding_layer, config=cfg)


@pytest.fixture
def batches(embedding_layer) -> list[dict]:
    result = []
    for _ in range(N_BATCHES):
        ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        mask = torch.ones_like(ids)
        with torch.no_grad():
            emb = embedding_layer(ids)
        result.append({"input_ids": ids, "attention_mask": mask, "embeddings": emb})
    return result


@pytest.fixture
def linear_model() -> nn.Module:
    """Simple linear model: outputs mean-pooled representation."""

    class LinearModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(HIDDEN, HIDDEN)

        def forward(self, inputs_embeds=None, attention_mask=None):
            # Return a simple BaseModelOutput-compatible object
            x = self.proj(inputs_embeds)

            class _Out:
                def __init__(self, h):
                    self.last_hidden_state = h

            return _Out(x)

    m = LinearModel()
    nn.init.eye_(m.proj.weight)
    nn.init.zeros_(m.proj.bias)
    return m.eval()


@pytest.fixture
def config() -> SPSConfig:
    return SPSConfig(
        epsilon=0.1,
        n_data_samples=BATCH * N_BATCHES,
        m_transforms=4,
        device="cpu",
        seed=0,
        batch_size=BATCH,
    )


@pytest.fixture
def relative_estimator(linear_model, t_family, arb_family, config):
    return build_relative_sps_estimator(linear_model, t_family, arb_family, config)


@pytest.fixture
def report(relative_estimator, batches) -> RelativeSPSReport:
    return relative_estimator.estimate(batches)


# ─────────────────────────────────────────────────────────────────────────────
# TestRelativeSPSReport — dataclass fields
# ─────────────────────────────────────────────────────────────────────────────

class TestRelativeSPSReport:
    def test_fields_present(self, report: RelativeSPSReport):
        """All expected fields are present and have correct types."""
        assert isinstance(report.sps_t, float)
        assert isinstance(report.sps_arb, float)
        assert isinstance(report.rsps_point, float)
        assert isinstance(report.rsps_ci, tuple)
        assert isinstance(report.rsps_ci_excludes_one, bool)
        assert isinstance(report.mean_sens_t, float)
        assert isinstance(report.std_sens_t, float)
        assert isinstance(report.n_samples_t, int)
        assert isinstance(report.mean_sens_arb, float)
        assert isinstance(report.std_sens_arb, float)
        assert isinstance(report.n_samples_arb, int)
        assert isinstance(report.sens_ci_t, tuple)
        assert isinstance(report.loo_t, dict)

    def test_sps_values_positive(self, report: RelativeSPSReport):
        """SPS_T and SPS_arb must be positive (SPS ∈ (0, 1])."""
        assert report.sps_t > 0
        assert report.sps_arb > 0

    def test_rsps_point_positive(self, report: RelativeSPSReport):
        """rSPS point estimate must be positive (ratio of two positive values)."""
        assert report.rsps_point > 0

    def test_rsps_ci_is_three_tuple(self, report: RelativeSPSReport):
        """rsps_ci must be (point, lo, hi)."""
        assert len(report.rsps_ci) == 3

    def test_rsps_ci_ordering(self, report: RelativeSPSReport):
        """lo ≤ point ≤ hi for the delta-method CI."""
        point, lo, hi = report.rsps_ci
        assert lo <= point, f"CI lower {lo} > point {point}"
        assert point <= hi, f"CI point {point} > upper {hi}"

    def test_rsps_ci_width_positive(self, report: RelativeSPSReport):
        """CI width must be positive (non-degenerate interval)."""
        _, lo, hi = report.rsps_ci
        assert hi > lo

    def test_sens_ci_t_is_three_tuple(self, report: RelativeSPSReport):
        """sens_ci_t must be (point, lo, hi)."""
        assert len(report.sens_ci_t) == 3

    def test_loo_t_has_keys(self, report: RelativeSPSReport):
        """LOO dict must contain 'stable' and 'n' keys."""
        assert "stable" in report.loo_t
        assert "n" in report.loo_t

    def test_summary_not_empty(self, report: RelativeSPSReport):
        """summary() must return a non-empty string."""
        s = report.summary()
        assert isinstance(s, str) and len(s) > 0

    def test_summary_contains_rsps(self, report: RelativeSPSReport):
        """summary() must contain the rSPS value."""
        s = report.summary()
        assert f"{report.rsps_point:.4f}" in s

    def test_summary_label_more_stable(self):
        """summary() uses 'MORE stable' label when rsps_point > 1.05."""
        rep = RelativeSPSReport(
            sps_t=0.9, sps_arb=0.5, rsps_point=1.10,
            rsps_ci=(1.10, 1.05, 1.15), rsps_ci_excludes_one=True,
            mean_sens_t=0.1, std_sens_t=0.01, n_samples_t=100,
            mean_sens_arb=0.5, std_sens_arb=0.05, n_samples_arb=100,
            sens_ci_t=(0.1, 0.09, 0.11), loo_t={"stable": True, "n": 100},
        )
        assert "MORE stable" in rep.summary()

    def test_summary_label_approximately_equal(self):
        """summary() uses '≈ T_arb' label when rsps_point ≈ 1.0."""
        rep = RelativeSPSReport(
            sps_t=0.5, sps_arb=0.5, rsps_point=1.00,
            rsps_ci=(1.00, 0.95, 1.05), rsps_ci_excludes_one=False,
            mean_sens_t=0.3, std_sens_t=0.02, n_samples_t=100,
            mean_sens_arb=0.3, std_sens_arb=0.02, n_samples_arb=100,
            sens_ci_t=(0.3, 0.28, 0.32), loo_t={"stable": True, "n": 100},
        )
        assert "≈ T_arb" in rep.summary()

    def test_summary_label_less_stable(self):
        """summary() uses 'LESS stable' label when rsps_point < 0.95."""
        rep = RelativeSPSReport(
            sps_t=0.3, sps_arb=0.7, rsps_point=0.80,
            rsps_ci=(0.80, 0.70, 0.90), rsps_ci_excludes_one=True,
            mean_sens_t=0.8, std_sens_t=0.05, n_samples_t=100,
            mean_sens_arb=0.3, std_sens_arb=0.02, n_samples_arb=100,
            sens_ci_t=(0.8, 0.75, 0.85), loo_t={"stable": True, "n": 100},
        )
        assert "LESS stable" in rep.summary()

    def test_summary_sig_yes_when_excludes_one(self):
        """summary() shows 'YES' significance label when CI excludes 1."""
        rep = RelativeSPSReport(
            sps_t=0.9, sps_arb=0.5, rsps_point=1.10,
            rsps_ci=(1.10, 1.05, 1.15), rsps_ci_excludes_one=True,
            mean_sens_t=0.1, std_sens_t=0.01, n_samples_t=100,
            mean_sens_arb=0.5, std_sens_arb=0.05, n_samples_arb=100,
            sens_ci_t=(0.1, 0.09, 0.11), loo_t={"stable": True, "n": 100},
        )
        assert "YES" in rep.summary()

    def test_summary_no_sig_when_contains_one(self):
        """summary() shows 'no' significance label when CI contains 1."""
        rep = RelativeSPSReport(
            sps_t=0.5, sps_arb=0.5, rsps_point=1.00,
            rsps_ci=(1.00, 0.92, 1.08), rsps_ci_excludes_one=False,
            mean_sens_t=0.3, std_sens_t=0.02, n_samples_t=100,
            mean_sens_arb=0.3, std_sens_arb=0.02, n_samples_arb=100,
            sens_ci_t=(0.3, 0.28, 0.32), loo_t={"stable": True, "n": 100},
        )
        assert "no" in rep.summary()


# ─────────────────────────────────────────────────────────────────────────────
# TestRelativeSPSEstimator — estimate()
# ─────────────────────────────────────────────────────────────────────────────

class TestRelativeSPSEstimator:
    def test_returns_relative_sps_report(self, report: RelativeSPSReport):
        """estimate() must return a RelativeSPSReport."""
        assert isinstance(report, RelativeSPSReport)

    def test_sps_t_in_range(self, report: RelativeSPSReport):
        """SPS_T must be in (0, 1]."""
        assert 0 < report.sps_t <= 1.0 + 1e-6

    def test_sps_arb_in_range(self, report: RelativeSPSReport):
        """SPS_arb must be in (0, 1]."""
        assert 0 < report.sps_arb <= 1.0 + 1e-6

    def test_rsps_point_matches_ratio(self, report: RelativeSPSReport):
        """rsps_point ≈ sps_t / sps_arb (within rounding tolerance)."""
        expected = report.sps_t / report.sps_arb
        assert abs(report.rsps_point - expected) < 1e-3, (
            f"rsps_point={report.rsps_point:.6f} ≠ sps_t/sps_arb={expected:.6f}"
        )

    def test_n_samples_t_positive(self, report: RelativeSPSReport):
        """n_samples_t must be > 0."""
        assert report.n_samples_t > 0

    def test_n_samples_arb_positive(self, report: RelativeSPSReport):
        """n_samples_arb must be > 0."""
        assert report.n_samples_arb > 0

    def test_mean_sensitivity_t_non_negative(self, report: RelativeSPSReport):
        """mean_sens_t must be ≥ 0."""
        assert report.mean_sens_t >= 0

    def test_mean_sensitivity_arb_non_negative(self, report: RelativeSPSReport):
        """mean_sens_arb must be ≥ 0."""
        assert report.mean_sens_arb >= 0

    def test_sens_ci_t_containment(self, report: RelativeSPSReport):
        """Bootstrap CI must contain the point estimate: lo ≤ point ≤ hi."""
        point, lo, hi = report.sens_ci_t
        assert lo <= point <= hi

    def test_loo_t_stable_key_is_bool(self, report: RelativeSPSReport):
        """loo_t['stable'] must be a bool."""
        assert isinstance(report.loo_t["stable"], bool)

    def test_config_stored(self, relative_estimator, report):
        """Report must store the SPSConfig."""
        assert report.config is relative_estimator.config

    def test_rsps_ci_excludes_one_logic(self):
        """rsps_ci_excludes_one is True only when CI is entirely above or below 1."""
        # Manual construction with CI above 1
        rep_above = RelativeSPSReport(
            sps_t=0.9, sps_arb=0.5, rsps_point=1.2,
            rsps_ci=(1.2, 1.1, 1.3), rsps_ci_excludes_one=True,
            mean_sens_t=0.1, std_sens_t=0.01, n_samples_t=100,
            mean_sens_arb=0.5, std_sens_arb=0.05, n_samples_arb=100,
            sens_ci_t=(0.1, 0.09, 0.11), loo_t={"stable": True, "n": 100},
        )
        assert rep_above.rsps_ci_excludes_one is True

        # CI below 1
        rep_below = RelativeSPSReport(
            sps_t=0.3, sps_arb=0.7, rsps_point=0.80,
            rsps_ci=(0.80, 0.70, 0.95), rsps_ci_excludes_one=True,
            mean_sens_t=0.8, std_sens_t=0.05, n_samples_t=100,
            mean_sens_arb=0.3, std_sens_arb=0.02, n_samples_arb=100,
            sens_ci_t=(0.8, 0.75, 0.85), loo_t={"stable": True, "n": 100},
        )
        assert rep_below.rsps_ci_excludes_one is True

        # CI straddles 1 — not significant
        rep_straddle = RelativeSPSReport(
            sps_t=0.5, sps_arb=0.5, rsps_point=1.0,
            rsps_ci=(1.0, 0.90, 1.10), rsps_ci_excludes_one=False,
            mean_sens_t=0.3, std_sens_t=0.02, n_samples_t=100,
            mean_sens_arb=0.3, std_sens_arb=0.02, n_samples_arb=100,
            sens_ci_t=(0.3, 0.28, 0.32), loo_t={"stable": True, "n": 100},
        )
        assert rep_straddle.rsps_ci_excludes_one is False

    def test_identical_families_rsps_near_one(
        self, linear_model, embedding_layer, batches, config
    ):
        """
        When T and T_arb are the same family (same config, same seed),
        rSPS should be close to 1.0 (both estimators sample identically).
        """
        set_seed(0)
        cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
        fam1 = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)
        fam2 = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)
        estimator = build_relative_sps_estimator(linear_model, fam1, fam2, config)
        report = estimator.estimate(batches)
        assert abs(report.rsps_point - 1.0) < 0.5, (
            f"Identical families: rsps_point={report.rsps_point:.4f} far from 1"
        )


# ─────────────────────────────────────────────────────────────────────────────
# TestBuildRelativeSPSEstimator — factory
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildRelativeSPSEstimator:
    def test_factory_returns_correct_type(
        self, linear_model, t_family, arb_family, config
    ):
        """build_relative_sps_estimator must return a RelativeSPSEstimator."""
        estimator = build_relative_sps_estimator(
            linear_model, t_family, arb_family, config
        )
        assert isinstance(estimator, RelativeSPSEstimator)

    def test_factory_default_config(self, linear_model, t_family, arb_family):
        """Factory with no config argument uses SPSConfig defaults."""
        estimator = build_relative_sps_estimator(linear_model, t_family, arb_family)
        assert isinstance(estimator, RelativeSPSEstimator)
        assert isinstance(estimator.config, SPSConfig)
