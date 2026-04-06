"""
Tests for sps.core: StructuredSensitivityEstimator and SPSEstimator.

Covers Proposition 1 (boundedness), Definition 1 (sensitivity = 0 for constant model),
and Definition 2 (SPS = 1 for invariant model).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps.core import SPSConfig, SPSEstimator, StructuredSensitivityEstimator, build_sps_estimator
from sps.transformations import EmbeddingPerturbationConfig, EmbeddingPerturbationFamily
from sps.utils import cosine_divergence, set_seed


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

HIDDEN = 32
SEQ_LEN = 8
BATCH = 4
VOCAB = 100


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
def input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


@pytest.fixture
def embeddings(embedding_layer, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return embedding_layer(input_ids)


@pytest.fixture
def attention_mask(input_ids) -> torch.Tensor:
    return torch.ones_like(input_ids)


# ---------------------------------------------------------------------------
# Dummy models
# ---------------------------------------------------------------------------

class ConstantModel(nn.Module):
    """Always returns a fixed vector — zero sensitivity, SPS = 1."""
    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.out = nn.Parameter(torch.randn(hidden), requires_grad=False)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        B = inputs_embeds.size(0)
        hidden = torch.zeros(B, inputs_embeds.size(1), self.out.size(0))
        hidden[:, 0, :] = self.out.unsqueeze(0).expand(B, -1)

        class _Out:
            last_hidden_state = hidden
        return _Out()


class LinearModel(nn.Module):
    """Linear projection on CLS token — known sensitivity."""
    def __init__(self, hidden: int, out_dim: int = 16) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden, out_dim, bias=False)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor):
        cls = inputs_embeds[:, 0, :]                               # (B, h)
        out = self.proj(cls)                                       # (B, out_dim)

        class _Out:
            last_hidden_state = inputs_embeds  # unused; we override below

        # Return shaped as (B, seq, out_dim) so CLS extraction still works
        full = torch.zeros(inputs_embeds.size(0), inputs_embeds.size(1), 16)
        full[:, 0, :] = out

        class _Out2:
            last_hidden_state = full
        return _Out2()


# ---------------------------------------------------------------------------
# Tests: StructuredSensitivityEstimator
# ---------------------------------------------------------------------------

class TestStructuredSensitivityEstimator:

    def test_zero_sensitivity_for_constant_model(self, t_emb, embeddings, input_ids, attention_mask):
        """Proposition 1 special case: constant model → Sens = 0."""
        model = ConstantModel(hidden=HIDDEN)
        estimator = StructuredSensitivityEstimator(
            model=model,
            transform_family=t_emb,
            divergence_fn=cosine_divergence,
            epsilon=0.1,
        )
        sens = estimator.estimate(embeddings, input_ids, attention_mask, m=8)
        assert sens.shape == (BATCH,)
        # Constant model: f(x) = f(Tx) always, so divergence = 0
        assert (sens < 1e-6).all(), f"Expected ~0 sensitivity, got {sens}"

    def test_sensitivity_nonnegative(self, t_emb, embeddings, input_ids, attention_mask):
        """Sensitivity is always >= 0 (Proposition 1 prerequisite)."""
        model = LinearModel(HIDDEN)
        estimator = StructuredSensitivityEstimator(
            model=model,
            transform_family=t_emb,
            divergence_fn=cosine_divergence,
            epsilon=0.1,
        )
        sens = estimator.estimate(embeddings, input_ids, attention_mask, m=8)
        assert (sens >= 0).all()

    def test_sensitivity_increases_with_epsilon(self, t_emb, embedding_layer, attention_mask):
        """Larger epsilon -> larger or equal max sensitivity (by superset argument)."""
        set_seed(1)
        input_ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
        with torch.no_grad():
            emb = embedding_layer(input_ids)

        model = LinearModel(HIDDEN)
        sens_small = StructuredSensitivityEstimator(
            model=model, transform_family=t_emb, divergence_fn=cosine_divergence, epsilon=0.05
        ).estimate(emb, input_ids, attention_mask, m=16)

        sens_large = StructuredSensitivityEstimator(
            model=model, transform_family=t_emb, divergence_fn=cosine_divergence, epsilon=0.3
        ).estimate(emb, input_ids, attention_mask, m=16)

        # On average, larger epsilon should not give strictly smaller sensitivity
        assert sens_large.mean() >= sens_small.mean() - 1e-4


# ---------------------------------------------------------------------------
# Tests: SPSEstimator
# ---------------------------------------------------------------------------

class TestSPSEstimator:

    def _make_batches(self, embedding_layer, n_batches: int = 3) -> list[dict]:
        batches = []
        for _ in range(n_batches):
            ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
            mask = torch.ones_like(ids)
            with torch.no_grad():
                emb = embedding_layer(ids)
            batches.append({"input_ids": ids, "attention_mask": mask, "embeddings": emb})
        return batches

    def test_sps_in_open_unit_interval(self, t_emb, embedding_layer):
        """Proposition 1: 0 < SPS <= 1."""
        batches = self._make_batches(embedding_layer)
        model = LinearModel(HIDDEN)
        config = SPSConfig(epsilon=0.1, m_transforms=8)
        estimator = build_sps_estimator(model, t_emb, config)
        result = estimator.estimate(iter(batches))

        assert 0.0 < result["sps"] <= 1.0 + 1e-9

    def test_sps_equals_one_for_constant_model(self, t_emb, embedding_layer):
        """
        Theorem 5: invariant representation -> SPS = 1.
        Constant model is trivially invariant.
        """
        batches = self._make_batches(embedding_layer)
        model = ConstantModel(hidden=HIDDEN)
        config = SPSConfig(epsilon=0.1, m_transforms=8)
        estimator = build_sps_estimator(model, t_emb, config)
        result = estimator.estimate(iter(batches))

        assert math.isclose(result["sps"], 1.0, abs_tol=1e-6), \
            f"Expected SPS=1 for constant model, got {result['sps']}"

    def test_sps_keys_present(self, t_emb, embedding_layer):
        batches = self._make_batches(embedding_layer)
        model = LinearModel(HIDDEN)
        config = SPSConfig(epsilon=0.1, m_transforms=4)
        estimator = build_sps_estimator(model, t_emb, config)
        result = estimator.estimate(iter(batches))

        assert set(result.keys()) == {"sps", "mean_sensitivity", "std_sensitivity", "n_samples"}
        assert result["n_samples"] == BATCH * 3

    def test_family_monotonicity_sps(self, embedding_layer, attention_mask):
        """
        Theorem 1: T1 ⊆ T2 -> SPS(T1) >= SPS(T2).
        Approximated by comparing small vs. large n_directions.
        """
        set_seed(2)
        batches_small, batches_large = [], []
        for _ in range(4):
            ids = torch.randint(0, VOCAB, (BATCH, SEQ_LEN))
            mask = torch.ones_like(ids)
            with torch.no_grad():
                emb = embedding_layer(ids)
            b = {"input_ids": ids, "attention_mask": mask, "embeddings": emb}
            batches_small.append(b)
            batches_large.append(b)

        model = LinearModel(HIDDEN)

        cfg_small = EmbeddingPerturbationConfig(n_directions=2, use_synonym_directions=False)
        t_small = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg_small)

        cfg_large = EmbeddingPerturbationConfig(n_directions=16, use_synonym_directions=False)
        t_large = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg_large)

        config = SPSConfig(epsilon=0.1, m_transforms=16, seed=0)

        sps_small = build_sps_estimator(model, t_small, config).estimate(iter(batches_small))["sps"]
        sps_large = build_sps_estimator(model, t_large, config).estimate(iter(batches_large))["sps"]

        # SPS with larger family should be <= SPS with smaller family (or close)
        assert sps_large <= sps_small + 0.05, \
            f"Monotonicity violated: SPS(large)={sps_large:.4f} > SPS(small)={sps_small:.4f}"
