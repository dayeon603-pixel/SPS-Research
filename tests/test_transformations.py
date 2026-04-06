"""
Tests for sps.transformations: EmbeddingPerturbationFamily and SynonymSubstitutionFamily.

Verifies perturbation constraints, magnitude computation, and direction properties.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps.transformations import (
    EmbeddingPerturbationConfig,
    EmbeddingPerturbationFamily,
    SynonymSubstitutionConfig,
    SynonymSubstitutionFamily,
)
from sps.utils import set_seed


HIDDEN = 32
SEQ_LEN = 8
BATCH = 4
VOCAB = 200


@pytest.fixture(autouse=True)
def fix_seed():
    set_seed(0)


@pytest.fixture
def embedding_layer() -> nn.Embedding:
    emb = nn.Embedding(VOCAB, HIDDEN)
    nn.init.normal_(emb.weight, std=0.1)
    return emb


@pytest.fixture
def input_ids() -> torch.Tensor:
    return torch.randint(0, VOCAB, (BATCH, SEQ_LEN))


@pytest.fixture
def embeddings(embedding_layer, input_ids) -> torch.Tensor:
    with torch.no_grad():
        return embedding_layer(input_ids)


@pytest.fixture
def t_emb(embedding_layer) -> EmbeddingPerturbationFamily:
    cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
    return EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)


@pytest.fixture
def synonym_map() -> dict[int, list[int]]:
    """Small synthetic synonym map: each token maps to [token+1, token+2] mod VOCAB."""
    return {i: [(i + 1) % VOCAB, (i + 2) % VOCAB] for i in range(VOCAB)}


@pytest.fixture
def t_syn(embedding_layer, synonym_map) -> SynonymSubstitutionFamily:
    cfg = SynonymSubstitutionConfig(substitution_prob=0.5, max_substitutions=3)
    return SynonymSubstitutionFamily(
        embedding_layer=embedding_layer,
        synonym_map=synonym_map,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# EmbeddingPerturbationFamily
# ---------------------------------------------------------------------------

class TestEmbeddingPerturbationFamily:

    def test_output_shapes(self, t_emb, embeddings, input_ids):
        perturbed, magnitudes = t_emb.sample(embeddings, input_ids, epsilon=0.1)
        assert perturbed.shape == embeddings.shape
        assert magnitudes.shape == (BATCH,)

    def test_magnitudes_within_epsilon(self, t_emb, embeddings, input_ids):
        """c(T, x) <= epsilon for all samples."""
        epsilon = 0.1
        _, magnitudes = t_emb.sample(embeddings, input_ids, epsilon)
        assert (magnitudes <= epsilon + 1e-6).all(), \
            f"Some magnitudes exceed epsilon: {magnitudes[magnitudes > epsilon + 1e-6]}"

    def test_magnitudes_positive(self, t_emb, embeddings, input_ids):
        """c(T, x) > 0: non-degenerate perturbation."""
        _, magnitudes = t_emb.sample(embeddings, input_ids, epsilon=0.5)
        assert (magnitudes > 1e-8).all(), f"Zero magnitudes found: {magnitudes}"

    def test_directions_shape(self, t_emb, embeddings, input_ids):
        """semantic_directions returns (B, K, seq_len, hidden)."""
        K = 4
        dirs = t_emb.semantic_directions(embeddings, input_ids)
        assert dirs.shape == (BATCH, K, SEQ_LEN, HIDDEN)

    def test_directions_unit_norm(self, t_emb, embeddings, input_ids):
        """Each direction vector should have unit Frobenius norm."""
        dirs = t_emb.semantic_directions(embeddings, input_ids)   # (B, K, seq, h)
        flat = dirs.flatten(start_dim=2)                           # (B, K, seq*h)
        norms = flat.norm(dim=-1)                                  # (B, K)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
            f"Non-unit directions found: min={norms.min():.4f}, max={norms.max():.4f}"

    def test_perturbation_is_deterministic_with_seed(self, embedding_layer, input_ids):
        """Same seed -> same perturbation."""
        cfg = EmbeddingPerturbationConfig(n_directions=4, use_synonym_directions=False)
        with torch.no_grad():
            emb = embedding_layer(input_ids)

        set_seed(42)
        t1 = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)
        p1, m1 = t1.sample(emb, input_ids, epsilon=0.1)

        set_seed(42)
        t2 = EmbeddingPerturbationFamily(embedding_layer=embedding_layer, config=cfg)
        p2, m2 = t2.sample(emb, input_ids, epsilon=0.1)

        assert torch.allclose(p1, p2), "Non-deterministic perturbation with same seed."
        assert torch.allclose(m1, m2)

    def test_epsilon_zero_gives_no_perturbation(self, t_emb, embeddings, input_ids):
        """epsilon -> 0 gives magnitudes -> 0."""
        _, mags = t_emb.sample(embeddings, input_ids, epsilon=1e-9)
        assert (mags < 1e-6).all()

    def test_synonym_directions_shape(self, embedding_layer, input_ids):
        """When synonym_map is provided, synonym directions are returned."""
        syn_map = {i: [(i + 1) % VOCAB] for i in range(VOCAB)}
        cfg = EmbeddingPerturbationConfig(n_directions=3, use_synonym_directions=True)
        t = EmbeddingPerturbationFamily(
            embedding_layer=embedding_layer, synonym_map=syn_map, config=cfg
        )
        with torch.no_grad():
            emb = embedding_layer(input_ids)
        dirs = t.semantic_directions(emb, input_ids)
        assert dirs.shape == (BATCH, 3, SEQ_LEN, HIDDEN)


# ---------------------------------------------------------------------------
# SynonymSubstitutionFamily
# ---------------------------------------------------------------------------

class TestSynonymSubstitutionFamily:

    def test_output_shapes(self, t_syn, embeddings, input_ids):
        perturbed, magnitudes = t_syn.sample(embeddings, input_ids, epsilon=1.0)
        assert perturbed.shape == embeddings.shape
        assert magnitudes.shape == (BATCH,)

    def test_rejects_large_perturbations(self, t_syn, embeddings, input_ids):
        """Samples exceeding epsilon are masked to zero magnitude."""
        epsilon = 1e-9                                             # almost nothing allowed
        _, magnitudes = t_syn.sample(embeddings, input_ids, epsilon)
        # All should be zero (perturbations rejected)
        assert (magnitudes < 1e-6).all(), \
            f"Expected all rejected at epsilon=1e-9, got {magnitudes}"

    def test_no_substitution_when_no_synonyms(self, embedding_layer, input_ids):
        """Empty synonym map raises ValueError (Assumption A5 violation)."""
        with pytest.raises(ValueError, match="non-empty synonym_map"):
            SynonymSubstitutionFamily(
                embedding_layer=embedding_layer,
                synonym_map={},
            )

    def test_perturbed_differs_from_original(self, t_syn, embeddings, input_ids):
        """At least some samples should be genuinely perturbed."""
        perturbed, magnitudes = t_syn.sample(embeddings, input_ids, epsilon=10.0)
        valid = magnitudes > 1e-8
        if valid.any():
            diff = (perturbed[valid] - embeddings[valid]).abs().sum()
            assert diff > 1e-6, "Perturbed embeddings identical to originals despite nonzero magnitude."
