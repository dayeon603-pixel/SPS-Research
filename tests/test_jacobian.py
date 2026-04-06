"""
Tests for sps.jacobian: restricted operator norm and spectral gap.

Covers Theorem 2 (both parts) and Definition 5 / Corollary 1.
Uses analytically tractable linear models for exact verification.
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps.jacobian import (
    SpectralGapResult,
    directional_derivative_norm,
    full_spectral_norm,
    restricted_operator_norm,
    spectral_gap,
    verify_spectral_gap_bound,
)
from sps.utils import normalize_directions, set_seed


HIDDEN = 16
SEQ_LEN = 4
BATCH = 3


@pytest.fixture(autouse=True)
def fix_seed():
    set_seed(0)


# ---------------------------------------------------------------------------
# Analytically known linear model: f(E) = W * E[:, 0, :]
# Jacobian wrt E[:, 0, :] is W; wrt other positions is 0.
# ||Jf v||_2 = ||W v_0||_2  where v_0 = v[:, 0, :].
# ---------------------------------------------------------------------------

class ExactLinearModel(nn.Module):
    """f(E) = W * cls_token.  Jacobian is exactly W (wrt CLS position)."""

    def __init__(self, in_dim: int, out_dim: int, W: torch.Tensor) -> None:
        super().__init__()
        assert W.shape == (out_dim, in_dim)
        self.W = nn.Parameter(W, requires_grad=False)

    def forward(self, inputs_embeds: torch.Tensor, **kwargs) -> torch.Tensor:
        cls = inputs_embeds[:, 0, :]               # (B, in_dim)
        return cls @ self.W.T                      # (B, out_dim)


# ---------------------------------------------------------------------------
# Theorem 2 Part (i): directional derivative norm
# ---------------------------------------------------------------------------

class TestDirectionalDerivativeNorm:

    def test_linear_model_exact(self):
        """
        Theorem 2(i): For linear f(E) = W*E_cls and unit direction v at CLS,
        ||Jf(x) v|| = ||W v_cls||.
        """
        in_dim, out_dim = HIDDEN, 8
        W = torch.randn(out_dim, in_dim)
        model = ExactLinearModel(in_dim, out_dim, W)

        x = torch.randn(BATCH, SEQ_LEN, in_dim, requires_grad=True)

        # Direction only in CLS position
        v_raw = torch.zeros(BATCH, SEQ_LEN, in_dim)
        v_raw[:, 0, :] = torch.randn(BATCH, in_dim)
        v = normalize_directions(v_raw)

        def fn(emb: torch.Tensor) -> torch.Tensor:
            return model(emb)

        jvp_norm = directional_derivative_norm(fn, x, v)           # (B,)

        # Expected: ||W * v_cls||_2 for each sample
        v_cls = v[:, 0, :]                                         # (B, in_dim) unit vectors
        expected = (v_cls @ W.T).norm(dim=-1)                      # (B,)

        assert torch.allclose(jvp_norm, expected, atol=1e-5), \
            f"JVP norm mismatch.\nGot: {jvp_norm}\nExpected: {expected}"

    def test_zero_direction_gives_zero(self):
        """Direction v = 0 gives ||Jf(x) 0|| = 0."""
        in_dim, out_dim = HIDDEN, 8
        W = torch.randn(out_dim, in_dim)
        model = ExactLinearModel(in_dim, out_dim, W)
        x = torch.randn(BATCH, SEQ_LEN, in_dim)
        v = torch.zeros(BATCH, SEQ_LEN, in_dim)

        def fn(emb): return model(emb)

        # normalize_directions handles zero vectors gracefully (returns zeros)
        jvp_norm = directional_derivative_norm(fn, x, v)
        assert (jvp_norm < 1e-6).all()


# ---------------------------------------------------------------------------
# Theorem 2 Part (ii): restricted operator norm
# ---------------------------------------------------------------------------

class TestRestrictedOperatorNorm:

    def test_upper_bounded_by_full_spectral_norm(self):
        """||Jf(x)||_{A_x} <= sigma_max(Jf(x)) by definition."""
        in_dim, out_dim = HIDDEN, 8
        W = torch.randn(out_dim, in_dim)
        model = ExactLinearModel(in_dim, out_dim, W)
        x = torch.randn(BATCH, SEQ_LEN, in_dim)

        # Random semantic directions
        K = 6
        dirs = normalize_directions(
            torch.randn(BATCH, K, SEQ_LEN, in_dim).flatten(start_dim=2)
        ).view(BATCH, K, SEQ_LEN, in_dim)

        def fn(emb): return model(emb)

        rest_norm = restricted_operator_norm(fn, x, dirs)           # (B,)
        full_norm = full_spectral_norm(fn, x, n_probe=32)           # (B,)

        # Allow small tolerance from stochastic full_norm estimation
        assert (rest_norm <= full_norm + 0.1).all(), \
            f"Restricted norm exceeds full norm:\nrest={rest_norm}\nfull={full_norm}"

    def test_exact_linear_with_known_direction(self):
        """
        For linear model and direction aligned with largest singular vector of W,
        ||Jf||_{A_x} should equal sigma_max(W).
        """
        in_dim, out_dim = HIDDEN, 8
        W = torch.randn(out_dim, in_dim)
        model = ExactLinearModel(in_dim, out_dim, W)

        # Largest right singular vector of W
        _, _, Vt = torch.linalg.svd(W, full_matrices=False)
        v_top = Vt[0, :]                                            # (in_dim,)  leading r.s.v.

        x = torch.randn(BATCH, SEQ_LEN, in_dim)
        # Direction: put v_top in CLS position, zero elsewhere
        dirs = torch.zeros(BATCH, 1, SEQ_LEN, in_dim)
        dirs[:, 0, 0, :] = v_top.unsqueeze(0).expand(BATCH, -1)
        dirs = normalize_directions(dirs.flatten(start_dim=2)).view(BATCH, 1, SEQ_LEN, in_dim)

        def fn(emb): return model(emb)

        rest_norm = restricted_operator_norm(fn, x, dirs)           # (B,)
        sigma_max = torch.linalg.svdvals(W)[0].item()

        assert torch.allclose(rest_norm, torch.full((BATCH,), sigma_max), atol=1e-4), \
            f"Expected sigma_max={sigma_max:.4f}, got {rest_norm}"


# ---------------------------------------------------------------------------
# Definition 5: Spectral Gap
# ---------------------------------------------------------------------------

class TestSpectralGap:

    def test_gap_in_unit_interval(self):
        """gamma-bar in [0, 1] for any model (Definition 5)."""
        in_dim, out_dim = HIDDEN, 8
        W = torch.randn(out_dim, in_dim)
        model = ExactLinearModel(in_dim, out_dim, W)
        x = torch.randn(BATCH, SEQ_LEN, in_dim)
        K = 4
        dirs = normalize_directions(
            torch.randn(BATCH, K, SEQ_LEN, in_dim).flatten(start_dim=2)
        ).view(BATCH, K, SEQ_LEN, in_dim)

        def fn(emb): return model(emb)

        result = spectral_gap(fn, x, dirs)
        assert isinstance(result, SpectralGapResult)
        assert (result.normalized_gap >= 0.0).all()
        assert (result.normalized_gap <= 1.0 + 1e-6).all()

    def test_gap_zero_when_semantic_dir_is_top_singular(self):
        """
        When A_x contains the top singular direction of W, the restricted norm
        equals sigma_max, so gamma-bar ≈ 0.
        """
        in_dim, out_dim = HIDDEN, 8
        W = torch.randn(out_dim, in_dim)
        model = ExactLinearModel(in_dim, out_dim, W)

        _, _, Vt = torch.linalg.svd(W, full_matrices=False)
        v_top = Vt[0]

        x = torch.randn(BATCH, SEQ_LEN, in_dim)
        dirs = torch.zeros(BATCH, 1, SEQ_LEN, in_dim)
        dirs[:, 0, 0, :] = v_top.unsqueeze(0).expand(BATCH, -1)
        dirs = normalize_directions(dirs.flatten(start_dim=2)).view(BATCH, 1, SEQ_LEN, in_dim)

        def fn(emb): return model(emb)

        result = spectral_gap(fn, x, dirs, n_probe_full=64)
        # gamma-bar should be near 0 since the semantic direction IS the top singular direction
        assert result.mean_gap < 0.15, \
            f"Expected gap ≈ 0 when semantic dir = top s.v., got {result.mean_gap:.4f}"

    def test_gap_large_when_semantic_dir_is_null(self):
        """
        When semantic directions are in the null space of W, restricted norm ≈ 0
        and gamma-bar ≈ 1.
        """
        in_dim = 16
        out_dim = 4
        # Construct W with explicit null space
        W_small = torch.randn(out_dim, out_dim)
        # Pad with zeros so last (in_dim - out_dim) directions are null
        W = torch.zeros(out_dim, in_dim)
        W[:, :out_dim] = W_small

        model = ExactLinearModel(in_dim, out_dim, W)
        x = torch.randn(BATCH, SEQ_LEN, in_dim)

        # Directions entirely in null space (indices out_dim onwards)
        dirs = torch.zeros(BATCH, 4, SEQ_LEN, in_dim)
        for k in range(4):
            idx = out_dim + k
            dirs[:, k, 0, idx] = 1.0

        def fn(emb): return model(emb)

        result = spectral_gap(fn, x, dirs, n_probe_full=32)
        assert result.mean_gap > 0.7, \
            f"Expected gap ≈ 1 for null-space semantic dirs, got {result.mean_gap:.4f}"


# ---------------------------------------------------------------------------
# Corollary 1: verify_spectral_gap_bound
# ---------------------------------------------------------------------------

class TestVerifySpectralGapBound:

    def test_bound_structure(self):
        result = verify_spectral_gap_bound(
            sps_value=0.7,
            lipschitz_constant=1.0,
            mean_gap=0.3,
        )
        assert set(result.keys()) == {
            "lower_bound", "theorem3_bound", "bound_satisfied",
            "improvement_factor", "sps_value"
        }

    def test_tighter_than_theorem3(self):
        """Corollary 1 bound is strictly tighter than Theorem 3 when gap > 0."""
        result = verify_spectral_gap_bound(
            sps_value=0.9,
            lipschitz_constant=2.0,
            mean_gap=0.4,
        )
        assert result["lower_bound"] > result["theorem3_bound"], \
            "Spectral gap bound should be strictly tighter than Theorem 3."
        assert result["improvement_factor"] > 1.0

    def test_degenerates_to_theorem3_at_zero_gap(self):
        """At gap = 0, Corollary 1 reduces to Theorem 3."""
        result = verify_spectral_gap_bound(
            sps_value=0.5,
            lipschitz_constant=1.5,
            mean_gap=0.0,
        )
        assert math.isclose(result["lower_bound"], result["theorem3_bound"], rel_tol=1e-6)
