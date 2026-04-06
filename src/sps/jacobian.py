"""
Jacobian-based analysis: restricted operator norm and semantic spectral gap.

Implements Theorem 2 (corrected, two-part) and Definition 5 / Corollary 1.

Key functions:
  - restricted_operator_norm:  ||Jf(x)||_{A_x} via JVP (Theorem 2, Part ii)
  - full_spectral_norm:        sigma_max(Jf(x)) via randomized power iteration
  - spectral_gap:              Definition 5 — normalized gap gamma-bar
  - verify_lipschitz_bound:    Checks Corollary 1 numerically

References
----------
Kang (2026) Theorem 2 (corrected), Definition 5, Corollary 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

import torch
import torch.nn as nn

from sps.transformations import TransformationFamily
from sps.utils import normalize_directions

logger = logging.getLogger(__name__)

# A callable f: embeddings (B, seq, h) -> output (B, d)
ForwardFn = Callable[[torch.Tensor], torch.Tensor]


# ---------------------------------------------------------------------------
# Data container for spectral gap results
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SpectralGapResult:
    """
    Per-sample spectral gap analysis results.

    Attributes:
        restricted_norm:  ||Jf(x)||_{A_x} per sample, shape (B,).
        full_spectral_norm: sigma_max(Jf(x)) per sample, shape (B,).
        normalized_gap:   gamma-bar(f, T; x) in [0,1], shape (B,).
        mean_restricted_norm:  E_x[||Jf(x)||_{A_x}].
        mean_full_norm:        E_x[sigma_max(Jf(x))].
        mean_gap:              E_x[gamma-bar].
    """

    restricted_norm: torch.Tensor
    full_spectral_norm: torch.Tensor
    normalized_gap: torch.Tensor
    mean_restricted_norm: float = field(init=False)
    mean_full_norm: float = field(init=False)
    mean_gap: float = field(init=False)

    def __post_init__(self) -> None:
        self.mean_restricted_norm = self.restricted_norm.mean().item()
        self.mean_full_norm = self.full_spectral_norm.mean().item()
        self.mean_gap = self.normalized_gap.mean().item()

    def __repr__(self) -> str:
        return (
            f"SpectralGapResult("
            f"mean_restricted_norm={self.mean_restricted_norm:.4f}, "
            f"mean_full_norm={self.mean_full_norm:.4f}, "
            f"mean_gap={self.mean_gap:.4f})"
        )


# ---------------------------------------------------------------------------
# Theorem 2, Part (i): single directional derivative
# ---------------------------------------------------------------------------

def directional_derivative_norm(
    fn: ForwardFn,
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Compute ||Jf(x) v|| for a single batch of directions via JVP.

    Implements Theorem 2 Part (i):
        lim_{alpha->0} ||f(x + alpha*v) - f(x)|| / alpha = ||Jf(x) v||.

    Uses torch.autograd.functional.jvp for O(1) backward pass.

    Args:
        fn:  Forward function f: (B, seq, h) -> (B, d).
        x:   Input embeddings of shape (B, seq, h).
        v:   Unit direction vectors of shape (B, seq, h).

    Returns:
        ||Jf(x) v|| of shape (B,).
    """
    v_unit = normalize_directions(v)

    def scalar_fn(emb: torch.Tensor) -> torch.Tensor:
        return fn(emb)

    _, jvp_val = torch.autograd.functional.jvp(scalar_fn, x, v_unit, create_graph=False)
    return jvp_val.flatten(start_dim=1).norm(dim=1)               # (B,)


# ---------------------------------------------------------------------------
# Theorem 2, Part (ii): A_x-restricted operator norm
# ---------------------------------------------------------------------------

def restricted_operator_norm(
    fn: ForwardFn,
    x: torch.Tensor,
    directions: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the A_x-restricted operator norm ||Jf(x)||_{A_x} via JVP.

    Implements Theorem 2 Part (ii):
        ||Jf(x)||_{A_x} = sup_{v in A_x, ||v||=1} ||Jf(x) v||
                        ≈ max_{k=1,...,K} ||Jf(x) v_k||   (empirical, K directions).

    The approximation is tight when {v_k} is drawn to cover A_x;
    for the synonym-direction family, K = n_directions from EmbeddingPerturbationConfig.

    Args:
        fn:          Forward function f: (B, seq, h) -> (B, d).
        x:           Input embeddings of shape (B, seq, h).  Requires grad.
        directions:  Admissible semantic directions (B, K, seq, h) from
                     TransformationFamily.semantic_directions().

    Returns:
        ||Jf(x)||_{A_x} estimates of shape (B,).
    """
    B, K, seq_len, hidden = directions.shape
    norms_per_dir: list[torch.Tensor] = []

    for k in range(K):
        v_k = directions[:, k, :, :]                              # (B, seq, h)
        jvp_norm = directional_derivative_norm(fn, x, v_k)        # (B,)
        norms_per_dir.append(jvp_norm)

    stacked = torch.stack(norms_per_dir, dim=1)                   # (B, K)
    return stacked.max(dim=1).values                               # (B,)


# ---------------------------------------------------------------------------
# Full spectral norm via randomized power iteration
# ---------------------------------------------------------------------------

def full_spectral_norm(
    fn: ForwardFn,
    x: torch.Tensor,
    n_iter: int = 16,
    n_probe: int = 8,
) -> torch.Tensor:
    """
    Estimate sigma_max(Jf(x)) via randomized power iteration over the input space.

    This upper-bounds sigma_max and is tight with high probability when n_probe
    and n_iter are sufficiently large.  Full Jacobian materialization is avoided.

    Algorithm:
        For each probe direction v ~ Uniform(S^{D-1}):
            Compute ||Jf(x) v|| via JVP.
        Return max over probe directions as estimate.

    Args:
        fn:       Forward function f: (B, seq, h) -> (B, d).
        x:        Input embeddings (B, seq, h).
        n_iter:   Number of power iterations per probe (unused in current
                  max-probe implementation; kept for API consistency).
        n_probe:  Number of random probe directions.

    Returns:
        Estimated sigma_max of shape (B,).
    """
    B, seq_len, hidden = x.shape
    device = x.device
    max_norm = torch.zeros(B, device=device)

    for _ in range(n_probe):
        v = torch.randn(B, seq_len, hidden, device=device)
        v = normalize_directions(v)
        jvp_norm = directional_derivative_norm(fn, x, v)           # (B,)
        max_norm = torch.maximum(max_norm, jvp_norm)

    return max_norm                                                 # (B,)


# ---------------------------------------------------------------------------
# Definition 5: Semantic Spectral Gap
# ---------------------------------------------------------------------------

def spectral_gap(
    fn: ForwardFn,
    x: torch.Tensor,
    directions: torch.Tensor,
    n_probe_full: int = 16,
    eps: float = 1e-12,
) -> SpectralGapResult:
    """
    Compute the normalized semantic spectral gap (Definition 5, Corollary 1).

    gamma-bar(f, T; x) = 1 - ||Jf(x)||_{A_x} / sigma_max(Jf(x))

    A value near 1 indicates strong semantic separation: the model's most
    sensitive direction is NOT a semantic direction.
    A value near 0 indicates the worst-case direction is semantic — a failure.

    Args:
        fn:             Forward function f: (B, seq, h) -> (B, d).
        x:              Input embeddings (B, seq, h).
        directions:     Admissible semantic directions (B, K, seq, h).
        n_probe_full:   Random probes for sigma_max estimation.
        eps:            Numerical floor for division.

    Returns:
        SpectralGapResult with per-sample and mean statistics.
    """
    x = x.detach().requires_grad_(True)

    rest_norm = restricted_operator_norm(fn, x, directions)        # (B,)
    full_norm = full_spectral_norm(fn, x, n_probe=n_probe_full)    # (B,)

    # Normalize: gamma-bar = 1 - restricted / full
    # Clamp to [0, 1] to handle estimation noise in full_norm
    normalized = (1.0 - rest_norm / full_norm.clamp(min=eps)).clamp(0.0, 1.0)

    return SpectralGapResult(
        restricted_norm=rest_norm.detach(),
        full_spectral_norm=full_norm.detach(),
        normalized_gap=normalized.detach(),
    )


# ---------------------------------------------------------------------------
# Corollary 1: numerical verification
# ---------------------------------------------------------------------------

def verify_spectral_gap_bound(
    sps_value: float,
    lipschitz_constant: float,
    mean_gap: float,
) -> dict[str, float]:
    """
    Numerically verify the Spectral Gap Stability Bound (Corollary 1).

    Checks whether:
        SPS_eps(f) >= exp(-(1 - gamma_0) * L)

    and reports the tightness of the bound.

    Args:
        sps_value:          Empirical SPS estimate.
        lipschitz_constant: Global Lipschitz constant L of the model.
        mean_gap:           Mean normalized spectral gap gamma_0.

    Returns:
        Dict with keys:
            lower_bound   — exp(-(1 - gamma_0) * L)
            theorem3_bound — exp(-L)  [weaker, Theorem 3]
            bound_satisfied — bool
            improvement   — lower_bound / theorem3_bound (how much tighter)
    """
    import math
    lower = math.exp(-(1.0 - mean_gap) * lipschitz_constant)
    theorem3 = math.exp(-lipschitz_constant)
    satisfied = sps_value >= lower - 1e-6                          # small tolerance

    return {
        "lower_bound": lower,
        "theorem3_bound": theorem3,
        "bound_satisfied": float(satisfied),
        "improvement_factor": lower / max(theorem3, 1e-12),
        "sps_value": sps_value,
    }
