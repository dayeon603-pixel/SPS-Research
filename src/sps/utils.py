"""
Utility functions: divergence metrics, seed management, tensor helpers.

All divergences follow the signature (a: Tensor, b: Tensor) -> Tensor,
mapping (..., d) × (..., d) → (...).
"""
from __future__ import annotations

import logging
import random
from typing import Callable, Literal

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

DivergenceType = Literal["cosine", "l2"]
DivergenceFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def set_seed(seed: int = 42) -> None:
    """Set all random seeds for reproducibility (random, numpy, torch, cuda)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    logger.debug("Global seed set to %d.", seed)


def cosine_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute 1 - cos(a, b) along the last dimension.

    Args:
        a: Tensor of shape (..., d).
        b: Tensor of shape (..., d).

    Returns:
        Divergence values in [0, 2] of shape (...).
    """
    a_n = F.normalize(a, dim=-1)
    b_n = F.normalize(b, dim=-1)
    return 1.0 - (a_n * b_n).sum(dim=-1).clamp(-1.0, 1.0)


def l2_divergence(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute ||a - b||_2 along the last dimension.

    Args:
        a: Tensor of shape (..., d).
        b: Tensor of shape (..., d).

    Returns:
        L2 distances of shape (...).
    """
    return torch.norm(a - b, dim=-1)


def get_divergence_fn(name: DivergenceType) -> DivergenceFn:
    """
    Return a divergence callable by name.

    Args:
        name: One of "cosine" | "l2".

    Returns:
        The corresponding divergence function.

    Raises:
        ValueError: If name is not recognized.
    """
    match name:
        case "cosine":
            return cosine_divergence
        case "l2":
            return l2_divergence
        case _:
            raise ValueError(
                f"Unknown divergence '{name}'. Valid options: 'cosine', 'l2'."
            )


def normalize_directions(v: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Normalize direction vectors to unit norm over all non-batch dimensions.

    Args:
        v: Tensor of shape (B, *spatial_dims).
        eps: Numerical stability floor.

    Returns:
        Unit-normalized tensor of the same shape.
    """
    flat = v.flatten(start_dim=1)                              # (B, D)
    norm = flat.norm(dim=1, keepdim=True).clamp(min=eps)       # (B, 1)
    return (flat / norm).view_as(v)


def batch_dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Batched dot product over all dimensions except the first (batch).

    Args:
        a: Tensor of shape (B, *).
        b: Tensor of shape (B, *).

    Returns:
        Scalar dot products of shape (B,).
    """
    return (a.flatten(start_dim=1) * b.flatten(start_dim=1)).sum(dim=1)
