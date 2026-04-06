"""
Structured Perturbation Stability (SPS) — public API.

Reference: Kang (2026), Structured Perturbation Stability:
An Operator-Restricted Framework for Measuring Semantic Invariance
in Transformer Architectures.
"""
from __future__ import annotations

from sps.core import (
    SPSConfig,
    SPSEstimator,
    StructuredSensitivityEstimator,
    build_sps_estimator,
)
from sps.jacobian import (
    SpectralGapResult,
    directional_derivative_norm,
    full_spectral_norm,
    restricted_operator_norm,
    spectral_gap,
    verify_spectral_gap_bound,
)
from sps.metrics import (
    LayerwiseSPSAnalyzer,
    SPSReport,
    estimate_arbitrary_sps,
    full_sps_analysis,
    relative_sps,
)
from sps.transformations import (
    EmbeddingPerturbationConfig,
    EmbeddingPerturbationFamily,
    SynonymSubstitutionConfig,
    SynonymSubstitutionFamily,
    TransformationFamily,
    build_wordnet_synonym_map,
)
from sps.utils import (
    cosine_divergence,
    get_divergence_fn,
    l2_divergence,
    normalize_directions,
    set_seed,
)

__all__ = [
    # core
    "SPSConfig",
    "SPSEstimator",
    "StructuredSensitivityEstimator",
    "build_sps_estimator",
    # jacobian
    "SpectralGapResult",
    "directional_derivative_norm",
    "full_spectral_norm",
    "restricted_operator_norm",
    "spectral_gap",
    "verify_spectral_gap_bound",
    # metrics
    "LayerwiseSPSAnalyzer",
    "SPSReport",
    "estimate_arbitrary_sps",
    "full_sps_analysis",
    "relative_sps",
    # transformations
    "EmbeddingPerturbationConfig",
    "EmbeddingPerturbationFamily",
    "SynonymSubstitutionConfig",
    "SynonymSubstitutionFamily",
    "TransformationFamily",
    "build_wordnet_synonym_map",
    # utils
    "cosine_divergence",
    "get_divergence_fn",
    "l2_divergence",
    "normalize_directions",
    "set_seed",
]
