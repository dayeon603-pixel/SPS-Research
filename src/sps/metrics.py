"""
High-level SPS metrics: relative SPS, layer-wise SPS profile, and summary reports.

Implements Definitions 7, 8 and provides a unified SPSReport dataclass.

References
----------
Kang (2026) Definitions 7, 8; Proposition 2.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch
import torch.nn as nn

from sps.core import SPSConfig, SPSEstimator, StructuredSensitivityEstimator, build_sps_estimator
from sps.jacobian import SpectralGapResult, spectral_gap
from sps.transformations import EmbeddingPerturbationFamily, TransformationFamily
from sps.utils import get_divergence_fn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

@dataclass
class SPSReport:
    """
    Full SPS analysis report for a model and transformation family.

    Attributes:
        sps:               SPS_eps(f) in (0, 1].
        mean_sensitivity:  E_x[Sens_{T,eps}(f; x)].
        std_sensitivity:   Std_x[Sens_{T,eps}(f; x)].
        n_samples:         Number of data samples used.
        relative_sps:      rSPS (Definition 8). None if arb_sps not provided.
        spectral_gap:      Mean normalized spectral gap gamma-bar. None if not computed.
        layerwise_profile: SPS_eps^{(l)} for l=0,...,L. None if not computed.
        config:            The SPSConfig used.
    """

    sps: float
    mean_sensitivity: float
    std_sensitivity: float
    n_samples: int
    relative_sps: Optional[float] = None
    spectral_gap_mean: Optional[float] = None
    spectral_gap_result: Optional[SpectralGapResult] = None
    layerwise_profile: Optional[list[float]] = None
    config: Optional[SPSConfig] = None

    def summary(self) -> str:
        """Return a formatted one-page summary of results."""
        lines = [
            "=" * 60,
            "  Structured Perturbation Stability (SPS) Report",
            "=" * 60,
            f"  SPS_eps(f)          : {self.sps:.4f}",
            f"  Mean Sensitivity    : {self.mean_sensitivity:.4f}",
            f"  Std  Sensitivity    : {self.std_sensitivity:.4f}",
            f"  N Samples           : {self.n_samples}",
        ]
        if self.relative_sps is not None:
            interpretation = (
                "> 1 (semantic > arbitrary stability)"
                if self.relative_sps > 1.0
                else "= 1 (indistinguishable)" if abs(self.relative_sps - 1.0) < 1e-3
                else "< 1 (FAILURE: higher semantic sensitivity)"
            )
            lines.append(f"  rSPS (Def 8)        : {self.relative_sps:.4f}  {interpretation}")
        if self.spectral_gap_mean is not None:
            lines.append(f"  Spectral Gap gamma-bar : {self.spectral_gap_mean:.4f}")
        if self.layerwise_profile is not None:
            profile_str = "  ".join(f"L{i}:{v:.3f}" for i, v in enumerate(self.layerwise_profile))
            lines.append(f"  Layerwise Profile   : {profile_str}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relative SPS (Definition 8)
# ---------------------------------------------------------------------------

def relative_sps(
    semantic_sps: float,
    arbitrary_sps: float,
    eps: float = 1e-12,
) -> float:
    """
    Compute rSPS = SPS^{(T)}(f) / SPS^{(T_arb)}(f)  (Definition 8).

    Args:
        semantic_sps:  SPS under the semantic transformation family T.
        arbitrary_sps: SPS under arbitrary l2-ball perturbations T_arb.
        eps:           Numerical floor for division.

    Returns:
        rSPS in (0, inf).
    """
    if arbitrary_sps < eps:
        raise ValueError(
            "arbitrary_sps is effectively zero; rSPS is undefined. "
            "Check that T_arb is correctly configured."
        )
    return semantic_sps / arbitrary_sps


def estimate_arbitrary_sps(
    model: nn.Module,
    data_iterator: Iterator[dict[str, torch.Tensor]],
    config: SPSConfig,
) -> float:
    """
    Estimate SPS under arbitrary l2-ball perturbations T_arb (the denominator of rSPS).

    Uses random isotropic Gaussian perturbations in embedding space, normalized
    to have magnitude uniformly in (0, epsilon].

    Args:
        model:         Transformer model.
        data_iterator: Batches with input_ids, attention_mask, embeddings.
        config:        SPSConfig.

    Returns:
        SPS_eps^{(T_arb)}(f).
    """
    from sps.transformations import EmbeddingPerturbationConfig

    arb_config = EmbeddingPerturbationConfig(
        n_directions=config.m_transforms,
        use_synonym_directions=False,       # random orthogonal directions = T_arb
    )
    embedding_layer = _get_embedding_layer(model)
    arb_family = EmbeddingPerturbationFamily(
        embedding_layer=embedding_layer,
        synonym_map=None,
        config=arb_config,
    )
    estimator = build_sps_estimator(model, arb_family, config)
    result = estimator.estimate(data_iterator)
    return result["sps"]


# ---------------------------------------------------------------------------
# Layer-wise SPS (Definition 7)
# ---------------------------------------------------------------------------

class LayerwiseSPSAnalyzer:
    """
    Compute the SPS depth profile {SPS_eps^{(l)}}_{l=0}^{L} (Definition 7).

    For each transformer layer l, registers a forward hook to capture the
    CLS token representation at that depth, then computes SPS on the
    truncated-at-depth-l representation function.

    Args:
        model:            HuggingFace transformer model with .encoder.layer or
                          .transformer.h attribute for layer enumeration.
        transform_family: Admissible transformation family T.
        config:           SPSConfig.
    """

    def __init__(
        self,
        model: nn.Module,
        transform_family: TransformationFamily,
        config: SPSConfig,
    ) -> None:
        self.model = model
        self.transform_family = transform_family
        self.config = config
        self._layers = self._find_layers(model)

        if not self._layers:
            raise ValueError(
                "Could not enumerate transformer layers from model. "
                "Ensure model has .encoder.layer, .transformer.h, or .layers attribute."
            )

    def compute_profile(
        self,
        data_iterator: Iterator[dict[str, torch.Tensor]],
    ) -> list[float]:
        """
        Compute SPS at each layer depth.

        Args:
            data_iterator: Iterable of batches (consumed once per layer).
                           NOTE: the iterator must be re-creatable; pass a
                           factory function and call it per layer, or preload
                           all batches into memory.

        Returns:
            SPS values [SPS^{(0)}, SPS^{(1)}, ..., SPS^{(L)}] of length L+1.
        """
        raise NotImplementedError(
            "LayerwiseSPSAnalyzer.compute_profile requires a re-iterable data source. "
            "Use compute_profile_from_batches() with preloaded data."
        )

    def compute_profile_from_batches(
        self,
        batches: list[dict[str, torch.Tensor]],
    ) -> list[float]:
        """
        Compute SPS depth profile from a preloaded list of batches.

        Args:
            batches: List of batch dicts (preloaded into memory).

        Returns:
            List of SPS values, one per layer including the embedding layer (l=0).
        """
        profile: list[float] = []

        for layer_idx, layer_module in enumerate(self._layers):
            logger.info("Computing SPS at layer %d / %d ...", layer_idx, len(self._layers) - 1)
            sps_l = self._compute_sps_at_layer(batches, layer_module, layer_idx)
            profile.append(sps_l)
            logger.info("  Layer %d SPS = %.4f", layer_idx, sps_l)

        return profile

    def _compute_sps_at_layer(
        self,
        batches: list[dict[str, torch.Tensor]],
        target_layer: nn.Module,
        layer_idx: int,
    ) -> float:
        """
        Compute SPS using the representation captured at target_layer.

        Uses a forward hook to extract intermediate representations.
        """
        captured: dict[str, Optional[torch.Tensor]] = {"hidden": None}

        def hook(module: nn.Module, input: tuple, output) -> None:
            # output may be a tuple (hidden_state, ...) for transformer layers
            if isinstance(output, tuple):
                captured["hidden"] = output[0][:, 0, :].detach()  # CLS token
            elif isinstance(output, torch.Tensor):
                if output.dim() == 3:
                    captured["hidden"] = output[:, 0, :].detach()
                else:
                    captured["hidden"] = output.detach()

        handle = target_layer.register_forward_hook(hook)

        def layer_fn(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """Forward pass that returns representation at layer_idx."""
            _ = self.model(inputs_embeds=embeddings, attention_mask=attention_mask)
            assert captured["hidden"] is not None, "Hook did not fire."
            return captured["hidden"]

        # Build a thin nn.Module wrapper for StructuredSensitivityEstimator
        class _LayerProxy(nn.Module):
            def forward(self_, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:  # noqa: N805
                return layer_fn(inputs_embeds, attention_mask)

        proxy = _LayerProxy()
        divergence_fn = get_divergence_fn(self.config.divergence)
        sens_est = StructuredSensitivityEstimator(
            model=proxy,
            transform_family=self.transform_family,
            divergence_fn=divergence_fn,
            epsilon=self.config.epsilon,
            device=self.config.device,
        )
        estimator = SPSEstimator(sensitivity_estimator=sens_est, config=self.config)

        result = estimator.estimate(iter(batches))
        handle.remove()
        return result["sps"]

    @staticmethod
    def _find_layers(model: nn.Module) -> list[nn.Module]:
        """
        Enumerate transformer layers from a HuggingFace model.

        Tries common attribute names across BERT/RoBERTa/GPT-2 family models.
        """
        # RoBERTa / BERT
        for attr in ("roberta", "bert"):
            sub = getattr(model, attr, None)
            if sub is not None:
                encoder = getattr(sub, "encoder", None)
                if encoder is not None:
                    layers = getattr(encoder, "layer", None)
                    if layers is not None:
                        return list(layers)
        # GPT-2
        transformer = getattr(model, "transformer", None)
        if transformer is not None:
            h = getattr(transformer, "h", None)
            if h is not None:
                return list(h)
        # Generic
        for attr in ("encoder", "layers", "blocks"):
            sub = getattr(model, attr, None)
            if isinstance(sub, nn.ModuleList):
                return list(sub)
        return []


# ---------------------------------------------------------------------------
# Convenience: single-call full analysis
# ---------------------------------------------------------------------------

def full_sps_analysis(
    model: nn.Module,
    transform_family: TransformationFamily,
    batches: list[dict[str, torch.Tensor]],
    config: Optional[SPSConfig] = None,
    compute_spectral_gap: bool = True,
    compute_layerwise: bool = True,
    compute_relative: bool = True,
) -> SPSReport:
    """
    Run the complete SPS analysis pipeline and return a unified SPSReport.

    Args:
        model:                 HuggingFace transformer model.
        transform_family:      Semantic transformation family T.
        batches:               Preloaded list of data batches.
        config:                SPSConfig (defaults if None).
        compute_spectral_gap:  Whether to compute Definition 5.
        compute_layerwise:     Whether to compute Definition 7.
        compute_relative:      Whether to compute Definition 8.

    Returns:
        SPSReport with all available metrics populated.
    """
    config = config or SPSConfig()

    # --- Core SPS (Definition 2) ---
    estimator = build_sps_estimator(model, transform_family, config)
    core_result = estimator.estimate(iter(batches))

    report = SPSReport(
        sps=core_result["sps"],
        mean_sensitivity=core_result["mean_sensitivity"],
        std_sensitivity=core_result["std_sensitivity"],
        n_samples=core_result["n_samples"],
        config=config,
    )

    # --- Spectral Gap (Definition 5) ---
    if compute_spectral_gap:
        logger.info("Computing spectral gap ...")
        batch = batches[0]
        emb = batch.get("embeddings")
        if emb is None:
            raise ValueError("Spectral gap requires 'embeddings' in the first batch.")

        emb = emb.to(config.device)
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)

        directions = transform_family.semantic_directions(emb, input_ids)

        def _fn(inputs_embeds: torch.Tensor) -> torch.Tensor:
            out = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state[:, 0, :]
            return out[:, 0, :] if out.dim() == 3 else out

        gap_result = spectral_gap(_fn, emb, directions)
        report.spectral_gap_mean = gap_result.mean_gap
        report.spectral_gap_result = gap_result

    # --- Relative SPS (Definition 8) ---
    if compute_relative:
        logger.info("Computing relative SPS (arbitrary perturbation baseline) ...")
        arb_sps = estimate_arbitrary_sps(model, iter(batches), config)
        report.relative_sps = relative_sps(report.sps, arb_sps)

    # --- Layer-wise SPS (Definition 7) ---
    if compute_layerwise:
        logger.info("Computing layerwise SPS profile ...")
        analyzer = LayerwiseSPSAnalyzer(model, transform_family, config)
        report.layerwise_profile = analyzer.compute_profile_from_batches(batches)

    return report


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_embedding_layer(model: nn.Module) -> nn.Embedding:
    """Extract the word embedding layer from a HuggingFace model."""
    get_emb = getattr(model, "get_input_embeddings", None)
    if callable(get_emb):
        return get_emb()
    for attr in ("roberta", "bert", "transformer", "model"):
        sub = getattr(model, attr, None)
        if sub is not None:
            emb = getattr(sub, "embeddings", None)
            if emb is not None:
                we = getattr(emb, "word_embeddings", None)
                if we is not None:
                    return we
    raise AttributeError("Cannot locate word embedding layer on model.")
