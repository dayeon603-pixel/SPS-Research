"""
Core SPS computation: Definitions 1 & 2, empirical estimator (Definition 6).

Implements:
  - StructuredSensitivityEstimator: pointwise Sens_{T,eps}(f; x) (Def 1)
  - SPSEstimator:                   global SPS_eps(f)             (Def 2)
  - SPSConfig:                      pydantic v2 configuration model

References
----------
Kang (2026) Definitions 1, 2, 6; Proposition 1; Assumptions A3, A4.
"""
from __future__ import annotations

import logging
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn
from pydantic import BaseModel, Field, field_validator

from sps.transformations import TransformationFamily
from sps.utils import DivergenceFn, DivergenceType, get_divergence_fn, set_seed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class SPSConfig(BaseModel):
    """
    Configuration for SPS estimation.

    Attributes:
        epsilon:           Perturbation radius for Definitions 1 & 2.
        n_data_samples:    n in the empirical estimator (Definition 6).
        m_transforms:      m in the empirical estimator (Definition 6).
        divergence:        Output space metric d_Y.
        device:            Torch device string.
        seed:              Random seed for reproducibility.
        batch_size:        Batch size for data processing.
    """

    epsilon: float = Field(0.1, gt=0.0, description="Perturbation radius epsilon.")
    n_data_samples: int = Field(512, ge=1)
    m_transforms: int = Field(32, ge=1, description="Transformations sampled per input.")
    divergence: DivergenceType = "cosine"
    device: str = "cpu"
    seed: int = 42
    batch_size: int = Field(32, ge=1)

    @field_validator("device")
    @classmethod
    def validate_device(cls, v: str) -> str:
        try:
            torch.device(v)
        except RuntimeError as e:
            raise ValueError(f"Invalid torch device '{v}': {e}") from e
        return v


# ---------------------------------------------------------------------------
# Structured Sensitivity Estimator  (Definition 1)
# ---------------------------------------------------------------------------

class StructuredSensitivityEstimator:
    """
    Empirical estimator for Definition 1 (Structured Local Sensitivity).

    Approximates:
        Sens_{T,eps}^{(m)}(f; x) = max_{j=1,...,m}  d_Y(f(x), f(tau_j(x))) / c(tau_j, x)

    by sampling m transformations from T for each input x.

    Args:
        model:           Callable mapping embeddings (B, seq, h) -> representations (B, d).
                         Must accept keyword argument `inputs_embeds`.
        transform_family: The admissible transformation family T.
        divergence_fn:   Divergence d_Y on output space Y.
        epsilon:         Perturbation radius.
        device:          Torch device.
    """

    def __init__(
        self,
        model: nn.Module,
        transform_family: TransformationFamily,
        divergence_fn: DivergenceFn,
        epsilon: float,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.transform_family = transform_family
        self.divergence_fn = divergence_fn
        self.epsilon = epsilon
        self.device = device

    @torch.no_grad()
    def estimate(
        self,
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        m: int = 32,
    ) -> torch.Tensor:
        """
        Estimate pointwise structured sensitivity for a batch.

        Args:
            embeddings:     (B, seq_len, hidden_dim).
            input_ids:      (B, seq_len).
            attention_mask: (B, seq_len).
            m:              Number of transformations to sample.

        Returns:
            Sensitivity estimates of shape (B,), each in [0, inf).
        """
        embeddings = embeddings.to(self.device)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        B = embeddings.size(0)

        fx = self._forward(embeddings, attention_mask)             # (B, d)
        max_ratios = torch.zeros(B, device=self.device)

        for _ in range(m):
            tx, cx = self.transform_family.sample(
                embeddings, input_ids, self.epsilon
            )                                                      # (B, seq, h), (B,)
            valid = cx > 1e-12                                     # mask: no valid transform

            if not valid.any():
                continue

            ftx = self._forward(tx, attention_mask)                # (B, d)
            div = self.divergence_fn(fx, ftx)                      # (B,)
            ratio = torch.where(
                valid,
                div / cx.clamp(min=1e-12),
                torch.zeros_like(div),
            )
            max_ratios = torch.maximum(max_ratios, ratio)

        return max_ratios                                           # (B,)

    def _forward(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the model from pre-computed embeddings.

        Returns the [CLS] token representation (index 0) of shape (B, hidden_dim).
        """
        out = self.model(
            inputs_embeds=embeddings,
            attention_mask=attention_mask,
        )
        # Support both BaseModelOutput and plain tensor outputs
        if hasattr(out, "last_hidden_state"):
            return out.last_hidden_state[:, 0, :]                  # (B, h) — CLS token
        if isinstance(out, torch.Tensor):
            if out.dim() == 3:
                return out[:, 0, :]
            return out
        raise TypeError(f"Unexpected model output type: {type(out)}")


# ---------------------------------------------------------------------------
# SPS Estimator  (Definition 2 / Definition 6)
# ---------------------------------------------------------------------------

class SPSEstimator:
    """
    Empirical estimator for Definition 2 (Structured Perturbation Stability).

    Computes:
        SPS_eps^{(n,m)}(f) = exp(-1/n * sum_i Sens_{T,eps}^{(m)}(f; x_i))

    which converges a.s. to SPS_eps(f) as n, m -> inf (Definition 6).

    Args:
        sensitivity_estimator: Configured StructuredSensitivityEstimator.
        config:                SPSConfig.
    """

    def __init__(
        self,
        sensitivity_estimator: StructuredSensitivityEstimator,
        config: SPSConfig,
    ) -> None:
        self.sens_estimator = sensitivity_estimator
        self.config = config

    def estimate(
        self,
        data_iterator: Iterator[dict[str, torch.Tensor]],
    ) -> dict[str, float]:
        """
        Compute empirical SPS over a dataset.

        Each batch from data_iterator must contain:
            "input_ids":      (B, seq_len) int64
            "attention_mask": (B, seq_len) int64
            "embeddings":     (B, seq_len, hidden_dim) float32   [optional;
                               computed on-the-fly if absent]

        Args:
            data_iterator: Iterable of batches.

        Returns:
            Dict with keys:
                sps              — SPS_eps^{(n,m)}(f), in (0, 1]
                mean_sensitivity — E[Sens]
                std_sensitivity  — Std[Sens]
                n_samples        — total samples processed
        """
        set_seed(self.config.seed)
        all_sens: list[torch.Tensor] = []

        for batch in data_iterator:
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)

            if "embeddings" in batch:
                embeddings = batch["embeddings"].to(self.config.device)
            else:
                embeddings = self._get_embeddings(input_ids)

            sens = self.sens_estimator.estimate(
                embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                m=self.config.m_transforms,
            )
            all_sens.append(sens.cpu())

        if not all_sens:
            raise RuntimeError("data_iterator yielded no batches.")

        sensitivities = torch.cat(all_sens)                        # (N,)
        mean_s = sensitivities.mean().item()
        import math
        sps_val = math.exp(-mean_s)

        logger.info(
            "SPS estimate: %.4f | mean_sensitivity=%.4f | std=%.4f | n=%d",
            sps_val, mean_s, sensitivities.std().item(), len(sensitivities),
        )

        return {
            "sps": sps_val,
            "mean_sensitivity": mean_s,
            "std_sensitivity": float(sensitivities.std().item()),
            "n_samples": int(len(sensitivities)),
        }

    def _get_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Derive embeddings from input_ids if not precomputed.

        Accesses the embedding layer via the sensitivity estimator's model.
        Falls back gracefully if the model does not expose get_input_embeddings().
        """
        model = self.sens_estimator.model
        get_emb = getattr(model, "get_input_embeddings", None)
        if callable(get_emb):
            with torch.no_grad():
                return get_emb()(input_ids)
        # Attempt attribute access for common HuggingFace naming conventions
        for attr in ("roberta", "bert", "transformer", "model"):
            sub = getattr(model, attr, None)
            if sub is not None:
                emb_layer = getattr(sub, "embeddings", None)
                if emb_layer is not None:
                    we = getattr(emb_layer, "word_embeddings", None)
                    if we is not None:
                        with torch.no_grad():
                            return we(input_ids)
        raise AttributeError(
            "Cannot locate embedding layer on model. "
            "Pass 'embeddings' explicitly in the data batch."
        )


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------

def build_sps_estimator(
    model: nn.Module,
    transform_family: TransformationFamily,
    config: Optional[SPSConfig] = None,
) -> SPSEstimator:
    """
    Convenience factory: wire up a full SPSEstimator from model + transform family.

    Args:
        model:            HuggingFace or custom transformer (nn.Module).
        transform_family: Admissible transformation family T.
        config:           SPSConfig; defaults used if None.

    Returns:
        A fully configured SPSEstimator.
    """
    config = config or SPSConfig()
    divergence_fn = get_divergence_fn(config.divergence)

    sens_estimator = StructuredSensitivityEstimator(
        model=model,
        transform_family=transform_family,
        divergence_fn=divergence_fn,
        epsilon=config.epsilon,
        device=config.device,
    )
    return SPSEstimator(sensitivity_estimator=sens_estimator, config=config)
