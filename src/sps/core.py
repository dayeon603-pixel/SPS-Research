"""
Core SPS computation: Definitions 1 & 2, empirical estimator (Definition 6),
adversarial SPS integration (Definition 3 / Proposition 3), and relative SPS
(Definition 8) with delta-method confidence intervals.

Implements:
  - StructuredSensitivityEstimator: pointwise Sens_{T,eps}(f; x) (Def 1)
  - SPSEstimator:                   global SPS_eps(f)             (Def 2)
  - AdversarialSPSReport:           dataclass for combined T_emb / T_adv results
  - AdversarialSPSEstimator:        computes SPS_emb, SPS_adv, and adversarial gap ratio
  - RelativeSPSReport:              dataclass for rSPS = SPS(T) / SPS(T_arb) + delta-method CI
  - RelativeSPSEstimator:           computes rSPS and its delta-method 95% CI (Def 8, §4.3)
  - SPSConfig:                      pydantic v2 configuration model

References
----------
Kang (2026) Definitions 1, 2, 3, 6, 8; Proposition 1, 3; §4.3; Assumptions A3, A4.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
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

        from sps.stats import bootstrap_ci, loo_spectral_gap
        sens_list = sensitivities.tolist()
        sens_ci = bootstrap_ci(sens_list, seed=self.config.seed)
        loo = loo_spectral_gap(sens_list)

        logger.info(
            "SPS estimate: %.4f | mean_sensitivity=%.4f | std=%.4f | n=%d | loo_stable=%s",
            sps_val, mean_s, sensitivities.std().item(), len(sensitivities),
            loo.get("stable", "N/A"),
        )

        return {
            "sps": sps_val,
            "mean_sensitivity": mean_s,
            "std_sensitivity": float(sensitivities.std().item()),
            "n_samples": int(len(sensitivities)),
            "sens_ci": sens_ci,
            "loo": loo,
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


# ---------------------------------------------------------------------------
# AdversarialSPSReport
# ---------------------------------------------------------------------------

@dataclass
class AdversarialSPSReport:
    """
    Combined report from adversarial (T_adv) and semantic (T_emb) SPS estimation.

    The core quantity is ``adv_gap_ratio = SPS_adv / SPS_emb``, which measures
    how close random direction sampling is to the adversarial worst case:

    - Ratio near 1.0 → T_emb is already near-adversarial; random sampling
      approximates the worst-case direction; adversarial attack adds little.
    - Ratio near 0.0 → adversarial reveals substantial additional sensitivity
      not captured by T_emb; the benchmark underestimates true vulnerability.

    Since T_adv maximises per-sample sensitivity over the direction set A_x,
    ``mean_sens_adv >= mean_sens_emb`` and ``sps_adv <= sps_emb`` always hold,
    giving ``adv_gap_ratio <= 1.0``.

    Attributes
    ----------
    sps_emb :        SPS_eps^{(T_emb)}(f) — under random semantic directions.
    sps_adv :        SPS_eps^{(T_adv)}(f) — under adversarial worst-case directions.
    adv_gap_ratio :  sps_adv / sps_emb ∈ (0, 1].
    mean_sens_emb :  E_x[Sens_{T_emb}(f; x)].
    std_sens_emb :   Std_x[Sens_{T_emb}(f; x)].
    mean_sens_adv :  E_x[Sens_{T_adv}(f; x)].
    std_sens_adv :   Std_x[Sens_{T_adv}(f; x)].
    n_samples :      Total samples processed.
    emb_sens_ci :    Bootstrap 95% CI tuple (point, lo, hi) for mean_sens_emb.
    adv_sens_ci :    Bootstrap 95% CI tuple (point, lo, hi) for mean_sens_adv.
    loo_emb :        LOO stability dict for T_emb per-sample sensitivities.
    loo_adv :        LOO stability dict for T_adv per-sample sensitivities.
    config :         SPSConfig used for both estimators.

    References
    ----------
    Kang (2026) Definition 3, Proposition 3 (adversarial upper bound on sensitivity).
    """

    sps_emb: float
    sps_adv: float
    adv_gap_ratio: float
    mean_sens_emb: float
    std_sens_emb: float
    mean_sens_adv: float
    std_sens_adv: float
    n_samples: int
    emb_sens_ci: tuple
    adv_sens_ci: tuple
    loo_emb: dict
    loo_adv: dict
    config: Optional[SPSConfig] = None

    def summary(self) -> str:
        """Return a formatted summary of the adversarial SPS analysis."""
        if self.adv_gap_ratio > 0.95:
            gap_label = "near-adversarial (random ≈ worst case)"
        elif self.adv_gap_ratio > 0.80:
            gap_label = "moderate adversarial gap"
        else:
            gap_label = "LARGE adversarial gap — T_emb understates vulnerability"

        lines = [
            "=" * 65,
            "  Adversarial SPS Analysis",
            "=" * 65,
            f"  SPS_emb (random T_emb)   : {self.sps_emb:.4f}",
            f"    mean_sensitivity        : {self.mean_sens_emb:.4f} ± {self.std_sens_emb:.4f}",
            f"    95% CI (bootstrap)      : [{self.emb_sens_ci[1]:.4f}, {self.emb_sens_ci[2]:.4f}]",
            f"    LOO stable              : {self.loo_emb.get('stable', 'N/A')}",
            f"  SPS_adv (adversarial)     : {self.sps_adv:.4f}",
            f"    mean_sensitivity        : {self.mean_sens_adv:.4f} ± {self.std_sens_adv:.4f}",
            f"    95% CI (bootstrap)      : [{self.adv_sens_ci[1]:.4f}, {self.adv_sens_ci[2]:.4f}]",
            f"    LOO stable              : {self.loo_adv.get('stable', 'N/A')}",
            f"  Adversarial gap ratio     : {self.adv_gap_ratio:.4f}  ({gap_label})",
            f"  N Samples                 : {self.n_samples}",
            "=" * 65,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# AdversarialSPSEstimator
# ---------------------------------------------------------------------------

class AdversarialSPSEstimator:
    """
    Integrates adversarial (T_adv) and semantic (T_emb) SPS estimation.

    Runs two SPSEstimators — one with random semantic directions (T_emb) and
    one with adversarial worst-case directions (T_adv) — and returns a combined
    AdversarialSPSReport including the adversarial gap ratio, bootstrap CIs on
    both per-sample sensitivity distributions, and LOO stability diagnostics.

    The adversarial gap ratio ``sps_adv / sps_emb`` answers the question:
    "How much additional sensitivity does the adversarial attack reveal beyond
    what random semantic direction sampling already captures?"

    Args:
        emb_estimator : SPSEstimator configured with T_emb (random semantic).
        adv_estimator : SPSEstimator configured with T_adv (adversarial).
        config :        SPSConfig for metadata; defaults to emb_estimator.config.

    References
    ----------
    Kang (2026) Definition 3, Proposition 3.
    """

    def __init__(
        self,
        emb_estimator: SPSEstimator,
        adv_estimator: SPSEstimator,
        config: Optional[SPSConfig] = None,
    ) -> None:
        self.emb_estimator = emb_estimator
        self.adv_estimator = adv_estimator
        self.config = config or emb_estimator.config

    def estimate(
        self,
        batches: list[dict[str, torch.Tensor]],
    ) -> AdversarialSPSReport:
        """
        Run both T_emb and T_adv estimators and return a combined report.

        Args:
            batches : Preloaded list of data batches. Each must contain
                      ``input_ids``, ``attention_mask``, and optionally
                      ``embeddings``. Consumed multiple times internally.

        Returns:
            AdversarialSPSReport with all metrics populated.
        """
        from sps.stats import bootstrap_ci, loo_spectral_gap

        # ── T_emb pass ───────────────────────────────────────────────────
        emb_result = self.emb_estimator.estimate(iter(batches))
        sps_emb = emb_result["sps"]
        mean_sens_emb = emb_result["mean_sensitivity"]
        std_sens_emb = emb_result["std_sensitivity"]
        n_samples = emb_result["n_samples"]

        # ── T_adv pass ───────────────────────────────────────────────────
        adv_result = self.adv_estimator.estimate(iter(batches))
        sps_adv = adv_result["sps"]
        mean_sens_adv = adv_result["mean_sensitivity"]
        std_sens_adv = adv_result["std_sensitivity"]

        # ── Adversarial gap ratio ─────────────────────────────────────────
        # SPS_adv <= SPS_emb always (adversarial finds worst case), so ratio in (0, 1].
        # ratio = exp(mean_sens_emb - mean_sens_adv) via log-space arithmetic.
        adv_gap_ratio = round(sps_adv / max(sps_emb, 1e-12), 6)

        # ── Per-sample CI and LOO ─────────────────────────────────────────
        emb_sens_list = self._collect_sensitivities(self.emb_estimator, batches)
        adv_sens_list = self._collect_sensitivities(self.adv_estimator, batches)

        emb_sens_ci = bootstrap_ci(emb_sens_list)
        adv_sens_ci = bootstrap_ci(adv_sens_list)
        loo_emb = loo_spectral_gap(emb_sens_list)
        loo_adv = loo_spectral_gap(adv_sens_list)

        logger.info(
            "AdversarialSPS: sps_emb=%.4f  sps_adv=%.4f  gap_ratio=%.4f  n=%d",
            sps_emb, sps_adv, adv_gap_ratio, n_samples,
        )

        return AdversarialSPSReport(
            sps_emb=sps_emb,
            sps_adv=sps_adv,
            adv_gap_ratio=adv_gap_ratio,
            mean_sens_emb=mean_sens_emb,
            std_sens_emb=std_sens_emb,
            mean_sens_adv=mean_sens_adv,
            std_sens_adv=std_sens_adv,
            n_samples=n_samples,
            emb_sens_ci=emb_sens_ci,
            adv_sens_ci=adv_sens_ci,
            loo_emb=loo_emb,
            loo_adv=loo_adv,
            config=self.config,
        )

    def _collect_sensitivities(
        self,
        estimator: SPSEstimator,
        batches: list[dict[str, torch.Tensor]],
    ) -> list[float]:
        """
        Re-run sensitivity estimation and collect per-sample values as a list.

        A separate pass is needed because SPSEstimator.estimate() only returns
        aggregate statistics; the per-sample values are required for bootstrap
        CI and LOO stability computations.
        """
        config = estimator.config
        all_sens: list[torch.Tensor] = []

        for batch in batches:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            if "embeddings" in batch:
                embeddings = batch["embeddings"].to(config.device)
            else:
                embeddings = estimator._get_embeddings(input_ids)

            sens = estimator.sens_estimator.estimate(
                embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                m=config.m_transforms,
            )
            all_sens.append(sens.cpu())

        if not all_sens:
            return []
        return torch.cat(all_sens).tolist()


# ---------------------------------------------------------------------------
# Factory helper for adversarial estimator
# ---------------------------------------------------------------------------

def build_adversarial_sps_estimator(
    model: nn.Module,
    emb_family: TransformationFamily,
    adv_family: TransformationFamily,
    config: Optional[SPSConfig] = None,
) -> AdversarialSPSEstimator:
    """
    Convenience factory: build AdversarialSPSEstimator from model + families.

    Args:
        model :       HuggingFace or custom transformer (nn.Module).
        emb_family :  T_emb — random semantic transformation family (e.g., T_emb).
        adv_family :  T_adv — adversarial worst-case family (e.g., AdversarialEmbeddingFamily).
        config :      SPSConfig; defaults used if None.

    Returns:
        Fully configured AdversarialSPSEstimator.

    Example
    -------
    >>> from sps import (build_adversarial_sps_estimator,
    ...                  EmbeddingPerturbationFamily, AdversarialEmbeddingFamily,
    ...                  SPSConfig)
    >>> estimator = build_adversarial_sps_estimator(
    ...     model, emb_family, adv_family, config=SPSConfig(epsilon=0.1)
    ... )
    >>> report = estimator.estimate(batches)
    >>> print(report.summary())
    """
    config = config or SPSConfig()
    emb_estimator = build_sps_estimator(model, emb_family, config)
    adv_estimator = build_sps_estimator(model, adv_family, config)
    return AdversarialSPSEstimator(emb_estimator, adv_estimator, config)


# ---------------------------------------------------------------------------
# RelativeSPSReport  (Definition 8 / §4.3)
# ---------------------------------------------------------------------------

@dataclass
class RelativeSPSReport:
    """
    Report for rSPS = SPS(T) / SPS(T_arb) with a delta-method CI.

    The relative SPS ratio (Definition 8) answers:
    "How much more stable is the model under semantic transformations T than
    under arbitrary l2-ball perturbations T_arb?"

    Interpretation
    --------------
    rSPS > 1 → T is more stable than T_arb; semantic structure constrains the
               transformation family below the arbitrary perturbation level.
    rSPS ≈ 1 → T and T_arb are equally challenging; no semantic invariance benefit.
    rSPS < 1 → T is less stable than T_arb; semantic transformations are
               systematically more adversarial than random perturbations.

    The delta-method CI is the log-Wald interval derived from:

        log(rSPS) = log(SPS_T) − log(SPS_arb) = mean_s_arb − mean_s_T

    which respects the positivity constraint on rSPS and is well-calibrated at
    moderate n. ``rsps_ci_excludes_one`` flags whether the CI is entirely above
    or below 1.0 — i.e., whether the T vs T_arb difference is statistically
    significant at the 5% level.

    Attributes
    ----------
    sps_t :                SPS under T (semantic transformation family).
    sps_arb :              SPS under T_arb (arbitrary reference perturbations).
    rsps_point :           rSPS point estimate = sps_t / sps_arb.
    rsps_ci :              Delta-method 95% CI tuple (point, lo, hi) for rSPS.
    rsps_ci_excludes_one : True if the CI excludes rSPS=1 (significant at α=0.05).
    mean_sens_t :          E_x[Sens_{T}(f; x)].
    std_sens_t :           Std_x[Sens_{T}(f; x)].
    n_samples_t :          Samples processed for T estimation.
    mean_sens_arb :        E_x[Sens_{T_arb}(f; x)].
    std_sens_arb :         Std_x[Sens_{T_arb}(f; x)].
    n_samples_arb :        Samples processed for T_arb estimation.
    sens_ci_t :            Bootstrap 95% CI (point, lo, hi) for mean_sens_t.
    loo_t :                LOO stability dict for T per-sample sensitivities.
    config :               SPSConfig used for estimation.

    References
    ----------
    Kang (2026) Definition 8 (rSPS), §4.3 (delta-method inference).
    """

    sps_t: float
    sps_arb: float
    rsps_point: float
    rsps_ci: tuple
    rsps_ci_excludes_one: bool
    mean_sens_t: float
    std_sens_t: float
    n_samples_t: int
    mean_sens_arb: float
    std_sens_arb: float
    n_samples_arb: int
    sens_ci_t: tuple
    loo_t: dict
    config: Optional[SPSConfig] = None

    def summary(self) -> str:
        """Return a formatted summary of the rSPS analysis."""
        if self.rsps_point > 1.05:
            interp = "T MORE stable than T_arb (semantic invariance present)"
        elif self.rsps_point > 0.95:
            interp = "T ≈ T_arb (no significant semantic structure benefit)"
        else:
            interp = "T LESS stable than T_arb (semantic transformations adversarial)"

        sig_label = "YES — CI excludes 1" if self.rsps_ci_excludes_one else "no — CI contains 1"

        lines = [
            "=" * 65,
            "  Relative SPS Analysis  (rSPS = SPS_T / SPS_arb)",
            "=" * 65,
            f"  SPS_T  (semantic T)        : {self.sps_t:.4f}",
            f"    mean_sensitivity          : {self.mean_sens_t:.4f} ± {self.std_sens_t:.4f}",
            f"    95% CI (bootstrap)        : [{self.sens_ci_t[1]:.4f}, {self.sens_ci_t[2]:.4f}]",
            f"    LOO stable                : {self.loo_t.get('stable', 'N/A')}",
            f"    n_samples                 : {self.n_samples_t}",
            f"  SPS_arb (arbitrary T_arb)  : {self.sps_arb:.4f}",
            f"    mean_sensitivity          : {self.mean_sens_arb:.4f} ± {self.std_sens_arb:.4f}",
            f"    n_samples                 : {self.n_samples_arb}",
            f"  rSPS point estimate         : {self.rsps_point:.4f}  ({interp})",
            f"  rSPS 95% CI (delta-method)  : [{self.rsps_ci[1]:.4f}, {self.rsps_ci[2]:.4f}]",
            f"  Significant (CI ≠ 1)        : {sig_label}",
            "=" * 65,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# RelativeSPSEstimator  (Definition 8 / §4.3)
# ---------------------------------------------------------------------------

class RelativeSPSEstimator:
    """
    Computes rSPS = SPS(T) / SPS(T_arb) with a delta-method CI (Definition 8).

    Runs two SPSEstimators — one with the tested semantic transformation family T
    and one with the arbitrary reference T_arb — and returns a RelativeSPSReport
    including the rSPS point estimate, its delta-method 95% CI, and bootstrap CI
    and LOO stability diagnostics for the T per-sample sensitivities.

    The delta-method CI is derived from the log-Wald interval:

        log(rSPS) = mean_s_arb − mean_s_T
        Var(log rSPS) ≈ Var_T / n_T + Var_arb / n_arb

    and exponentiating the resulting interval to obtain (rsps_lo, rsps_hi).

    Args:
        t_estimator :   SPSEstimator configured with T (semantic family, numerator).
        arb_estimator : SPSEstimator configured with T_arb (reference, denominator).
        config :        SPSConfig for metadata; defaults to t_estimator.config.

    References
    ----------
    Kang (2026) Definition 8, §4.3.
    """

    def __init__(
        self,
        t_estimator: SPSEstimator,
        arb_estimator: SPSEstimator,
        config: Optional[SPSConfig] = None,
    ) -> None:
        self.t_estimator = t_estimator
        self.arb_estimator = arb_estimator
        self.config = config or t_estimator.config

    def estimate(
        self,
        batches: list[dict[str, torch.Tensor]],
    ) -> RelativeSPSReport:
        """
        Run both T and T_arb estimators and return a combined rSPS report.

        Args:
            batches : Preloaded list of data batches. Each must contain
                      ``input_ids``, ``attention_mask``, and optionally
                      ``embeddings``. The list is consumed multiple times
                      internally (T pass, T_arb pass, T LOO/CI collection).

        Returns:
            RelativeSPSReport with all metrics populated.
        """
        from sps.stats import bootstrap_ci, delta_method_rsps_ci, loo_spectral_gap

        # ── T pass ───────────────────────────────────────────────────────────
        t_result = self.t_estimator.estimate(iter(batches))
        sps_t = t_result["sps"]
        mean_sens_t = t_result["mean_sensitivity"]
        std_sens_t = t_result["std_sensitivity"]
        n_samples_t = t_result["n_samples"]

        # ── T_arb pass ────────────────────────────────────────────────────────
        arb_result = self.arb_estimator.estimate(iter(batches))
        sps_arb = arb_result["sps"]
        mean_sens_arb = arb_result["mean_sensitivity"]
        std_sens_arb = arb_result["std_sensitivity"]
        n_samples_arb = arb_result["n_samples"]

        # ── rSPS: delta-method CI ─────────────────────────────────────────────
        # delta_method_rsps_ci uses parameter names (emb_*, arb_*) corresponding
        # to T and T_arb respectively in the rSPS = SPS(T) / SPS(T_arb) formula.
        rsps_point, rsps_lo, rsps_hi = delta_method_rsps_ci(
            emb_mean_s=mean_sens_t,
            emb_std_s=std_sens_t,
            emb_n=n_samples_t,
            arb_mean_s=mean_sens_arb,
            arb_std_s=std_sens_arb,
            arb_n=n_samples_arb,
        )
        rsps_ci = (rsps_point, rsps_lo, rsps_hi)
        rsps_ci_excludes_one = rsps_lo > 1.0 or rsps_hi < 1.0

        # ── Per-sample CI and LOO for T (3rd pass) ────────────────────────────
        t_sens_list = self._collect_sensitivities(self.t_estimator, batches)
        sens_ci_t = bootstrap_ci(t_sens_list, seed=self.config.seed)
        loo_t = loo_spectral_gap(t_sens_list)

        logger.info(
            "RelativeSPS: sps_t=%.4f  sps_arb=%.4f  rsps=%.4f  "
            "CI=[%.4f, %.4f]  excl_one=%s  n_t=%d",
            sps_t, sps_arb, rsps_point, rsps_lo, rsps_hi,
            rsps_ci_excludes_one, n_samples_t,
        )

        return RelativeSPSReport(
            sps_t=sps_t,
            sps_arb=sps_arb,
            rsps_point=rsps_point,
            rsps_ci=rsps_ci,
            rsps_ci_excludes_one=rsps_ci_excludes_one,
            mean_sens_t=mean_sens_t,
            std_sens_t=std_sens_t,
            n_samples_t=n_samples_t,
            mean_sens_arb=mean_sens_arb,
            std_sens_arb=std_sens_arb,
            n_samples_arb=n_samples_arb,
            sens_ci_t=sens_ci_t,
            loo_t=loo_t,
            config=self.config,
        )

    def _collect_sensitivities(
        self,
        estimator: SPSEstimator,
        batches: list[dict[str, torch.Tensor]],
    ) -> list[float]:
        """Re-run sensitivity estimation and collect per-sample values as list."""
        config = estimator.config
        all_sens: list[torch.Tensor] = []

        for batch in batches:
            input_ids = batch["input_ids"].to(config.device)
            attention_mask = batch["attention_mask"].to(config.device)

            if "embeddings" in batch:
                embeddings = batch["embeddings"].to(config.device)
            else:
                embeddings = estimator._get_embeddings(input_ids)

            sens = estimator.sens_estimator.estimate(
                embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                m=config.m_transforms,
            )
            all_sens.append(sens.cpu())

        if not all_sens:
            return []
        return torch.cat(all_sens).tolist()


# ---------------------------------------------------------------------------
# Factory helper for relative SPS estimator
# ---------------------------------------------------------------------------

def build_relative_sps_estimator(
    model: nn.Module,
    t_family: TransformationFamily,
    arb_family: TransformationFamily,
    config: Optional[SPSConfig] = None,
) -> RelativeSPSEstimator:
    """
    Convenience factory: build RelativeSPSEstimator from model + families.

    Args:
        model :      HuggingFace or custom transformer (nn.Module).
        t_family :   T — semantic transformation family (numerator of rSPS).
        arb_family : T_arb — arbitrary reference family (denominator of rSPS).
        config :     SPSConfig; defaults used if None.

    Returns:
        Fully configured RelativeSPSEstimator.

    Example
    -------
    >>> from sps import (build_relative_sps_estimator,
    ...                  EmbeddingPerturbationFamily, SPSConfig)
    >>> estimator = build_relative_sps_estimator(
    ...     model, t_family, arb_family, config=SPSConfig(epsilon=0.1)
    ... )
    >>> report = estimator.estimate(batches)
    >>> print(report.summary())
    """
    config = config or SPSConfig()
    t_estimator = build_sps_estimator(model, t_family, config)
    arb_estimator = build_sps_estimator(model, arb_family, config)
    return RelativeSPSEstimator(t_estimator, arb_estimator, config)
