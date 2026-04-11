"""
Statistical utilities for SPS inference: bootstrap CIs, LOO stability,
and delta-method confidence intervals for the relative SPS ratio.

Implements the inference tools needed to report stable estimates with
uncertainty bounds, as described in Kang (2026) §4.

Functions
---------
bootstrap_ci :          Nonparametric 95% CI for any sample mean.
loo_spectral_gap :      Leave-one-out stability diagnostic for spectral gap.
delta_method_rsps_ci :  Delta-method 95% CI for rSPS = SPS(T) / SPS(T_arb).
"""
from __future__ import annotations

import math
import random

__all__ = [
    "bootstrap_ci",
    "loo_spectral_gap",
    "delta_method_rsps_ci",
]


def bootstrap_ci(
    values: list[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Nonparametric bootstrap 95% CI for the mean of ``values``.

    Uses percentile bootstrap (not BCa) — appropriate when the sample
    size is small and the distribution of the mean is unknown.

    Parameters
    ----------
    values :
        Sample of scalar observations.
    n_boot :
        Number of bootstrap resamples.
    alpha :
        Two-sided coverage level. Default 0.05 gives a 95% CI.
    seed :
        RNG seed for reproducibility.

    Returns
    -------
    (mean, lower_bound, upper_bound) : tuple[float, float, float]
        All values rounded to 6 decimal places.

    Notes
    -----
    The percentile CI is asymptotically consistent. For n < 10 the
    coverage may be below nominal — report alongside the sample size.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    boot_means = sorted(
        sum(rng.choices(values, k=n)) / n for _ in range(n_boot)
    )
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = int((1 - alpha / 2) * n_boot)
    return (
        round(sum(values) / n, 6),
        round(boot_means[lo_idx], 6),
        round(boot_means[hi_idx], 6),
    )


def loo_spectral_gap(per_sample_gaps: list[float]) -> dict:
    """
    Leave-one-out (LOO) stability diagnostic for the mean spectral gap.

    Computes the LOO mean estimate for each sample and reports the range
    [loo_min, loo_max]. A narrow range (< 0.05) indicates the mean is
    not driven by any single sample.

    Parameters
    ----------
    per_sample_gaps :
        Per-sample normalized spectral gap values gamma-bar(f, T; x_i).

    Returns
    -------
    dict with keys:
        n         — sample count
        mean      — full-sample mean
        loo_min   — minimum LOO mean
        loo_max   — maximum LOO mean
        loo_range — loo_max − loo_min
        stable    — True if loo_range < 0.05

    If n < 3, returns ``{"stable": False, "n": n}`` (LOO undefined at n ≤ 2).
    """
    n = len(per_sample_gaps)
    if n < 3:
        return {"stable": False, "n": n}
    total = sum(per_sample_gaps)
    loo_means = [(total - g) / (n - 1) for g in per_sample_gaps]
    return {
        "n": n,
        "mean": round(total / n, 6),
        "loo_min": round(min(loo_means), 6),
        "loo_max": round(max(loo_means), 6),
        "loo_range": round(max(loo_means) - min(loo_means), 6),
        "stable": (max(loo_means) - min(loo_means)) < 0.05,
    }


def delta_method_rsps_ci(
    emb_mean_s: float,
    emb_std_s: float,
    emb_n: int,
    arb_mean_s: float,
    arb_std_s: float,
    arb_n: int,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Delta-method 95% CI for rSPS = SPS(T_emb) / SPS(T_arb).

    Derivation
    ----------
    SPS is defined as exp(−mean_sensitivity), so:

        log(rSPS) = log(SPS_emb) − log(SPS_arb)
                  = −mean_s_emb − (−mean_s_arb)
                  = mean_s_arb − mean_s_emb

    By independence of the two estimators and the delta method:

        Var(log rSPS) ≈ std_s_emb² / n_emb + std_s_arb² / n_arb

    The 95% CI for rSPS is then:

        exp( log(rSPS) ± z_{α/2} × SE(log rSPS) )

    This is the log-space Wald interval, which respects the positivity
    constraint on rSPS and is better calibrated than the linear-space
    interval at small n.

    Parameters
    ----------
    emb_mean_s :   E_x[Sens_{T_emb}(f; x)]
    emb_std_s :    Std_x[Sens_{T_emb}(f; x)]
    emb_n :        Number of data samples used for T_emb estimation
    arb_mean_s :   E_x[Sens_{T_arb}(f; x)]
    arb_std_s :    Std_x[Sens_{T_arb}(f; x)]
    arb_n :        Number of data samples used for T_arb estimation
    alpha :        Two-sided coverage level (default 0.05 → 95% CI)

    Returns
    -------
    (rsps_point, rsps_lo, rsps_hi) : tuple[float, float, float]
        All values rounded to 6 decimal places.
    """
    # z-score for two-sided CI
    if abs(alpha - 0.05) < 1e-9:
        z = 1.96
    elif abs(alpha - 0.01) < 1e-9:
        z = 2.576
    else:
        z = 1.645  # 90%

    log_rsps = arb_mean_s - emb_mean_s
    var_log = (emb_std_s ** 2) / max(emb_n, 1) + (arb_std_s ** 2) / max(arb_n, 1)
    se_log = math.sqrt(var_log)

    rsps_point = math.exp(log_rsps)
    rsps_lo = math.exp(log_rsps - z * se_log)
    rsps_hi = math.exp(log_rsps + z * se_log)
    return round(rsps_point, 6), round(rsps_lo, 6), round(rsps_hi, 6)
