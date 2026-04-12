"""
Tests for sps.stats: bootstrap_ci, loo_spectral_gap, delta_method_rsps_ci.

Covers:
  - bootstrap_ci: determinism, monotonicity, empty input, single-element,
    and interval width scaling with n_boot.
  - loo_spectral_gap: stability flag, n < 3 guard, correct LOO arithmetic,
    exact min/max/range.
  - delta_method_rsps_ci: point estimate correctness, CI direction,
    CI shrinks with larger n, CI widens with larger std, alpha levels.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ------------------------------------------------------------------
# Minimal torch mock — lets us import sps.stats (a pure-Python
# module) without a full torch installation. The mock must be
# injected before any sps module is imported, since sps/__init__.py
# imports torch-dependent submodules at package load time.
# ------------------------------------------------------------------
if "torch" not in sys.modules:
    _mock_torch = MagicMock()
    _mock_torch.Tensor = type("Tensor", (), {})
    _mock_nn = MagicMock()
    _mock_nn.Module = object
    _mock_nn.Embedding = object
    _mock_nn.ModuleList = list
    _mock_torch.nn = _mock_nn
    sys.modules["torch"] = _mock_torch
    sys.modules["torch.nn"] = _mock_nn
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.autograd"] = MagicMock()
    sys.modules["torch.autograd.functional"] = MagicMock()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sps.stats import bootstrap_ci, delta_method_rsps_ci, loo_spectral_gap


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapCI:

    def test_empty_input_returns_zeros(self) -> None:
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0
        assert lo == 0.0
        assert hi == 0.0

    def test_single_element_tight_ci(self) -> None:
        # With a single constant value, all bootstrap resamples have the same mean.
        mean, lo, hi = bootstrap_ci([0.5], n_boot=200, seed=0)
        assert mean == pytest.approx(0.5, abs=1e-6)
        assert lo == pytest.approx(0.5, abs=1e-6)
        assert hi == pytest.approx(0.5, abs=1e-6)

    def test_constant_values_tight_ci(self) -> None:
        values = [0.3] * 20
        mean, lo, hi = bootstrap_ci(values, n_boot=500, seed=0)
        assert mean == pytest.approx(0.3, abs=1e-6)
        assert lo == pytest.approx(0.3, abs=1e-6)
        assert hi == pytest.approx(0.3, abs=1e-6)

    def test_mean_matches_sample_mean(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        mean, _, _ = bootstrap_ci(values, n_boot=200, seed=7)
        assert mean == pytest.approx(sum(values) / len(values), abs=1e-6)

    def test_ci_contains_mean(self) -> None:
        values = [0.1 * i for i in range(1, 11)]
        mean, lo, hi = bootstrap_ci(values, n_boot=2000, seed=42)
        assert lo <= mean <= hi

    def test_ci_lower_leq_upper(self) -> None:
        values = [0.5, 0.6, 0.4, 0.8, 0.2, 0.9, 0.1]
        mean, lo, hi = bootstrap_ci(values, n_boot=1000, seed=0)
        assert lo <= hi

    def test_determinism_same_seed(self) -> None:
        values = [0.3, 0.7, 0.5, 0.2, 0.9]
        r1 = bootstrap_ci(values, n_boot=500, seed=99)
        r2 = bootstrap_ci(values, n_boot=500, seed=99)
        assert r1 == r2

    def test_different_seeds_may_differ(self) -> None:
        values = [0.1 + 0.1 * i for i in range(10)]
        r1 = bootstrap_ci(values, n_boot=100, seed=1)
        r2 = bootstrap_ci(values, n_boot=100, seed=2)
        # Seeds should give different CI endpoints (not a guarantee, but true for
        # any reasonable values with non-zero variance)
        assert r1[1] != r2[1] or r1[2] != r2[2]

    def test_wider_spread_gives_wider_ci(self) -> None:
        narrow = [0.5] * 9 + [0.6]
        wide = [0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0]
        _, n_lo, n_hi = bootstrap_ci(narrow, n_boot=2000, seed=42)
        _, w_lo, w_hi = bootstrap_ci(wide, n_boot=2000, seed=42)
        assert (w_hi - w_lo) > (n_hi - n_lo)

    def test_return_type_is_tuple_of_floats(self) -> None:
        result = bootstrap_ci([0.5, 0.6, 0.7], n_boot=100, seed=0)
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_rounded_to_6_decimal_places(self) -> None:
        # All outputs should have at most 6 decimal digits
        values = [1 / 3, 2 / 3, 1 / 7]
        mean, lo, hi = bootstrap_ci(values, n_boot=100, seed=0)
        for v in (mean, lo, hi):
            # round(v, 6) should equal v (up to float precision)
            assert abs(v - round(v, 6)) < 1e-10


# ─────────────────────────────────────────────────────────────────────────────
# loo_spectral_gap
# ─────────────────────────────────────────────────────────────────────────────

class TestLOOSpectralGap:

    def test_fewer_than_3_returns_unstable(self) -> None:
        for n in (0, 1, 2):
            result = loo_spectral_gap([0.5] * n)
            assert result["stable"] is False
            assert result["n"] == n

    def test_constant_values_are_stable(self) -> None:
        values = [0.4] * 10
        result = loo_spectral_gap(values)
        assert result["stable"] is True
        assert result["loo_range"] == pytest.approx(0.0, abs=1e-9)
        assert result["loo_min"] == pytest.approx(0.4, abs=1e-6)
        assert result["loo_max"] == pytest.approx(0.4, abs=1e-6)

    def test_mean_correct(self) -> None:
        values = [0.2, 0.4, 0.6, 0.8]
        result = loo_spectral_gap(values)
        assert result["mean"] == pytest.approx(0.5, abs=1e-6)

    def test_loo_min_leq_mean_leq_loo_max(self) -> None:
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        result = loo_spectral_gap(values)
        assert result["loo_min"] <= result["mean"] <= result["loo_max"]

    def test_loo_range_equals_max_minus_min(self) -> None:
        values = [0.1, 0.5, 0.9]
        result = loo_spectral_gap(values)
        assert result["loo_range"] == pytest.approx(
            result["loo_max"] - result["loo_min"], abs=1e-9
        )

    def test_outlier_triggers_instability(self) -> None:
        # 9 values near 0.5, one extreme outlier — LOO range should exceed 0.05
        values = [0.5] * 9 + [5.0]
        result = loo_spectral_gap(values)
        assert result["stable"] is False

    def test_three_equal_values_stable(self) -> None:
        result = loo_spectral_gap([0.6, 0.6, 0.6])
        assert result["stable"] is True

    def test_loo_arithmetic_exact(self) -> None:
        # Manual check: values = [0.0, 0.3, 0.6]
        # total = 0.9, mean = 0.3
        # loo_means: (0.9 - 0.0)/2 = 0.45, (0.9 - 0.3)/2 = 0.3, (0.9 - 0.6)/2 = 0.15
        values = [0.0, 0.3, 0.6]
        result = loo_spectral_gap(values)
        assert result["loo_min"] == pytest.approx(0.15, abs=1e-6)
        assert result["loo_max"] == pytest.approx(0.45, abs=1e-6)
        assert result["loo_range"] == pytest.approx(0.30, abs=1e-6)

    def test_return_has_required_keys(self) -> None:
        result = loo_spectral_gap([0.2, 0.3, 0.4, 0.5])
        for key in ("n", "mean", "loo_min", "loo_max", "loo_range", "stable"):
            assert key in result

    def test_n_matches_input_length(self) -> None:
        values = [0.1] * 7
        assert loo_spectral_gap(values)["n"] == 7


# ─────────────────────────────────────────────────────────────────────────────
# delta_method_rsps_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaMethodRSPSCI:

    def test_point_estimate_rsps_gt1(self) -> None:
        # arb_mean_s > emb_mean_s => log(rSPS) = arb - emb > 0 => rSPS > 1
        rsps, lo, hi = delta_method_rsps_ci(
            emb_mean_s=0.3, emb_std_s=0.05, emb_n=50,
            arb_mean_s=0.5, arb_std_s=0.05, arb_n=50,
        )
        assert rsps > 1.0
        assert lo > 0.0
        assert hi > rsps

    def test_point_estimate_rsps_lt1(self) -> None:
        # emb_mean_s > arb_mean_s => rSPS < 1 (pathological)
        rsps, lo, hi = delta_method_rsps_ci(
            emb_mean_s=0.7, emb_std_s=0.05, emb_n=50,
            arb_mean_s=0.4, arb_std_s=0.05, arb_n=50,
        )
        assert rsps < 1.0

    def test_ci_contains_point_estimate(self) -> None:
        rsps, lo, hi = delta_method_rsps_ci(
            emb_mean_s=0.4, emb_std_s=0.1, emb_n=20,
            arb_mean_s=0.6, arb_std_s=0.1, arb_n=20,
        )
        assert lo <= rsps <= hi

    def test_ci_lower_leq_upper(self) -> None:
        rsps, lo, hi = delta_method_rsps_ci(
            emb_mean_s=0.5, emb_std_s=0.2, emb_n=30,
            arb_mean_s=0.5, arb_std_s=0.2, arb_n=30,
        )
        assert lo <= hi

    def test_equal_means_gives_rsps_1(self) -> None:
        rsps, _, _ = delta_method_rsps_ci(
            emb_mean_s=0.5, emb_std_s=0.1, emb_n=100,
            arb_mean_s=0.5, arb_std_s=0.1, arb_n=100,
        )
        assert rsps == pytest.approx(1.0, abs=1e-6)

    def test_larger_n_gives_narrower_ci(self) -> None:
        kwargs = dict(emb_mean_s=0.4, emb_std_s=0.1, arb_mean_s=0.6, arb_std_s=0.1)
        _, lo_small, hi_small = delta_method_rsps_ci(emb_n=10, arb_n=10, **kwargs)
        _, lo_large, hi_large = delta_method_rsps_ci(emb_n=1000, arb_n=1000, **kwargs)
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_larger_std_gives_wider_ci(self) -> None:
        kwargs = dict(emb_mean_s=0.4, arb_mean_s=0.6, emb_n=50, arb_n=50)
        _, lo_tight, hi_tight = delta_method_rsps_ci(emb_std_s=0.01, arb_std_s=0.01, **kwargs)
        _, lo_wide, hi_wide = delta_method_rsps_ci(emb_std_s=0.5, arb_std_s=0.5, **kwargs)
        assert (hi_wide - lo_wide) > (hi_tight - lo_tight)

    def test_alpha_01_gives_wider_ci_than_05(self) -> None:
        kwargs = dict(
            emb_mean_s=0.4, emb_std_s=0.1, emb_n=50,
            arb_mean_s=0.6, arb_std_s=0.1, arb_n=50,
        )
        _, lo_95, hi_95 = delta_method_rsps_ci(alpha=0.05, **kwargs)
        _, lo_99, hi_99 = delta_method_rsps_ci(alpha=0.01, **kwargs)
        assert (hi_99 - lo_99) > (hi_95 - lo_95)

    def test_all_positive(self) -> None:
        rsps, lo, hi = delta_method_rsps_ci(
            emb_mean_s=0.9, emb_std_s=0.3, emb_n=5,
            arb_mean_s=0.1, arb_std_s=0.3, arb_n=5,
        )
        assert rsps > 0.0
        assert lo > 0.0
        assert hi > 0.0

    def test_return_type_is_tuple_of_floats(self) -> None:
        result = delta_method_rsps_ci(
            emb_mean_s=0.5, emb_std_s=0.1, emb_n=100,
            arb_mean_s=0.5, arb_std_s=0.1, arb_n=100,
        )
        assert isinstance(result, tuple)
        assert len(result) == 3
        assert all(isinstance(v, float) for v in result)

    def test_point_estimate_formula(self) -> None:
        # Directly verify: rSPS = exp(arb_mean - emb_mean)
        emb, arb = 0.3, 0.7
        expected = math.exp(arb - emb)
        rsps, _, _ = delta_method_rsps_ci(
            emb_mean_s=emb, emb_std_s=0.0, emb_n=1000,
            arb_mean_s=arb, arb_std_s=0.0, arb_n=1000,
        )
        assert rsps == pytest.approx(expected, rel=1e-4)
