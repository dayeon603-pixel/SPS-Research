"""
Tests for sps.metrics: SPSReport dataclass, CI fields, and summary output.

Covers:
  - SPSReport default field values (gap_ci=None, rsps_ci=None)
  - SPSReport accepts and stores gap_ci and rsps_ci tuples
  - summary() omits CI lines when fields are None
  - summary() includes CI lines when gap_ci is set
  - summary() includes CI lines when rsps_ci is set
  - summary() CI line format matches expected pattern
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ------------------------------------------------------------------
# Minimal torch mock — lets us import SPSReport (a pure dataclass)
# without a full torch installation.  Must be injected before any
# sps module is imported.
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

from sps.metrics import SPSReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _minimal_report(**kwargs) -> SPSReport:
    """Build an SPSReport with minimal required fields."""
    defaults = dict(
        sps=0.85,
        mean_sensitivity=0.12,
        std_sensitivity=0.03,
        n_samples=64,
    )
    defaults.update(kwargs)
    return SPSReport(**defaults)


# ---------------------------------------------------------------------------
# Default field values
# ---------------------------------------------------------------------------

class TestSPSReportDefaults:

    def test_gap_ci_defaults_to_none(self) -> None:
        r = _minimal_report()
        assert r.gap_ci is None

    def test_rsps_ci_defaults_to_none(self) -> None:
        r = _minimal_report()
        assert r.rsps_ci is None

    def test_relative_sps_defaults_to_none(self) -> None:
        r = _minimal_report()
        assert r.relative_sps is None

    def test_spectral_gap_mean_defaults_to_none(self) -> None:
        r = _minimal_report()
        assert r.spectral_gap_mean is None


# ---------------------------------------------------------------------------
# Field assignment and storage
# ---------------------------------------------------------------------------

class TestSPSReportCIFields:

    def test_gap_ci_stored_correctly(self) -> None:
        ci = (0.65, 0.58, 0.72)
        r = _minimal_report(gap_ci=ci)
        assert r.gap_ci == ci

    def test_rsps_ci_stored_correctly(self) -> None:
        ci = (1.42, 1.10, 1.78)
        r = _minimal_report(rsps_ci=ci)
        assert r.rsps_ci == ci

    def test_gap_ci_elements_are_floats(self) -> None:
        ci = (0.65, 0.58, 0.72)
        r = _minimal_report(gap_ci=ci)
        assert isinstance(r.gap_ci[0], float)
        assert isinstance(r.gap_ci[1], float)
        assert isinstance(r.gap_ci[2], float)

    def test_gap_ci_lo_leq_point_leq_hi(self) -> None:
        ci = (0.65, 0.58, 0.72)
        r = _minimal_report(gap_ci=ci)
        point, lo, hi = r.gap_ci
        assert lo <= point <= hi

    def test_rsps_ci_lo_leq_point_leq_hi(self) -> None:
        ci = (1.42, 1.10, 1.78)
        r = _minimal_report(rsps_ci=ci)
        point, lo, hi = r.rsps_ci
        assert lo <= point <= hi

    def test_gap_ci_and_rsps_ci_independent(self) -> None:
        gap = (0.65, 0.58, 0.72)
        rsps = (1.42, 1.10, 1.78)
        r = _minimal_report(gap_ci=gap, rsps_ci=rsps)
        assert r.gap_ci == gap
        assert r.rsps_ci == rsps


# ---------------------------------------------------------------------------
# summary() output — CI lines included/omitted correctly
# ---------------------------------------------------------------------------

class TestSPSReportSummary:

    def test_summary_no_ci_by_default(self) -> None:
        r = _minimal_report()
        s = r.summary()
        assert "CI" not in s
        assert "gap_ci" not in s
        assert "rsps_ci" not in s

    def test_summary_gap_ci_line_present_when_set(self) -> None:
        r = _minimal_report(
            spectral_gap_mean=0.65,
            gap_ci=(0.65, 0.58, 0.72),
        )
        s = r.summary()
        assert "Gap 95% CI" in s

    def test_summary_gap_ci_line_absent_when_none(self) -> None:
        r = _minimal_report(spectral_gap_mean=0.65)
        s = r.summary()
        assert "Gap 95% CI" not in s

    def test_summary_rsps_ci_line_present_when_set(self) -> None:
        r = _minimal_report(
            relative_sps=1.42,
            rsps_ci=(1.42, 1.10, 1.78),
        )
        s = r.summary()
        assert "rSPS 95% CI" in s

    def test_summary_rsps_ci_line_absent_when_none(self) -> None:
        r = _minimal_report(relative_sps=1.42)
        s = r.summary()
        assert "rSPS 95% CI" not in s

    def test_summary_gap_ci_values_in_output(self) -> None:
        r = _minimal_report(
            spectral_gap_mean=0.65,
            gap_ci=(0.65, 0.58, 0.72),
        )
        s = r.summary()
        assert "0.5800" in s
        assert "0.7200" in s

    def test_summary_rsps_ci_values_in_output(self) -> None:
        r = _minimal_report(
            relative_sps=1.42,
            rsps_ci=(1.42, 1.10, 1.78),
        )
        s = r.summary()
        assert "1.1000" in s
        assert "1.7800" in s

    def test_summary_rsps_ci_after_rsps_line(self) -> None:
        """CI line must follow the rSPS line, not float free in the summary."""
        r = _minimal_report(
            relative_sps=1.42,
            rsps_ci=(1.42, 1.10, 1.78),
        )
        lines = r.summary().splitlines()
        rsps_idx = next(i for i, l in enumerate(lines) if "rSPS (Def 8)" in l)
        ci_idx   = next(i for i, l in enumerate(lines) if "rSPS 95% CI" in l)
        assert ci_idx == rsps_idx + 1, "CI line must immediately follow the rSPS line"

    def test_summary_gap_ci_after_gap_line(self) -> None:
        """Gap CI line must follow the spectral gap mean line."""
        r = _minimal_report(
            spectral_gap_mean=0.65,
            gap_ci=(0.65, 0.58, 0.72),
        )
        lines = r.summary().splitlines()
        gap_idx = next(i for i, l in enumerate(lines) if "Spectral Gap gamma-bar" in l)
        ci_idx  = next(i for i, l in enumerate(lines) if "Gap 95% CI" in l)
        assert ci_idx == gap_idx + 1, "Gap CI line must immediately follow the spectral gap line"

    def test_summary_both_cis_present_simultaneously(self) -> None:
        r = _minimal_report(
            relative_sps=1.42,
            rsps_ci=(1.42, 1.10, 1.78),
            spectral_gap_mean=0.65,
            gap_ci=(0.65, 0.58, 0.72),
        )
        s = r.summary()
        assert "rSPS 95% CI" in s
        assert "Gap 95% CI" in s
