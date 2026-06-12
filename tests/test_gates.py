"""Gate tests for reproduce_acm.py — self-contained, no conftest required.

All test inputs are constructed inline.  Each test exercises a single gate
condition and uses a descriptive name rather than a sequential number.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure the repo root is importable from within the tests/ sub-directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reproduce_acm import (
    PUBLISHED_MATURITIES,
    check_coverage_gaps,
    check_identity_gate,
    check_schema_gate,
    classify_tail_gap,
    maturity_suffix,
)

# ---------------------------------------------------------------------------
# Helpers shared across tests
# ---------------------------------------------------------------------------


def _make_date(date_str: str) -> pd.Timestamp:
    return pd.Timestamp(date_str)


def _make_dti(*dates: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex([pd.Timestamp(d) for d in dates])


def _clean_panel(dates: list[str]) -> pd.DataFrame:
    """Build a small valid generated panel (ACMY=ACMRNY+ACMTP by construction)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d in dates])
    idx.name = "DATE"
    recs = {}
    for n in PUBLISHED_MATURITIES:
        suffix = maturity_suffix(n)
        rny_vals = np.full(len(dates), 1.5)
        tp_vals = np.full(len(dates), 0.5)
        recs[f"ACMRNY{suffix}"] = rny_vals
        recs[f"ACMTP{suffix}"] = tp_vals
        recs[f"ACMY{suffix}"] = rny_vals + tp_vals  # identity holds exactly
    return pd.DataFrame(recs, index=idx)


# ---------------------------------------------------------------------------
# Coverage helper: classify_tail_gap
# ---------------------------------------------------------------------------


def test_classify_tail_gap_splits_interior_from_tail():
    gsw_last = _make_date("2026-06-05")
    missing = _make_dti("2026-06-03", "2026-06-08")

    interior, tail = classify_tail_gap(missing, gsw_last)

    assert pd.Timestamp("2026-06-03") in interior
    assert pd.Timestamp("2026-06-08") in tail
    assert pd.Timestamp("2026-06-08") not in interior
    assert pd.Timestamp("2026-06-03") not in tail


# ---------------------------------------------------------------------------
# Coverage helper: check_coverage_gaps — interior hole must fail
# ---------------------------------------------------------------------------


def test_coverage_gap_interior_fails():
    gsw_last = _make_date("2026-06-05")
    # 2026-05-01 is well before gsw_last — interior gap
    missing = _make_dti("2026-05-01")
    failures: list[str] = []

    check_coverage_gaps(missing, gsw_last, "daily", max_tail_gap_bd=5, failures=failures)

    assert len(failures) == 1
    assert "interior gaps" in failures[0]


# ---------------------------------------------------------------------------
# Coverage helper: tail gap within tolerance passes silently
# ---------------------------------------------------------------------------


def test_coverage_gap_tail_within_tolerance_passes():
    # gsw_last = 2026-06-05 (Friday); 2026-06-06/07 are the weekend, and
    # np.busday_count counts Mon-Fri, so 2026-06-08 (Monday) is 1 business
    # day after gsw_last.
    gsw_last = _make_date("2026-06-05")
    missing = _make_dti("2026-06-08")  # 1 bd gap
    failures: list[str] = []

    check_coverage_gaps(missing, gsw_last, "daily", max_tail_gap_bd=5, failures=failures)

    assert failures == [], f"Expected no failures; got: {failures}"


# ---------------------------------------------------------------------------
# Coverage helper: tail gap beyond 15 bd must fail
# ---------------------------------------------------------------------------


def test_coverage_gap_tail_beyond_limit_fails():
    gsw_last = _make_date("2026-01-01")
    # 2026-03-01 is many business days after 2026-01-01 (well > 15 bd)
    missing = _make_dti("2026-03-01")
    failures: list[str] = []

    check_coverage_gaps(missing, gsw_last, "daily", max_tail_gap_bd=5, failures=failures)

    assert len(failures) == 1
    assert "exceeds 15-bd maximum" in failures[0]


# ---------------------------------------------------------------------------
# Coverage helper: the 15-bd hard cap holds even with a larger tolerance
# ---------------------------------------------------------------------------


def test_coverage_gap_hard_cap_not_bypassed_by_large_tolerance():
    gsw_last = _make_date("2026-01-01")
    # 2026-01-29 is exactly 20 business days after 2026-01-01: beyond the
    # documented 15-bd hard maximum, but within a tolerance of 30.
    missing = _make_dti("2026-01-29")
    failures: list[str] = []

    check_coverage_gaps(missing, gsw_last, "daily", max_tail_gap_bd=30, failures=failures)

    assert len(failures) == 1
    assert "exceeds 15-bd maximum" in failures[0]


# ---------------------------------------------------------------------------
# Per-family gate: ACMY fine, ACMRNY rmse over limit → fail
# ---------------------------------------------------------------------------


def test_per_family_rny_rmse_exceeded_produces_failure():
    """A synthetic summary where ACMRNY rmse exceeds --max-rmse-bp should fail."""
    from reproduce_acm import assert_official_reproduced

    dates = ["2020-01-31", "2020-02-28", "2020-03-31"]
    idx = pd.DatetimeIndex(dates)
    idx.name = "DATE"

    # Build generated and reference panels that differ by 0.1 bp in ACMRNY01
    # so rmse is 0.1 bp (> default 0.005 threshold).
    gen = _clean_panel(dates)
    off = gen.copy()
    # Shift ACMRNY01 by 0.001 in panel units (percent) = 0.1 bp in the reference panel
    off["ACMRNY01"] = gen["ACMRNY01"] + 0.001

    from reproduce_acm import compare_panel

    monthly_summary, _ = compare_panel(gen, off)

    gsw_last = _make_date("2020-03-31")

    with pytest.raises(SystemExit) as exc_info:
        assert_official_reproduced(
            monthly_summary=monthly_summary,
            daily_summary=None,
            missing_monthly=pd.DatetimeIndex([]),
            missing_daily=pd.DatetimeIndex([]),
            max_abs_diff_bp=0.5,  # generous — won't trip
            gsw_last=gsw_last,
            max_tail_gap_bd=5,
            acmy_max_abs_diff_bp=1e-4,
            max_rmse_bp=0.005,  # 0.1 bp rmse > 0.005 → should fail
            max_bias_bp=0.001,
            generated_monthly=gen,
            generated_daily=None,
        )

    assert "rmse" in str(exc_info.value).lower() or "ACMRNY" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Per-family gate: all within limits → pass
# ---------------------------------------------------------------------------


def test_per_family_all_within_limits_passes():
    from reproduce_acm import assert_official_reproduced

    dates = ["2020-01-31", "2020-02-28", "2020-03-31"]
    gen = _clean_panel(dates)
    off = gen.copy()
    # Tiny diff: 1e-7 (1e-5 bp) — well within all thresholds
    off["ACMRNY01"] = gen["ACMRNY01"] + 1e-7

    from reproduce_acm import compare_panel

    monthly_summary, _ = compare_panel(gen, off)

    gsw_last = _make_date("2020-03-31")
    # Should not raise
    assert_official_reproduced(
        monthly_summary=monthly_summary,
        daily_summary=None,
        missing_monthly=pd.DatetimeIndex([]),
        missing_daily=pd.DatetimeIndex([]),
        max_abs_diff_bp=0.01,
        gsw_last=gsw_last,
        max_tail_gap_bd=5,
        acmy_max_abs_diff_bp=1e-4,
        max_rmse_bp=0.005,
        max_bias_bp=0.001,
        generated_monthly=gen,
        generated_daily=None,
    )


# ---------------------------------------------------------------------------
# Bias gate: RNY with signed_mean exceeding --max-bias-bp → fail
# ---------------------------------------------------------------------------


def test_bias_gate_detects_dense_systematic_offset():
    """A column where max/rmse are within limits but |signed_mean| is too large.

    This proves the bias gate catches the dense-bias false-negative that the
    plain max-abs-diff gate misses when the error is small but consistently
    signed in one direction.
    """
    from reproduce_acm import assert_official_reproduced

    # 200 monthly observations with a constant 0.02 bp offset on ACMRNY01.
    # rmse = 0.02 bp (> 0.005 default) so we raise max_rmse_bp to 0.1 to let
    # rmse pass; the bias gate (|signed_mean| = 0.02 bp > 0.001) should trip.
    n = 200
    dates = pd.date_range("2000-01-31", periods=n, freq="ME")
    idx = pd.DatetimeIndex(dates)
    idx.name = "DATE"

    gen = _clean_panel([d.strftime("%Y-%m-%d") for d in dates])
    off = gen.copy()
    # systematic offset of 0.0002 in panel units (percent) = 0.02 bp on one ACMRNY column
    off["ACMRNY01"] = gen["ACMRNY01"] + 0.0002

    from reproduce_acm import compare_panel

    monthly_summary, _ = compare_panel(gen, off)

    gsw_last = dates[-1]

    with pytest.raises(SystemExit) as exc_info:
        assert_official_reproduced(
            monthly_summary=monthly_summary,
            daily_summary=None,
            missing_monthly=pd.DatetimeIndex([]),
            missing_daily=pd.DatetimeIndex([]),
            max_abs_diff_bp=0.5,
            gsw_last=gsw_last,
            max_tail_gap_bd=5,
            acmy_max_abs_diff_bp=1e-4,
            max_rmse_bp=0.1,  # generous — rmse passes
            max_bias_bp=0.001,  # |signed_mean| = 0.02 bp > 0.001 → fail
            generated_monthly=gen,
            generated_daily=None,
        )

    assert "|signed_mean|" in str(exc_info.value) or "bias" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# Per-family summary: bias statistic is included in the aggregate
# ---------------------------------------------------------------------------


def test_by_family_summary_includes_bias_statistic():
    from reproduce_acm import compare_panel

    dates = ["2020-01-31", "2020-02-28", "2020-03-31"]
    gen = _clean_panel(dates)
    off = gen.copy()
    # systematic offset of 0.0002 in panel units (percent) = 0.02 bp
    off["ACMRNY01"] = gen["ACMRNY01"] + 0.0002

    _, by_family = compare_panel(gen, off)

    assert "max_abs_signed_mean_bp" in by_family.columns
    rny = by_family[by_family["family"] == "ACMRNY"]
    assert float(rny["max_abs_signed_mean_bp"].iloc[0]) == pytest.approx(0.02, rel=1e-6)


# ---------------------------------------------------------------------------
# Identity gate: ACMY ≠ ACMRNY + ACMTP → fail
# ---------------------------------------------------------------------------


def test_identity_gate_detects_violation():
    dates = ["2020-01-31", "2020-02-28"]
    panel = _clean_panel(dates)
    # Corrupt one maturity so identity breaks
    panel["ACMY01"] = panel["ACMY01"] + 1.0  # 1 percent = 100 bp — huge violation
    failures: list[str] = []

    check_identity_gate(panel, "test panel", failures)

    assert len(failures) == 1
    assert "Identity gate" in failures[0]


# ---------------------------------------------------------------------------
# Identity gate: ACMY == ACMRNY + ACMTP → pass
# ---------------------------------------------------------------------------


def test_identity_gate_passes_when_satisfied():
    dates = ["2020-01-31", "2020-02-28"]
    panel = _clean_panel(dates)  # built with identity satisfied
    failures: list[str] = []

    check_identity_gate(panel, "test panel", failures)

    assert failures == [], f"Expected no failures; got: {failures}"


# ---------------------------------------------------------------------------
# Identity gate: residual is measured in basis points, not panel units
# ---------------------------------------------------------------------------


def test_identity_gate_residual_is_in_basis_points():
    dates = ["2020-01-31", "2020-02-28"]
    panel = _clean_panel(dates)
    # Violate the identity by 1e-9 in panel units (percent) = 1e-7 bp.
    # This exceeds the default 1e-8 bp threshold, but would slip through
    # unnoticed if the percent → bp conversion were dropped.
    panel["ACMY01"] = panel["ACMY01"] + 1e-9
    failures: list[str] = []

    residual_bp = check_identity_gate(panel, "test panel", failures)

    assert residual_bp == pytest.approx(1e-7, rel=1e-3)
    assert len(failures) == 1
    assert "Identity gate" in failures[0]


# ---------------------------------------------------------------------------
# Schema gate: panel with a NaN value → fail
# ---------------------------------------------------------------------------


def test_schema_gate_detects_nan():
    dates = ["2020-01-31", "2020-02-28"]
    panel = _clean_panel(dates)
    panel.at[pd.Timestamp("2020-01-31"), "ACMY01"] = float("nan")
    failures: list[str] = []

    check_schema_gate(panel, "test panel", PUBLISHED_MATURITIES, failures)

    assert any("NaN" in f or "Inf" in f for f in failures)


# ---------------------------------------------------------------------------
# Schema gate: panel with a missing expected column → fail
# ---------------------------------------------------------------------------


def test_schema_gate_detects_missing_column():
    dates = ["2020-01-31", "2020-02-28"]
    panel = _clean_panel(dates)
    panel = panel.drop(columns=["ACMY01"])
    failures: list[str] = []

    check_schema_gate(panel, "test panel", PUBLISHED_MATURITIES, failures)

    assert any("missing" in f.lower() for f in failures)


# ---------------------------------------------------------------------------
# Schema gate: panel with duplicate dates → fail
# ---------------------------------------------------------------------------


def test_schema_gate_detects_duplicate_dates():
    dates = ["2020-01-31", "2020-01-31", "2020-02-28"]
    panel = _clean_panel(dates)
    # Panel already has duplicate index; reset numeric index so concat works
    panel = panel.reset_index(drop=False)
    panel = panel.set_index("DATE")
    failures: list[str] = []

    check_schema_gate(panel, "test panel", PUBLISHED_MATURITIES, failures)

    assert any("duplicate" in f.lower() for f in failures)


# ---------------------------------------------------------------------------
# Schema gate: valid panel passes all schema checks
# ---------------------------------------------------------------------------


def test_schema_gate_passes_on_clean_panel():
    dates = ["2020-01-31", "2020-02-28", "2020-03-31"]
    panel = _clean_panel(dates)
    failures: list[str] = []

    check_schema_gate(panel, "test panel", PUBLISHED_MATURITIES, failures)

    assert failures == [], f"Expected no failures; got: {failures}"
