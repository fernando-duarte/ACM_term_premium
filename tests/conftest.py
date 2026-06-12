"""Shared pytest fixtures for the offline ACM test suite.

All fixtures produce synthetic, deterministic data (seeded with
np.random.default_rng(0)).  No network I/O is performed here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reproduce_acm

# ---------------------------------------------------------------------------
# Constants that drive synthetic data generation
# ---------------------------------------------------------------------------

_RNG_SEED = 0
_N_MONTHS = 300  # ≈ 25 years of monthly observations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_month_ends(n: int, start: str = "1995-01-31") -> pd.DatetimeIndex:
    """Return *n* month-end timestamps starting from *start*."""
    base = pd.date_range(start, periods=n, freq="ME")
    return base


def _make_nss_params(rng: np.random.Generator, n: int) -> pd.DataFrame:
    """Return a DataFrame of smooth NSS params over *n* dates.

    Parameters vary slowly (random-walk with small steps) so the curve is
    always realistic and well-conditioned.
    """
    # Start from plausible US-treasury-like values (rates in %).
    beta0 = np.cumsum(rng.normal(0, 0.01, n)) + 5.0  # long-run level
    beta1 = np.cumsum(rng.normal(0, 0.01, n)) - 2.0  # slope
    beta2 = np.cumsum(rng.normal(0, 0.02, n)) + 1.0  # curvature
    beta3 = np.cumsum(rng.normal(0, 0.02, n)) + 0.5  # 2nd curvature
    tau1 = np.clip(np.cumsum(rng.normal(0, 0.01, n)) + 1.5, 0.5, 5.0)
    tau2 = np.clip(np.cumsum(rng.normal(0, 0.01, n)) + 5.0, 2.0, 15.0)

    return pd.DataFrame(
        {
            "BETA0": beta0,
            "BETA1": beta1,
            "BETA2": beta2,
            "BETA3": beta3,
            "TAU1": tau1,
            "TAU2": tau2,
        }
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def synthetic_nss_params_monthly() -> pd.DataFrame:
    """NSS params DataFrame indexed by month-end dates (session scope)."""
    rng = np.random.default_rng(_RNG_SEED)
    month_ends = _make_month_ends(_N_MONTHS)
    params = _make_nss_params(rng, _N_MONTHS)
    params.index = month_ends
    params.index.name = "DATE"
    return params


@pytest.fixture(scope="session")
def synthetic_nss_params_daily(synthetic_nss_params_monthly) -> pd.DataFrame:
    """NSS params reindexed to a dense business-day calendar (session scope).

    We forward-fill the monthly params to every business day so we get a
    realistic daily panel without separately generating per-day betas.
    """
    month_ends = synthetic_nss_params_monthly.index
    start = month_ends.min()
    end = month_ends.max()
    bdays = pd.bdate_range(start, end)
    daily = synthetic_nss_params_monthly.reindex(bdays, method="ffill")
    daily.index.name = "DATE"
    return daily


@pytest.fixture(scope="session")
def tau_years_all() -> np.ndarray:
    """Maturity array in years for all 120 maturities (1..120 months)."""
    return np.array(reproduce_acm.ALL_MATURITIES, dtype=float) / 12.0


@pytest.fixture(scope="session")
def synthetic_curve_monthly(synthetic_nss_params_monthly, tau_years_all) -> pd.DataFrame:
    """Monthly yield curve (120 columns, month-end index) in decimal form."""
    values = reproduce_acm.nss_curve_matrix(tau_years_all, synthetic_nss_params_monthly)
    curve = pd.DataFrame(
        values / 100.0,
        index=synthetic_nss_params_monthly.index,
        columns=reproduce_acm.ALL_MATURITIES,
    )
    curve.index.name = "DATE"
    return curve


@pytest.fixture(scope="session")
def synthetic_curve_daily(synthetic_nss_params_daily, tau_years_all) -> pd.DataFrame:
    """Daily yield curve (120 columns, business-day index) in decimal form."""
    values = reproduce_acm.nss_curve_matrix(tau_years_all, synthetic_nss_params_daily)
    curve = pd.DataFrame(
        values / 100.0,
        index=synthetic_nss_params_daily.index,
        columns=reproduce_acm.ALL_MATURITIES,
    )
    curve.index.name = "DATE"
    return curve


@pytest.fixture(scope="session")
def synthetic_fedfunds(synthetic_curve_monthly) -> pd.Series:
    """Monthly FEDFUNDS series loosely tracking the 1-month yield."""
    rng = np.random.default_rng(_RNG_SEED + 1)
    one_month = synthetic_curve_monthly[1].copy()
    # Add small noise around the 1M yield (both expressed in decimals).
    noise = rng.normal(0, 0.001, len(one_month))
    ff = (one_month.values + noise).clip(0.0001)
    series = pd.Series(ff, index=one_month.index, name="FEDFUNDS")
    return series


@pytest.fixture(scope="session")
def nominal_acm_model(synthetic_curve_daily, synthetic_curve_monthly) -> reproduce_acm.NominalACM:
    """An estimated NominalACM model on the synthetic curves."""
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        model = reproduce_acm.NominalACM(
            curve_d=synthetic_curve_daily,
            curve_m=synthetic_curve_monthly,
            n_factors=reproduce_acm.N_FACTORS,
            selected_maturities=reproduce_acm.SELECTED_RETURN_MATURITIES,
        )
    return model
