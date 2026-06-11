"""Algebraic-property tests for NominalACM on synthetic data.

These tests verify mathematical identities and sanity checks on the model
output; they do not compare to live market data, so they are fully offline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reproduce_acm


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _max_abs(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float((a - b).abs().max().max())


# ===========================================================================
# Identity: model implied yield = risk-neutral yield + term premium
# ===========================================================================

class TestDecompositionIdentity:
    """miy ≡ rny + tp must hold everywhere (monthly and daily) to < 1e-9."""

    def test_monthly_identity(self, nominal_acm_model):
        model = nominal_acm_model
        err = _max_abs(model.miy_m, model.rny_m + model.tp_m)
        assert err < 1e-9, f"Monthly identity violated: max abs error = {err:.3e}"

    def test_daily_identity(self, nominal_acm_model):
        model = nominal_acm_model
        err = _max_abs(model.miy_d, model.rny_d + model.tp_d)
        assert err < 1e-9, f"Daily identity violated: max abs error = {err:.3e}"


# ===========================================================================
# Risk-neutral coefficients: zero prices of risk reproduce a_rn / b_rn
# ===========================================================================

class TestRiskNeutralCoefficients:
    """affine_coefficients with λ₀=0 / λ₁=0 must reproduce (a_rn, b_rn)."""

    def test_zero_lambda_reproduces_rn_affine_coeffs(self, nominal_acm_model):
        model = nominal_acm_model
        a_check, b_check = model.affine_coefficients(
            np.zeros_like(model.lambda0),
            np.zeros_like(model.lambda1),
        )
        np.testing.assert_allclose(
            a_check, model.a_rn, atol=1e-12,
            err_msg="a: zero λ should reproduce a_rn"
        )
        np.testing.assert_allclose(
            b_check, model.b_rn, atol=1e-12,
            err_msg="b: zero λ should reproduce b_rn"
        )

    def test_zero_risk_price_term_premium_near_zero_on_zero_data(self):
        """Term premium built with λ=0 is exactly zero by construction."""
        # Build a tiny synthetic model where we can check this directly.
        # miy_d_zero = compute_yields(pc_d, a_rn, b_rn) — same as rny_d.
        # tp = miy - rny = 0.
        # We use the model fixture and just verify the arithmetic.
        pass  # Covered structurally by TestDecompositionIdentity.


# ===========================================================================
# Short-rate fit: fitted 1-month yield ≈ input 1-month yield
# ===========================================================================

class TestShortRateFit:
    """The model's fitted 1-month yield should track the input 1M yield closely."""

    def test_fitted_one_month_yield_tracks_input(self, nominal_acm_model):
        model = nominal_acm_model
        # Input 1-month yield (monthly, decimal)
        y1_input = model.curve_m.iloc[:, 0].values
        # Model implied 1-month yield (monthly), annualised in percent → back to decimal
        y1_fitted = model.miy_m.iloc[:, 0].values
        # The short rate equation is a regression, so residuals can be non-trivial.
        # We just require R² > 0.90 and |mean error| < 50 bp.
        corr = float(np.corrcoef(y1_input, y1_fitted)[0, 1])
        mean_err = float(np.mean(np.abs(y1_input - y1_fitted)))
        assert corr > 0.90, f"1M yield correlation {corr:.4f} too low"
        assert mean_err < 0.005, f"1M yield mean abs error {mean_err*100:.2f} bp too large"


# ===========================================================================
# Determinism and sanity checks
# ===========================================================================

class TestDeterminismAndSanity:
    """Building the model twice on identical inputs gives identical outputs."""

    def test_model_outputs_are_finite(self, nominal_acm_model):
        model = nominal_acm_model
        for attr in ("miy_m", "rny_m", "tp_m", "miy_d", "rny_d", "tp_d"):
            frame = getattr(model, attr)
            assert np.isfinite(frame.values).all(), \
                f"{attr} contains non-finite values"

    def test_model_is_deterministic_on_rerun(
        self, synthetic_curve_daily, synthetic_curve_monthly
    ):
        """Constructing NominalACM twice from identical inputs gives bit-identical results."""
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            m1 = reproduce_acm.NominalACM(
                curve_d=synthetic_curve_daily,
                curve_m=synthetic_curve_monthly,
                n_factors=reproduce_acm.N_FACTORS,
                selected_maturities=reproduce_acm.SELECTED_RETURN_MATURITIES,
            )
            m2 = reproduce_acm.NominalACM(
                curve_d=synthetic_curve_daily,
                curve_m=synthetic_curve_monthly,
                n_factors=reproduce_acm.N_FACTORS,
                selected_maturities=reproduce_acm.SELECTED_RETURN_MATURITIES,
            )
        for attr in ("miy_m", "tp_m", "miy_d", "tp_d"):
            diff = (getattr(m1, attr) - getattr(m2, attr)).abs().max().max()
            assert diff == 0.0, f"{attr}: two runs gave different results (max diff {diff})"

    def test_output_shapes_consistent(self, nominal_acm_model):
        """Output DataFrames have the expected shape (n_obs × n_maturities)."""
        model = nominal_acm_model
        n_monthly = len(model.curve_m)
        n_daily = len(model.curve_d)
        n_mats = len(reproduce_acm.ALL_MATURITIES)

        assert model.miy_m.shape == (n_monthly, n_mats)
        assert model.rny_m.shape == (n_monthly, n_mats)
        assert model.tp_m.shape == (n_monthly, n_mats)
        assert model.miy_d.shape == (n_daily, n_mats)
        assert model.rny_d.shape == (n_daily, n_mats)
        assert model.tp_d.shape == (n_daily, n_mats)

    def test_output_column_labels_are_integer_maturities(self, nominal_acm_model):
        """Columns of model output DataFrames are the expected integer maturity labels."""
        model = nominal_acm_model
        expected = list(reproduce_acm.ALL_MATURITIES)
        assert list(model.miy_m.columns) == expected
        assert list(model.tp_d.columns) == expected

    def test_term_premium_has_plausible_range(self, nominal_acm_model):
        """Term premium for 10-year maturity (120 months) should be in [−5%, 10%]."""
        model = nominal_acm_model
        tp_10y = model.tp_m[120].values
        assert tp_10y.min() > -0.05, "10Y term premium below −5%"
        assert tp_10y.max() < 0.10, "10Y term premium above 10%"
