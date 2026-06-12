"""Pure-function unit tests for reproduce_acm helper functions.

All tests are fully offline — no network calls, no disk writes outside
pytest's tmp_path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import reproduce_acm

# ===========================================================================
# nss_curve_matrix
# ===========================================================================


class TestNssCurveMatrix:
    """Tests for the Nelson-Siegel-Svensson vectorised yield computation."""

    def _single_row_params(self, b0, b1, b2, b3, t1, t2) -> pd.DataFrame:
        return pd.DataFrame(
            {"BETA0": [b0], "BETA1": [b1], "BETA2": [b2], "BETA3": [b3], "TAU1": [t1], "TAU2": [t2]}
        )

    def _nss_closed_form(self, tau_yr, b0, b1, b2, b3, t1, t2) -> float:
        """Scalar reference implementation of the NSS formula."""
        x1 = tau_yr / t1
        term1 = (1.0 - np.exp(-x1)) / x1
        term2 = term1 - np.exp(-x1)
        y = b0 + b1 * term1 + b2 * term2
        if not np.isnan(t2) and t2 > 0:
            x2 = tau_yr / t2
            term3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
            y += b3 * term3
        return y

    def test_known_values_match_closed_form(self):
        """Matrix output equals the scalar closed-form at two chosen maturities."""
        b0, b1, b2, b3, t1, t2 = 5.0, -1.5, 0.8, 0.4, 1.5, 6.0
        tau_yrs = np.array([1.0, 5.0])
        params = self._single_row_params(b0, b1, b2, b3, t1, t2)
        result = reproduce_acm.nss_curve_matrix(tau_yrs, params)  # (1, 2)
        for j, tau in enumerate(tau_yrs):
            expected = self._nss_closed_form(tau, b0, b1, b2, b3, t1, t2)
            assert abs(result[0, j] - expected) < 1e-12, (
                f"tau={tau}: got {result[0, j]:.12f}, expected {expected:.12f}"
            )

    def test_flat_curve_when_only_beta0(self):
        """When beta1=beta2=beta3=0 every maturity yields exactly beta0 (÷100 not applied here)."""
        b0 = 4.5
        tau_yrs = np.array([1.0, 2.0, 5.0, 10.0])
        params = self._single_row_params(b0, 0.0, 0.0, 0.0, 1.5, 6.0)
        result = reproduce_acm.nss_curve_matrix(tau_yrs, params)
        np.testing.assert_allclose(
            result[0], b0, atol=1e-12, err_msg="Flat curve: all yields should equal BETA0"
        )

    def test_nan_tau2_uses_three_factor_form(self):
        """Rows with NaN tau2 should fall back to Nelson-Siegel (no beta3 term)."""
        b0, b1, b2, b3, t1 = 5.0, -1.0, 0.5, 99.0, 1.5
        tau_yrs = np.array([2.0])
        params_nan = self._single_row_params(b0, b1, b2, b3, t1, np.nan)
        params_ref = self._single_row_params(b0, b1, b2, 0.0, t1, np.nan)
        result_nan = reproduce_acm.nss_curve_matrix(tau_yrs, params_nan)
        result_ref = reproduce_acm.nss_curve_matrix(tau_yrs, params_ref)
        np.testing.assert_allclose(
            result_nan, result_ref, atol=1e-12, err_msg="NaN tau2: beta3 term should be suppressed"
        )

    def test_zero_tau2_uses_three_factor_form(self):
        """Rows with tau2 ≤ 0 should fall back to three-factor Nelson-Siegel."""
        b0, b1, b2, b3, t1 = 5.0, -1.0, 0.5, 99.0, 1.5
        tau_yrs = np.array([2.0])
        params_zero = self._single_row_params(b0, b1, b2, b3, t1, 0.0)
        params_ref = self._single_row_params(b0, b1, b2, 0.0, t1, np.nan)
        result_zero = reproduce_acm.nss_curve_matrix(tau_yrs, params_zero)
        result_ref = reproduce_acm.nss_curve_matrix(tau_yrs, params_ref)
        np.testing.assert_allclose(
            result_zero, result_ref, atol=1e-12, err_msg="tau2=0: beta3 term should be suppressed"
        )

    def test_vectorised_equals_row_by_row(self, synthetic_nss_params_monthly, tau_years_all):
        """Matrix output equals stacking single-row calls (first 10 rows for speed)."""
        params_sub = synthetic_nss_params_monthly.iloc[:10]
        batch_result = reproduce_acm.nss_curve_matrix(tau_years_all, params_sub)
        for i in range(len(params_sub)):
            single = reproduce_acm.nss_curve_matrix(tau_years_all, params_sub.iloc[[i]])
            np.testing.assert_allclose(
                batch_result[i], single[0], atol=1e-14, err_msg=f"Row {i}: vectorised != single-row"
            )

    def test_output_shape(self):
        """Output shape is (n_dates, n_maturities)."""
        n_dates, n_mats = 7, 12
        tau_yrs = np.linspace(1 / 12, 1.0, n_mats)
        rows = [
            {"BETA0": 5.0, "BETA1": -1.0, "BETA2": 0.3, "BETA3": 0.2, "TAU1": 1.5, "TAU2": 6.0}
        ] * n_dates
        params = pd.DataFrame(rows)
        result = reproduce_acm.nss_curve_matrix(tau_yrs, params)
        assert result.shape == (n_dates, n_mats)


# ===========================================================================
# gsw_header_offset
# ===========================================================================


class TestGswHeaderOffset:
    """Tests for locating the 'Date,' header row in raw GSW CSV bytes."""

    def test_header_at_first_line(self):
        raw = b"Date,BETA0,BETA1\n1990-01-01,5.0,-1.0\n"
        assert reproduce_acm.gsw_header_offset(raw) == 0

    def test_header_offset_five_lines(self):
        preamble = b"# comment\n# comment\n# comment\n# comment\n# comment\n"
        header = b"Date,BETA0,BETA1\n1990-01-01,5.0,-1.0\n"
        raw = preamble + header
        assert reproduce_acm.gsw_header_offset(raw) == 5

    def test_header_offset_single_preamble_line(self):
        raw = b"Unique identifier\nDate,BETA0\n2000-06-30,4.0\n"
        assert reproduce_acm.gsw_header_offset(raw) == 1

    def test_missing_header_raises(self):
        raw = b"no header here\njust garbage\n"
        with pytest.raises(ValueError, match="Date,"):
            reproduce_acm.gsw_header_offset(raw)

    def test_empty_bytes_raises(self):
        with pytest.raises(ValueError):
            reproduce_acm.gsw_header_offset(b"")


# ===========================================================================
# parse_fedfunds
# ===========================================================================


class TestParseFedFunds:
    """Tests for both FRED and H.15 package CSV formats."""

    # --- FRED format --------------------------------------------------------

    def _fred_bytes(self, rows: list[tuple[str, str]]) -> bytes:
        lines = ["observation_date,FEDFUNDS"]
        for date, val in rows:
            lines.append(f"{date},{val}")
        return "\n".join(lines).encode()

    def test_fred_format_parses_values(self):
        raw = self._fred_bytes([("2020-01-01", "1.55"), ("2020-02-01", "1.58")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert isinstance(result, pd.Series)
        assert result.name == "FEDFUNDS"
        assert len(result) == 2
        np.testing.assert_allclose(result.iloc[0], 1.55 / 100, atol=1e-12)
        np.testing.assert_allclose(result.iloc[1], 1.58 / 100, atol=1e-12)

    def test_fred_format_index_is_month_end(self):
        raw = self._fred_bytes([("2020-03-01", "1.00")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert result.index[0] == pd.Timestamp("2020-03-31")

    def test_fred_format_dot_dropped(self):
        """A '.' entry should be treated as NaN and dropped."""
        raw = self._fred_bytes([("2020-01-01", "1.55"), ("2020-02-01", ".")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert len(result) == 1

    def test_fred_format_nd_dropped(self):
        """An 'ND' entry should be treated as NaN and dropped."""
        raw = self._fred_bytes([("2020-01-01", "ND"), ("2020-02-01", "1.58")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert len(result) == 1
        np.testing.assert_allclose(result.iloc[0], 1.58 / 100, atol=1e-12)

    def test_fred_format_sorted(self):
        raw = self._fred_bytes([("2020-06-01", "0.08"), ("2019-12-01", "1.55")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert result.index.is_monotonic_increasing

    # --- H.15 package format ------------------------------------------------

    def _h15_bytes(self, rows: list[tuple[str, str]]) -> bytes:
        """Build a minimal H.15 package CSV.

        The real file has a 'Series Description' column in the first 5 rows,
        then a blank skip block, then row 6 starts with "Time Period,RIFSPFF_N.M".
        reproduce_acm skips 5 rows and reads from there.
        """
        header_block = (
            "Series Description,Federal funds (effective),,,\n"
            "Unit:,Percent:_Annual,,,\n"
            "Multiplier:,1,,,\n"
            "Currency:,NA,,,\n"
            "Unique Identifier:,H15/H15/RIFSPFF_N.M,,,\n"
        )
        data_lines = ["Time Period,RIFSPFF_N.M"]
        for date, val in rows:
            data_lines.append(f"{date},{val}")
        return (header_block + "\n".join(data_lines)).encode()

    def test_h15_format_parses_values(self):
        raw = self._h15_bytes([("1990-01", "8.11"), ("1990-02", "8.24")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert len(result) == 2
        np.testing.assert_allclose(result.iloc[0], 8.11 / 100, atol=1e-12)

    def test_h15_format_index_is_month_end(self):
        raw = self._h15_bytes([("2000-06", "6.54")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert result.index[0] == pd.Timestamp("2000-06-30")

    def test_h15_format_dot_dropped(self):
        raw = self._h15_bytes([("1990-01", "."), ("1990-02", "8.24")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert len(result) == 1

    def test_h15_format_nd_dropped(self):
        raw = self._h15_bytes([("1990-01", "ND"), ("1990-02", "8.24")])
        result = reproduce_acm.parse_fedfunds(raw)
        assert len(result) == 1

    def test_unknown_format_raises(self):
        raw = b"col_a,col_b\n1,2\n3,4\n"
        with pytest.raises(ValueError, match="Unrecognized"):
            reproduce_acm.parse_fedfunds(raw)


# ===========================================================================
# get_excess_returns
# ===========================================================================


class TestGetExcessReturns:
    """Tests for the excess return calculation."""

    def _minimal_curve(self, n_rows: int = 30) -> pd.DataFrame:
        """Return a small synthetic monthly curve with all 120 maturities."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2000-01-31", periods=n_rows, freq="ME")
        # Reasonably shaped yield curve in decimal form
        mats = np.arange(1, 121)
        base = 0.04 + 0.01 * (1.0 - np.exp(-mats / 30))
        noise = rng.normal(0, 0.001, (n_rows, 120))
        values = base[None, :] + noise
        values = np.clip(values, 1e-4, None)
        curve = pd.DataFrame(values, index=dates, columns=reproduce_acm.ALL_MATURITIES)
        curve.index.name = "DATE"
        return curve

    def test_column_one_is_zero(self):
        """Column 1 (1-month maturity) must be exactly 0.0 everywhere."""
        rx = reproduce_acm.get_excess_returns(self._minimal_curve())
        assert (rx[1] == 0.0).all(), "Column 1 excess return must be identically 0"

    def test_first_all_nan_row_dropped(self):
        """The function should drop the all-NaN first row from the result."""
        curve = self._minimal_curve(n_rows=20)
        rx = reproduce_acm.get_excess_returns(curve)
        # No all-NaN rows after the drop
        assert not rx.isnull().all(axis=1).any(), "All-NaN rows should have been dropped"

    def test_columns_span_all_maturities(self):
        rx = reproduce_acm.get_excess_returns(self._minimal_curve())
        assert list(rx.columns) == reproduce_acm.ALL_MATURITIES

    def test_flat_constant_curve_gives_near_zero_excess_returns(self):
        """Perfectly flat, constant curve: excess returns ≈ 0."""
        n_rows = 40
        dates = pd.date_range("2000-01-31", periods=n_rows, freq="ME")
        level = 0.05  # 5 % flat
        mats = reproduce_acm.ALL_MATURITIES
        values = np.full((n_rows, len(mats)), level)
        curve = pd.DataFrame(values, index=dates, columns=mats)
        curve.index.name = "DATE"
        rx = reproduce_acm.get_excess_returns(curve)
        # Exclude column 1 (pinned to 0) and check non-NaN entries ≈ 0
        non_first = rx.drop(columns=[1]).dropna(how="all")
        np.testing.assert_allclose(
            non_first.values,
            0.0,
            atol=1e-10,
            err_msg="Flat constant curve: excess returns should be ≈ 0",
        )


# ===========================================================================
# maturity_suffix
# ===========================================================================


class TestMaturitySuffix:
    """Tests for the maturity label formatter."""

    def test_six_months(self):
        assert reproduce_acm.maturity_suffix(6) == "006M"

    def test_eleven_months(self):
        assert reproduce_acm.maturity_suffix(11) == "011M"

    def test_twelve_months(self):
        assert reproduce_acm.maturity_suffix(12) == "01"

    def test_twenty_four_months(self):
        assert reproduce_acm.maturity_suffix(24) == "02"

    def test_one_hundred_twenty_months(self):
        assert reproduce_acm.maturity_suffix(120) == "10"

    def test_non_multiple_three_digit(self):
        assert reproduce_acm.maturity_suffix(3) == "003M"

    def test_non_multiple_two_digit(self):
        # 18 months is not a multiple of 12 in a way that gives an integer year
        # Actually 18 % 12 == 6 != 0, so sub-annual suffix
        assert reproduce_acm.maturity_suffix(18) == "018M"


# ===========================================================================
# completed_monthly_dates
# ===========================================================================


class TestCompletedMonthlyDates:
    """Tests for selecting completed (vs. partial) months from a daily curve."""

    def _curve_for_index(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        """Minimal DataFrame with the given DatetimeIndex."""
        return pd.DataFrame(
            {"val": np.ones(len(index))},
            index=index,
        )

    def test_mid_month_end_excludes_partial_month_by_default(self):
        """Index ending mid-month ⇒ current partial month excluded."""
        # Build a daily index that ends on the 15th of the last month.
        dates = pd.bdate_range("2020-01-02", "2020-03-15")
        curve = self._curve_for_index(dates)
        result = reproduce_acm.completed_monthly_dates(curve, include_partial_current_month=False)
        # Should include Jan and Feb month-ends only, not March
        periods = result.to_period("M")
        assert pd.Period("2020-03", "M") not in periods
        assert pd.Period("2020-02", "M") in periods

    def test_mid_month_end_includes_partial_when_flag_set(self):
        """include_partial_current_month=True includes the partial month."""
        dates = pd.bdate_range("2020-01-02", "2020-03-15")
        curve = self._curve_for_index(dates)
        result = reproduce_acm.completed_monthly_dates(curve, include_partial_current_month=True)
        periods = result.to_period("M")
        assert pd.Period("2020-03", "M") in periods

    def test_business_month_end_includes_that_month(self):
        """Index ending on last business day of a month ⇒ that month included."""
        # 2020-01-31 is a Friday — last business day of January 2020.
        dates = pd.bdate_range("2020-01-02", "2020-01-31")
        curve = self._curve_for_index(dates)
        result = reproduce_acm.completed_monthly_dates(curve, include_partial_current_month=False)
        periods = result.to_period("M")
        assert pd.Period("2020-01", "M") in periods

    def test_empty_curve_returns_empty(self):
        empty_idx = pd.DatetimeIndex([], name="DATE")
        curve = pd.DataFrame({"val": []}, index=empty_idx)
        result = reproduce_acm.completed_monthly_dates(curve, include_partial_current_month=False)
        assert len(result) == 0


# ===========================================================================
# smooth_pre_1982_one_month_rate
# ===========================================================================


class TestSmoothPre1982OneMonthRate:
    """Tests for the FEDFUNDS-based smoothing of the 1M GSW yield pre-1982."""

    def _build_inputs(self, rng: np.random.Generator):
        """Return (curve_m, fedfunds, true_intercept, true_slope).

        Post-1982 monthly observations: 1M yield = intercept + slope * FEDFUNDS + noise.
        Pre-1982 monthly observations:  1M yield = arbitrary values.
        """
        true_intercept = 0.003
        true_slope = 0.85

        # Post-1982 dates (120 months starting 1982-01)
        post_dates = pd.date_range("1982-01-31", periods=120, freq="ME")
        ff_post = rng.uniform(0.03, 0.12, 120)
        yield_post = true_intercept + true_slope * ff_post + rng.normal(0, 0.0005, 120)

        # Pre-1982 dates (24 months)
        pre_dates = pd.date_range("1980-01-31", periods=24, freq="ME")
        ff_pre = rng.uniform(0.06, 0.18, 24)
        yield_pre = rng.uniform(0.05, 0.15, 24)  # arbitrary

        all_dates = pre_dates.append(post_dates)
        all_ff = np.concatenate([ff_pre, ff_post])
        all_1m = np.concatenate([yield_pre, yield_post])

        # Build curve_m with all 120 maturity columns; fill cols 2..120 simply
        mats = reproduce_acm.ALL_MATURITIES
        n = len(all_dates)
        other_yields = (
            0.04
            + 0.001 * np.arange(len(mats) - 1)[None, :]
            + rng.normal(0, 0.0001, (n, len(mats) - 1))
        )
        data = np.column_stack([all_1m, other_yields])
        curve_m = pd.DataFrame(data, index=all_dates, columns=mats)
        curve_m.index.name = "DATE"

        fedfunds = pd.Series(all_ff, index=all_dates, name="FEDFUNDS")
        return curve_m, fedfunds, true_intercept, true_slope

    def test_beta_recovers_truth_loosely(self):
        """Fitted beta ≈ true intercept/slope (loose tolerance due to noise)."""
        rng = np.random.default_rng(7)
        curve_m, fedfunds, true_intercept, true_slope = self._build_inputs(rng)
        _, beta, _, _ = reproduce_acm.smooth_pre_1982_one_month_rate(curve_m, fedfunds)
        assert abs(beta[0] - true_intercept) < 0.005, (
            f"Intercept {beta[0]:.6f} far from truth {true_intercept}"
        )
        assert abs(beta[1] - true_slope) < 0.1, f"Slope {beta[1]:.6f} far from truth {true_slope}"

    def test_only_pre_1982_column_1_changed(self):
        """Smoothing modifies only column 1, and only for pre-1982 dates."""
        rng = np.random.default_rng(7)
        curve_m, fedfunds, _, _ = self._build_inputs(rng)
        out, _, _, _ = reproduce_acm.smooth_pre_1982_one_month_rate(curve_m, fedfunds)

        smoothing_start = pd.Timestamp("1982-01-31")
        pre_mask = out.index < smoothing_start
        post_mask = out.index >= smoothing_start

        # Post-1982 column 1: unchanged
        pd.testing.assert_series_equal(
            out.loc[post_mask, 1],
            curve_m.loc[post_mask, 1],
            check_names=False,
        )
        # Columns 2..120: entirely unchanged everywhere
        for col in reproduce_acm.ALL_MATURITIES[1:]:
            pd.testing.assert_series_equal(out[col], curve_m[col], check_names=False)
        # Pre-1982 column 1: values are now different from original
        changed = (out.loc[pre_mask, 1] != curve_m.loc[pre_mask, 1]).any()
        assert changed, "Pre-1982 column 1 should have been modified"

    def test_returns_fit_and_replace_counts(self):
        """The function returns plausible fit and replace month counts."""
        rng = np.random.default_rng(7)
        curve_m, fedfunds, _, _ = self._build_inputs(rng)
        _, _, n_fit, n_replace = reproduce_acm.smooth_pre_1982_one_month_rate(curve_m, fedfunds)
        assert n_fit > 0
        assert n_replace > 0


# ===========================================================================
# read_or_download
# ===========================================================================


class TestReadOrDownload:
    def test_download_writes_cache_atomically(self, tmp_path, monkeypatch):
        monkeypatch.setattr(reproduce_acm, "fetch_url", lambda url: b"payload")
        cache_path = tmp_path / "cache.csv"
        data = reproduce_acm.read_or_download("https://example.com/x.csv", cache_path, refresh=False)
        assert data == b"payload"
        assert cache_path.read_bytes() == b"payload"
        leftovers = [p for p in tmp_path.iterdir() if p != cache_path]
        assert leftovers == []

    def test_cache_hit_skips_download(self, tmp_path, monkeypatch):
        def boom(url):
            raise AssertionError("must not download on cache hit")

        monkeypatch.setattr(reproduce_acm, "fetch_url", boom)
        cache_path = tmp_path / "cache.csv"
        cache_path.write_bytes(b"cached")
        data = reproduce_acm.read_or_download("https://example.com/x.csv", cache_path, refresh=False)
        assert data == b"cached"


# ===========================================================================
# load_fedfunds stale-cache fallback
# ===========================================================================


class TestLoadFedfundsFallback:
    def test_refresh_parse_failure_falls_back_to_prior_cache(self, tmp_path, monkeypatch):
        good_csv = b"observation_date,FEDFUNDS\n2020-01-01,1.55\n2020-02-01,1.58\n"
        cache_path = tmp_path / "H15_FEDFUNDS_monthly.csv"
        cache_path.write_bytes(good_csv)
        # Fresh download succeeds at the HTTP layer but is unparseable.
        monkeypatch.setattr(reproduce_acm, "fetch_url", lambda url: b"<html>maintenance</html>")

        fedfunds, used_path, source = reproduce_acm.load_fedfunds(tmp_path, refresh=True)

        assert "stale cache fallback" in source
        assert used_path == cache_path
        assert len(fedfunds) == 2
        assert fedfunds.iloc[0] == 1.55 / 100.0
        # The refresh wrote the broken payload to disk; the fallback must
        # restore the last good copy or the next run starts from poison.
        assert cache_path.read_bytes() == good_csv


# ===========================================================================
# ensure_finite
# ===========================================================================


class TestEnsureFinite:
    def test_passes_on_finite_frame(self):
        frame = pd.DataFrame({"a": [1.0, 2.0]})
        reproduce_acm.ensure_finite("panel", frame)

    def test_raises_on_nan(self):
        frame = pd.DataFrame({"a": [1.0, np.nan]})
        with pytest.raises(ValueError, match="panel"):
            reproduce_acm.ensure_finite("panel", frame)

    def test_raises_on_inf(self):
        frame = pd.DataFrame({"a": [1.0, np.inf]})
        with pytest.raises(ValueError, match="panel"):
            reproduce_acm.ensure_finite("panel", frame)
