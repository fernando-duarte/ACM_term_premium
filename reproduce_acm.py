from __future__ import annotations

import argparse
import io
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATE_COLUMN = "DATE"
MONTHLY_SHEET = "ACM Monthly"
DAILY_SHEET = "ACM Daily"
EXCEL_DATE_FORMAT = "m/d/yyyy"

DEFAULT_REPRODUCTION_OUTPUT = Path("outputs/ACMTermPremium_reproduced.xlsx")
DEFAULT_UPDATE_OUTPUT = Path("outputs/ACMTermPremium_updated.xlsx")
DEFAULT_CACHE_DIR = Path("data_cache")

OFFICIAL_ACM_URL = (
    "https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls"
)
GSW_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv"
FEDFUNDS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
# H.15 data-download "package" CSV for the monthly effective federal funds rate
# (series RIFSPFF_N.M). The opaque ``series`` token is the Fed's stable handle
# for that selection; if it ever changes, ``load_fedfunds`` falls back to FRED.
H15_MONTHLY_URL = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?"
    "rel=H15&series=d7e27b7b09a3a7feae95b9c61781fcd8&lastObs=&from=&to=&"
    "filetype=csv&label=include&layout=seriescolumn&type=package"
)

N_FACTORS = 5
PC_MATURITIES = list(range(3, 121))
SELECTED_RETURN_MATURITIES = [6, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120]
PUBLISHED_MATURITIES = list(range(12, 121, 12))
EXPANDED_MONTHLY_MATURITIES = list(range(6, 121))
ALL_MATURITIES = list(range(1, 121))
SMOOTHING_START = pd.Period("1982-01", freq="M")


def fetch_url(url: str) -> bytes:
    last_error = None
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
            if not data:
                raise ValueError(f"Downloaded empty response from {url}")
            return data
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(2**attempt)

    raise RuntimeError(f"Could not download {url}: {last_error!r}")


def read_or_download(url: str, cache_path: Path, refresh: bool) -> bytes:
    if cache_path.exists() and not refresh:
        data = cache_path.read_bytes()
        if data:
            age_days = (time.time() - cache_path.stat().st_mtime) / 86400.0
            print(f"Using cached {cache_path.name} ({age_days:.0f} days old; --refresh to update)")
            return data
        cache_path.unlink()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    data = fetch_url(url)
    # Write-then-rename so an interrupted run never leaves a truncated cache.
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
    temp_path.write_bytes(data)
    os.replace(temp_path, cache_path)
    return data


def is_url(value: str) -> bool:
    parsed = urllib.parse.urlparse(value)
    return parsed.scheme in {"http", "https"}


def cached_filename_from_url(url: str, fallback: str) -> str:
    name = Path(urllib.parse.urlparse(url).path).name
    return name or fallback


def nss_curve_matrix(tau_years: np.ndarray, params: pd.DataFrame) -> np.ndarray:
    """Nelson-Siegel-Svensson zero-coupon yields for every date at once.

    Returns a ``(n_dates, n_maturities)`` array. The Svensson (``beta3`` /
    ``tau2``) term is included only for rows with a valid positive ``tau2``;
    the remaining rows fall back to the three-factor Nelson-Siegel form.
    """
    beta0 = params["BETA0"].to_numpy()[:, None]
    beta1 = params["BETA1"].to_numpy()[:, None]
    beta2 = params["BETA2"].to_numpy()[:, None]
    beta3 = params["BETA3"].to_numpy()[:, None]
    tau1 = params["TAU1"].to_numpy()[:, None]
    tau2 = params["TAU2"].to_numpy()[:, None]
    tau = tau_years[None, :]

    x1 = tau / tau1
    term1 = (1.0 - np.exp(-x1)) / x1
    term2 = term1 - np.exp(-x1)
    yields = beta0 + beta1 * term1 + beta2 * term2

    has_tau2 = ~np.isnan(tau2) & (tau2 > 0)
    x2 = tau / np.where(has_tau2, tau2, 1.0)
    term3 = (1.0 - np.exp(-x2)) / x2 - np.exp(-x2)
    yields = yields + np.where(has_tau2, beta3, 0.0) * term3
    return yields


def gsw_header_offset(raw: bytes) -> int:
    for index, line in enumerate(raw.decode("utf-8", "replace").splitlines()):
        if line.startswith("Date,"):
            return index
    raise ValueError("Could not locate the 'Date,' header row in feds200628.csv.")


def load_gsw_curve(cache_dir: Path, refresh: bool) -> tuple[pd.DataFrame, Path]:
    cache_path = cache_dir / "feds200628.csv"
    raw = read_or_download(GSW_URL, cache_path, refresh)
    params = pd.read_csv(
        io.BytesIO(raw),
        skiprows=gsw_header_offset(raw),
        index_col="Date",
        na_values=["NA", "-999.99"],
    )
    params.index = pd.to_datetime(params.index)
    params.index.name = DATE_COLUMN
    params = params[["BETA0", "BETA1", "BETA2", "BETA3", "TAU1", "TAU2"]]
    params = params.dropna(subset=["BETA0", "BETA1", "BETA2", "TAU1"])

    tau_years = np.array(ALL_MATURITIES, dtype=float) / 12.0
    values = nss_curve_matrix(tau_years, params)
    curve = pd.DataFrame(values / 100.0, index=params.index, columns=ALL_MATURITIES)
    curve.index.name = DATE_COLUMN
    return curve.sort_index(), cache_path


def parse_fedfunds(raw: bytes) -> pd.Series:
    data = pd.read_csv(io.BytesIO(raw))

    if {"observation_date", "FEDFUNDS"}.issubset(data.columns):
        data = data.rename(columns={"observation_date": DATE_COLUMN, "FEDFUNDS": "FEDFUNDS"})
        data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN]) + pd.offsets.MonthEnd(0)
        value_col = "FEDFUNDS"
    elif "Series Description" in data.columns:
        data = pd.read_csv(io.BytesIO(raw), skiprows=5)
        data = data.rename(columns={"Time Period": DATE_COLUMN, "RIFSPFF_N.M": "FEDFUNDS"})
        data[DATE_COLUMN] = pd.PeriodIndex(data[DATE_COLUMN], freq="M").to_timestamp("M")
        value_col = "FEDFUNDS"
    else:
        raise ValueError("Unrecognized FEDFUNDS CSV format.")

    fedfunds = (
        data.set_index(DATE_COLUMN)[value_col].replace([".", "ND"], np.nan).dropna().astype(float)
        / 100.0
    )
    fedfunds.name = "FEDFUNDS"
    return fedfunds.sort_index()


def load_fedfunds(cache_dir: Path, refresh: bool) -> tuple[pd.Series, Path, str]:
    sources = [
        ("federal_reserve_h15_monthly", H15_MONTHLY_URL, cache_dir / "H15_FEDFUNDS_monthly.csv"),
        ("fred_fedfunds", FEDFUNDS_URL, cache_dir / "FEDFUNDS.csv"),
    ]

    errors = []
    for source_name, url, cache_path in sources:
        # Snapshot the prior cache before a refresh can overwrite it, so a
        # download that succeeds at the HTTP layer but fails to parse can
        # still fall back to the last good copy. The fallback is deliberately
        # low-stakes: FEDFUNDS only feeds the pre-1982 one-month-rate
        # smoothing, so a slightly stale series cannot move current-period
        # term premia.
        stale = cache_path.read_bytes() if refresh and cache_path.exists() else b""
        try:
            raw = read_or_download(url, cache_path, refresh)
            return parse_fedfunds(raw), cache_path, url
        except Exception as exc:
            errors.append(f"{source_name}: {exc!r}")
            if stale:
                try:
                    source = f"{url} (stale cache fallback)"
                    fedfunds = parse_fedfunds(stale)
                    # The failed refresh overwrote the cache; restore the last
                    # good bytes (atomically, like read_or_download) so the
                    # next run does not start from poison.
                    restore_temp = cache_path.with_suffix(cache_path.suffix + ".tmp")
                    restore_temp.write_bytes(stale)
                    os.replace(restore_temp, cache_path)
                    return fedfunds, cache_path, source
                except Exception as cache_exc:
                    errors.append(f"{source_name} cache fallback: {cache_exc!r}")

    raise RuntimeError("Could not load FEDFUNDS. " + " | ".join(errors))


def load_official(path: Path, sheet_name: str) -> pd.DataFrame:
    data = pd.read_excel(path, sheet_name=sheet_name)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data.set_index(DATE_COLUMN).sort_index().apply(pd.to_numeric)


def official_workbook_path(
    source: str,
    cache_dir: Path,
    refresh: bool,
) -> tuple[Path, str]:
    if is_url(source):
        cache_name = cached_filename_from_url(source, "ACMTermPremium.xls")
        cache_path = cache_dir / cache_name
        read_or_download(source, cache_path, refresh)
        return cache_path, source

    local_path = Path(source).expanduser()
    return local_path, str(local_path.resolve())


def completed_monthly_dates(
    curve: pd.DataFrame,
    include_partial_current_month: bool,
) -> pd.DatetimeIndex:
    last_by_month = curve.groupby(curve.index.to_period("M")).tail(1)
    if include_partial_current_month or last_by_month.empty:
        return last_by_month.index

    latest_date = last_by_month.index.max()
    business_month_end = latest_date + pd.offsets.BMonthEnd(0)
    if latest_date == business_month_end:
        return last_by_month.index

    latest_period = latest_date.to_period("M")
    return last_by_month.index[last_by_month.index.to_period("M") < latest_period]


def smooth_pre_1982_one_month_rate(
    curve_m: pd.DataFrame,
    fedfunds: pd.Series,
) -> tuple[pd.DataFrame, np.ndarray, int, int]:
    out = curve_m.copy()

    one_month = out[1].copy()
    one_month.index = one_month.index.to_period("M")

    fedfunds_m = fedfunds.copy()
    fedfunds_m.index = fedfunds_m.index.to_period("M")

    common_periods = one_month.index.intersection(fedfunds_m.index)
    fit_periods = common_periods[common_periods >= SMOOTHING_START]
    if len(fit_periods) == 0:
        raise ValueError("No post-1982 overlap between monthly GSW 1M yield and FEDFUNDS.")

    x = np.column_stack([np.ones(len(fit_periods)), fedfunds_m.loc[fit_periods].values])
    beta = np.linalg.lstsq(x, one_month.loc[fit_periods].values, rcond=None)[0]

    replace_periods = one_month.index[one_month.index < SMOOTHING_START]
    replace_periods = replace_periods.intersection(fedfunds_m.index)
    predicted = beta[0] + beta[1] * fedfunds_m.loc[replace_periods].values

    period_to_date = pd.Series(out.index, index=out.index.to_period("M"))
    out.loc[period_to_date.loc[replace_periods].values, 1] = predicted

    return out, beta, len(fit_periods), len(replace_periods)


def get_pc_factors(
    curve_m: pd.DataFrame,
    curve_d: pd.DataFrame,
    n_factors: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    curve_m_cut = curve_m.loc[:, PC_MATURITIES]
    curve_d_cut = curve_d.loc[:, PC_MATURITIES]

    mean_yields = curve_m_cut.mean()
    monthly_demeaned = curve_m_cut - mean_yields
    daily_demeaned = curve_d_cut - mean_yields

    _, singular_values, vt = np.linalg.svd(monthly_demeaned.values, full_matrices=False)
    loadings = pd.DataFrame(
        vt[:n_factors].T,
        index=PC_MATURITIES,
        columns=[f"PC {i + 1}" for i in range(n_factors)],
    )

    pc_m = monthly_demeaned @ loadings
    sigma_factor = pc_m.std()
    pc_m = pc_m / sigma_factor
    loadings = loadings / sigma_factor

    sign_changes = np.sign(loadings.mean()).replace(0, 1)
    pc_m = pc_m * sign_changes
    loadings = loadings * sign_changes
    pc_d = daily_demeaned @ loadings

    explained = pd.Series(
        (singular_values[:n_factors] ** 2) / np.sum(singular_values**2),
        index=loadings.columns,
        name="Explained Variance",
    )
    return pc_m, pc_d, loadings, explained


def get_excess_returns(curve_m: pd.DataFrame) -> pd.DataFrame:
    ttm = np.arange(1, curve_m.shape[1] + 1, dtype=float) / 12.0
    log_prices = -curve_m * ttm
    rf = -log_prices.iloc[:, 0].shift(1)
    rx = (log_prices - log_prices.shift(1, axis=0).shift(-1, axis=1)).subtract(rf, axis=0)
    rx = rx.shift(1, axis=1)
    rx = rx.dropna(how="all", axis=0)
    rx[1] = 0.0
    return rx


class NominalACM:
    def __init__(
        self,
        curve_d: pd.DataFrame,
        curve_m: pd.DataFrame,
        n_factors: int = N_FACTORS,
        selected_maturities: list[int] | None = None,
    ) -> None:
        self.curve_d = curve_d
        self.curve_m = curve_m
        self.n_factors = n_factors
        self.selected_maturities = selected_maturities or list(curve_d.columns)
        self.n_maturities = curve_d.shape[1]
        self.t_m = curve_m.shape[0] - 1

        self.pc_m, self.pc_d, self.pc_loadings, self.pc_explained = get_pc_factors(
            curve_m,
            curve_d,
            n_factors,
        )
        self.rx_m = get_excess_returns(curve_m)
        self.mu, self.phi, self.v, self.s0 = self.estimate_var()
        self.beta, self.omega, self.beta_star = self.excess_return_regression()
        self.lambda0, self.lambda1, self.mu_star, self.phi_star = self.retrieve_lambda()
        self.delta0, self.delta1 = self.short_rate_equation()
        self.a, self.b = self.affine_coefficients(self.lambda0, self.lambda1)
        self.a_rn, self.b_rn = self.affine_coefficients(
            np.zeros_like(self.lambda0),
            np.zeros_like(self.lambda1),
        )
        self.miy_d = self.compute_yields(self.pc_d, self.a, self.b)
        self.rny_d = self.compute_yields(self.pc_d, self.a_rn, self.b_rn)
        self.tp_d = self.miy_d - self.rny_d
        self.miy_m = self.compute_yields(self.pc_m, self.a, self.b)
        self.rny_m = self.compute_yields(self.pc_m, self.a_rn, self.b_rn)
        self.tp_m = self.miy_m - self.rny_m

    def estimate_var(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x = self.pc_m.T.values
        x_lhs = x[:, 1:]
        x_rhs = np.vstack([np.ones((1, self.t_m)), x[:, :-1]])
        var_coeffs = x_lhs @ np.linalg.pinv(x_rhs)
        phi = var_coeffs[:, 1:]

        mu = np.zeros((self.n_factors, 1))
        var_coeffs[:, [0]] = 0.0
        v = x_lhs - var_coeffs @ x_rhs
        s0 = np.cov(v).reshape((-1, 1))
        return mu, phi, v, s0

    def excess_return_regression(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rx = self.rx_m.loc[:, self.selected_maturities].values
        x = self.pc_m.T.values[:, :-1]
        z = np.vstack([np.ones((1, self.t_m)), x, self.v]).T
        abc = np.linalg.inv(z.T @ z) @ (z.T @ rx)
        residuals = rx - z @ abc

        omega = np.var(residuals.reshape(-1, 1)) * np.eye(len(self.selected_maturities))
        abc = abc.T
        beta = abc[:, -self.n_factors :]

        beta_star = np.zeros((len(self.selected_maturities), self.n_factors**2))
        for i in range(len(self.selected_maturities)):
            beta_star[i, :] = np.kron(beta[i, :], beta[i, :]).T

        return beta, omega, beta_star

    def retrieve_lambda(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rx = self.rx_m.loc[:, self.selected_maturities]
        factors = np.hstack([np.ones((self.t_m, 1)), self.pc_m.iloc[:-1].values])

        v_proj = self.v.T @ np.linalg.pinv(self.v @ self.v.T) @ self.v
        factors = factors - v_proj @ factors

        adjustment = self.beta_star @ self.s0 + np.diag(self.omega).reshape(-1, 1)
        rx_adjusted = rx.values + 0.5 * np.tile(adjustment, (1, self.t_m)).T
        y = (np.linalg.inv(factors.T @ factors) @ factors.T @ rx_adjusted).T

        lambda_matrix = np.linalg.inv(self.beta.T @ self.beta) @ self.beta.T @ y
        lambda0 = lambda_matrix[:, 0]
        lambda1 = lambda_matrix[:, 1:]
        mu_star = self.mu.reshape(-1) - lambda0
        phi_star = self.phi - lambda1
        return lambda0, lambda1, mu_star, phi_star

    def short_rate_equation(self) -> tuple[float, np.ndarray]:
        r1 = self.curve_m.iloc[:, 0].values / 12.0
        x = np.column_stack([np.ones(len(self.pc_m)), self.pc_m.values])
        delta = np.linalg.inv(x.T @ x) @ x.T @ r1
        return float(delta[0]), delta[1:]

    def affine_coefficients(
        self,
        lambda0: np.ndarray,
        lambda1: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        lambda0 = lambda0.reshape(-1, 1)
        a = np.zeros(self.n_maturities)
        b = np.zeros((self.n_maturities, self.n_factors))

        a[0] = -self.delta0
        b[0, :] = -self.delta1

        for n in range(1, self.n_maturities):
            bpb = np.kron(b[n - 1, :], b[n - 1, :])
            s0term = 0.5 * (bpb @ self.s0 + self.omega[0, 0])
            a[n] = (a[n - 1] + b[n - 1, :] @ (self.mu - lambda0) + s0term + a[0])[0]
            b[n, :] = b[n - 1, :] @ (self.phi - lambda1) + b[0, :]

        return a, b

    def compute_yields(self, factors: pd.DataFrame, a: np.ndarray, b: np.ndarray) -> pd.DataFrame:
        n_obs = len(factors)
        multiplier = np.tile(np.array(self.curve_d.columns, dtype=float) / 12.0, (n_obs, 1)).T
        values = (-((np.tile(a.reshape(-1, 1), (1, n_obs)) + b @ factors.T) / multiplier).T).values
        return pd.DataFrame(values, index=factors.index, columns=self.curve_d.columns)


def maturity_suffix(maturity_months: int) -> str:
    if maturity_months % 12 == 0:
        return f"{maturity_months // 12:02d}"
    return f"{maturity_months:03d}M"


def acm_panel(
    miy: pd.DataFrame,
    tp: pd.DataFrame,
    rny: pd.DataFrame,
    maturities: list[int],
) -> pd.DataFrame:
    pieces = []
    for prefix, frame in (("ACMY", miy), ("ACMTP", tp), ("ACMRNY", rny)):
        selected = frame.loc[:, maturities].copy() * 100.0
        selected.columns = [f"{prefix}{maturity_suffix(m)}" for m in maturities]
        pieces.append(selected)
    out = pd.concat(pieces, axis=1)
    out.index.name = DATE_COLUMN
    return out


def official_panel(miy: pd.DataFrame, tp: pd.DataFrame, rny: pd.DataFrame) -> pd.DataFrame:
    return acm_panel(miy, tp, rny, PUBLISHED_MATURITIES)


def expanded_monthly_panel(
    miy: pd.DataFrame,
    tp: pd.DataFrame,
    rny: pd.DataFrame,
) -> pd.DataFrame:
    return acm_panel(miy, tp, rny, EXPANDED_MONTHLY_MATURITIES)


def _column_family(column: str) -> str:
    """Return the ACM family prefix for a column name."""
    if column.startswith("ACMRNY"):
        return "ACMRNY"
    if column.startswith("ACMTP"):
        return "ACMTP"
    return "ACMY"


def compare_panel(
    generated: pd.DataFrame,
    official: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_index = generated.index.intersection(official.index)
    common_columns = generated.columns.intersection(official.columns)
    diff = generated.loc[common_index, common_columns] - official.loc[common_index, common_columns]

    rows = []
    for column in common_columns:
        series = diff[column].dropna()
        if series.empty:
            continue
        abs_series = series.abs()
        max_date = abs_series.idxmax()
        rows.append(
            {
                "family": _column_family(column),
                "column": column,
                "n": int(series.shape[0]),
                "max_abs_diff_bp": float(abs_series.loc[max_date] * 100.0),
                "mean_abs_diff_bp": float(abs_series.mean() * 100.0),
                "rmse_bp": float(np.sqrt(np.mean(np.square(series))) * 100.0),
                "signed_mean_bp": float(series.mean() * 100.0),
                "max_abs_diff_date": pd.Timestamp(max_date).strftime("%Y-%m-%d"),
                "generated_at_max": float(generated.loc[max_date, column]),
                "official_at_max": float(official.loc[max_date, column]),
                "diff_at_max_bp": float(series.loc[max_date] * 100.0),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        by_family = pd.DataFrame()
    else:
        by_family = (
            summary.groupby("family")
            .agg(
                columns=("column", "count"),
                observations_per_column_min=("n", "min"),
                observations_per_column_max=("n", "max"),
                max_abs_diff_bp=("max_abs_diff_bp", "max"),
                mean_abs_diff_bp=("mean_abs_diff_bp", "mean"),
                max_rmse_bp=("rmse_bp", "max"),
                max_abs_signed_mean_bp=("signed_mean_bp", lambda s: s.abs().max()),
            )
            .reset_index()
        )
    return summary, by_family


def classify_tail_gap(
    missing: pd.DatetimeIndex,
    gsw_last: pd.Timestamp,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split missing dates into interior (≤ gsw_last) and tail (> gsw_last)."""
    interior = missing[missing <= gsw_last]
    tail = missing[missing > gsw_last]
    return interior, tail


def check_coverage_gaps(
    missing: pd.DatetimeIndex,
    gsw_last: pd.Timestamp,
    label: str,
    max_tail_gap_bd: int,
    failures: list[str],
) -> None:
    """Check coverage gaps and append failure messages to failures list.

    Interior gaps (≤ gsw_last) always fail.  Tail gaps (> gsw_last) are
    tolerated silently when ≤ max_tail_gap_bd business days old, warned
    when in (max_tail_gap_bd, 15] bd, and failed when > 15 bd.
    """
    if not len(missing):
        return

    interior, tail = classify_tail_gap(missing, gsw_last)

    if len(interior):
        dates_str = ", ".join(d.strftime("%Y-%m-%d") for d in interior)
        failures.append(
            f"Official {label} dates missing from GSW input (interior gaps): {dates_str}"
        )

    for missing_date in tail:
        age_bd = int(np.busday_count(gsw_last.date(), missing_date.date()))
        date_str = missing_date.strftime("%Y-%m-%d")
        if age_bd > 15:
            # Hard cap: fails regardless of the configured silent tolerance.
            failures.append(
                f"Official {label} date {date_str} missing from GSW "
                f"(tail gap {age_bd} bd from gsw_last {gsw_last.date()} "
                f"exceeds 15-bd maximum)."
            )
        elif age_bd <= max_tail_gap_bd:
            pass  # tolerate silently
        else:
            print(
                f"WARNING: Official {label} date {date_str} missing from GSW "
                f"(tail gap {age_bd} bd from gsw_last {gsw_last.date()}); "
                f"within 15-bd warning window."
            )


def check_identity_gate(
    panel: pd.DataFrame,
    label: str,
    failures: list[str],
    threshold_bp: float = 1e-8,
) -> float:
    """Verify ACMY{n} - ACMRNY{n} - ACMTP{n} ≈ 0 for every maturity.

    Panel columns are in percent, so the residual is converted to basis
    points (1% = 100 bp) before comparison and reporting.

    Returns the maximum residual in basis points across all maturities.
    Appends a failure message to failures if any residual exceeds threshold_bp.
    """
    max_residual = 0.0
    for n in PUBLISHED_MATURITIES:
        suffix = maturity_suffix(n)
        col_y = f"ACMY{suffix}"
        col_rny = f"ACMRNY{suffix}"
        col_tp = f"ACMTP{suffix}"
        if (
            col_y not in panel.columns
            or col_rny not in panel.columns
            or col_tp not in panel.columns
        ):
            failures.append(
                f"Identity gate ({label}): columns {col_y}, {col_rny}, or {col_tp} missing."
            )
            return float("nan")
        residual_bp = float((panel[col_y] - panel[col_rny] - panel[col_tp]).abs().max()) * 100.0
        if residual_bp > max_residual:
            max_residual = residual_bp

    if max_residual >= threshold_bp:
        failures.append(
            f"Identity gate ({label}): max |ACMY - ACMRNY - ACMTP| = "
            f"{max_residual:.6g} bp >= {threshold_bp:.6g} bp."
        )
    return max_residual


def check_schema_gate(
    panel: pd.DataFrame,
    label: str,
    maturities: list[int],
    failures: list[str],
) -> None:
    """Check schema, uniqueness, sorting, and finiteness of a generated panel.

    Appends precise failure messages to failures if any check fails.
    """
    expected_cols = []
    for prefix in ("ACMY", "ACMTP", "ACMRNY"):
        for n in maturities:
            expected_cols.append(f"{prefix}{maturity_suffix(n)}")

    missing_cols = [c for c in expected_cols if c not in panel.columns]
    if missing_cols:
        failures.append(
            f"Schema gate ({label}): missing expected columns: {missing_cols[:5]}"
            + ("..." if len(missing_cols) > 5 else "")
        )

    if not panel.index.is_monotonic_increasing:
        failures.append(f"Schema gate ({label}): DATE index is not sorted ascending.")

    if not panel.index.is_unique:
        dupes = panel.index[panel.index.duplicated()].tolist()
        failures.append(
            f"Schema gate ({label}): duplicate DATE entries: "
            + ", ".join(str(d) for d in dupes[:5])
        )

    present_cols = [c for c in expected_cols if c in panel.columns]
    for col in present_cols:
        bad = ~np.isfinite(panel[col].to_numpy(dtype=float))
        if bad.any():
            first_bad = panel.index[bad][0]
            failures.append(
                f"Schema gate ({label}): column {col} has NaN/Inf "
                f"(first occurrence at {first_bad})."
            )


def panel_with_date_column(panel: pd.DataFrame, csv_dates: bool = False) -> pd.DataFrame:
    out = panel.reset_index()
    dates = pd.to_datetime(out[DATE_COLUMN])
    out[DATE_COLUMN] = dates.dt.strftime("%Y-%m-%d") if csv_dates else dates.dt.date
    return out


def write_official_workbook(path: Path, monthly: pd.DataFrame, daily: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        date_format=EXCEL_DATE_FORMAT,
        datetime_format=EXCEL_DATE_FORMAT,
    ) as writer:
        writer.book.iso_dates = True
        panel_with_date_column(monthly).to_excel(writer, sheet_name=MONTHLY_SHEET, index=False)
        panel_with_date_column(daily).to_excel(writer, sheet_name=DAILY_SHEET, index=False)

        for sheet_name in (MONTHLY_SHEET, DAILY_SHEET):
            worksheet = writer.sheets[sheet_name]
            for cell in worksheet["A"][1:]:
                cell.number_format = EXCEL_DATE_FORMAT


def write_monthly_only_workbook(path: Path, monthly: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(
        path,
        engine="openpyxl",
        date_format=EXCEL_DATE_FORMAT,
        datetime_format=EXCEL_DATE_FORMAT,
    ) as writer:
        writer.book.iso_dates = True
        panel_with_date_column(monthly).to_excel(writer, sheet_name=MONTHLY_SHEET, index=False)

        worksheet = writer.sheets[MONTHLY_SHEET]
        for cell in worksheet["A"][1:]:
            cell.number_format = EXCEL_DATE_FORMAT


def write_panel_csvs(path: Path, panel: pd.DataFrame) -> tuple[Path, Path]:
    gzip_path = path.with_suffix(path.suffix + ".gz")
    path.parent.mkdir(parents=True, exist_ok=True)
    out = panel_with_date_column(panel, csv_dates=True)
    out.to_csv(path, index=False, float_format="%.12f")
    out.to_csv(
        gzip_path,
        index=False,
        float_format="%.12f",
        compression={"method": "gzip", "mtime": 0},
    )
    return path, gzip_path


def write_csv_outputs(
    outdir: Path,
    metadata: pd.DataFrame,
    monthly: pd.DataFrame,
    daily: pd.DataFrame,
    curve_m_raw: pd.DataFrame,
    curve_m_smoothed: pd.DataFrame,
    fedfunds: pd.Series,
    monthly_summary: pd.DataFrame | None,
    monthly_by_family: pd.DataFrame | None,
    daily_summary: pd.DataFrame | None,
    daily_by_family: pd.DataFrame | None,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(outdir / "run_metadata.csv", index=False)
    monthly.to_csv(outdir / "acm_monthly.csv", float_format="%.12f")
    daily.to_csv(outdir / "acm_daily.csv", float_format="%.12f")
    curve_m_raw.to_csv(outdir / "monthly_gsw_raw.csv", float_format="%.12f")
    curve_m_smoothed.to_csv(outdir / "monthly_gsw_ffr_smoothed.csv", float_format="%.12f")
    fedfunds.to_csv(outdir / "fedfunds.csv", float_format="%.12f")
    if monthly_summary is not None:
        monthly_summary.to_csv(
            outdir / "monthly_comparison_summary.csv",
            index=False,
            float_format="%.12g",
        )
    if monthly_by_family is not None:
        monthly_by_family.to_csv(
            outdir / "monthly_comparison_by_family.csv",
            index=False,
            float_format="%.12g",
        )
    if daily_summary is not None:
        daily_summary.to_csv(
            outdir / "daily_comparison_summary.csv",
            index=False,
            float_format="%.12g",
        )
    if daily_by_family is not None:
        daily_by_family.to_csv(
            outdir / "daily_comparison_by_family.csv",
            index=False,
            float_format="%.12g",
        )


def ensure_finite(name: str, frame: pd.DataFrame) -> None:
    bad = ~np.isfinite(frame.to_numpy())
    if bad.any():
        row, col = np.argwhere(bad)[0]
        raise ValueError(
            f"The {name} contains non-finite values (first at "
            f"{frame.index[row]}, column {frame.columns[col]}); "
            "refusing to write outputs."
        )


def metadata_frame(items: list[tuple[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([(key, value) for key, value in items], columns=["setting", "value"])


def assert_official_reproduced(
    monthly_summary: pd.DataFrame | None,
    daily_summary: pd.DataFrame | None,
    missing_monthly: pd.DatetimeIndex,
    missing_daily: pd.DatetimeIndex,
    max_abs_diff_bp: float,
    gsw_last: pd.Timestamp,
    max_tail_gap_bd: int = 5,
    acmy_max_abs_diff_bp: float = 1e-4,
    max_rmse_bp: float = 0.005,
    max_bias_bp: float = 0.001,
    generated_monthly: pd.DataFrame | None = None,
    generated_daily: pd.DataFrame | None = None,
) -> None:
    failures: list[str] = []

    # Coverage gap checks (interior = hard fail, tail = tolerance by age)
    check_coverage_gaps(missing_monthly, gsw_last, "monthly", max_tail_gap_bd, failures)
    check_coverage_gaps(missing_daily, gsw_last, "daily", max_tail_gap_bd, failures)

    # Per-family reproduction checks
    panels_named = (("monthly", monthly_summary), ("daily", daily_summary))
    for panel_name, summary in panels_named:
        if summary is None:
            # Not applicable for this run (e.g. no official source provided)
            continue
        if summary.empty:
            failures.append(f"{panel_name.capitalize()} comparison summary is empty.")
            continue

        for _, row in summary.iterrows():
            family = row["family"]
            col = row["column"]
            max_diff = float(row["max_abs_diff_bp"])
            rmse = float(row["rmse_bp"])
            bias = abs(float(row.get("signed_mean_bp", 0.0)))

            if family == "ACMY":
                if max_diff >= acmy_max_abs_diff_bp:
                    failures.append(
                        f"{panel_name.capitalize()} {col} (ACMY): "
                        f"max_abs_diff={max_diff:.12g} bp >= {acmy_max_abs_diff_bp:.12g} bp "
                        f"at {row['max_abs_diff_date']}."
                    )
            else:
                if max_diff >= max_abs_diff_bp:
                    failures.append(
                        f"{panel_name.capitalize()} {col} ({family}): "
                        f"max_abs_diff={max_diff:.12g} bp >= {max_abs_diff_bp:.12g} bp "
                        f"at {row['max_abs_diff_date']}."
                    )
                if rmse >= max_rmse_bp:
                    failures.append(
                        f"{panel_name.capitalize()} {col} ({family}): "
                        f"rmse={rmse:.12g} bp >= {max_rmse_bp:.12g} bp."
                    )
                if bias >= max_bias_bp:
                    failures.append(
                        f"{panel_name.capitalize()} {col} ({family}): "
                        f"|signed_mean|={bias:.12g} bp >= {max_bias_bp:.12g} bp."
                    )

    # Identity gate: ACMY - ACMRNY - ACMTP = 0 for generated panels
    identity_max = 0.0
    for gen_panel, gen_label in (
        (generated_monthly, "generated monthly"),
        (generated_daily, "generated daily"),
    ):
        if gen_panel is not None and not gen_panel.empty:
            residual = check_identity_gate(gen_panel, gen_label, failures)
            if np.isfinite(residual):
                identity_max = max(identity_max, residual)

    # Schema/finiteness gate on generated panels
    for gen_panel, gen_label in (
        (generated_monthly, "generated monthly"),
        (generated_daily, "generated daily"),
    ):
        if gen_panel is not None and not gen_panel.empty:
            check_schema_gate(gen_panel, gen_label, PUBLISHED_MATURITIES, failures)

    if failures:
        message = "\n".join(f" - {failure}" for failure in failures)
        raise SystemExit(f"Official ACM reproduction check FAILED:\n{message}")

    # Build per-family summary for the PASS message
    family_stats: dict[str, dict[str, float]] = {}
    for _, summary in panels_named:
        if summary is None or summary.empty:
            continue
        for _, row in summary.iterrows():
            fam = row["family"]
            if fam not in family_stats:
                family_stats[fam] = {"max_diff": 0.0, "max_rmse": 0.0, "max_bias": 0.0}
            family_stats[fam]["max_diff"] = max(
                family_stats[fam]["max_diff"], float(row["max_abs_diff_bp"])
            )
            family_stats[fam]["max_rmse"] = max(
                family_stats[fam]["max_rmse"], float(row["rmse_bp"])
            )
            family_stats[fam]["max_bias"] = max(
                family_stats[fam]["max_bias"], abs(float(row.get("signed_mean_bp", 0.0)))
            )

    fam_lines = []
    for fam, stats in sorted(family_stats.items()):
        fam_lines.append(
            f"  {fam}: max_abs_diff={stats['max_diff']:.6g} bp, "
            f"rmse={stats['max_rmse']:.6g} bp, "
            f"|bias|={stats['max_bias']:.6g} bp"
        )
    fam_summary = "\n".join(fam_lines)

    print(
        f"\nOfficial ACM reproduction check PASSED.\n"
        f"Per-family stats:\n{fam_summary}\n"
        f"Identity residual (max |ACMY-ACMRNY-ACMTP|): {identity_max:.6g} bp"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce or update nominal ACM yields and term premia.",
    )
    parser.add_argument(
        "--official",
        default=None,
        help=(
            "Optional official ACMTermPremium.xls URL or local path for exact "
            "sample dates and comparison."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output workbook path. Defaults to "
            f"{DEFAULT_REPRODUCTION_OUTPUT} with --official and {DEFAULT_UPDATE_OUTPUT} otherwise."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help="Raw input download cache directory.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of GSW and FEDFUNDS inputs.",
    )
    parser.add_argument(
        "--include-partial-current-month",
        action="store_true",
        help="In update mode, include the current incomplete month in ACM Monthly.",
    )
    parser.add_argument(
        "--assert-official-reproduced",
        action="store_true",
        help=(
            "With --official, fail unless the official monthly and daily ACM "
            "panels are fully reproduced within --max-abs-diff-bp."
        ),
    )
    parser.add_argument(
        "--max-abs-diff-bp",
        type=float,
        default=0.01,
        help=(
            "Maximum allowed per-column absolute difference in basis points for "
            "ACMRNY and ACMTP families. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--acmy-max-abs-diff-bp",
        type=float,
        default=1e-4,
        help=(
            "Maximum allowed per-column absolute difference in basis points for "
            "the ACMY family (fitted yields). Default: 1e-4."
        ),
    )
    parser.add_argument(
        "--max-rmse-bp",
        type=float,
        default=0.005,
        help=(
            "Maximum allowed per-column RMSE in basis points for "
            "ACMRNY and ACMTP families. Default: 0.005."
        ),
    )
    parser.add_argument(
        "--max-bias-bp",
        type=float,
        default=0.001,
        help=(
            "Maximum allowed per-column |signed mean| in basis points for "
            "ACMRNY and ACMTP families. Default: 0.001."
        ),
    )
    parser.add_argument(
        "--max-tail-gap-business-days",
        type=int,
        default=5,
        help=(
            "Maximum recent tail gap (in business days from the latest GSW date) "
            "that is silently tolerated when official dates are missing from GSW. "
            "Gaps above this tolerance but within 15 bd emit a WARNING. "
            "Gaps > 15 bd always fail. Default: 5."
        ),
    )
    parser.add_argument(
        "--check-tolerance-bp",
        type=float,
        default=None,
        help=(
            "Deprecated compatibility option. With --official, run the full "
            "reproduction verification (coverage, identity, schema, and "
            "per-family gates) using this value as the maximum absolute "
            "difference threshold in basis points."
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    official_source = args.official
    exact_mode = official_source is not None
    output_path = Path(
        args.output or (DEFAULT_REPRODUCTION_OUTPUT if exact_mode else DEFAULT_UPDATE_OUTPUT)
    )
    expanded_monthly_output_path = output_path.with_name(
        f"{output_path.stem}_monthly_6m_120m{output_path.suffix}"
    )
    expanded_monthly_csv_path = expanded_monthly_output_path.with_suffix(".csv")
    expanded_monthly_csv_gz_path = expanded_monthly_csv_path.with_suffix(".csv.gz")
    diagnostics_dir = output_path.with_suffix("")

    curve_d_all, gsw_cache = load_gsw_curve(cache_dir, args.refresh)
    fedfunds, fedfunds_cache, fedfunds_source = load_fedfunds(cache_dir, args.refresh)

    official_m = None
    official_d = None
    official_path = None
    official_source_label = ""
    if exact_mode:
        official_path, official_source_label = official_workbook_path(
            official_source,
            cache_dir,
            args.refresh,
        )
        official_m = load_official(official_path, MONTHLY_SHEET)
        official_d = load_official(official_path, DAILY_SHEET)
        monthly_dates = official_m.index
        daily_dates = official_d.index.intersection(curve_d_all.index)
        sample_mode = "official workbook dates"
    else:
        monthly_dates = completed_monthly_dates(curve_d_all, args.include_partial_current_month)
        daily_dates = curve_d_all.index
        sample_mode = "updated GSW dates"

    missing_monthly = monthly_dates.difference(curve_d_all.index)
    if len(missing_monthly):
        gsw_last = curve_d_all.index.max()
        interior_m, tail_m = classify_tail_gap(missing_monthly, gsw_last)
        if len(interior_m):
            interior_str = ", ".join(d.strftime("%Y-%m-%d") for d in interior_m)
            raise ValueError(
                f"GSW curve is missing required monthly dates (interior gaps): {interior_str}"
            )
        # Tail-only gaps: warn and continue (the assertion gate will apply limits)
        for missing_date in tail_m:
            age_bd = int(np.busday_count(gsw_last.date(), missing_date.date()))
            print(
                f"INFO: Official monthly date {missing_date.strftime('%Y-%m-%d')} "
                f"not in GSW (tail gap: {age_bd} bd after gsw_last {gsw_last.date()})."
            )
        # Drop tail-missing dates so we only compute on available GSW dates
        monthly_dates = monthly_dates.difference(tail_m)

    curve_m_raw = curve_d_all.loc[monthly_dates].copy()

    # numpy 2.x on Apple's Accelerate BLAS raises spurious floating-point
    # warnings ("divide by zero / invalid value / overflow encountered in
    # matmul") on well-conditioned matrix products. The matmuls are spread
    # across the whole estimation, so the suppression has to cover it all;
    # ensure_finite() below catches any genuine non-finite result before
    # outputs are written.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        curve_m, smoothing_beta, smoothing_fit_months, smoothed_months = (
            smooth_pre_1982_one_month_rate(curve_m_raw, fedfunds)
        )
        model = NominalACM(
            curve_d=curve_d_all,
            curve_m=curve_m,
            n_factors=N_FACTORS,
            selected_maturities=SELECTED_RETURN_MATURITIES,
        )

    generated_monthly = official_panel(model.miy_m, model.tp_m, model.rny_m)
    generated_expanded_monthly = expanded_monthly_panel(model.miy_m, model.tp_m, model.rny_m)
    generated_daily = official_panel(
        model.miy_d.loc[daily_dates],
        model.tp_d.loc[daily_dates],
        model.rny_d.loc[daily_dates],
    )

    for panel_name, panel in (
        ("generated monthly panel", generated_monthly),
        ("expanded monthly panel", generated_expanded_monthly),
        ("generated daily panel", generated_daily),
    ):
        ensure_finite(panel_name, panel)

    monthly_summary = monthly_by_family = daily_summary = daily_by_family = None
    missing_daily = pd.DatetimeIndex([])
    if exact_mode:
        missing_daily = official_d.index.difference(curve_d_all.index)
        monthly_summary, monthly_by_family = compare_panel(generated_monthly, official_m)
        daily_summary, daily_by_family = compare_panel(generated_daily, official_d)

    metadata = metadata_frame(
        [
            ("sample_mode", sample_mode),
            ("official_workbook", official_source_label),
            ("official_workbook_cache", str(official_path.resolve()) if official_path else ""),
            ("output_workbook", str(output_path.resolve())),
            ("expanded_monthly_output_workbook", str(expanded_monthly_output_path.resolve())),
            ("expanded_monthly_output_csv", str(expanded_monthly_csv_path.resolve())),
            ("expanded_monthly_output_csv_gz", str(expanded_monthly_csv_gz_path.resolve())),
            ("gsw_url", GSW_URL),
            ("fedfunds_url", fedfunds_source),
            ("gsw_cache", str(gsw_cache.resolve())),
            ("fedfunds_cache", str(fedfunds_cache.resolve())),
            ("gsw_daily_first", curve_d_all.index.min().strftime("%Y-%m-%d")),
            ("gsw_daily_last", curve_d_all.index.max().strftime("%Y-%m-%d")),
            ("generated_daily_first", generated_daily.index.min().strftime("%Y-%m-%d")),
            ("generated_daily_last", generated_daily.index.max().strftime("%Y-%m-%d")),
            ("generated_daily_rows", len(generated_daily)),
            ("generated_monthly_first", generated_monthly.index.min().strftime("%Y-%m-%d")),
            ("generated_monthly_last", generated_monthly.index.max().strftime("%Y-%m-%d")),
            ("generated_monthly_rows", len(generated_monthly)),
            (
                "missing_official_daily_dates",
                ", ".join(d.strftime("%Y-%m-%d") for d in missing_daily),
            ),
            ("pc_maturities", "3..120 months"),
            ("n_factors", N_FACTORS),
            ("selected_return_maturities", ", ".join(map(str, SELECTED_RETURN_MATURITIES))),
            ("published_output_maturities", "12, 24, ..., 120 months"),
            ("expanded_monthly_output_maturities", "6, 7, ..., 120 months"),
            ("short_rate_preprocessing", "1M GSW yield smoothed with FEDFUNDS before 1982-01"),
            ("smoothing_beta_intercept", smoothing_beta[0]),
            ("smoothing_beta_fedfunds", smoothing_beta[1]),
            ("smoothing_fit_months", smoothing_fit_months),
            ("smoothed_pre_1982_months", smoothed_months),
            ("pc_explained_variance_json", json.dumps(model.pc_explained.to_dict())),
        ]
    )

    write_official_workbook(output_path, generated_monthly, generated_daily)
    write_monthly_only_workbook(expanded_monthly_output_path, generated_expanded_monthly)
    write_panel_csvs(expanded_monthly_csv_path, generated_expanded_monthly)
    write_csv_outputs(
        diagnostics_dir,
        metadata,
        generated_monthly,
        generated_daily,
        curve_m_raw,
        curve_m,
        fedfunds,
        monthly_summary,
        monthly_by_family,
        daily_summary,
        daily_by_family,
    )

    print(f"Wrote workbook: {output_path}")
    print(f"Wrote expanded monthly workbook: {expanded_monthly_output_path}")
    print(f"Wrote expanded monthly CSV: {expanded_monthly_csv_path}")
    print(f"Wrote expanded monthly CSV gzip: {expanded_monthly_csv_gz_path}")
    monthly_range = (
        f"{generated_monthly.index.min().date()} to {generated_monthly.index.max().date()}"
    )
    daily_range = f"{generated_daily.index.min().date()} to {generated_daily.index.max().date()}"
    print(f"Monthly rows: {len(generated_monthly)} ({monthly_range})")
    print(f"Daily rows: {len(generated_daily)} ({daily_range})")
    if exact_mode:
        print("\nMonthly comparison by family:")
        print(monthly_by_family.to_string(index=False))
        print("\nDaily comparison by family:")
        print(daily_by_family.to_string(index=False))
        if len(missing_daily):
            print("\nMissing official daily dates in current GSW download:")
            print(", ".join(d.strftime("%Y-%m-%d") for d in missing_daily))

        if args.check_tolerance_bp is not None:
            assert_official_reproduced(
                monthly_summary,
                daily_summary,
                missing_monthly,
                pd.DatetimeIndex([]),
                args.check_tolerance_bp,
                gsw_last=curve_d_all.index.max(),
                max_tail_gap_bd=args.max_tail_gap_business_days,
                acmy_max_abs_diff_bp=args.acmy_max_abs_diff_bp,
                max_rmse_bp=args.max_rmse_bp,
                max_bias_bp=args.max_bias_bp,
                generated_monthly=generated_monthly,
                generated_daily=generated_daily,
            )
        if args.assert_official_reproduced:
            assert_official_reproduced(
                monthly_summary,
                daily_summary,
                missing_monthly,
                missing_daily,
                args.max_abs_diff_bp,
                gsw_last=curve_d_all.index.max(),
                max_tail_gap_bd=args.max_tail_gap_business_days,
                acmy_max_abs_diff_bp=args.acmy_max_abs_diff_bp,
                max_rmse_bp=args.max_rmse_bp,
                max_bias_bp=args.max_bias_bp,
                generated_monthly=generated_monthly,
                generated_daily=generated_daily,
            )
    elif args.assert_official_reproduced or args.check_tolerance_bp is not None:
        raise SystemExit("Official ACM reproduction check requires --official.")


if __name__ == "__main__":
    main()
