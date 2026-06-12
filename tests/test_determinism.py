"""Fixture-output regression and end-to-end pipeline tests.

Gate-1 (offline): these tests detect any change in the model's numerical
output relative to a committed fixture CSV, and exercise the main() CLI
pipeline using only synthetic, in-memory data — no network calls.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import reproduce_acm

# ---------------------------------------------------------------------------
# Fixture path (committed into tests/fixtures/)
# ---------------------------------------------------------------------------

_FIXTURES_DIR = Path(__file__).parent / "fixtures"
_GOLDEN_PATH = _FIXTURES_DIR / "golden_synthetic_panel.csv"


# ===========================================================================
# Fixture-output regression
# ===========================================================================


class TestFixtureOutputRegression:
    """Panel output must match the committed fixture CSV to < 1e-9."""

    def _build_panel(self, nominal_acm_model) -> pd.DataFrame:
        model = nominal_acm_model
        return reproduce_acm.official_panel(model.miy_m, model.tp_m, model.rny_m)

    def test_panel_matches_golden(self, nominal_acm_model):
        panel = self._build_panel(nominal_acm_model)

        if os.environ.get("REBUILD_GOLDEN") == "1":
            # Deliberate, opt-in regeneration: rebuild the fixture, then fall
            # through to the comparison so the rebuilt file is checked.
            _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
            panel.to_csv(_GOLDEN_PATH, float_format="%.12f")

        assert _GOLDEN_PATH.exists(), (
            f"Fixture file missing at {_GOLDEN_PATH}. It must be committed to the "
            "repo; regenerate with REBUILD_GOLDEN=1 python -m pytest "
            "tests/test_determinism.py and review the diff before committing."
        )

        expected = pd.read_csv(_GOLDEN_PATH, index_col="DATE", parse_dates=True)
        # Compare schema and values: added/removed dates or columns must fail too,
        # not just value drift on the overlap.
        assert panel.index.equals(expected.index), (
            f"Panel dates differ from fixture: {len(panel.index)} rows generated "
            f"vs {len(expected.index)} in fixture."
        )
        assert list(panel.columns) == list(expected.columns), (
            f"Panel columns differ from fixture: {list(panel.columns)} vs {list(expected.columns)}."
        )

        diff = panel.values - expected.values
        max_diff = float(np.abs(diff).max())
        assert max_diff < 1e-9, (
            f"Panel deviates from fixture file by {max_diff:.3e} "
            f"(threshold 1e-9). Rebuild the fixture if the model change is intentional."
        )

    def test_expanded_monthly_csv_round_trip_is_lossless(self, nominal_acm_model):
        """Saving the expanded panel to CSV and re-loading must preserve values."""
        model = nominal_acm_model
        panel = reproduce_acm.expanded_acm_panel(model.miy_m, model.tp_m, model.rny_m)
        # Verify self-consistency: saving then re-loading gives same values
        buf = io.StringIO()
        panel.to_csv(buf, float_format="%.12f")
        buf.seek(0)
        reloaded = pd.read_csv(buf, index_col="DATE", parse_dates=True)
        diff = (panel.values - reloaded.values).ravel()
        assert np.abs(diff).max() < 1e-9, "Expanded panel CSV round-trip is not lossless"


# ===========================================================================
# End-to-end main() pipeline test (offline, uses tmp_path)
# ===========================================================================


class TestMainEndToEnd:
    """Exercise reproduce_acm.main() from scratch with synthetic local files."""

    # ------------------------------------------------------------------
    # Helpers to build synthetic files that main() will load
    # ------------------------------------------------------------------

    def _write_synthetic_gsw_csv(
        self,
        path: Path,
        params: pd.DataFrame,
    ) -> None:
        """Write a GSW-style CSV that load_gsw_curve can parse."""
        # The real feds200628.csv has some preamble lines; one of them must
        # start "Date," — gsw_header_offset finds that line index.
        lines = [
            "Unique identifier: feds200628\n",
            "Description: Fitted yield curve parameters\n",
        ]
        header = "Date," + ",".join(params.columns) + "\n"
        lines.append(header)
        with path.open("w") as fh:
            fh.writelines(lines)
            params.to_csv(fh, header=False)

    def _write_synthetic_fedfunds_csv(
        self,
        path: Path,
        fedfunds: pd.Series,
    ) -> None:
        """Write a FRED-format FEDFUNDS CSV."""
        dates = fedfunds.index.strftime("%Y-%m-%d")
        rows = ["observation_date,FEDFUNDS\n"]
        for date, val in zip(dates, fedfunds.values):
            rows.append(f"{date},{val * 100:.6f}\n")
        path.write_text("".join(rows))

    # ------------------------------------------------------------------
    # The test itself
    # ------------------------------------------------------------------

    def test_main_runs_without_error_update_mode(
        self,
        tmp_path: Path,
        synthetic_nss_params_daily: pd.DataFrame,
        synthetic_fedfunds: pd.Series,
    ):
        """main() in update mode (no --official) completes without exception."""
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        output_path = tmp_path / "out" / "ACMTermPremium_updated.xlsx"

        # Write GSW params file — main() will call load_gsw_curve which calls
        # gsw_header_offset then pd.read_csv(..., skiprows=N)
        gsw_path = cache_dir / "feds200628.csv"
        self._write_synthetic_gsw_csv(gsw_path, synthetic_nss_params_daily)

        # Write FEDFUNDS file
        ff_path = cache_dir / "H15_FEDFUNDS_monthly.csv"
        self._write_synthetic_fedfunds_csv(ff_path, synthetic_fedfunds)

        # Patch sys.argv for argparse
        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "reproduce_acm",
                "--cache-dir",
                str(cache_dir),
                "--output",
                str(output_path),
            ]
            # main() calls load_gsw_curve and load_fedfunds which call
            # read_or_download; since the cache files exist they will be read
            # directly (no network).
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                reproduce_acm.main()
        finally:
            sys.argv = old_argv

        # Output workbook should have been created
        assert output_path.exists(), "main() did not write the output workbook"

        daily_gzip = output_path.with_name(f"{output_path.stem}_daily_6m_120m.csv.gz")
        assert daily_gzip.exists(), "main() did not write the expanded daily gzip CSV"
        expected_cols = ["DATE"] + [
            f"{prefix}{reproduce_acm.maturity_suffix(maturity)}"
            for prefix in ("ACMY", "ACMTP", "ACMRNY")
            for maturity in reproduce_acm.EXPANDED_OUTPUT_MATURITIES
        ]
        daily_header = pd.read_csv(daily_gzip, nrows=0)
        assert list(daily_header.columns) == expected_cols

    def test_main_does_not_write_to_repo_outputs(
        self,
        tmp_path: Path,
        synthetic_nss_params_daily: pd.DataFrame,
        synthetic_fedfunds: pd.Series,
    ):
        """main() must not write to the repo's outputs/ or data_cache/ directories."""
        repo_root = Path(__file__).parent.parent
        repo_outputs = repo_root / "outputs"
        repo_cache = repo_root / "data_cache"

        # Record pre-test state
        before_outputs = set(repo_outputs.rglob("*")) if repo_outputs.exists() else set()
        before_cache = set(repo_cache.rglob("*")) if repo_cache.exists() else set()

        cache_dir = tmp_path / "cache2"
        cache_dir.mkdir()
        output_path = tmp_path / "out2" / "test_out.xlsx"

        gsw_path = cache_dir / "feds200628.csv"
        self._write_synthetic_gsw_csv(gsw_path, synthetic_nss_params_daily)
        ff_path = cache_dir / "H15_FEDFUNDS_monthly.csv"
        self._write_synthetic_fedfunds_csv(ff_path, synthetic_fedfunds)

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "reproduce_acm",
                "--cache-dir",
                str(cache_dir),
                "--output",
                str(output_path),
            ]
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                reproduce_acm.main()
        finally:
            sys.argv = old_argv

        after_outputs = set(repo_outputs.rglob("*")) if repo_outputs.exists() else set()
        after_cache = set(repo_cache.rglob("*")) if repo_cache.exists() else set()

        new_outputs = after_outputs - before_outputs
        new_cache = after_cache - before_cache
        assert not new_outputs, f"main() wrote to repo outputs/: {new_outputs}"
        assert not new_cache, f"main() wrote to repo data_cache/: {new_cache}"
