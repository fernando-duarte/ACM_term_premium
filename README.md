# Official ACM Reproduction

Standalone Python code to reproduce and update the NY Fed nominal ACM workbook.

Generated workbook sheets:

- `ACM Monthly`
- `ACM Daily`

Generated workbook columns:

- `DATE`
- `ACMY01` to `ACMY10`
- `ACMTP01` to `ACMTP10`
- `ACMRNY01` to `ACMRNY10`

The important replication detail is the original ACM short-rate preprocessing:
before estimation, the monthly 1-month GSW yield before January 1982 is replaced
with a fitted value from monthly effective federal funds.

## Setup

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce The Official NY Fed Workbook

```bash
make reproduce
```

Equivalent direct command:

```bash
python reproduce_acm.py \
  --official https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls
```

This downloads the official NY Fed workbook, uses its exact monthly and daily
dates, and writes:

- `outputs/ACMTermPremium_reproduced.xlsx`
- `outputs/ACMTermPremium_reproduced_monthly_6m_120m.xlsx`
- `outputs/ACMTermPremium_reproduced_monthly_6m_120m.csv`
- `outputs/ACMTermPremium_reproduced_monthly_6m_120m.csv.gz`
- `outputs/ACMTermPremium_reproduced/run_metadata.csv`
- `outputs/ACMTermPremium_reproduced/monthly_comparison_by_family.csv`
- `outputs/ACMTermPremium_reproduced/daily_comparison_by_family.csv`

The expanded monthly files contain one-month maturity spacing from 6M through
120M for each of `ACMY`, `ACMTP`, and `ACMRNY`. Sub-annual columns use month
suffixes, for example `ACMY006M`; whole-year maturities keep the official
suffix style, for example `ACMY01`, `ACMY02`, ..., `ACMY10`. The CSV and gzip
CSV use `YYYY-MM-DD` dates and the same columns as the expanded monthly
workbook.

If a date in the official daily workbook is not available in the current GSW
download, the date is omitted from the generated daily panel and recorded in
`run_metadata.csv`.

To compare against a local copy instead, override the source:

```bash
make reproduce OFFICIAL=../ACMTermPremium.xls
```

## Verify The Reproduction

```bash
make verify
```

This runs the reproduction and additionally fails (non-zero exit) if the
official monthly and daily workbooks are not fully reproduced within the
configured basis-point tolerance (default `0.01`, override with
`MAX_ABS_DIFF_BP`). It is a regression guard suitable for CI:

```bash
make verify MAX_ABS_DIFF_BP=0.01
```

Equivalent direct command:

```bash
python reproduce_acm.py \
  --official https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls \
  --assert-official-reproduced \
  --max-abs-diff-bp 0.01
```

### Verification gates

The assertion check applies the following gates, all accumulated before
reporting a single failure message:

- **Coverage (official dates missing from GSW):** Missing dates earlier than
  the latest GSW date are *interior gaps* and always fail.  Missing dates
  after the latest GSW date are *tail gaps*: silently tolerated when ≤
  `--max-tail-gap-business-days` (default 5) business days old; a `WARNING`
  is printed for gaps above that tolerance up to 15 business days; gaps
  beyond 15 business days always fail, regardless of the configured
  tolerance.  This handles the typical 1-business-day publication lag between the
  official NY Fed workbook and the GSW feed without breaking the CI check.
- **ACMY family (fitted yields):** per-column max absolute diff <
  `--acmy-max-abs-diff-bp` (default `1e-4` bp).
- **ACMRNY and ACMTP families:** per-column max absolute diff <
  `--max-abs-diff-bp` (default `0.01` bp), per-column RMSE <
  `--max-rmse-bp` (default `0.005` bp), and per-column |signed mean| <
  `--max-bias-bp` (default `0.001` bp).
- **Identity:** for every maturity, |ACMY − ACMRNY − ACMTP| < 1e-8 bp in
  the generated panels.
- **Schema and finiteness:** the expected 30 columns are present in the
  generated panels, the DATE index is sorted and unique, and no NaN or Inf
  values are present.

A successful run prints per-family max/RMSE/|bias| statistics and the
identity residual.

### Additional CLI flags

| Flag | Default | Description |
|---|---|---|
| `--acmy-max-abs-diff-bp` | `1e-4` | Max allowed per-column abs diff (bp) for ACMY |
| `--max-rmse-bp` | `0.005` | Max allowed per-column RMSE (bp) for ACMRNY/ACMTP |
| `--max-bias-bp` | `0.001` | Max allowed per-column \|signed mean\| (bp) for ACMRNY/ACMTP |
| `--max-tail-gap-business-days` | `5` | Tail-gap tolerance in business days |

## Tests

The gate logic has a fast, self-contained unit test suite that does not
require network access or cached data:

```bash
pytest tests/test_gates.py -q
```

Coverage includes: interior vs tail gap classification, per-family
tolerance checks, the systematic-bias gate, the ACMY identity constraint,
and schema/finiteness validation.

## GitHub Actions Releases

The `reproduce-official-acm` workflow runs on pushes and pull requests for
`develop` and `main`, on manual dispatch, and at 08:00 UTC on the first day of
each month. Successful runs upload
`outputs/ACMTermPremium_reproduced_monthly_6m_120m.csv.gz` as an artifact.

Scheduled runs checkout `main`, run strict verification, and create or update a
monthly GitHub release named `ACM Term Premium YYYY-MM` with the gzip CSV
attached. Manual runs can also create or update that release by enabling the
`make_release` input.

## Create An Updated Workbook

```bash
make update
```

Equivalent direct command:

```bash
python reproduce_acm.py
```

This downloads current inputs and writes:

- `outputs/ACMTermPremium_updated.xlsx`
- `outputs/ACMTermPremium_updated_monthly_6m_120m.xlsx`
- `outputs/ACMTermPremium_updated_monthly_6m_120m.csv`
- `outputs/ACMTermPremium_updated_monthly_6m_120m.csv.gz`
- `outputs/ACMTermPremium_updated/run_metadata.csv`
- raw and smoothed input CSVs under `outputs/ACMTermPremium_updated/`

Update-mode date rules:

- `ACM Daily` uses every available GSW daily date.
- `ACM Monthly` uses the last available GSW observation in each completed month.
- A current partial month is excluded by default.
- Pass `--include-partial-current-month` to include the current partial month.

## Inputs

The script downloads and caches:

- Official NY Fed ACM workbook:
  `https://www.newyorkfed.org/medialibrary/media/research/data_indicators/ACMTermPremium.xls`
- GSW nominal yield curve parameters:
  `https://www.federalreserve.gov/data/yield-curve-tables/feds200628.csv`
- Monthly effective federal funds from Federal Reserve H.15 (data-download
  package, series `RIFSPFF_N.M`):
  `https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=d7e27b7b09a3a7feae95b9c61781fcd8&lastObs=&from=&to=&filetype=csv&label=include&layout=seriescolumn&type=package`
- Fallback monthly effective federal funds from FRED:
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS`

Raw downloads are cached under `data_cache/`. Use `--refresh` to force a new
download.

## Cleaning

```bash
make clean
```

Removes generated outputs and cached downloads.

```bash
make distclean
```

Also removes the local virtual environment. If the virtual environment is
active, deactivate it first:

```bash
deactivate
make distclean
```

## License

Released under the MIT License. See [`LICENSE`](LICENSE).
