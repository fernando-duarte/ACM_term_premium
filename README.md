# Official ACM Reproduction

Standalone Python code to reproduce and update the NY Fed nominal ACM term
premium workbook (`ACM Monthly` and `ACM Daily` sheets, columns `ACMY01`–`ACMY10`,
`ACMTP01`–`ACMTP10`, `ACMRNY01`–`ACMRNY10`).

## Usage

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt

make reproduce   # reproduce the official NY Fed workbook
make verify      # reproduce and fail if not within tolerance (CI regression guard)
make update      # build an updated workbook from current inputs
make clean       # remove outputs and cached downloads
```

Outputs are written under `outputs/`, including the reproduced/updated
workbook, an expanded monthly panel with one-month maturity spacing (6M–120M)
in `.xlsx`/`.csv`/`.csv.gz`, run metadata, and comparison evidence. Raw
downloads are cached under `data_cache/` (`--refresh` forces a new download).
Run `python reproduce_acm.py --help` for all flags and tolerances. Unit tests:
`pytest tests/test_gates.py -q` (no network required).

A GitHub Actions workflow verifies the reproduction on pushes, pull requests,
and a monthly schedule, and publishes a monthly GitHub release with the
workbook, CSVs, and a `SHA256SUMS` manifest.

## Sources

- Adrian, Tobias, Richard K. Crump, and Emanuel Moench (2013), "Pricing the
  Term Structure with Linear Regressions," *Journal of Financial Economics*
  110(1): 110–138. Paper and official data:
  [NY Fed term premia page](https://www.newyorkfed.org/research/data_indicators/term-premia-tabs).
- Gürkaynak, Refet S., Brian Sack, and Jonathan H. Wright (2007), "The U.S.
  Treasury Yield Curve: 1961 to the Present," *Journal of Monetary Economics*
  54(8): 2291–2304. Paper and data:
  [Federal Reserve Board nominal yield curve page](https://www.federalreserve.gov/data/nominal-yield-curve.htm).
- Monthly effective federal funds rate from Federal Reserve
  [H.15](https://www.federalreserve.gov/releases/h15/), with
  [FRED FEDFUNDS](https://fred.stlouisfed.org/series/FEDFUNDS) as fallback.

## License and Data Attribution

The code in this repository is released under the MIT License. See
[`LICENSE`](LICENSE). The MIT License covers the code only; data
redistributed via this repository's releases remains subject to its
sources' own terms:

- **ACM term premia** (validation target): © Federal Reserve Bank of
  New York. Content from the New York Fed subject to the Terms of Use at
  [newyorkfed.org](https://www.newyorkfed.org/privacy/termsofuse). The
  series published in this repository's releases are *reproduced* by this
  project's code — modified/derived content, not the New York Fed's
  official series — and the modifications must not be attributed to the
  New York Fed. Redistribution of that content is subject to the same New
  York Fed Terms of Use, which take precedence over the MIT License for
  the data.
- **GSW yield-curve parameters and H.15 federal funds rate**: produced by
  the Board of Governors of the Federal Reserve System and in the public
  domain; the Board should be cited as the source.

This project is not affiliated with, endorsed by, or sponsored by the
Federal Reserve Bank of New York or the Board of Governors of the Federal
Reserve System.
