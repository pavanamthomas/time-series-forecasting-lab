# Data policy

This repository contains **no observational microdata, market extracts, or personal data**. Every series is generated in code from an explicit data-generating process (DGP).

## Labelling

Simulated objects are named with the prefix `simulated_`. Plot titles and table columns repeat that the series is simulated. Do not rename outputs in a way that suggests an empirical source.

## Generation

- Implementation: `src/tsforecast/dgp.py` and `src/tsforecast/volatility.py`.
- Default seed: 42, used by `scripts/run_all.py` and `generate_catalog()`.
- Regeneration: `python scripts/run_all.py` writes CSV copies under `data/simulated/` and summaries under `outputs/tables/`. Those files are artifacts. They are not a data release.

## What may be stored

- DGP equations, parameter values, and seeds (source-controlled).
- Regenerated CSV and PNG artifacts from a local or CI run (not required in version control).

## What must not be stored

- Live or historical market data, client series, or scraped prices.
- Files that mix simulated paths with real identifiers.
- Credentials, API keys, or `.env` contents.

## Use in analysis

In-sample statistics computed on a simulated path describe that path. They are not estimates of a real-world parameter. Out-of-sample scores are scores against a withheld segment of the **same** simulation, not forecasts of an external series.

When this laboratory is cited next to the finance and statistical-reasoning repositories, keep the data-generating assumptions separate. A GARCH path here is not a return series from `quantitative-finance-models`. An ADF rejection here is not a claim in `statistical-reasoning-validation` about observational data.

## License

Generated artifacts inherit the repository MIT license. Copyright 2026 Dr. Pavanam Thomas.
