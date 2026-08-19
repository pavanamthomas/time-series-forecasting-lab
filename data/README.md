# Simulated data

This directory is the landing place for CSV copies of the laboratory series after `python scripts/run_all.py`.

There is no observational dataset here. The generating processes, parameter values, and default seed (42) are documented in `src/tsforecast/dgp.py`.

Expected files after a run (all simulated):

- `simulated_trend_only.csv`
- `simulated_seasonal.csv`
- `simulated_stationary_ar.csv`
- `simulated_structural_break.csv`
- `simulated_volatility_clustering.csv`

Regenerate rather than edit. Policy: `docs/data_policy.md`.
