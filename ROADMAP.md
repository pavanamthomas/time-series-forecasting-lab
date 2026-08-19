# Roadmap

Current as of August 2026.

## In scope now

- Five simulated series (trend, seasonal, stationary AR, known break, GARCH-like clustering).
- Expanding/rolling-origin evaluation with a no-future-data contract.
- Naive, seasonal naive, and mean benchmarks on comparisons that claim forecast skill.
- Compact Gaussian GARCH(1,1) QMLE without the `arch` package.
- CI: `python -m pytest` and `python scripts/run_all.py`.

## Failures that are part of the design

- Undifferenced ARIMA on a linear trend: AR root near one in sample; worse RMSE than trend extrapolation on a held-out window.
- Pooling through a known mean shift: a single origin that ignores the break is the wrong information set.
- In-sample fitted values presented as if they were h-step forecasts.

Details: `docs/failures_and_corrections.md`.

## Open (issues)

1. Break date is a DGP parameter. A search for an unknown T_b (Bai–Perron or related) is a different procedure and is not implemented.
2. GARCH QMLE uses a Gaussian quasi-likelihood. Sandwich standard errors and t-innovations are not implemented.
3. Interval coverage is not Monte Carlo-calibrated for every forecaster; intervals inherit the model’s innovation assumption.
4. Multivariate series, exogenous regressors, and genuine market data are out of scope.

## Explicitly not in scope

- Declaring a complex model superior because in-sample AIC is lower.
- Committing regenerated `data/simulated/*.csv` or figure PNGs as source.
- Live trading or volatility-target performance.

Close an issue only with a new test or with a sentence that the item remains a limitation.
