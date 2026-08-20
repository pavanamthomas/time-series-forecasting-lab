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

## Remaining bounds

Issues #2–#4 were closed after `sup_chow` search, ARIMA one-step coverage,
and numerical GARCH sandwich SEs were added. Still unimplemented:

1. The sup-F search locates a level shift; Andrews critical values for the sup statistic are not tabulated
   ([issue #5](https://github.com/pavanamthomas/time-series-forecasting-lab/issues/5)).
2. Interval coverage is checked for ARIMA(1,0,0) one-step Gaussian bands on a stationary AR DGP, not for every forecaster.
3. GARCH sandwich SEs are numerical (finite-difference scores). t-innovations and leverage are not implemented.
4. Multivariate series and genuine market data remain out of scope.

## Explicitly not in scope

- Declaring a complex model superior because in-sample AIC is lower.
- Committing regenerated `data/simulated/*.csv` or figure PNGs as source.
- Live trading or volatility-target performance.

Close an issue only with a new test or with a sentence that the item remains a limitation.
