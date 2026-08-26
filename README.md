# Time-series forecasting lab

[![CI](https://github.com/pavanamthomas/time-series-forecasting-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/time-series-forecasting-lab/actions)

Dr. Pavanam Thomas · [pavanamthomas](https://github.com/pavanamthomas) · thomaspavanam@gmail.com  
MIT License · Copyright 2026

Univariate forecasts on series I generate myself. Every path is **simulated** from a documented DGP. Nothing here is market data, survey data, or a claim about a real economy.

I care about the usual trap: a model that looks fine in sample and then loses to a naive benchmark once the origin is held fixed. Complex models are not treated as automatically better.

## How I evaluate a forecast

1. State the decision (horizon, loss, what must not leak).
2. Write down the DGP or the maintained statistical model.
3. Put a naive benchmark on the table before any named model.
4. Estimate candidates on data available at the forecast origin.
5. Inspect residuals as specification checks, not as performance.
6. Evaluate with expanding or rolling origins.
7. Separate in-sample fit from out-of-sample forecast error.
8. Report interval assumptions and the limitations that follow from them.

Remaining bounds: [`ROADMAP.md`](ROADMAP.md) (issues #2–#5 are closed). Recorded failures: [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md). Process notes: [`docs/lab_process.md`](docs/lab_process.md). The same sequence, written out at more length: [`FORECAST_VALIDATION_PLAYBOOK.md`](FORECAST_VALIDATION_PLAYBOOK.md).

## The five series

Given a univariate series, produce h-step-ahead point forecasts and interval estimates, and say what would falsify the procedure. The five processes are chosen because they break different habits:

| Series name | DGP (simulated) | Typical modelling error |
|---|---|---|
| `simulated_trend_only` | Linear trend plus iid noise | Undifferenced ARIMA absorbs the trend as a near-unit-root AR |
| `simulated_seasonal` | Stable monthly additive seasonality | Non-seasonal ARIMA ignores the repeating cycle |
| `simulated_stationary_ar` | Causal Gaussian AR(1) | Treating in-sample R² as forecast skill |
| `simulated_structural_break` | Known mean shift | Pooling pre- and post-break observations |
| `simulated_volatility_clustering` | Gaussian GARCH(1,1) returns | Reading clustered |r_t| as a level ARMA problem |

## What is implemented

- Horizon, origin, information set.
- Exploratory description of trend, seasonality, and transformations without calling that description a forecast.
- Augmented Dickey–Fuller with an explicit regression specification.
- ACF/PACF as order-selection evidence, not validation.
- ARIMA and SARIMA via `statsmodels`.
- Holt–Winters via `statsmodels`.
- Analytic and simulation-based forecast intervals, with the assumption written next to the interval.
- Expanding- and rolling-origin evaluation with a no-future-data contract.
- Residual diagnostics including Ljung–Box.
- Known-break Chow, split-sample comparison, a trimmed-window `sup_chow` search, and a Monte Carlo null quantile of the sup statistic under iid Gaussian (not an Andrews table).
- Naive, seasonal naive, and mean benchmarks on every comparison that claims forecast skill.
- Compact Gaussian GARCH(1,1) QMLE with numerical sandwich standard errors (SciPy, no `arch` package).
- RMSE, MAE, MAPE (when scale-appropriate), and MASE.

At origin t the information set is {y_s : s ≤ t}. A forecast ŷ_{t+h|t} is a function of that set only.

Estimation uses `statsmodels` for ARIMA/SARIMA and Holt–Winters, NumPy/SciPy for OLS trend and GARCH(1,1) QMLE, and `scikit-learn`'s `TimeSeriesSplit` only as an independent expanding-origin cross-check. Residual diagnostics stay on the estimation window.

## In-sample versus out-of-sample

An ARIMA fit on the full sample can track a linear trend closely and still produce a poor h-step path once the origin is held fixed. That is interpolation versus forecasting, not a software bug.

- **In-sample:** fitted values, residual ACF, Ljung–Box, ADF, AIC/BIC, Chow statistics, GARCH filtered variance on the estimation sample.
- **Out-of-sample:** forecasts formed at origin t using only y_s for s ≤ t, scored on y_{t+1}, …, y_{t+h}.

If a figure overlays a fitted line on the whole series, the caption says so. Forecast figures use a held-out window after a stated origin.

## Data

Generation lives in `src/tsforecast/dgp.py` with seed **42**. Policy: [`docs/data_policy.md`](docs/data_policy.md). After `python scripts/run_all.py`, CSV copies appear under `data/simulated/` for inspection; they are regenerated artifacts. Do not cite them as empirical results.

## Assumptions

- Innovations in the DGPs are Gaussian and independent of the past, except that GARCH makes the *conditional scale* dependent on the past.
- Seasonal period is 12 when seasonality is present.
- The break DGP still records a known date for Chow illustrations. `sup_chow` searches a trimmed grid inside the estimation window. Rejection of the sup uses a Monte Carlo quantile under iid N(0,1) (`sup_chow_null_critical_value`), not tabulated Andrews (1993) values.
- MAPE is omitted when any realised value is too close to zero.
- GARCH(1,1) QMLE assumes a zero conditional mean and uses a Gaussian quasi-likelihood. Numerical sandwich standard errors (`garch11_sandwich_se`) are finite-difference OPG/Hessian, not analytic scores. t-innovations and leverage are not implemented.
- Forecast intervals inherit the model's innovation assumption. They are not distribution-free.

## Reproduce

Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
python scripts/run_all.py
```

`scripts/run_all.py` sets Matplotlib to Agg, uses seed 42, and writes `outputs/figures/` and `outputs/tables/`. This README does not quote those numbers.

Walkthrough notebook: `notebooks/01_forecast_workflow.ipynb`.

CI on Python 3.11: `pip install -e ".[dev]"`, `pytest -q`, `python scripts/run_all.py`.

## Tests

| Test module | What it protects |
|---|---|
| `tests/test_dgp.py` | Five labelled simulated series; DGP properties |
| `tests/test_stationarity.py` | ADF rejects a unit root on a stationary AR (usual case) |
| `tests/test_arima.py` | ARIMA(1,0,0) recovers a plausible AR(1) coefficient; undifferenced AR on a trend has a large AR root |
| `tests/test_trend_misspecification.py` | Trend extrapolation beats undifferenced ARIMA out of sample on the trend DGP |
| `tests/test_metrics.py` | Naive-forecast errors are finite; MAPE/MASE guards |
| `tests/test_volatility.py` | Squared-return ACF lag 1 is positive more often than white noise |
| `tests/test_forecast_no_leakage.py` | Fits at origin t cannot see y_s for s > t |

The leakage tests plant a sentinel in the future and assert that mean, naive, trend, and rolling evaluators are unchanged.

## Limits

- Synthetic DGPs. External validity is not claimed.
- Univariate only. No cointegration, no hierarchical forecasting, no ensembles.
- ARMA orders are chosen to match or to misspecify a known DGP. There is no automated order search.
- ADF has low power near the unit-root boundary; a non-rejection is not proof of integration.
- Holt–Winters intervals are Monte Carlo under residual simulation, not analytic ARIMA intervals.
- GARCH is Gaussian (1,1) QMLE with numerical sandwich SEs. No leverage, no t-innovations, no realised-volatility comparison.
- Known-date Chow still conditions on the DGP break and iid Gaussian errors. The sup-Chow critical value is a Monte Carlo quantile for the sample size and trim used in the test, not an Andrews table.
- Rolling evaluation uses a modest number of origins so CI stays tractable.

```text
src/tsforecast/     library
scripts/run_all.py  regenerate figures and tables
tests/              specification and leakage tests
notebooks/          workflow notebook
docs/data_policy.md synthetic-data rules
docs/failures_and_corrections.md recorded misspecification
ROADMAP.md          open bounds
```

See `CITATION.cff`.
