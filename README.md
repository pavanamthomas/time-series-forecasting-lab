# Time-Series Forecasting Lab

[![CI](https://github.com/pavanamthomas/time-series-forecasting-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/pavanamthomas/time-series-forecasting-lab/actions)

Reproducible time-series modelling, forecasting, diagnostics, rolling validation, and volatility analysis.

**Author:** Dr. Pavanam Thomas ([GitHub](https://github.com/pavanamthomas), thomaspavanam@gmail.com)

**License:** MIT. Copyright 2026 Dr. Pavanam Thomas.

**Data:** Every series in this repository is **simulated** from a documented data-generating process (DGP). Nothing here is market data, survey data, or an empirical claim about a real economy.

Related laboratories: [quantitative-finance-models](https://github.com/pavanamthomas/quantitative-finance-models), [statistical-reasoning-validation](https://github.com/pavanamthomas/statistical-reasoning-validation).

---

## Forecast validation sequence

This laboratory answers a forecasting question by separating specification from evaluation:

1. State the decision (horizon, loss, what must not leak).
2. Write down the DGP or the maintained statistical model.
3. Put a naive benchmark on the table before any named model.
4. Estimate candidates on data available at the forecast origin.
5. Inspect residuals as specification checks, not as performance.
6. Evaluate with expanding or rolling origins.
7. Separate in-sample fit from out-of-sample forecast error.
8. Report interval assumptions and the limitations that follow from them.

The code is organised so that a misspecified model can be shown to look acceptable in sample and then lose to a simpler benchmark out of sample. That comparison is the point of the repository. Complex models are not treated as automatically superior.

Open work: [`ROADMAP.md`](ROADMAP.md) and GitHub Issues. Recorded failures: [`docs/failures_and_corrections.md`](docs/failures_and_corrections.md). Process: [`docs/lab_process.md`](docs/lab_process.md).

---

## Problem

Given a univariate series, produce h-step-ahead point forecasts and interval estimates, and say what would falsify the procedure.

The laboratory uses five synthetic processes chosen because they break different modelling habits:

| Series name | DGP (simulated) | Typical modelling error |
|---|---|---|
| `simulated_trend_only` | Linear trend plus iid noise | Undifferenced ARIMA absorbs the trend as a near-unit-root AR |
| `simulated_seasonal` | Stable monthly additive seasonality | Non-seasonal ARIMA ignores the repeating cycle |
| `simulated_stationary_ar` | Causal Gaussian AR(1) | Treating in-sample R² as forecast skill |
| `simulated_structural_break` | Known mean shift | Pooling pre- and post-break observations |
| `simulated_volatility_clustering` | Gaussian GARCH(1,1) returns | Reading clustered |r_t| as a level ARMA problem |

---

## Skills demonstrated

- Problem formalisation: horizon, origin, information set.
- Exploratory description of trend, seasonality, and transformations without calling that description a forecast.
- Augmented Dickey–Fuller diagnostics with an explicit regression specification.
- ACF/PACF as order-selection evidence, not validation.
- ARIMA and SARIMA via `statsmodels`.
- Holt–Winters exponential smoothing via `statsmodels`.
- Analytic and simulation-based forecast intervals, with the assumption written next to the interval.
- Expanding- and rolling-origin evaluation with a no-future-data contract.
- Residual diagnostics including Ljung–Box.
- Known-break Chow illustration, split-sample comparison, and a trimmed-window `sup_chow` search (no Andrews p-values).
- Naive, seasonal naive, and mean benchmarks on every comparison that claims forecast skill.
- Compact Gaussian GARCH(1,1) QMLE with numerical sandwich standard errors (SciPy, no `arch` package).
- RMSE, MAE, MAPE (when scale-appropriate), and MASE.

---

## Methods

At origin t the information set is {y_s : s ≤ t}. A forecast ŷ_{t+h|t} is a function of that set only.

Assumptions are model-specific and are stated in the module docstrings. Shared maintained conditions for the synthetic DGPs are Gaussian innovations and a correct seasonality period when seasonality is present. The break example still ships a known DGP date; `sup_chow` can search for T_b on a trimmed estimation window.

Estimation uses `statsmodels` for ARIMA/SARIMA and Holt–Winters, NumPy/SciPy for OLS trend and GARCH(1,1) QMLE, and `scikit-learn`'s `TimeSeriesSplit` only as an independent expanding-origin cross-check.

Validation is rolling or expanding origin. Residual diagnostics are computed on the estimation window. They are not out-of-sample scores.

Interpretation is comparative: a candidate is discussed relative to a naive benchmark on the same origins and the same horizon. In-sample fitted values are labelled as in-sample.

---

## In-sample fit versus out-of-sample forecast performance

An ARIMA fit on the full sample can track a linear trend closely and still produce a poor h-step path once the origin is held fixed and the future is withheld. That is not a software bug; it is the difference between interpolation and forecasting.

This repository keeps the two exercises separate:

- **In-sample:** fitted values, residual ACF, Ljung–Box, ADF, AIC/BIC, Chow statistics, GARCH filtered variance on the estimation sample.
- **Out-of-sample:** forecasts formed at origin t using only y_s for s ≤ t, scored on y_{t+1}, …, y_{t+h}.

If a figure overlays a fitted line on the whole series, the caption states that it is in-sample. Forecast figures use a held-out window after a stated origin.

---

## Data

All series are simulated. Generation is implemented in `src/tsforecast/dgp.py` with seed **42** as the laboratory default. Policy: [`docs/data_policy.md`](docs/data_policy.md). File-level note: [`data/README.md`](data/README.md).

Do not cite these paths as empirical results. After `python scripts/run_all.py`, CSV copies appear under `data/simulated/` for inspection; they are regenerated artifacts.

---

## Assumptions (laboratory-wide)

- Innovations in the DGPs are Gaussian and independent of the past, except that GARCH makes the *conditional scale* dependent on the past.
- The seasonal period is 12 when seasonality is present.
- The break DGP still records a known date for Chow illustrations. `sup_chow` searches a trimmed grid inside the estimation window; Andrews critical values for the sup statistic are not tabulated ([issue #5](https://github.com/pavanamthomas/time-series-forecasting-lab/issues/5)).
- MAPE is omitted when any realised value is too close to zero.
- GARCH(1,1) QMLE assumes a zero conditional mean and uses a Gaussian quasi-likelihood. Numerical sandwich standard errors (`garch11_sandwich_se`) sit beside the QMLE; they are finite-difference OPG/Hessian, not analytic scores. t-innovations and leverage are not implemented.
- Forecast intervals inherit the model's innovation assumption. They are not distribution-free.

---

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

`scripts/run_all.py` sets the Matplotlib backend to Agg, uses seed 42, and writes:

- `outputs/figures/` — diagnostic and forecast plots, all labelled simulated
- `outputs/tables/` — ADF, rolling-origin errors, misspecification illustrations, GARCH estimates

This README does not quote those numbers. Read the generated files after a local run or CI run.

Walkthrough notebook: `notebooks/01_forecast_workflow.ipynb`.

Interview-style protocol: [`FORECAST_VALIDATION_PLAYBOOK.md`](FORECAST_VALIDATION_PLAYBOOK.md).

---

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

The leakage tests plant a sentinel value in the future and assert that mean, naive, trend, and rolling evaluators are unchanged. They fail if training accidentally uses the full sample.

---

## Continuous integration

`.github/workflows/ci.yml` runs on Python 3.11:

```text
pip install -e ".[dev]"
pytest -q
python scripts/run_all.py
```

---

## Limitations

- The DGPs are synthetic. External validity is not claimed.
- Univariate models only. No cointegration, no hierarchical forecasting, no machine-learning ensembles.
- ARMA orders are chosen to match or to misspecify a known DGP. There is no automated order search.
- ADF has low power near the unit-root boundary; a non-rejection is not proof of integration.
- Holt–Winters intervals are Monte Carlo intervals under residual simulation, not analytic ARIMA intervals.
- GARCH is Gaussian (1,1) QMLE with numerical sandwich SEs. No leverage, no t-innovations, no realised-volatility comparison.
- Known-date Chow illustrations still condition on the DGP break and iid Gaussian errors. The sup-Chow search does not supply Andrews p-values.
- Rolling evaluation uses a modest number of origins so that CI remains tractable. It is a validation design, not an exhaustive backtest.

---

## Package layout

```text
src/tsforecast/     library (DGP, EDA, ADF, ARIMA, smoothing, validation, GARCH, metrics, plots)
scripts/run_all.py  regenerate figures and tables
tests/              specification and leakage tests
notebooks/          workflow notebook
docs/data_policy.md synthetic-data rules
docs/failures_and_corrections.md recorded misspecification
docs/lab_process.md issue and test discipline
ROADMAP.md open bounds
```

---

## Citation

See `CITATION.cff`.
