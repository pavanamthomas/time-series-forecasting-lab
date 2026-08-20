# Failures and corrections

Forecast skill is an out-of-sample statement. Several modelling habits fail on the laboratory DGPs; those failures are retained on purpose.

| What was tried | How it failed | Diagnostic | Correction | Locked by | What remains unknown |
| --- | --- | --- | --- | --- | --- |
| ARIMA(1,0,0) on a linear trend plus noise | AR coefficient near one; the model absorbs the trend as persistence | `ar_lag1_coefficient` > 0.9 | Difference, or use a trend model; do not call the AR root a unit-root test | `tests/test_arima.py::test_undifferenced_arima_on_trend_has_large_ar_root` | ADF size/power on this short trend DGP is a separate question |
| ARIMA(1,0,0) or ARIMA(2,0,2) fitted only up to origin t, then 24-step path | Higher RMSE than linear-trend extrapolation (and than naive) on the held-out window | Origin-t split; RMSE on test | Report the comparison; do not prefer ARIMA because in-sample fit looks smooth | `tests/test_trend_misspecification.py::test_trend_extrapolation_beats_undifferenced_arima_out_of_sample` | Other trend shapes; one linear DGP is not a ranking of ARIMA in general |
| Features or fits using observations after the origin | Would leak future information into ŷ_{t+h\|t} | Sentinel after origin must not change the fit | `forecast_from_origin` uses `y.iloc[: origin+1]` only | `tests/test_forecast_no_leakage.py` | Leakage through exogenous features if those are added later |
| Mean or last-value forecast omitted from the comparison | Named models can look good without a benchmark | Rolling tables | Always put a naive next to a named model | `tests/test_metrics.py`; `FORECAST_VALIDATION_PLAYBOOK.md` | Other loss functions |
| Pooling through a known mean shift | A single origin that ignores the break is the wrong information set | Known-break Chow; sup-F search | Search T_b on the estimation window only | `tests/test_corrections.py::test_sup_chow_search_stays_inside_the_estimation_window` | Other error distributions |
| Treating a pointwise Chow p-value on the grid as the sup test | False-rejection exceeds the nominal level under no break | Monte Carlo null quantile of sup F | Reject only if sup F exceeds that quantile | `tests/test_corrections.py::test_pointwise_chow_on_the_grid_is_not_the_sup_test` | Tabulated Andrews values for other trimming |
| Gaussian ARIMA bands treated as distribution-free | They inherit the innovation assumption | One-step coverage on a simulated AR(1) | Report the nominal level and the DGP | `tests/test_corrections.py::test_arima_one_step_interval_coverage_near_nominal` | Other models and horizons |
| Gaussian QMLE intervals without a sandwich | Conventional curvature can understate uncertainty | Finite-difference OPG/Hessian | Report sandwich SEs next to the QMLE | `tests/test_corrections.py::test_garch_sandwich_standard_errors_are_positive` | t-innovations; analytic scores |
| GARCH-like clustering read as a level ARMA problem | Squared-return dependence is the object, not the conditional mean | Squared-return ACF in the volatility module | Model the scale, or say the level model is not a volatility model | `tests/test_volatility.py` | Realised-kernel estimators; this QMLE is Gaussian |

Process: `docs/lab_process.md`. Open extensions: `ROADMAP.md`.
