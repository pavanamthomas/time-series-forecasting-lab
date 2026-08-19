"""Unknown-break search, interval coverage, and GARCH sandwich corrections."""

from __future__ import annotations

import numpy as np

from tsforecast.arima_models import ARIMAForecaster
from tsforecast.dgp import simulate_stationary_ar, simulate_structural_break
from tsforecast.interval_coverage import arima100_one_step_coverage
from tsforecast.metrics import rmse
from tsforecast.validation import LinearTrendForecaster, forecast_from_origin, sup_chow
from tsforecast.volatility import fit_garch11, garch11_sandwich_se, simulate_garch11


def test_sup_chow_locates_known_level_shift() -> None:
    item = simulate_structural_break(n=300, break_index=150, mu_post=3.0, seed=42)
    found = sup_chow(item.values, min_frac=0.2)
    assert abs(found.break_index - 150) <= 8


def test_sup_chow_search_stays_inside_the_estimation_window() -> None:
    item = simulate_structural_break(n=300, break_index=220, mu_post=4.0, seed=7)
    origin = 140
    train = item.values.iloc[: origin + 1]
    found = sup_chow(train, min_frac=0.2)
    assert found.break_index < len(train)
    assert found.break_index != 220


def test_arima_one_step_interval_coverage_near_nominal() -> None:
    out = arima100_one_step_coverage(n_reps=40, n=140, phi=0.45, seed=42)
    assert out["n_used"] >= 30
    assert abs(out["coverage"] - out["nominal"]) < 0.20


def test_garch_sandwich_standard_errors_are_positive() -> None:
    r = simulate_garch11(n=1200, omega=0.05, alpha=0.12, beta=0.83, seed=42)
    fit = fit_garch11(r)
    se = garch11_sandwich_se(fit)
    assert se["se_omega"] > 0.0
    assert se["se_alpha"] > 0.0
    assert se["se_beta"] > 0.0
    assert np.isfinite([se["se_omega"], se["se_alpha"], se["se_beta"]]).all()


def test_linear_trend_is_the_correction_on_the_trend_dgp() -> None:
    from tsforecast.dgp import simulate_trend_only

    y = simulate_trend_only(n=240, slope=0.08, sigma=0.4, seed=15).values
    origin = len(y) - 25
    fc_tr, _, test = forecast_from_origin(y, LinearTrendForecaster(), origin, horizon=24)
    fc_bad, _, _ = forecast_from_origin(
        y,
        ARIMAForecaster(order=(1, 0, 0), name="ARIMA(1,0,0)_no_difference"),
        origin,
        horizon=24,
    )
    assert rmse(test, fc_tr.point) < rmse(test, fc_bad.point)
