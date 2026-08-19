"""Metrics: finite naive errors, MAPE guard, MASE scale."""

from __future__ import annotations

import numpy as np
import pytest

from tsforecast.dgp import simulate_seasonal, simulate_trend_only
from tsforecast.metrics import error_summary, mae, mape, mase, rmse
from tsforecast.validation import NaiveForecaster, forecast_from_origin


def test_identical_vectors_have_zero_rmse_and_mae() -> None:
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == 0.0
    assert mae(y, y) == 0.0
    assert mape(y, y) == 0.0


def test_mae_of_unit_errors() -> None:
    assert mae([0.0, 0.0], [1.0, -1.0]) == 1.0


def test_mape_is_nan_when_realised_values_are_near_zero() -> None:
    assert np.isnan(mape([0.0, 1.0], [0.1, 1.1]))


def test_mase_equals_one_for_in_sample_naive_on_the_same_window() -> None:
    train = np.array([1.0, 2.0, 3.0, 4.0, 6.0], dtype=float)
    # Non-seasonal naive forecasts of the last four points using lag-1:
    # this checks the scale construction, not a forecasting theorem.
    y_true = train[1:]
    y_pred = train[:-1]
    value = mase(y_true, y_pred, train, seasonality=1)
    assert value == pytest.approx(1.0)


def test_naive_metrics_are_finite_on_simulated_seasonal_series() -> None:
    y = simulate_seasonal(n=120, seed=5).values
    fc, train, test = forecast_from_origin(y, NaiveForecaster(), origin=96, horizon=12)
    summary = error_summary(test.to_numpy(), fc.point, train.to_numpy(), seasonality=12)
    assert np.isfinite(summary["rmse"])
    assert np.isfinite(summary["mae"])
    assert np.isfinite(summary["mape"])
    assert np.isfinite(summary["mase"])
    assert summary["rmse"] >= 0.0


def test_rmse_finite_on_trend_naive() -> None:
    y = simulate_trend_only(n=100, seed=6).values
    fc, train, test = forecast_from_origin(y, NaiveForecaster(), origin=80, horizon=8)
    assert np.isfinite(rmse(test, fc.point))
    assert rmse(test, fc.point) > 0.0


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])
