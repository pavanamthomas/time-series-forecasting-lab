"""ARIMA recovery on a simulated AR(1)."""

from __future__ import annotations

import numpy as np

from tsforecast.arima_models import ar_lag1_coefficient, fit_arima, forecast_arima
from tsforecast.dgp import simulate_stationary_ar, simulate_trend_only


def test_arima100_recovers_ar1_phi_in_a_reasonable_range() -> None:
    item = simulate_stationary_ar(n=800, phi=0.7, sigma=1.0, seed=13)
    fit = fit_arima(item.values, order=(1, 0, 0))
    phi_hat = ar_lag1_coefficient(fit)
    assert 0.45 < phi_hat < 0.90


def test_arima_forecast_horizon_matches_request() -> None:
    item = simulate_stationary_ar(n=200, phi=0.4, seed=14)
    fit = fit_arima(item.values.iloc[:180], order=(1, 0, 0))
    fc = forecast_arima(fit, horizon=7)
    assert len(fc.point) == 7
    assert fc.lower is not None and len(fc.lower) == 7
    assert np.all(fc.lower <= fc.upper)


def test_undifferenced_arima_on_trend_has_large_ar_root() -> None:
    """A linear trend, estimated as ARIMA(1,0,0), typically loads onto phi near 1.

    This is a specification warning, not a unit-root test.
    """

    y = simulate_trend_only(n=240, slope=0.08, sigma=0.4, seed=15).values
    fit = fit_arima(y, order=(1, 0, 0))
    phi_hat = ar_lag1_coefficient(fit)
    assert phi_hat > 0.9
