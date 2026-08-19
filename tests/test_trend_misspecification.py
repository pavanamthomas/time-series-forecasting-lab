"""Out-of-sample cost of absorbing a linear trend as undifferenced ARIMA.

In-sample AR roots near one are a specification warning. This test asks
whether a correctly specified trend extrapolation has lower RMSE on a
held-out window than ARIMA(2,0,2) and ARIMA(1,0,0) fitted only up to the
origin. A pass does not rank ARIMA on other DGPs.
"""

from tsforecast.arima_models import ARIMAForecaster
from tsforecast.dgp import simulate_trend_only
from tsforecast.metrics import rmse
from tsforecast.validation import LinearTrendForecaster, NaiveForecaster, forecast_from_origin


def test_trend_extrapolation_beats_undifferenced_arima_out_of_sample() -> None:
    y = simulate_trend_only(n=240, slope=0.08, sigma=0.4, seed=15).values
    origin = len(y) - 25
    fc_tr, _, test = forecast_from_origin(
        y, LinearTrendForecaster(), origin, horizon=24
    )
    fc_ari, _, _ = forecast_from_origin(
        y,
        ARIMAForecaster(order=(1, 0, 0), name="ARIMA(1,0,0)_no_difference"),
        origin,
        horizon=24,
    )
    fc_ma, _, _ = forecast_from_origin(
        y,
        ARIMAForecaster(order=(2, 0, 2), name="ARIMA(2,0,2)_no_difference"),
        origin,
        horizon=24,
    )
    fc_nv, _, _ = forecast_from_origin(y, NaiveForecaster(), origin, horizon=24)
    err_tr = rmse(test, fc_tr.point)
    assert err_tr < rmse(test, fc_ari.point)
    assert err_tr < rmse(test, fc_ma.point)
    assert err_tr < rmse(test, fc_nv.point)
