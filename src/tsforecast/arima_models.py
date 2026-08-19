"""ARIMA and SARIMA estimation and forecasting via statsmodels.

In-sample fitted values measure how well the model interpolates the
estimation window. They are not out-of-sample forecasts. Coefficients
recovered on a correctly specified simulated AR(1) should be close to
the DGP in large samples; that recovery is not a licence to treat ARIMA
as automatically superior to a naive benchmark on other series.

Forecasts at origin t use only observations up to t when the caller
passes that slice. Fitting on the full sample and then 'forecasting'
inside the sample is in-sample fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from tsforecast.stationarity import ljung_box
from tsforecast.validation import PointForecast


@dataclass
class ARIMAFit:
    result: Any
    order: tuple[int, int, int]
    seasonal_order: tuple[int, int, int, int]
    aic: float
    bic: float
    ar_params: dict[str, float]
    ma_params: dict[str, float]
    n_obs: int


class ARIMAForecaster:
    """ARIMA/SARIMA wrapper with a causal fit/forecast interface."""

    def __init__(
        self,
        order: tuple[int, int, int] = (1, 0, 0),
        seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
        trend: str | None = None,
        name: str | None = None,
    ) -> None:
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.name = name or _default_name(order, seasonal_order)
        self._result: Any | None = None
        self._endog_index: pd.Index | None = None

    def fit(self, y: pd.Series) -> ARIMAForecaster:
        model = ARIMA(
            y.astype(float),
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = model.fit(method_kwargs={"maxiter": 200})
        self._endog_index = y.index
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
        if self._result is None:
            raise RuntimeError("Must call fit before forecast.")
        pred = self._result.get_forecast(steps=horizon)
        mean = np.asarray(pred.predicted_mean, dtype=float)
        ci = pred.conf_int(alpha=alpha)
        lower = np.asarray(ci.iloc[:, 0], dtype=float)
        upper = np.asarray(ci.iloc[:, 1], dtype=float)
        return PointForecast(
            point=mean,
            lower=lower,
            upper=upper,
            name=self.name,
            interval_note="statsmodels ARIMA analytic prediction intervals, Gaussian innovations.",
        )

    @property
    def result(self) -> Any:
        if self._result is None:
            raise RuntimeError("Must call fit first.")
        return self._result


def _default_name(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
) -> str:
    if seasonal_order[3] == 0 and seasonal_order[:3] == (0, 0, 0):
        return f"ARIMA{order}"
    return f"SARIMA{order}x{seasonal_order}"


def fit_arima(
    y: pd.Series,
    order: tuple[int, int, int] = (1, 0, 0),
    seasonal_order: tuple[int, int, int, int] = (0, 0, 0, 0),
    trend: str | None = None,
) -> ARIMAFit:
    fitted = ARIMAForecaster(
        order=order, seasonal_order=seasonal_order, trend=trend
    ).fit(y)
    res = fitted.result
    params = res.params
    ar = {k: float(params[k]) for k in params.index if k.startswith("ar.")}
    ma = {k: float(params[k]) for k in params.index if k.startswith("ma.")}
    return ARIMAFit(
        result=res,
        order=order,
        seasonal_order=seasonal_order,
        aic=float(res.aic),
        bic=float(res.bic),
        ar_params=ar,
        ma_params=ma,
        n_obs=int(res.nobs),
    )


def forecast_arima(
    fit: ARIMAFit,
    horizon: int,
    alpha: float = 0.05,
) -> PointForecast:
    pred = fit.result.get_forecast(steps=horizon)
    ci = pred.conf_int(alpha=alpha)
    return PointForecast(
        point=np.asarray(pred.predicted_mean, dtype=float),
        lower=np.asarray(ci.iloc[:, 0], dtype=float),
        upper=np.asarray(ci.iloc[:, 1], dtype=float),
        name=_default_name(fit.order, fit.seasonal_order),
        interval_note="statsmodels ARIMA analytic prediction intervals, Gaussian innovations.",
    )


def residual_diagnostics(fit: ARIMAFit, *, lags: int = 12) -> pd.DataFrame:
    resid = np.asarray(fit.result.resid, dtype=float)
    model_df = sum(fit.order[0:3:2]) + sum(fit.seasonal_order[0:3:2])
    lb = ljung_box(resid, lags=lags, model_df=model_df)
    lb.insert(0, "model", _default_name(fit.order, fit.seasonal_order))
    lb.insert(1, "resid_mean", float(np.nanmean(resid)))
    lb.insert(2, "resid_sd", float(np.nanstd(resid, ddof=1)))
    lb.insert(3, "aic", fit.aic)
    lb.insert(4, "bic", fit.bic)
    return lb


def ar_lag1_coefficient(fit: ARIMAFit) -> float:
    """Return the lag-1 AR coefficient if present, else NaN."""

    for key, value in fit.ar_params.items():
        if key.endswith("L1") or key.endswith("L1.ar"):
            return float(value)
        if key in {"ar.L1", "ar.S.L1"}:
            return float(value)
    if fit.ar_params:
        return float(next(iter(fit.ar_params.values())))
    return float("nan")
