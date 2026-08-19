"""Exponential smoothing (Holt-Winters) via statsmodels.

Holt-Winters is a recursive smoother, not a statement about a unique DGP.
Additive seasonality matches the laboratory seasonal series. Multiplicative
seasonality is left unused because the seasonal DGP is additive.

Prediction intervals are obtained from residual simulation under the
fitted model. They are not analytic ARIMA intervals and inherit the
Gaussian/iid residual assumption used by ``simulate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from tsforecast.validation import PointForecast


@dataclass
class HoltWintersFit:
    result: Any
    trend: str | None
    seasonal: str | None
    seasonal_periods: int | None
    aic: float
    bic: float
    n_obs: int


class HoltWintersForecaster:
    def __init__(
        self,
        *,
        trend: str | None = "add",
        seasonal: str | None = None,
        seasonal_periods: int | None = None,
        damped_trend: bool = False,
        name: str | None = None,
        simulation_reps: int = 400,
        rng_seed: int = 42,
    ) -> None:
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.damped_trend = damped_trend
        self.simulation_reps = simulation_reps
        self.rng_seed = rng_seed
        if name is None:
            seas = seasonal or "none"
            tr = trend or "none"
            self.name = f"holt_winters_{tr}_{seas}"
        else:
            self.name = name
        self._result: Any | None = None

    def fit(self, y: pd.Series) -> HoltWintersForecaster:
        if self.seasonal is not None and self.seasonal_periods is None:
            raise ValueError("seasonal_periods is required when seasonal is set.")
        model = ExponentialSmoothing(
            y.astype(float),
            trend=self.trend,
            seasonal=self.seasonal,
            seasonal_periods=self.seasonal_periods,
            damped_trend=self.damped_trend,
            initialization_method="estimated",
        )
        self._result = model.fit(optimized=True)
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
        if self._result is None:
            raise RuntimeError("Must call fit before forecast.")
        point = np.asarray(self._result.forecast(horizon), dtype=float)
        lower, upper = _simulated_intervals(
            self._result,
            horizon=horizon,
            alpha=alpha,
            repetitions=self.simulation_reps,
            seed=self.rng_seed,
        )
        return PointForecast(
            point=point,
            lower=lower,
            upper=upper,
            name=self.name,
            interval_note=(
                "Monte Carlo intervals from HoltWintersResults.simulate "
                "under fitted residual draws."
            ),
        )


def fit_holtwinters(
    y: pd.Series,
    *,
    trend: str | None = "add",
    seasonal: str | None = None,
    seasonal_periods: int | None = None,
    damped_trend: bool = False,
) -> HoltWintersFit:
    fitted = HoltWintersForecaster(
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        damped_trend=damped_trend,
    ).fit(y)
    res = fitted._result
    aic = float(getattr(res, "aic", np.nan))
    bic = float(getattr(res, "bic", np.nan))
    return HoltWintersFit(
        result=res,
        trend=trend,
        seasonal=seasonal,
        seasonal_periods=seasonal_periods,
        aic=aic,
        bic=bic,
        n_obs=int(len(y)),
    )


def forecast_holtwinters(
    fit: HoltWintersFit,
    horizon: int,
    *,
    alpha: float = 0.05,
    repetitions: int = 400,
    seed: int = 42,
) -> PointForecast:
    point = np.asarray(fit.result.forecast(horizon), dtype=float)
    lower, upper = _simulated_intervals(
        fit.result,
        horizon=horizon,
        alpha=alpha,
        repetitions=repetitions,
        seed=seed,
    )
    seas = fit.seasonal or "none"
    tr = fit.trend or "none"
    return PointForecast(
        point=point,
        lower=lower,
        upper=upper,
        name=f"holt_winters_{tr}_{seas}",
        interval_note="Monte Carlo intervals from HoltWintersResults.simulate.",
    )


def _simulated_intervals(
    result: Any,
    *,
    horizon: int,
    alpha: float,
    repetitions: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        sims = np.asarray(
            result.simulate(
                nsimulations=horizon,
                repetitions=repetitions,
                random_errors="bootstrap",
                random_state=seed,
            ),
            dtype=float,
        )
        if sims.ndim == 1:
            sims = sims.reshape(horizon, 1)
        # statsmodels returns (horizon, repetitions) or a DataFrame
        sims = np.reshape(sims, (horizon, -1))
        lower = np.quantile(sims, alpha / 2.0, axis=1)
        upper = np.quantile(sims, 1.0 - alpha / 2.0, axis=1)
        return lower, upper
    except (ValueError, TypeError, AttributeError):
        resid = np.asarray(result.resid, dtype=float)
        resid = resid[np.isfinite(resid)]
        sigma = float(np.std(resid, ddof=1)) if resid.size > 1 else 0.0
        point = np.asarray(result.forecast(horizon), dtype=float)
        from scipy import stats as _stats

        z = float(_stats.norm.ppf(1.0 - alpha / 2.0))
        half = z * sigma * np.sqrt(np.arange(1, horizon + 1, dtype=float))
        return point - half, point + half
