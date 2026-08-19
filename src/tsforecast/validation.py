"""Causal train/test splits, naive benchmarks, and rolling-origin evaluation.

Contract
--------
For an origin index ``origin`` (inclusive last training location, 0-based
``iloc``), every fit and every feature is computed on ``y.iloc[: origin + 1]``.
The held-out path is ``y.iloc[origin + 1 : origin + 1 + horizon]``.

If a caller passes the full sample into ``fit``, subsequent forecasts can
silently use future information. ``forecast_from_origin`` exists to make
that mistake fail tests: a sentinel planted after the origin must not
affect the fitted mean, last value, or ARIMA coefficients.

Sklearn's ``TimeSeriesSplit`` is used as an independent expanding-origin
cross-check. It is not the primary API; the origin integer is.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Protocol

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit

from tsforecast.metrics import error_summary

ForecasterFactory = Callable[[], "Forecaster"]


class Forecaster(Protocol):
    name: str

    def fit(self, y: pd.Series) -> "Forecaster":
        ...

    def forecast(self, horizon: int, alpha: float = 0.05) -> "PointForecast":
        ...


@dataclass
class PointForecast:
    point: np.ndarray
    name: str
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    interval_note: str = ""


def slice_train_test(
    y: pd.Series,
    origin: int,
    horizon: int,
) -> tuple[pd.Series, pd.Series]:
    """Return causal train and test slices for a single origin.

    ``origin`` is the last training position (iloc). Training never includes
    observations strictly after that position.
    """

    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    n = len(y)
    if origin < 0 or origin >= n:
        raise IndexError(f"origin={origin} is outside [0, {n - 1}].")
    train = y.iloc[: origin + 1]
    test = y.iloc[origin + 1 : origin + 1 + horizon]
    if train.empty:
        raise ValueError("Empty training window.")
    return train, test


def expanding_origins(
    n_obs: int,
    min_train: int,
    horizon: int,
    step: int = 1,
) -> np.ndarray:
    """Inclusive last-train positions for an expanding-origin scheme."""

    _check_window_args(n_obs, min_train, horizon, step)
    start = min_train - 1
    stop = n_obs - horizon
    if stop <= start:
        return np.array([], dtype=int)
    return np.arange(start, stop, step, dtype=int)


def rolling_origins(
    n_obs: int,
    train_size: int,
    horizon: int,
    step: int = 1,
) -> np.ndarray:
    """Inclusive last-train positions for a fixed-width rolling window."""

    _check_window_args(n_obs, train_size, horizon, step)
    start = train_size - 1
    stop = n_obs - horizon
    if stop <= start:
        return np.array([], dtype=int)
    return np.arange(start, stop, step, dtype=int)


def _check_window_args(n_obs: int, min_train: int, horizon: int, step: int) -> None:
    if n_obs < 3:
        raise ValueError("n_obs must be at least 3.")
    if min_train < 2:
        raise ValueError("Training length must be at least 2.")
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if step < 1:
        raise ValueError("step must be >= 1.")
    if min_train + horizon > n_obs:
        raise ValueError("min_train + horizon exceeds sample length.")


def train_slice_for_origin(
    y: pd.Series,
    origin: int,
    *,
    scheme: str = "expanding",
    train_size: int | None = None,
) -> pd.Series:
    """Training window ending at ``origin``, never looking past it."""

    if scheme not in {"expanding", "rolling"}:
        raise ValueError("scheme must be 'expanding' or 'rolling'.")
    if origin < 0 or origin >= len(y):
        raise IndexError("origin out of range.")
    if scheme == "expanding":
        return y.iloc[: origin + 1]
    if train_size is None:
        raise ValueError("rolling scheme requires train_size.")
    start = origin - train_size + 1
    if start < 0:
        raise ValueError("train_size longer than available history at origin.")
    return y.iloc[start : origin + 1]


def forecast_from_origin(
    y: pd.Series,
    forecaster: Forecaster,
    origin: int,
    horizon: int,
    *,
    scheme: str = "expanding",
    train_size: int | None = None,
    alpha: float = 0.05,
) -> tuple[PointForecast, pd.Series, pd.Series]:
    """Fit on data at or before ``origin`` and forecast the next ``horizon`` steps."""

    train = train_slice_for_origin(
        y, origin, scheme=scheme, train_size=train_size
    )
    _, test = slice_train_test(y, origin, horizon)
    fitted = forecaster.fit(train)
    fc = fitted.forecast(horizon, alpha=alpha)
    if len(fc.point) != horizon:
        raise ValueError("Forecaster returned the wrong horizon.")
    return fc, train, test


def evaluate_forecaster(
    y: pd.Series,
    factory: ForecasterFactory,
    *,
    min_train: int,
    horizon: int,
    step: int = 1,
    scheme: str = "expanding",
    train_size: int | None = None,
    seasonality: int = 1,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Rolling or expanding evaluation. A new forecaster is created at each origin."""

    n = len(y)
    if scheme == "expanding":
        origins = expanding_origins(n, min_train, horizon, step)
        size = None
    else:
        size = train_size if train_size is not None else min_train
        origins = rolling_origins(n, size, horizon, step)

    rows: list[dict[str, float | int | str]] = []
    for origin in origins:
        model = factory()
        fc, train, test = forecast_from_origin(
            y,
            model,
            int(origin),
            horizon,
            scheme=scheme,
            train_size=size,
            alpha=alpha,
        )
        if len(test) < horizon:
            continue
        metrics = error_summary(
            test.to_numpy(),
            fc.point,
            train.to_numpy(),
            seasonality=seasonality,
        )
        rows.append(
            {
                "origin": int(origin),
                "origin_label": str(y.index[origin]),
                "train_end_value": float(train.iloc[-1]),
                "train_length": int(len(train)),
                "model": fc.name,
                "horizon": int(horizon),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def sklearn_expanding_train_indices(
    n_obs: int,
    n_splits: int,
    *,
    test_size: int = 1,
    min_train: int | None = None,
) -> Iterator[np.ndarray]:
    """Expanding training indices from sklearn.TimeSeriesSplit.

    Each yielded array is a strictly causal training set: its maximum
    index is always less than every test index of that split.
    """

    kwargs: dict[str, int] = {"n_splits": n_splits, "test_size": test_size}
    if min_train is not None:
        kwargs["max_train_size"] = None  # expanding
    splitter = TimeSeriesSplit(**kwargs)
    dummy = np.arange(n_obs)
    for train_idx, test_idx in splitter.split(dummy):
        if min_train is not None and len(train_idx) < min_train:
            continue
        yield train_idx, test_idx


def naive_forecast(train: np.ndarray | pd.Series, horizon: int) -> np.ndarray:
    last = float(np.asarray(train, dtype=float).ravel()[-1])
    return np.repeat(last, horizon)


def seasonal_naive_forecast(
    train: np.ndarray | pd.Series,
    horizon: int,
    period: int,
) -> np.ndarray:
    y = np.asarray(train, dtype=float).ravel()
    if period < 1:
        raise ValueError("period must be >= 1.")
    if y.size < period:
        raise ValueError("Training series shorter than the seasonal period.")
    last_season = y[-period:]
    reps = int(np.ceil(horizon / period))
    return np.tile(last_season, reps)[:horizon]


def mean_forecast(train: np.ndarray | pd.Series, horizon: int) -> np.ndarray:
    mu = float(np.mean(np.asarray(train, dtype=float)))
    return np.repeat(mu, horizon)


class NaiveForecaster:
    """Last-value (random-walk) forecast.

    Intervals treat first differences as iid Gaussian. That assumption is
    part of the interval, not a claim about the DGP.
    """

    name = "naive"

    def __init__(self) -> None:
        self._last = np.nan
        self._diff_std = np.nan

    def fit(self, y: pd.Series) -> NaiveForecaster:
        v = np.asarray(y, dtype=float).ravel()
        self._last = float(v[-1])
        if v.size >= 2:
            self._diff_std = float(np.std(np.diff(v), ddof=1))
        else:
            self._diff_std = 0.0
        if not np.isfinite(self._diff_std):
            self._diff_std = 0.0
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
        point = np.repeat(self._last, horizon)
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        step = np.sqrt(np.arange(1, horizon + 1, dtype=float))
        half = z * self._diff_std * step
        return PointForecast(
            point=point,
            lower=point - half,
            upper=point + half,
            name=self.name,
            interval_note="Random-walk Gaussian intervals using in-sample difference SD.",
        )


class SeasonalNaiveForecaster:
    name = "seasonal_naive"

    def __init__(self, period: int = 12) -> None:
        if period < 1:
            raise ValueError("period must be >= 1.")
        self.period = period
        self._train: np.ndarray | None = None
        self._resid_std = np.nan

    def fit(self, y: pd.Series) -> SeasonalNaiveForecaster:
        self._train = np.asarray(y, dtype=float).ravel()
        if self._train.size > self.period:
            fitted = self._train[: -self.period]
            target = self._train[self.period :]
            self._resid_std = float(np.std(target - fitted, ddof=1))
        else:
            self._resid_std = float(np.std(self._train, ddof=1)) if self._train.size > 1 else 0.0
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
        if self._train is None:
            raise RuntimeError("Must call fit before forecast.")
        point = seasonal_naive_forecast(self._train, horizon, self.period)
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        half = z * (self._resid_std if np.isfinite(self._resid_std) else 0.0)
        return PointForecast(
            point=point,
            lower=point - half,
            upper=point + half,
            name=self.name,
            interval_note="Constant-width intervals from seasonal-naive in-sample residuals.",
        )


class MeanForecaster:
    name = "mean"

    def __init__(self) -> None:
        self._mu = np.nan
        self._std = np.nan
        self._n = 0

    def fit(self, y: pd.Series) -> MeanForecaster:
        v = np.asarray(y, dtype=float).ravel()
        self._n = int(v.size)
        self._mu = float(np.mean(v))
        self._std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
        point = np.repeat(self._mu, horizon)
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        se = self._std * np.sqrt(1.0 + 1.0 / max(self._n, 1))
        half = z * se
        return PointForecast(
            point=point,
            lower=point - half,
            upper=point + half,
            name=self.name,
            interval_note="iid Gaussian mean-model predictive intervals.",
        )


class LinearTrendForecaster:
    """OLS intercept and slope in calendar time, extrapolated out of sample.

    Time is the integer position within the training window, then continued
    forward. This is the correctly specified mean for the trend-only DGP.
    """

    name = "linear_trend"

    def __init__(self) -> None:
        self._intercept = np.nan
        self._slope = np.nan
        self._n = 0
        self._sigma = np.nan
        self._t_mean = np.nan
        self._sst = np.nan

    def fit(self, y: pd.Series) -> LinearTrendForecaster:
        v = np.asarray(y, dtype=float).ravel()
        n = v.size
        t = np.arange(n, dtype=float)
        X = np.column_stack([np.ones(n), t])
        beta, *_ = np.linalg.lstsq(X, v, rcond=None)
        resid = v - X @ beta
        self._intercept = float(beta[0])
        self._slope = float(beta[1])
        self._n = n
        self._t_mean = float(t.mean())
        self._sst = float(np.sum((t - self._t_mean) ** 2))
        df = max(n - 2, 1)
        self._sigma = float(np.sqrt(np.sum(resid**2) / df))
        return self

    def forecast(self, horizon: int, alpha: float = 0.05) -> PointForecast:
        t_f = np.arange(self._n, self._n + horizon, dtype=float)
        point = self._intercept + self._slope * t_f
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        if self._sst <= 0.0:
            half = np.repeat(z * self._sigma, horizon)
        else:
            leverage = 1.0 / self._n + (t_f - self._t_mean) ** 2 / self._sst
            se = self._sigma * np.sqrt(1.0 + leverage)
            half = z * se
        return PointForecast(
            point=point,
            lower=point - half,
            upper=point + half,
            name=self.name,
            interval_note="OLS predictive intervals, homoskedastic Gaussian errors.",
        )


@dataclass
class ChowResult:
    break_index: int
    specification: str
    f_stat: float
    p_value: float
    rss_pooled: float
    rss_split: float
    df_num: int
    df_den: int
    note: str = field(default="")


def chow_test(
    y: pd.Series,
    break_index: int,
    *,
    trend: bool = False,
) -> ChowResult:
    """Chow test for a known break in an intercept (and optional trend) OLS model.

    The break date is treated as exogenous. That is appropriate here because
    the laboratory DGP supplies T_b. Searching over break dates would require
    a different null distribution (Quandt/Andrews), which is not implemented.
    """

    v = np.asarray(y, dtype=float).ravel()
    n = v.size
    if not 2 < break_index < n - 2:
        raise ValueError("break_index leaves a regime too short for OLS.")
    t = np.arange(n, dtype=float)

    def design(segment: np.ndarray, t_seg: np.ndarray) -> np.ndarray:
        if trend:
            return np.column_stack([np.ones(segment.size), t_seg])
        return np.ones((segment.size, 1))

    def rss(segment: np.ndarray, t_seg: np.ndarray) -> float:
        X = design(segment, t_seg)
        beta, *_ = np.linalg.lstsq(X, segment, rcond=None)
        resid = segment - X @ beta
        return float(resid @ resid)

    rss_p = rss(v, t)
    rss_1 = rss(v[:break_index], t[:break_index])
    rss_2 = rss(v[break_index:], t[break_index:])
    k = 2 if trend else 1
    df_num = k
    df_den = n - 2 * k
    if df_den <= 0 or rss_1 + rss_2 <= 0.0:
        f_stat = float("nan")
        p_value = float("nan")
    else:
        f_stat = ((rss_p - (rss_1 + rss_2)) / df_num) / ((rss_1 + rss_2) / df_den)
        p_value = float(stats.f.sf(f_stat, df_num, df_den))
    spec = "intercept_and_trend" if trend else "intercept_only"
    return ChowResult(
        break_index=break_index,
        specification=spec,
        f_stat=float(f_stat),
        p_value=p_value,
        rss_pooled=rss_p,
        rss_split=rss_1 + rss_2,
        df_num=df_num,
        df_den=df_den,
        note=(
            "Known-break Chow test under iid Gaussian errors and a linear "
            "specification. Heteroskedasticity or a misstated break date "
            "invalidates the F reference distribution. For an unknown break, "
            "use sup_chow; the single-date p-value does not apply to the sup."
        ),
    )


@dataclass(frozen=True)
class SupChowResult:
    break_index: int
    f_stat: float
    path: pd.DataFrame
    min_frac: float
    note: str


def sup_chow(
    y: pd.Series,
    *,
    min_frac: float = 0.2,
    trend: bool = False,
) -> SupChowResult:
    """Search for a break date by maximising the Chow F on a trimmed grid.

    This is a Quandt-style sup-F search. Conventional Chow p-values are not
    valid for the maximised statistic. The returned date is an estimate of
    T_b under a single mean (or trend) shift, not a proof of a unique break.
    """

    v = np.asarray(y, dtype=float).ravel()
    n = v.size
    if not 0.05 <= min_frac < 0.45:
        raise ValueError("min_frac must lie in [0.05, 0.45).")
    lo = max(4, int(np.floor(min_frac * n)))
    hi = n - lo
    if hi <= lo:
        raise ValueError("series too short for a trimmed break search.")
    rows: list[dict[str, float]] = []
    best_i = lo
    best_f = -np.inf
    series = pd.Series(v)
    for i in range(lo, hi):
        res = chow_test(series, i, trend=trend)
        f_stat = float(res.f_stat)
        rows.append({"break_index": float(i), "f_stat": f_stat})
        if np.isfinite(f_stat) and f_stat > best_f:
            best_f = f_stat
            best_i = i
    return SupChowResult(
        break_index=int(best_i),
        f_stat=float(best_f),
        path=pd.DataFrame(rows),
        min_frac=float(min_frac),
        note=(
            "Sup-F search over break dates. Do not use a single Chow p-value "
            "as the null distribution of the maximised F."
        ),
    )


def iter_causal_windows(
    n_obs: int,
    min_train: int,
    horizon: int,
    step: int = 1,
) -> Iterable[tuple[int, int, int]]:
    """Yield (train_end_exclusive, test_start, test_end_exclusive)."""

    for origin in expanding_origins(n_obs, min_train, horizon, step):
        train_end = int(origin) + 1
        yield train_end, train_end, train_end + horizon
