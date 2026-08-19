"""Point-forecast error metrics.

RMSE and MAE are always well-defined for finite series. MAPE is returned
only when every realised value is bounded away from zero. MASE scales
absolute error by an in-sample naive (or seasonal naive) MAE; it is
undefined when that scale is zero.

These statistics describe a given forecast path. They are not a model
selection theorem, and they do not replace a rolling-origin comparison
against a naive benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_MIN_ABS_FOR_MAPE = 1e-8


def _as_float_arrays(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(y_true, dtype=float).ravel()
    b = np.asarray(y_pred, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError(f"Length mismatch: true={a.shape[0]}, pred={b.shape[0]}.")
    if a.size == 0:
        raise ValueError("Empty arrays.")
    return a, b


def rmse(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    a, b = _as_float_arrays(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    a, b = _as_float_arrays(y_true, y_pred)
    return float(np.mean(np.abs(a - b)))


def mape(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    min_abs: float = _MIN_ABS_FOR_MAPE,
) -> float:
    """Mean absolute percentage error, or NaN if any |y| is below min_abs."""

    a, b = _as_float_arrays(y_true, y_pred)
    if np.any(np.abs(a) < min_abs):
        return float("nan")
    return float(np.mean(np.abs((a - b) / a)) * 100.0)


def mase(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series,
    *,
    seasonality: int = 1,
) -> float:
    """Mean absolute scaled error using in-sample seasonal naive MAE.

    seasonality=1 recovers the non-seasonal naive scale of Hyndman and
    Koehler. If the in-sample naive MAE is zero, the result is NaN.
    """

    if seasonality < 1:
        raise ValueError("seasonality must be >= 1.")
    a, b = _as_float_arrays(y_true, y_pred)
    train = np.asarray(y_train, dtype=float).ravel()
    if train.size <= seasonality:
        raise ValueError("Training series is shorter than the naive lag.")
    scale = np.mean(np.abs(train[seasonality:] - train[:-seasonality]))
    if not np.isfinite(scale) or scale <= 0.0:
        return float("nan")
    return float(np.mean(np.abs(a - b)) / scale)


def error_summary(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    y_train: np.ndarray | pd.Series | None = None,
    *,
    seasonality: int = 1,
) -> dict[str, float]:
    """Compact error dictionary for a single forecast window."""

    out = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
    }
    if y_train is not None:
        out["mase"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    return out
