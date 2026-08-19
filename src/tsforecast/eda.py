"""Exploratory summaries that stay within the observed sample.

These helpers describe trend, seasonal profile, rolling moments, and
simple transformations. They do not produce forecasts. Any statistic
computed on the full sample is an in-sample description and must not be
reported as out-of-sample performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def series_summary(y: pd.Series) -> pd.Series:
    v = np.asarray(y, dtype=float).ravel()
    n = v.size
    if n == 0:
        raise ValueError("Empty series.")
    q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
    return pd.Series(
        {
            "n": n,
            "mean": float(np.mean(v)),
            "std": float(np.std(v, ddof=1)) if n > 1 else float("nan"),
            "min": float(np.min(v)),
            "q05": float(q[0]),
            "q25": float(q[1]),
            "median": float(q[2]),
            "q75": float(q[3]),
            "q95": float(q[4]),
            "max": float(np.max(v)),
            "n_missing": int(np.isnan(v).sum()),
        },
        name=getattr(y, "name", "series"),
    )


def rolling_moments(y: pd.Series, window: int) -> pd.DataFrame:
    if window < 2:
        raise ValueError("window must be >= 2.")
    return pd.DataFrame(
        {
            "rolling_mean": y.rolling(window, min_periods=window).mean(),
            "rolling_std": y.rolling(window, min_periods=window).std(ddof=1),
        }
    )


def linear_time_fit(y: pd.Series) -> dict[str, float]:
    """OLS intercept and slope on integer time. In-sample description only."""

    v = np.asarray(y, dtype=float).ravel()
    n = v.size
    t = np.arange(n, dtype=float)
    X = np.column_stack([np.ones(n), t])
    beta, *_ = np.linalg.lstsq(X, v, rcond=None)
    fitted = X @ beta
    resid = v - fitted
    sst = float(np.sum((v - v.mean()) ** 2))
    ssr = float(np.sum(resid**2))
    r2 = float("nan") if sst <= 0 else 1.0 - ssr / sst
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "r_squared": r2,
        "residual_sd": float(np.sqrt(ssr / max(n - 2, 1))),
    }


def seasonal_profile(y: pd.Series, period: int = 12) -> pd.DataFrame:
    """Mean and SD of each season index. Requires at least one full cycle."""

    if period < 2:
        raise ValueError("period must be >= 2.")
    v = np.asarray(y, dtype=float).ravel()
    if v.size < period:
        raise ValueError("Series shorter than one seasonal cycle.")
    season = np.arange(v.size) % period
    rows = []
    for s in range(period):
        part = v[season == s]
        rows.append(
            {
                "season": s,
                "n": int(part.size),
                "mean": float(np.mean(part)),
                "std": float(np.std(part, ddof=1)) if part.size > 1 else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def difference(
    y: pd.Series,
    *,
    order: int = 1,
    seasonal_period: int | None = None,
) -> pd.Series:
    """Regular and optional seasonal difference. Leading NaNs are dropped."""

    if order < 1:
        raise ValueError("order must be >= 1.")
    out = y.copy()
    for _ in range(order):
        out = out.diff()
    if seasonal_period is not None:
        if seasonal_period < 2:
            raise ValueError("seasonal_period must be >= 2.")
        out = out.diff(seasonal_period)
    return out.dropna()


def log_if_positive(y: pd.Series) -> pd.Series:
    v = np.asarray(y, dtype=float)
    if np.any(v <= 0.0):
        raise ValueError("Log transform requires a strictly positive series.")
    return pd.Series(np.log(v), index=y.index, name=f"log_{y.name}")


def transform_report(y: pd.Series, period: int = 12) -> dict[str, float | bool | str]:
    """Lightweight notes for later modelling; not an automatic model choice."""

    v = np.asarray(y, dtype=float).ravel()
    positive = bool(np.all(v > 0.0))
    fit = linear_time_fit(y)
    roll = rolling_moments(y, window=min(24, max(8, len(y) // 6)))
    std_start = float(roll["rolling_std"].dropna().iloc[:5].mean()) if roll["rolling_std"].notna().any() else float("nan")
    std_end = float(roll["rolling_std"].dropna().iloc[-5:].mean()) if roll["rolling_std"].notna().any() else float("nan")
    variance_ratio = (
        std_end / std_start if np.isfinite(std_start) and std_start > 0 else float("nan")
    )
    note = "levels"
    if abs(fit["slope"]) > 2.0 * fit["residual_sd"] / max(np.sqrt(len(y)), 1.0):
        note = "linear_trend_visible_in_sample"
    if period and len(y) >= 2 * period:
        prof = seasonal_profile(y, period=period)
        seasonal_range = float(prof["mean"].max() - prof["mean"].min())
        if seasonal_range > 2.0 * float(np.nanmean(prof["std"])):
            note = note + "+seasonal_profile"
    return {
        "all_positive": positive,
        "log_admissible": positive,
        "ols_slope": fit["slope"],
        "ols_r_squared": fit["r_squared"],
        "rolling_std_end_over_start": variance_ratio,
        "in_sample_note": note,
    }
