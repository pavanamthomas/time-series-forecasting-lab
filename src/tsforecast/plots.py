"""Figures for simulated series, forecasts, residuals, and volatility.

Every title states that the underlying series is simulated. Plots are
written to disk; they are not a substitute for the numerical tables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from tsforecast.validation import PointForecast

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    }
)


def _save(fig: plt.Figure, path: Path | None) -> None:
    if path is None:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_series_gallery(
    series_map: dict[str, pd.Series],
    path: Path | None = None,
) -> plt.Figure:
    names = list(series_map.keys())
    fig, axes = plt.subplots(len(names), 1, figsize=(9.5, 2.1 * len(names)), sharex=False)
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        y = series_map[name]
        ax.plot(y.index, y.to_numpy(), color="#1f4e79", linewidth=1.0)
        ax.set_title(f"{name} (simulated)")
        ax.set_ylabel("level")
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_acf_pacf(
    y: pd.Series,
    path: Path | None = None,
    *,
    nlags: int = 24,
    title: str | None = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    plot_acf(y.dropna(), ax=axes[0], lags=nlags)
    plot_pacf(y.dropna(), ax=axes[1], lags=nlags, method="ywm")
    fig.suptitle(title or f"ACF/PACF: {y.name} (simulated)")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_forecast(
    train: pd.Series,
    test: pd.Series,
    forecast: PointForecast,
    path: Path | None = None,
    *,
    title: str | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.plot(train.index, train.to_numpy(), color="#1f4e79", linewidth=1.0, label="train (simulated)")
    ax.plot(test.index, test.to_numpy(), color="#000000", linewidth=1.1, label="held-out (simulated)")
    ax.plot(test.index, forecast.point, color="#c45911", linewidth=1.2, label=forecast.name)
    if forecast.lower is not None and forecast.upper is not None:
        ax.fill_between(
            test.index,
            forecast.lower,
            forecast.upper,
            color="#c45911",
            alpha=0.18,
            label="interval",
        )
    ax.set_title(title or f"Out-of-sample path: {forecast.name}")
    ax.set_ylabel("level")
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_rolling_rmse(
    table: pd.DataFrame,
    path: Path | None = None,
    *,
    title: str = "Rolling-origin RMSE (simulated series)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    for model, grp in table.groupby("model"):
        ax.plot(grp["origin"], grp["rmse"], linewidth=1.1, label=str(model))
    ax.set_xlabel("origin (iloc)")
    ax.set_ylabel("RMSE")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_residuals(
    resid: pd.Series | np.ndarray,
    path: Path | None = None,
    *,
    title: str = "Residuals (in-sample)",
) -> plt.Figure:
    r = pd.Series(np.asarray(resid, dtype=float)).dropna()
    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
    axes[0, 0].plot(r.to_numpy(), color="#1f4e79", linewidth=0.9)
    axes[0, 0].set_title("Residual path")
    axes[0, 1].hist(r.to_numpy(), bins=24, color="#1f4e79", edgecolor="white")
    axes[0, 1].set_title("Histogram")
    plot_acf(r, ax=axes[1, 0], lags=min(24, max(5, r.size // 8)))
    axes[1, 0].set_title("Residual ACF")
    lags = np.arange(1, r.size + 1)
    axes[1, 1].scatter(lags, r.to_numpy(), s=8, color="#1f4e79")
    axes[1, 1].set_title("Residual vs index")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_break_split(
    y: pd.Series,
    break_index: int,
    path: Path | None = None,
    *,
    title: str = "Known structural break (simulated)",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.plot(y.index, y.to_numpy(), color="#1f4e79", linewidth=1.0)
    ax.axvline(y.index[break_index], color="#c00000", linestyle="--", linewidth=1.1, label="DGP break")
    pre = y.iloc[:break_index]
    post = y.iloc[break_index:]
    ax.hlines(pre.mean(), y.index[0], y.index[break_index - 1], colors="#548235", linewidth=1.4, label="pre-break mean")
    ax.hlines(post.mean(), y.index[break_index], y.index[-1], colors="#833c0c", linewidth=1.4, label="post-break mean")
    ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_garch(
    returns: pd.Series,
    sigma2: np.ndarray,
    path: Path | None = None,
    *,
    title: str = "Simulated GARCH(1,1): returns and filtered variance",
) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.4), sharex=True)
    axes[0].plot(returns.index, returns.to_numpy(), color="#1f4e79", linewidth=0.8)
    axes[0].set_title("Simulated returns")
    axes[1].plot(returns.index, np.sqrt(np.maximum(sigma2, 0.0)), color="#c45911", linewidth=1.0)
    axes[1].set_title("Filtered conditional standard deviation")
    axes[1].set_xlabel("time")
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)
    return fig


def plot_misleading_trend_arima(
    y: pd.Series,
    fitted_levels: np.ndarray,
    forecast: PointForecast,
    origin: int,
    path: Path | None = None,
) -> plt.Figure:
    """In-sample ARIMA(p,0,q) fit versus held-out path on a trend-only series."""

    train = y.iloc[: origin + 1]
    test = y.iloc[origin + 1 : origin + 1 + len(forecast.point)]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.plot(y.index, y.to_numpy(), color="#9aa7b2", linewidth=1.0, label="simulated series")
    ax.plot(train.index, fitted_levels[: len(train)], color="#1f4e79", linewidth=1.1, label="in-sample ARIMA fit")
    ax.plot(test.index, forecast.point, color="#c00000", linewidth=1.2, label=f"{forecast.name} forecast")
    ax.axvline(train.index[-1], color="black", linestyle=":", linewidth=1.0)
    ax.set_title("Undifferenced ARIMA on a simulated trend: in-sample fit is not a forecast")
    ax.legend(loc="best")
    fig.tight_layout()
    _save(fig, path)
    return fig
