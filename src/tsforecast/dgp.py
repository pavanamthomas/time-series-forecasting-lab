"""Documented synthetic data-generating processes.

Every series produced here is simulated. Labels, metadata, and index
names are chosen so that a later reader cannot mistake the output for
observational data.

DGPs
----
1. Trend-only
       y_t = intercept + slope * t + e_t,  e_t ~ iid N(0, sigma^2)
2. Seasonal (additive monthly factors)
       y_t = mu + s_{t mod 12} + e_t
3. Stationary AR(1)
       y_t = c + phi * y_{t-1} + e_t,  |phi| < 1
4. Structural break (mean shift at a known time)
       y_t = mu_1 + e_t  for t < T_b
       y_t = mu_2 + e_t  for t >= T_b
5. Volatility clustering (Gaussian GARCH(1,1))
       r_t = sigma_t z_t,  z_t ~ iid N(0, 1)
       sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2

The same processes are used to illustrate how a misspecified model
can look adequate in sample and fail out of sample.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from tsforecast.volatility import simulate_garch11

DEFAULT_SEED = 42

_MONTHLY_FACTORS = np.array(
    [4.0, 7.5, 11.0, 8.0, 2.5, -3.0, -8.5, -11.0, -6.5, -1.5, 1.5, 3.5]
)


@dataclass(frozen=True)
class SimulatedSeries:
    """A labelled synthetic series with an explicit DGP description."""

    name: str
    values: pd.Series
    dgp: str
    parameters: dict[str, Any]
    notes: str
    simulated: bool = field(default=True, init=False)

    def __post_init__(self) -> None:
        if not self.name.startswith("simulated_"):
            raise ValueError(
                f"Synthetic series must be labelled simulated_*; got {self.name!r}."
            )


def _monthly_index(n: int, start: str = "2000-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="MS", name="time")


def _daily_index(n: int, start: str = "2000-01-03") -> pd.DatetimeIndex:
    return pd.date_range(start=start, periods=n, freq="B", name="time")


def _as_series(
    values: np.ndarray,
    index: pd.Index,
    name: str,
) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=index, name=name)


def simulate_trend_only(
    n: int = 240,
    intercept: float = 10.0,
    slope: float = 0.08,
    sigma: float = 0.6,
    seed: int = DEFAULT_SEED,
) -> SimulatedSeries:
    """Linear trend plus iid Gaussian noise. No mean reversion."""

    if n < 10:
        raise ValueError("n must be at least 10.")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    y = intercept + slope * t + rng.normal(0.0, sigma, size=n)
    series = _as_series(y, _monthly_index(n), "simulated_trend_only")
    return SimulatedSeries(
        name="simulated_trend_only",
        values=series,
        dgp="y_t = intercept + slope * t + e_t,  e_t iid N(0, sigma^2)",
        parameters={
            "n": n,
            "intercept": intercept,
            "slope": slope,
            "sigma": sigma,
            "seed": seed,
        },
        notes=(
            "Correctly specified forecasts extrapolate the linear trend. "
            "An undifferenced ARIMA fit typically absorbs the trend as a "
            "near-unit-root autoregression and is not a substitute for a "
            "trend model."
        ),
    )


def simulate_seasonal(
    n: int = 240,
    mu: float = 20.0,
    sigma: float = 0.9,
    period: int = 12,
    seed: int = DEFAULT_SEED,
) -> SimulatedSeries:
    """Stable additive seasonal pattern plus iid Gaussian noise."""

    if n < 2 * period:
        raise ValueError("n must cover at least two seasonal cycles.")
    if period != 12:
        raise ValueError("This laboratory DGP uses a fixed period of 12.")
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonal = _MONTHLY_FACTORS[t % period]
    y = mu + seasonal + rng.normal(0.0, sigma, size=n)
    series = _as_series(y, _monthly_index(n), "simulated_seasonal")
    return SimulatedSeries(
        name="simulated_seasonal",
        values=series,
        dgp="y_t = mu + s_{t mod 12} + e_t,  e_t iid N(0, sigma^2)",
        parameters={
            "n": n,
            "mu": mu,
            "sigma": sigma,
            "period": period,
            "seasonal_factors": _MONTHLY_FACTORS.tolist(),
            "seed": seed,
        },
        notes=(
            "Seasonal naive and Holt-Winters (additive) match the DGP. "
            "A non-seasonal ARIMA fit will understate repeating monthly structure."
        ),
    )


def simulate_stationary_ar(
    n: int = 400,
    phi: float = 0.6,
    c: float = 0.0,
    sigma: float = 1.0,
    burnin: int = 80,
    seed: int = DEFAULT_SEED,
) -> SimulatedSeries:
    """Causal Gaussian AR(1) with |phi| < 1."""

    if abs(phi) >= 1.0:
        raise ValueError("Stationary AR(1) requires |phi| < 1.")
    if n < 20:
        raise ValueError("n must be at least 20.")
    rng = np.random.default_rng(seed)
    total = n + burnin
    y = np.zeros(total, dtype=float)
    e = rng.normal(0.0, sigma, size=total)
    for t in range(1, total):
        y[t] = c + phi * y[t - 1] + e[t]
    series = _as_series(y[burnin:], _monthly_index(n), "simulated_stationary_ar")
    return SimulatedSeries(
        name="simulated_stationary_ar",
        values=series,
        dgp="y_t = c + phi * y_{t-1} + e_t,  |phi| < 1,  e_t iid N(0, sigma^2)",
        parameters={
            "n": n,
            "phi": phi,
            "c": c,
            "sigma": sigma,
            "burnin": burnin,
            "seed": seed,
        },
        notes=(
            "The ADF regression with a constant is expected to reject a unit "
            "root in large samples. ARIMA(1,0,0) is correctly specified up "
            "to intercept."
        ),
    )


def simulate_structural_break(
    n: int = 300,
    break_index: int = 150,
    mu_pre: float = 0.0,
    mu_post: float = 3.0,
    sigma: float = 0.7,
    seed: int = DEFAULT_SEED,
) -> SimulatedSeries:
    """Level shift at a known index (first observation of the second regime)."""

    if not 10 < break_index < n - 10:
        raise ValueError("break_index must leave both regimes identifiable.")
    rng = np.random.default_rng(seed)
    y = rng.normal(0.0, sigma, size=n)
    y[:break_index] += mu_pre
    y[break_index:] += mu_post
    series = _as_series(y, _monthly_index(n), "simulated_structural_break")
    return SimulatedSeries(
        name="simulated_structural_break",
        values=series,
        dgp=(
            "y_t = mu_pre + e_t for t < T_b; "
            "y_t = mu_post + e_t for t >= T_b; e_t iid N(0, sigma^2)"
        ),
        parameters={
            "n": n,
            "break_index": break_index,
            "break_label": str(series.index[break_index].date()),
            "mu_pre": mu_pre,
            "mu_post": mu_post,
            "sigma": sigma,
            "seed": seed,
        },
        notes=(
            "The break date is known because it is part of the DGP. A pooled "
            "mean (or a single ARIMA fit) averages incompatible regimes. "
            "Split-sample or Chow-style comparison is the relevant check."
        ),
    )


def simulate_volatility_clustered(
    n: int = 800,
    omega: float = 0.05,
    alpha: float = 0.12,
    beta: float = 0.83,
    seed: int = DEFAULT_SEED,
) -> SimulatedSeries:
    """Gaussian GARCH(1,1) returns. This is a simulated return series, not prices."""

    r, sigma2 = simulate_garch11(
        n=n,
        omega=omega,
        alpha=alpha,
        beta=beta,
        seed=seed,
        return_variance=True,
    )
    series = _as_series(r, _daily_index(n), "simulated_volatility_clustering")
    var = pd.Series(sigma2, index=series.index, name="simulated_conditional_variance")
    return SimulatedSeries(
        name="simulated_volatility_clustering",
        values=series,
        dgp=(
            "r_t = sigma_t z_t, z_t iid N(0,1); "
            "sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2"
        ),
        parameters={
            "n": n,
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": alpha + beta,
            "seed": seed,
            "conditional_variance": var,
        },
        notes=(
            "Squared returns are serially correlated even though the levels "
            "need not be. A Gaussian GARCH(1,1) quasi-likelihood is the "
            "matching estimator in this laboratory; it is not a general "
            "volatility toolkit."
        ),
    )


def generate_catalog(seed: int = DEFAULT_SEED) -> dict[str, SimulatedSeries]:
    """Build the five laboratory series from a single documented seed."""

    return {
        "simulated_trend_only": simulate_trend_only(seed=seed),
        "simulated_seasonal": simulate_seasonal(seed=seed),
        "simulated_stationary_ar": simulate_stationary_ar(seed=seed),
        "simulated_structural_break": simulate_structural_break(seed=seed),
        "simulated_volatility_clustering": simulate_volatility_clustered(seed=seed),
    }
