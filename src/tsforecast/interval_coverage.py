"""Monte Carlo coverage of ARIMA one-step analytic intervals.

The bands inherit Gaussian innovations from statsmodels. Coverage near
the nominal level on a correctly specified AR(1) is a check of that
assumption, not a distribution-free warranty.
"""

from __future__ import annotations

import numpy as np

from tsforecast.arima_models import fit_arima, forecast_arima
from tsforecast.dgp import simulate_stationary_ar


def arima100_one_step_coverage(
    *,
    n_reps: int = 50,
    n: int = 160,
    phi: float = 0.5,
    sigma: float = 1.0,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    """Fraction of held-out one-step outcomes inside ARIMA(1,0,0) intervals."""

    rng = np.random.default_rng(seed)
    hits = 0
    used = 0
    for _ in range(n_reps):
        sim_seed = int(rng.integers(1, 1_000_000_000))
        y = simulate_stationary_ar(n=n, phi=phi, sigma=sigma, seed=sim_seed).values
        origin = n - 2
        train = y.iloc[: origin + 1]
        realised = float(y.iloc[origin + 1])
        fit = fit_arima(train, order=(1, 0, 0))
        fc = forecast_arima(fit, horizon=1, alpha=alpha)
        lower = float(fc.lower[0])
        upper = float(fc.upper[0])
        if not (np.isfinite(lower) and np.isfinite(upper)):
            continue
        used += 1
        if lower <= realised <= upper:
            hits += 1
    if used == 0:
        return {"coverage": float("nan"), "n_used": 0.0, "nominal": 1.0 - alpha}
    return {
        "coverage": float(hits / used),
        "n_used": float(used),
        "nominal": 1.0 - alpha,
    }
