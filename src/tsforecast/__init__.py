"""Time-series forecasting laboratory.

This package implements diagnostics, estimation, and rolling-origin
evaluation for synthetic univariate series. In-sample fit is never
treated as a substitute for out-of-sample forecast performance.

Copyright 2026 Dr. Pavanam Thomas. MIT License.
"""

from tsforecast.dgp import (
    DEFAULT_SEED,
    SimulatedSeries,
    generate_catalog,
    simulate_seasonal,
    simulate_stationary_ar,
    simulate_structural_break,
    simulate_trend_only,
    simulate_volatility_clustered,
)
from tsforecast.metrics import mae, mape, mase, rmse
from tsforecast.validation import (
    expanding_origins,
    forecast_from_origin,
    rolling_origins,
    slice_train_test,
)

__version__ = "0.1.0"
__author__ = "Dr. Pavanam Thomas"

__all__ = [
    "DEFAULT_SEED",
    "SimulatedSeries",
    "__author__",
    "__version__",
    "expanding_origins",
    "forecast_from_origin",
    "generate_catalog",
    "mae",
    "mape",
    "mase",
    "rmse",
    "rolling_origins",
    "simulate_seasonal",
    "simulate_stationary_ar",
    "simulate_structural_break",
    "simulate_trend_only",
    "simulate_volatility_clustered",
    "slice_train_test",
]
