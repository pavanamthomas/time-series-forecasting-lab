"""Unit-root diagnostics and serial-correlation descriptions.

The Augmented Dickey-Fuller statistic is a diagnostic, not a decision
procedure that 'proves' stationarity. Finite-sample power is low near
the unit-root boundary; the default regression specification must match
the alternative of interest (constant versus constant and trend).

ACF and PACF are in-sample descriptions of linear dependence. They inform
candidate ARMA orders; they do not validate forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, adfuller, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


@dataclass(frozen=True)
class ADFResult:
    statistic: float
    p_value: float
    used_lags: int
    n_obs: int
    critical_values: dict[str, float]
    regression: str
    autolag: str
    hypothesis: str
    note: str

    @property
    def rejects_unit_root_5pct(self) -> bool:
        return bool(self.p_value < 0.05)


def adf_unit_root(
    y: pd.Series | np.ndarray,
    *,
    regression: str = "c",
    autolag: str = "AIC",
    maxlag: int | None = None,
) -> ADFResult:
    """ADF test via statsmodels.

    regression:
        'c'  constant only (appropriate for a mean-stationary alternative)
        'ct' constant and linear trend
        'n'  no constant (rarely appropriate here)
    """

    v = np.asarray(y, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size < 20:
        raise ValueError("ADF requires a longer sample.")
    stat, pvalue, usedlag, nobs, crit, _ = adfuller(
        v,
        maxlag=maxlag,
        regression=regression,
        autolag=autolag,
    )
    return ADFResult(
        statistic=float(stat),
        p_value=float(pvalue),
        used_lags=int(usedlag),
        n_obs=int(nobs),
        critical_values={str(k): float(val) for k, val in crit.items()},
        regression=regression,
        autolag=autolag,
        hypothesis="H0: unit root (series is integrated of order at least 1)",
        note=(
            "Rejection is evidence against a unit root under the chosen "
            "deterministic specification. Failure to reject is not confirmation "
            "of a unit root. A trending series tested with regression='c' is "
            "a misspecified ADF regression."
        ),
    )


def acf_pacf_table(
    y: pd.Series | np.ndarray,
    *,
    nlags: int = 24,
) -> pd.DataFrame:
    v = np.asarray(y, dtype=float).ravel()
    v = v[np.isfinite(v)]
    nlags = int(min(nlags, v.size - 2))
    if nlags < 1:
        raise ValueError("Series too short for ACF/PACF.")
    acf_vals = acf(v, nlags=nlags, fft=True)
    pacf_vals = pacf(v, nlags=nlags, method="ywm")
    se = 1.0 / np.sqrt(v.size)
    lags = np.arange(0, nlags + 1)
    return pd.DataFrame(
        {
            "lag": lags,
            "acf": acf_vals,
            "pacf": pacf_vals,
            "approx_se": np.repeat(se, lags.size),
        }
    )


def ljung_box(
    resid: pd.Series | np.ndarray,
    *,
    lags: int = 12,
    model_df: int = 0,
) -> pd.DataFrame:
    """Ljung-Box test on residuals. ``model_df`` should count estimated ARMA parameters."""

    v = np.asarray(resid, dtype=float).ravel()
    v = v[np.isfinite(v)]
    df = acorr_ljungbox(v, lags=[lags], model_df=model_df, return_df=True)
    df = df.rename(
        columns={
            "lb_stat": "ljung_box_stat",
            "lb_pvalue": "ljung_box_pvalue",
        }
    )
    df.insert(0, "lags", lags)
    df.insert(1, "model_df", model_df)
    return df.reset_index(drop=True)


def adf_to_row(name: str, result: ADFResult) -> dict[str, float | int | str | bool]:
    return {
        "series": name,
        "adf_stat": result.statistic,
        "adf_pvalue": result.p_value,
        "used_lags": result.used_lags,
        "n_obs": result.n_obs,
        "regression": result.regression,
        "rejects_unit_root_5pct": result.rejects_unit_root_5pct,
        "crit_1pct": result.critical_values.get("1%", float("nan")),
        "crit_5pct": result.critical_values.get("5%", float("nan")),
        "crit_10pct": result.critical_values.get("10%", float("nan")),
    }
