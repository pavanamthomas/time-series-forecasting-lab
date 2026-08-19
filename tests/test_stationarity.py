"""ADF behaviour on a stationary simulated AR process."""

from __future__ import annotations

import numpy as np

from tsforecast.dgp import simulate_stationary_ar, simulate_trend_only
from tsforecast.stationarity import acf_pacf_table, adf_unit_root


def test_adf_rejects_unit_root_on_stationary_ar() -> None:
    """With |phi| well below 1 and a moderate sample, ADF with a constant rejects."""

    item = simulate_stationary_ar(n=500, phi=0.4, sigma=1.0, seed=7)
    result = adf_unit_root(item.values, regression="c")
    assert result.p_value < 0.05
    assert result.rejects_unit_root_5pct


def test_acf_lag1_of_ar_is_positive() -> None:
    item = simulate_stationary_ar(n=400, phi=0.55, seed=8)
    table = acf_pacf_table(item.values, nlags=8)
    acf1 = float(table.loc[table["lag"] == 1, "acf"].iloc[0])
    assert acf1 > 0.25


def test_trend_series_adf_with_constant_only_is_a_different_regression() -> None:
    """Document the specification: trend DGP + regression='c' is not regression='ct'."""

    y = simulate_trend_only(n=240, seed=9).values
    with_c = adf_unit_root(y, regression="c")
    with_ct = adf_unit_root(y, regression="ct")
    assert with_c.regression == "c"
    assert with_ct.regression == "ct"
    assert with_c.n_obs > 100
    assert with_ct.n_obs > 100


def test_white_noise_acf_lag1_is_small() -> None:
    rng = np.random.default_rng(11)
    y = rng.normal(size=800)
    table = acf_pacf_table(y, nlags=5)
    acf1 = abs(float(table.loc[table["lag"] == 1, "acf"].iloc[0]))
    assert acf1 < 0.1
