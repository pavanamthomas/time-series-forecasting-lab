"""Tests for documented synthetic data-generating processes."""

from __future__ import annotations

import numpy as np
import pandas as pd

from tsforecast.dgp import (
    DEFAULT_SEED,
    generate_catalog,
    simulate_seasonal,
    simulate_stationary_ar,
    simulate_structural_break,
    simulate_trend_only,
    simulate_volatility_clustered,
)


def test_default_seed_is_documented() -> None:
    assert DEFAULT_SEED == 42


def test_catalog_contains_five_labelled_simulated_series() -> None:
    catalog = generate_catalog(seed=DEFAULT_SEED)
    expected = {
        "simulated_trend_only",
        "simulated_seasonal",
        "simulated_stationary_ar",
        "simulated_structural_break",
        "simulated_volatility_clustering",
    }
    assert set(catalog) == expected
    for name, item in catalog.items():
        assert item.simulated is True
        assert item.name.startswith("simulated_")
        assert item.values.name == name
        assert item.values.notna().all()
        assert len(item.values) >= 200


def test_trend_only_correlates_with_time() -> None:
    item = simulate_trend_only(n=240, slope=0.08, sigma=0.4, seed=0)
    t = np.arange(len(item.values), dtype=float)
    corr = np.corrcoef(t, item.values.to_numpy())[0, 1]
    assert corr > 0.95


def test_seasonal_monthly_means_are_not_flat() -> None:
    item = simulate_seasonal(n=240, sigma=0.4, seed=1)
    y = item.values.to_numpy()
    means = np.array([y[i::12].mean() for i in range(12)])
    assert means.max() - means.min() > 10.0


def test_stationary_ar_sample_acf1_matches_sign_of_phi() -> None:
    item = simulate_stationary_ar(n=500, phi=0.6, sigma=1.0, seed=2)
    y = item.values.to_numpy()
    y = y - y.mean()
    acf1 = float(np.dot(y[1:], y[:-1]) / np.dot(y, y))
    assert 0.35 < acf1 < 0.8


def test_structural_break_shifts_the_mean() -> None:
    item = simulate_structural_break(
        n=300, break_index=150, mu_pre=0.0, mu_post=3.0, sigma=0.5, seed=3
    )
    y = item.values
    pre = y.iloc[:150].mean()
    post = y.iloc[150:].mean()
    assert post - pre > 2.0


def test_reproducible_under_fixed_seed() -> None:
    a = simulate_trend_only(seed=42).values.to_numpy()
    b = simulate_trend_only(seed=42).values.to_numpy()
    np.testing.assert_array_equal(a, b)


def test_volatility_series_is_a_pandas_series() -> None:
    item = simulate_volatility_clustered(n=400, seed=4)
    assert isinstance(item.values, pd.Series)
    assert "conditional_variance" in item.parameters
    assert len(item.parameters["conditional_variance"]) == len(item.values)
