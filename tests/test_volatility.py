"""GARCH(1,1) simulation exhibits clustered squared returns."""

from __future__ import annotations

import numpy as np

from tsforecast.volatility import fit_garch11, simulate_garch11, squared_acf_lag1


def test_garch_path_length() -> None:
    r = simulate_garch11(n=300, seed=21)
    assert len(r) == 300
    assert np.isfinite(r).all()


def test_squared_return_acf_lag1_positive_more_often_than_white_noise() -> None:
    """Volatility clustering: corr(r_t^2, r_{t-1}^2) is usually positive.

    Independent Gaussian noise has lag-1 squared ACF centred near zero, so
    the sign is positive only about half the time. A GARCH(1,1) DGP with
    non-trivial alpha produces positive lag-1 squared ACF much more often.
    """

    n_sims = 24
    n = 700
    garch_positive = 0
    noise_positive = 0
    for i in range(n_sims):
        r = simulate_garch11(n=n, omega=0.05, alpha=0.20, beta=0.70, seed=300 + i)
        z = np.random.default_rng(800 + i).normal(size=n)
        if squared_acf_lag1(r) > 0.0:
            garch_positive += 1
        if squared_acf_lag1(z) > 0.0:
            noise_positive += 1
    assert garch_positive > noise_positive
    assert garch_positive >= 18


def test_garch_mle_persistence_is_inside_the_unit_interval() -> None:
    r = simulate_garch11(n=1500, omega=0.05, alpha=0.12, beta=0.83, seed=42)
    est = fit_garch11(r)
    assert est.success
    assert 0.0 <= est.alpha < 0.5
    assert 0.4 < est.beta < 0.99
    assert est.persistence < 1.0


def test_long_garch_path_has_positive_squared_acf() -> None:
    r = simulate_garch11(n=2000, omega=0.05, alpha=0.15, beta=0.80, seed=42)
    assert squared_acf_lag1(r) > 0.05
