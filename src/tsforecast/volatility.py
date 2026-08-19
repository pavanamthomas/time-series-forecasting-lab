"""Gaussian GARCH(1,1) simulation and quasi-maximum likelihood.

Model
-----
    r_t = sigma_t z_t,               z_t iid N(0, 1)
    sigma_t^2 = omega + alpha r_{t-1}^2 + beta sigma_{t-1}^2

Assumptions used by the estimator
---------------------------------
1. Conditional mean is zero. If a non-zero mean is present, demean first.
2. The Gaussian log-likelihood is treated as a quasi-likelihood (QMLE).
   Point estimates remain consistent under weaker conditional distributions
   with finite fourth moments under regularity conditions; interval
   estimates here are not sandwich-robust.
3. Parameter constraints: omega > 0, alpha >= 0, beta >= 0, and
   alpha + beta < 1 for covariance stationarity of the variance process.
4. The specification is GARCH(1,1) with no leverage, no jumps, and no
   variance targeting beyond the unconstrained MLE.
5. sigma_0^2 is initialised at the unconditional variance when
   alpha + beta < 1, otherwise at the sample variance of r.

This module does not depend on the `arch` package. Numerical optimisation
uses SciPy bounded L-BFGS-B with a stationarity penalty in the objective.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-12
_STATIONARITY_CAP = 0.999


@dataclass(frozen=True)
class GARCH11Result:
    omega: float
    alpha: float
    beta: float
    persistence: float
    nll: float
    success: bool
    message: str
    n_obs: int
    sigma2: np.ndarray
    returns: np.ndarray

    @property
    def unconditional_variance(self) -> float:
        gap = 1.0 - self.persistence
        if gap <= 0.0:
            return float("nan")
        return float(self.omega / gap)


def simulate_garch11(
    n: int,
    omega: float = 0.05,
    alpha: float = 0.12,
    beta: float = 0.83,
    *,
    seed: int = 42,
    burnin: int = 250,
    return_variance: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Simulate a Gaussian GARCH(1,1) path after a burn-in period."""

    if n < 2:
        raise ValueError("n must be at least 2.")
    if omega <= 0 or alpha < 0 or beta < 0:
        raise ValueError("Require omega > 0 and alpha, beta >= 0.")
    if alpha + beta >= 1.0:
        raise ValueError("Covariance stationarity requires alpha + beta < 1.")
    rng = np.random.default_rng(seed)
    total = n + burnin
    r = np.empty(total, dtype=float)
    sigma2 = np.empty(total, dtype=float)
    sigma2[0] = omega / (1.0 - alpha - beta)
    z = rng.normal(size=total)
    r[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, total):
        sigma2[t] = omega + alpha * r[t - 1] ** 2 + beta * sigma2[t - 1]
        r[t] = np.sqrt(sigma2[t]) * z[t]
    if return_variance:
        return r[burnin:], sigma2[burnin:]
    return r[burnin:]


def garch11_variance(
    returns: np.ndarray,
    omega: float,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """Filter conditional variance given parameters and a return path."""

    r = np.asarray(returns, dtype=float).ravel()
    n = r.size
    sigma2 = np.empty(n, dtype=float)
    persist = alpha + beta
    sample_var = float(np.var(r, ddof=1)) if n > 1 else float(np.mean(r**2))
    if persist < _STATIONARITY_CAP:
        sigma2[0] = omega / max(1.0 - persist, _EPS)
    else:
        sigma2[0] = max(sample_var, _EPS)
    r2 = r**2
    for t in range(1, n):
        sigma2[t] = omega + alpha * r2[t - 1] + beta * sigma2[t - 1]
    return np.maximum(sigma2, _EPS)


def _gaussian_nll(params: np.ndarray, r: np.ndarray) -> float:
    omega, alpha, beta = (float(p) for p in params)
    if omega <= 0.0 or alpha < 0.0 or beta < 0.0:
        return 1e12
    if alpha + beta >= _STATIONARITY_CAP:
        return 1e12 + 1e6 * (alpha + beta - _STATIONARITY_CAP + 1.0)
    sigma2 = garch11_variance(r, omega, alpha, beta)
    return float(0.5 * np.sum(np.log(2.0 * np.pi) + np.log(sigma2) + r**2 / sigma2))


def fit_garch11(
    returns: np.ndarray,
    *,
    start: tuple[float, float, float] | None = None,
) -> GARCH11Result:
    """Estimate Gaussian GARCH(1,1) by QMLE (SciPy L-BFGS-B)."""

    r = np.asarray(returns, dtype=float).ravel()
    if r.size < 50:
        raise ValueError("GARCH(1,1) QMLE needs a reasonably long sample.")
    var = float(np.var(r, ddof=1))
    starts: list[tuple[float, float, float]]
    if start is None:
        starts = [
            (max(var * 0.05, 1e-6), 0.05, 0.90),
            (max(var * 0.10, 1e-6), 0.10, 0.80),
            (max(var * 0.02, 1e-6), 0.02, 0.95),
        ]
    else:
        starts = [start]

    bounds = [(1e-8, None), (0.0, 0.999), (0.0, 0.999)]
    best = None
    for x0 in starts:
        result = minimize(
            _gaussian_nll,
            x0=np.array(x0, dtype=float),
            args=(r,),
            method="L-BFGS-B",
            bounds=bounds,
        )
        if best is None or result.fun < best.fun:
            best = result

    assert best is not None
    omega, alpha, beta = (float(x) for x in best.x)
    persist = alpha + beta
    sigma2 = garch11_variance(r, omega, alpha, beta)
    return GARCH11Result(
        omega=omega,
        alpha=alpha,
        beta=beta,
        persistence=persist,
        nll=float(best.fun),
        success=bool(best.success) and persist < 1.0,
        message=str(best.message),
        n_obs=int(r.size),
        sigma2=sigma2,
        returns=r,
    )


def _obs_nll(params: np.ndarray, r: np.ndarray) -> np.ndarray:
    omega, alpha, beta = (float(p) for p in params)
    sigma2 = garch11_variance(r, omega, alpha, beta)
    return 0.5 * (np.log(2.0 * np.pi) + np.log(sigma2) + r**2 / sigma2)


def garch11_sandwich_se(result: GARCH11Result, *, step: float = 1e-5) -> dict[str, float]:
    """Outer-product / Hessian sandwich standard errors for GARCH(1,1) QMLE.

    Scores are obtained by finite differences of the observation-wise
    Gaussian quasi-log-likelihood. The Hessian is the Jacobian of the
    summed scores. This is a numerical sandwich, not an analytic score
    recursion, and it inherits the Gaussian quasi-likelihood.
    """

    r = np.asarray(result.returns, dtype=float).ravel()
    theta = np.array([result.omega, result.alpha, result.beta], dtype=float)
    k = theta.size
    n = r.size
    scores = np.empty((n, k), dtype=float)
    for j in range(k):
        bumped = theta.copy()
        bumped[j] += step
        scores[:, j] = (_obs_nll(bumped, r) - _obs_nll(theta, r)) / step
    meat = scores.T @ scores
    # Hessian of total nll ≈ J^T of mean scores times n, via another difference
    hessian = np.empty((k, k), dtype=float)
    total = scores.sum(axis=0)
    for j in range(k):
        bumped = theta.copy()
        bumped[j] += step
        scores_b = np.empty((n, k), dtype=float)
        for m in range(k):
            db = bumped.copy()
            db[m] += step
            scores_b[:, m] = (_obs_nll(db, r) - _obs_nll(bumped, r)) / step
        hessian[:, j] = (scores_b.sum(axis=0) - total) / step
    try:
        h_inv = np.linalg.inv(hessian)
        vcov = h_inv @ meat @ h_inv
        se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    return {
        "se_omega": float(se[0]),
        "se_alpha": float(se[1]),
        "se_beta": float(se[2]),
    }


def squared_acf_lag1(returns: np.ndarray) -> float:
    """Lag-1 sample autocorrelation of squared returns."""

    r2 = np.asarray(returns, dtype=float).ravel() ** 2
    r2 = r2 - r2.mean()
    if r2.size < 3:
        return float("nan")
    num = float(np.dot(r2[1:], r2[:-1]))
    den = float(np.dot(r2, r2))
    if den <= 0.0:
        return float("nan")
    return num / den
