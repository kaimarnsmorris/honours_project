"""nbe.samplers — MCMC posterior samplers.

Provides:
  mh_normal(y, n_iter, burn_in, prop_scale, rng) -> (mu_hat, sigma_hat)
      Random-walk Metropolis-Hastings for a Normal(mu, sigma^2) likelihood
      with prior mu ~ N(0,1), sigma ~ U(0,5).  Returns the post-burn-in
      posterior-mean estimates.  Ported from SRC/final/normal_cost_benchmark.py.

  gibbs_ar1(x, A, sweeps, burnin, rng) -> (rho_hat, sigma_hat)
      Gibbs sampler for an AR(1) posterior with sigma truncated to (0, A).
      Returns post-burn-in posterior-mean estimates.  Ported from
      SRC/final/sweep_importance_sampling/run.py (truncated_normal_sample,
      truncated_invgamma_sample, gibbs_posterior_mean).
"""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm, gamma as gamma_dist

# ---- priors for the Normal model (match SRC/final/normal_cost_benchmark.py) ----
_MU0 = 0.0
_SIGMA0_PRIOR = 1.0
_SIGMA_HI = 5.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log_post_normal(mu: float, sigma: float, ybar: float, S: float, n: int) -> float:
    """Unnormalised log-posterior for Normal(mu, sigma^2) model.

    Prior: mu ~ N(0,1), sigma ~ U(0, SIGMA_HI).
    Returns -inf outside sigma in (0, SIGMA_HI).
    """
    if sigma <= 0.0 or sigma >= _SIGMA_HI:
        return -np.inf
    return (
        -n * math.log(sigma)
        - 0.5 / sigma ** 2 * (S + n * (ybar - mu) ** 2)
        - (mu - _MU0) ** 2 / (2.0 * _SIGMA0_PRIOR ** 2)
    )


def _truncated_normal_sample(
    mean: float, sd: float, lo: float, hi: float, rng: np.random.Generator
) -> float:
    """Sample from N(mean, sd^2) truncated to (lo, hi) via inverse-CDF."""
    a = norm.cdf((lo - mean) / sd)
    b = norm.cdf((hi - mean) / sd)
    u = rng.uniform(a, b)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return float(mean + sd * norm.ppf(u))


def _truncated_invgamma_sample(
    shape: float,
    scale: float,
    lo2: float,
    hi2: float,
    rng: np.random.Generator,
) -> float:
    """Sample sigma^2 ~ Inv-Gamma(shape, scale) truncated to (lo2, hi2).

    Uses the duality: sigma^2 ~ InvGamma(a, b) <=> 1/sigma^2 ~ Gamma(a, rate=b).
    """
    prec_lo = 1.0 / hi2
    prec_hi = 1.0 / lo2 if lo2 > 0 else np.inf
    a, b = shape, scale
    cdf_lo = gamma_dist.cdf(prec_lo, a, scale=1.0 / b)
    cdf_hi = (
        gamma_dist.cdf(prec_hi, a, scale=1.0 / b) if np.isfinite(prec_hi) else 1.0
    )
    u = rng.uniform(cdf_lo, cdf_hi)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    prec = gamma_dist.ppf(u, a, scale=1.0 / b)
    prec = np.clip(prec, 1e-12, None)
    return float(1.0 / prec)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mh_normal(
    y: np.ndarray,
    n_iter: int,
    burn_in: int,
    prop_scale: float,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Random-walk MH posterior-mean estimator for a Normal model.

    Parameters
    ----------
    y : array of shape (n,)
        Observed data.
    n_iter : int
        Total number of MCMC iterations (including burn-in).
    burn_in : int
        Number of iterations to discard as burn-in.
    prop_scale : float
        Standard deviation of the Gaussian random-walk proposals for both mu
        and sigma.
    rng : numpy Generator
        Random-number generator (e.g. ``np.random.default_rng(seed)``).

    Returns
    -------
    (mu_hat, sigma_hat) : tuple of float
        Post-burn-in posterior means of mu and sigma.
    """
    n = y.shape[0]
    ybar = float(y.mean())
    S = float(((y - ybar) ** 2).sum())

    mu = ybar
    sigma = max(float(y.std()), 0.5)
    lp = _log_post_normal(mu, sigma, ybar, S, n)

    draws = np.empty((n_iter - burn_in, 2))
    for it in range(n_iter):
        mu_p = mu + prop_scale * rng.standard_normal()
        sigma_p = sigma + prop_scale * rng.standard_normal()
        lp_p = _log_post_normal(mu_p, sigma_p, ybar, S, n)
        if math.log(rng.random()) < lp_p - lp:
            mu, sigma, lp = mu_p, sigma_p, lp_p
        if it >= burn_in:
            draws[it - burn_in] = (mu, sigma)

    post_means = draws.mean(0)
    return float(post_means[0]), float(post_means[1])


def gibbs_ar1(
    x: np.ndarray,
    A: float,
    sweeps: int,
    burnin: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Gibbs-sampler posterior-mean estimator for an AR(1) model.

    Model: x_t = rho * x_{t-1} + eps_t, eps_t ~ N(0, sigma^2),
    with rho truncated to (-1, 1) and sigma truncated to (0, A).

    Parameters
    ----------
    x : array of shape (T,)
        One observed AR(1) series.
    A : float
        Upper bound on sigma (sigma ~ U(0, A) prior reflected in truncation).
    sweeps : int
        Total number of Gibbs sweeps (including burn-in).
    burnin : int
        Number of sweeps to discard as burn-in.
    rng : numpy Generator
        Random-number generator.

    Returns
    -------
    (rho_hat, sigma_hat) : tuple of float
        Post-burn-in posterior means of rho and sigma.
    """
    T = x.shape[0]
    S1 = float(np.sum(x[:-1] ** 2))
    S01 = float(np.sum(x[1:] * x[:-1]))
    Sxx = float(np.sum(x[1:] ** 2))

    # Initialise at OLS estimate
    rho = float(np.clip(S01 / max(S1, 1e-8), -0.99, 0.99))
    ssr = Sxx - 2 * rho * S01 + rho ** 2 * S1
    sigma2 = max(ssr / max(T - 1, 1), 1e-4)

    rs: list[float] = []
    ss: list[float] = []

    for it in range(sweeps):
        # rho | sigma^2, x  ~  N(S01/S1, sigma2/S1) truncated to (-1, 1)
        m = S01 / max(S1, 1e-8)
        sd = math.sqrt(sigma2 / max(S1, 1e-8))
        rho = _truncated_normal_sample(m, sd, -1.0, 1.0, rng)

        # sigma^2 | rho, x ~ Inv-Gamma((T-2)/2, SSR/2) truncated to (0, A^2)
        ssr = Sxx - 2 * rho * S01 + rho ** 2 * S1
        shape = max(T / 2.0 - 1.0, 1e-3)
        scale = max(ssr / 2.0, 1e-8)
        sigma2 = _truncated_invgamma_sample(shape, scale, 1e-8, A ** 2, rng)

        if it >= burnin:
            rs.append(rho)
            ss.append(math.sqrt(sigma2))

    return float(np.mean(rs)), float(np.mean(ss))
