"""nbe.simulators — data simulators and prior samplers.

Prior bounds are copied verbatim from the source research scripts:

Normal model (normal_cost_benchmark.py):
    mu   ~ N(0, 1)        (MU0=0.0, SIGMA0_PRIOR=1.0)
    sigma ~ U(0, 5)        (SIGMA_HI=5.0)

NIG linear-regression model (sweeps2/rb_vs_mc_sweep_tau.py):
    beta  ~ N(0, sigma0^2 * I_p)   sigma0=5.0
    tau = 1/sigma^2 ~ U(tau_low, tau_high) = U(0.01, 1.0)
    => sigma = 1/sqrt(tau)

AR(1) model (sweep_importance_sampling/run.py lines 85-97):
    simulate_ar1 ported verbatim; prior bounds (rho, sigma) supplied by caller
    via sample_prior_ar1(B, A, rng): rho ~ U(-1, 1), sigma ~ U(0, A).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------

def simulate_normal(mu, sigma, n, rng):
    """Simulate i.i.d. Normal observations.

    Parameters
    ----------
    mu, sigma : (B,) arrays
    n         : number of observations per dataset
    rng       : numpy.random.Generator

    Returns
    -------
    y : (B, n) array
    """
    B = mu.shape[0]
    return rng.normal(mu[:, None], sigma[:, None], (B, n))


def simulate_ar1(rho, sigma, T, rng):
    """Simulate AR(1) series, initialised from the stationary distribution.

    Ported verbatim from SRC/final/sweep_importance_sampling/run.py lines 85-97.

    Parameters
    ----------
    rho, sigma : (B,) arrays
    T          : series length
    rng        : numpy.random.Generator

    Returns
    -------
    x : (B, T) array
    """
    B = rho.shape[0]
    x = np.empty((B, T), dtype=np.float64)
    stat_sd = sigma / np.sqrt(np.clip(1.0 - rho**2, 1e-6, None))
    x[:, 0] = rng.normal(0.0, stat_sd)
    for t in range(1, T):
        eps = rng.normal(0.0, sigma)
        x[:, t] = rho * x[:, t - 1] + eps
    return x


def simulate_linreg(beta, sigma, X, rng):
    """Simulate linear-regression responses.

    Parameters
    ----------
    beta  : (B, p) array — regression coefficients
    sigma : (B,) array   — noise standard deviations
    X     : (n, p) array — design matrix
    rng   : numpy.random.Generator

    Returns
    -------
    y : (B, n) array
    """
    B, n = beta.shape[0], X.shape[0]
    return (beta @ X.T) + rng.normal(0, 1, (B, n)) * sigma[:, None]


# ---------------------------------------------------------------------------
# Prior samplers
# ---------------------------------------------------------------------------

def sample_prior_normal(B, rng):
    """Sample (mu, sigma) from the normal-model prior.

    Source: SRC/final/normal_cost_benchmark.py
        mu    ~ N(0, 1)    (MU0=0.0, SIGMA0_PRIOR=1.0)
        sigma ~ U(0, 5)    (SIGMA_HI=5.0)

    Parameters
    ----------
    B   : batch size
    rng : numpy.random.Generator

    Returns
    -------
    mu    : (B,) array
    sigma : (B,) array  — all positive
    """
    mu = rng.normal(0.0, 1.0, size=B)
    sigma = rng.uniform(0.0, 5.0, size=B)
    return mu, sigma


def sample_prior_ar1(B, A, rng):
    """Sample (rho, sigma) from a uniform AR(1) prior.

    Source: SRC/final/sweep_importance_sampling/run.py (draw_params interface).
        rho   ~ U(-1, 1)
        sigma ~ U(0, A)

    Parameters
    ----------
    B   : batch size
    A   : upper bound for sigma
    rng : numpy.random.Generator

    Returns
    -------
    rho   : (B,) array in (-1, 1)
    sigma : (B,) array in (0, A)
    """
    rho = rng.uniform(-1.0, 1.0, size=B)
    sigma = rng.uniform(0.0, A, size=B)
    return rho, sigma


def sample_prior_nig(B, p, rng):
    """Sample (beta, sigma) from the NIG linear-regression prior.

    Source: SRC/final/sweeps2/rb_vs_mc_sweep_tau.py
        beta  ~ N(0, sigma0^2 * I_p)   with sigma0 = 5.0
        tau = 1/sigma^2 ~ U(0.01, 1.0)  =>  sigma = 1/sqrt(tau)

    Parameters
    ----------
    B   : batch size
    p   : number of regression coefficients
    rng : numpy.random.Generator

    Returns
    -------
    beta  : (B, p) array
    sigma : (B,) array
    """
    SIGMA0 = 5.0
    TAU_LOW = 0.01
    TAU_HIGH = 1.0

    beta = rng.normal(0.0, SIGMA0, size=(B, p))
    tau = rng.uniform(TAU_LOW, TAU_HIGH, size=B)
    sigma = 1.0 / np.sqrt(tau)
    return beta, sigma
