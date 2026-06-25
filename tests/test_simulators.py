import numpy as np, sys; sys.path.insert(0, "src")
from nbe import simulators as S

def test_ar1_shape_and_stationarity():
    rng = np.random.default_rng(0)
    rho = np.full(64, 0.5); sigma = np.ones(64)
    x = S.simulate_ar1(rho, sigma, 100, rng)
    assert x.shape == (64, 100)
    assert abs(x.var() - 1/0.75) < 0.3   # stationary var = sigma^2/(1-rho^2) = 1/0.75

def test_normal_shape():
    rng = np.random.default_rng(0)
    mu = np.zeros(10); sigma = np.full(10, 2.0)
    y = S.simulate_normal(mu, sigma, 50, rng)
    assert y.shape == (10, 50) and abs(y.std() - 2.0) < 0.4

def test_linreg_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((40, 5)); beta = rng.standard_normal((8, 5))
    y = S.simulate_linreg(beta, np.ones(8), X, rng)
    assert y.shape == (8, 40)

def test_priors():
    rng = np.random.default_rng(0)
    mu, sig = S.sample_prior_normal(100, rng); assert mu.shape == (100,) and (sig > 0).all()
    rho, s2 = S.sample_prior_ar1(100, 2.0, rng); assert ((rho > -1) & (rho < 1)).all()
    b, s3 = S.sample_prior_nig(100, 5, rng); assert b.shape == (100, 5)
