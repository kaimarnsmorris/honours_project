import numpy as np
import sys
sys.path.insert(0, "src")
from nbe import samplers as M, simulators as S


def test_mh_normal_recovers():
    rng = np.random.default_rng(1)
    y = rng.normal(3.0, 2.0, 200)
    mu, sig = M.mh_normal(y, n_iter=3000, burn_in=1000, prop_scale=0.3, rng=rng)
    assert abs(mu - 3.0) < 0.5 and abs(sig - 2.0) < 0.5


def test_gibbs_ar1_recovers():
    rng = np.random.default_rng(2)
    x = S.simulate_ar1(np.array([0.6]), np.array([1.0]), 300, rng)[0]
    rho, sig = M.gibbs_ar1(x, A=2.0, sweeps=1500, burnin=500, rng=rng)
    assert abs(rho - 0.6) < 0.2 and abs(sig - 1.0) < 0.3
