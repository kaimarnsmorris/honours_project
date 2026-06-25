"""Smoke tests for nbe.training — train loop and gradient-variance measurement."""
import sys
import numpy as np
import torch

sys.path.insert(0, "src")
from nbe import training, models, simulators as S, losses


# ---------------------------------------------------------------------------
#  train smoke test
# ---------------------------------------------------------------------------

def test_train_smoke_reduces_loss():
    dev = "cpu"
    net = models.AR1Estimator()

    def sim(rho, sig, rng):
        x = S.simulate_ar1(rho, sig, 100, rng)
        return x

    def prior(B, rng):
        return S.sample_prior_ar1(B, 2.0, rng)

    net, curve, ess = training.train(
        net, sim, prior, losses.mc_loss,
        epochs=2, steps=20, batch=64, lr=1e-3,
        device=dev, seed=0,
    )
    assert len(curve) == 2, f"expected 2 validation points, got {len(curve)}"
    assert curve[-1] < curve[0], (
        f"loss did not decrease: curve[0]={curve[0]:.6f}, curve[-1]={curve[-1]:.6f}"
    )


# ---------------------------------------------------------------------------
#  gradient_variance smoke test: returns a non-negative float
# ---------------------------------------------------------------------------

def test_gradient_variance_returns_nonneg_float():
    dev = "cpu"
    net = models.AR1Estimator()

    def alpha_draw(B, rng):
        return S.sample_prior_ar1(B, 2.0, rng)

    def sim(rho, sig, rng):
        return S.simulate_ar1(rho, sig, 100, rng)

    rng = np.random.default_rng(42)
    v = training.gradient_variance(net, alpha_draw, sim, batch=32, M=10,
                                   device=dev, rng=rng)
    assert isinstance(v, float), f"expected float, got {type(v)}"
    assert v >= 0.0, f"variance must be non-negative, got {v}"
