import numpy as np
import torch
import sys

sys.path.insert(0, "src")
from nbe import losses as L


def test_ess_uniform_and_degenerate():
    assert abs(L.effective_sample_size(np.ones(100)) - 100) < 1e-6
    w = np.zeros(100)
    w[0] = 1.0
    assert abs(L.effective_sample_size(w) - 1.0) < 1e-6


def test_snis_equals_mean_when_uniform():
    ps = torch.tensor([1.0, 2.0, 3.0, 4.0])
    w = torch.ones(4)
    assert torch.allclose(L.snis_loss(ps, w), ps.mean())


def test_rb_loss_zero_at_posterior_mean():
    pm = torch.randn(8, 3)
    assert L.rb_loss(pm.clone(), pm).item() < 1e-8
