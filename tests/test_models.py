import torch, sys; sys.path.insert(0, "src")
from nbe import models


def test_ar1_forward_shapes():
    net = models.AR1Estimator()
    rho, sig = net(torch.randn(8, 100))
    assert rho.shape == (8,) and sig.shape == (8,)
    assert (rho.abs() <= 1).all() and (sig >= 0).all()


def test_deepsets_permutation_invariance():
    net = models.DeepSetsEstimator(in_dim=1, out_dim=2)
    y = torch.randn(4, 20, 1)
    a = net(y); b = net(y[:, torch.randperm(20)])
    assert torch.allclose(a, b, atol=1e-5)


def test_deepsets_output_shape():
    net = models.DeepSetsEstimator(in_dim=3, out_dim=4)
    y = torch.randn(6, 15, 3)
    out = net(y)
    assert out.shape == (6, 4)


def test_normalnbe_sigma_positive():
    net = models.NormalNBE()
    y = torch.randn(5, 30, 1)
    out = net(y)
    assert out.shape == (5, 2)
    # sigma (second column) must be positive due to softplus
    assert (out[:, 1] > 0).all()
