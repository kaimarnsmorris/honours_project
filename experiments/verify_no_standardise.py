#!/usr/bin/env python3
# Verification: does dropping the standardise() step restore sigma recovery?
# Trains run.py's AR1Estimator on the RAW series (uniform prior, no IS) and
# reports sigma correlation / MSE vs the prior-variance floor.
import numpy as np
import torch
from torch import nn
import run

A, T = 2.0, 100
EPOCHS, STEPS, BATCH = 60, 128, 128


def _standardise(x):
    # The buggy preprocessing this script exists to demonstrate: dividing each
    # series by its own SD is scale-invariant and erases the sigma information.
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    return x / sd


def train_raw(standardise):
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    net = run.AR1Estimator(64, 64, A)
    opt = torch.optim.Adam(net.parameters(), lr=0.002)
    for ep in range(EPOCHS):
        net.train()
        for _ in range(STEPS):
            rho = rng.uniform(-1.0, 1.0, BATCH)
            sigma = rng.uniform(0.0, A, BATCH)
            x = run.simulate_ar1(rho, sigma, T, rng)
            xt = torch.as_tensor(x, dtype=torch.float32)
            if standardise:
                xt = _standardise(xt)
            rt = torch.as_tensor(rho, dtype=torch.float32)
            st = torch.as_tensor(sigma, dtype=torch.float32)
            rh, sh = net(xt)
            loss = ((rh - rt) ** 2 + (sh - st) ** 2).mean()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
    return net


def evaluate(net, standardise):
    rng = np.random.default_rng(2024)
    rho = rng.uniform(-1.0, 1.0, 4000)
    sigma = rng.uniform(0.0, A, 4000)
    x = run.simulate_ar1(rho, sigma, T, rng)
    xt = torch.as_tensor(x, dtype=torch.float32)
    if standardise:
        xt = _standardise(xt)
    net.eval()
    with torch.no_grad():
        rh, sh = net(xt)
    rh, sh = rh.numpy(), sh.numpy()
    print(f"  sigma_hat mean {sh.mean():.3f} sd {sh.std():.3f} (prior mean {A/2:.3f})")
    print(f"  corr(sigma_hat,sigma) = {np.corrcoef(sh,sigma)[0,1]:.3f}")
    print(f"  sigma-MSE = {np.mean((sh-sigma)**2):.3f}  prior var = {A**2/12:.3f}")
    print(f"  corr(rho_hat,rho)     = {np.corrcoef(rh,rho)[0,1]:.3f}")


if __name__ == "__main__":
    print("=== WITH standardise (current run.py behaviour) ===")
    evaluate(train_raw(True), True)
    print("=== WITHOUT standardise (notebook behaviour) ===")
    evaluate(train_raw(False), False)
