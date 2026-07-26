"""Per-dataset cost benchmark for the normal-model estimators (Chapter 3 table).

Measures wall-clock cost per test dataset for the two estimators compared in
Section 'Example: estimating normal parameters':

  * Metropolis-Hastings: a random-walk MH chain run SEPARATELY on each dataset,
    its post-burn-in average estimating the posterior mean E[(mu, sigma) | y].
  * Neural Bayes estimator: a single DeepSets forward pass (timing is
    weight-independent, so an untrained net of representative width is used).

The MSE accuracy numbers in the thesis are NOT recomputed here -- this script only
supplies the concrete cost column. Prints ms/dataset (MH), us/dataset (NBE) and the
speedup ratio.

Model:  mu ~ N(0,1), sigma ~ U(0,5), y_i ~ N(mu, sigma^2), n obs, 256 test datasets.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

# ---- settings (documented in the design spec) ----
N_OBS = 100
N_DATASETS = 256
MU0, SIGMA0_PRIOR = 0.0, 1.0
SIGMA_HI = 5.0
N_ITER, BURN_IN = 5000, 1000
PROP_SCALE = 0.15          # random-walk std for (mu, sigma) proposal
SEED = 0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def simulate(rng):
    """256 datasets; returns y (D, n), and true (mu, sigma)."""
    mu = rng.normal(MU0, SIGMA0_PRIOR, size=N_DATASETS)
    sigma = rng.uniform(0.0, SIGMA_HI, size=N_DATASETS)
    y = rng.normal(mu[:, None], sigma[:, None], size=(N_DATASETS, N_OBS))
    return y, mu, sigma


def log_post(mu, sigma, ybar, S, n):
    """Unnormalised log-posterior (eq:normal-logpost); -inf outside sigma in (0, A)."""
    if sigma <= 0.0 or sigma >= SIGMA_HI:
        return -np.inf
    return (-n * np.log(sigma)
            - 0.5 / sigma**2 * (S + n * (ybar - mu) ** 2)
            - (mu - MU0) ** 2 / (2.0 * SIGMA0_PRIOR**2))


def mh_one(y, rng):
    """Random-walk MH for one dataset; returns post-burn-in mean (mu, sigma)."""
    n = y.shape[0]
    ybar = y.mean()
    S = ((y - ybar) ** 2).sum()
    mu, sigma = ybar, max(y.std(), 0.5)            # sensible start
    lp = log_post(mu, sigma, ybar, S, n)
    draws = np.empty((N_ITER - BURN_IN, 2))
    for it in range(N_ITER):
        mu_p = mu + PROP_SCALE * rng.standard_normal()
        sigma_p = sigma + PROP_SCALE * rng.standard_normal()
        lp_p = log_post(mu_p, sigma_p, ybar, S, n)
        if np.log(rng.random()) < lp_p - lp:
            mu, sigma, lp = mu_p, sigma_p, lp_p
        if it >= BURN_IN:
            draws[it - BURN_IN] = (mu, sigma)
    return draws.mean(0)


# --------------------------------------------------------------------------- #
# DeepSets
# --------------------------------------------------------------------------- #
class DeepSets(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.psi = nn.Sequential(nn.Linear(1, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU())
        self.phi = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, y):                 # y: (D, n)
        h = self.psi(y.unsqueeze(-1))     # (D, n, H)
        agg = h.sum(dim=1)                # (D, H)
        return self.phi(agg)              # (D, 2)


# --------------------------------------------------------------------------- #
def main():
    rng = np.random.default_rng(SEED)
    y, mu_t, sigma_t = simulate(rng)

    # ---- MH timing: chain run separately per dataset ----
    t0 = time.perf_counter()
    for d in range(N_DATASETS):
        mh_one(y[d], rng)
    t_mh_total = time.perf_counter() - t0
    ms_per_dataset_mh = 1e3 * t_mh_total / N_DATASETS

    # ---- NBE timing: DeepSets forward pass ----
    torch.manual_seed(SEED)
    net = DeepSets().eval()
    yt = torch.tensor(y, dtype=torch.float32)
    with torch.no_grad():
        net(yt)                                    # warm-up
        reps = 200
        t0 = time.perf_counter()
        for _ in range(reps):
            net(yt)
        t_nbe_batch = (time.perf_counter() - t0) / reps
    us_per_dataset_nbe = 1e6 * t_nbe_batch / N_DATASETS

    ratio = (ms_per_dataset_mh * 1e3) / us_per_dataset_nbe

    print("=== Normal-model per-dataset cost ===")
    print(f"settings: n_obs={N_OBS}, datasets={N_DATASETS}, "
          f"MH iters={N_ITER} (burn-in {BURN_IN})")
    print(f"MH   : {t_mh_total:7.3f} s total  ->  {ms_per_dataset_mh:8.3f} ms/dataset")
    print(f"NBE  : {1e3*t_nbe_batch:7.3f} ms/batch ->  {us_per_dataset_nbe:8.2f} us/dataset")
    print(f"speedup (MH / NBE per dataset): {ratio:,.0f}x")


if __name__ == "__main__":
    main()
