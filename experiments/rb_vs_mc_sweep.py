"""Rao-Blackwellised vs Monte-Carlo Bayes-risk sweep for Bayesian linear regression.

Model (per dataset):

    beta  ~ N(0, sigma0^2 I_p)
    sigma ~ U(sigma_low, sigma_high)
    Y     = X beta + eps,   eps ~ N(0, sigma^2 I_n)

A network phi(Y) -> (beta_hat, sigma_hat) is trained to minimise a Bayes-risk
estimate of ||phi(Y) - theta||^2 with theta = (beta, sigma).

* MC target: the noisy parameter draw beta itself.
* RB target: the analytic conditional mean beta_n = E[beta | sigma, Y] plus the
  constant tr(Sigma_n), so the RB loss is an *unbiased estimate of the same Bayes
  risk* as MC but with lower per-sample variance. Both share a minimiser; RB wins
  when the gradient is noise-limited (small batch, high dimension).

This script runs a 2-D sweep over (p, batch_size) -- the dimension axis controls
how noise-limited the problem is, the batch axis controls how much that noise is
averaged out per step. For each (p, batch_size, seed) it trains an MC net and an
RB net from *identical* initial weights and scores both on a common held-out test
set. Curves are averaged over seeds.

Methodology matches ../4_rb_vs_mc_longrun.ipynb and ../../sweeps/sweep_p.py:
Cholesky solve (not an explicit inverse), floored sigma, .sum() over the beta
dimensions, gradient clipping, fixed seeded design matrix.

Outputs (written next to this file):
  * rb_vs_mc_sweep_results.npz  -- raw per-(p,bs) seed curves
  * rb_vs_mc_sweep_curves.png   -- convergence curves
  * rb_vs_mc_sweep_summary.png  -- final-loss and RB-advantage heatmaps

Usage:
  python rb_vs_mc_sweep.py                 # full sweep (slow on CPU)
  python rb_vs_mc_sweep.py --quick         # tiny config for a smoke test
  python rb_vs_mc_sweep.py --p 2 10 20 --batch-sizes 16 64 --seeds 2 --epochs 64
  python rb_vs_mc_sweep.py --no-plots      # skip matplotlib
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

# Windows: numpy(MKL) and torch can each ship an OpenMP runtime; allow the duplicate
# rather than aborting. Must be set before torch is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# Model + problem setup
# --------------------------------------------------------------------------- #
class MLP(nn.Module):
    """Flat-y MLP: (B, n_obs) -> (B, p+1). Last column is a softplus-sigma head."""

    def __init__(self, n_obs: int, p: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.p = p
        layers: list[nn.Module] = [nn.Linear(n_obs, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, p + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out = self.net(y)
        sigma_head = nn.functional.softplus(out[:, self.p:]) + 1e-4
        return torch.cat([out[:, :self.p], sigma_head], dim=-1)


def build_design(p: int, n_obs: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed (seeded) design matrix X = [1, N(0,1), ...] and cached XtX, prior precision.

    Seeded so the design is identical across batch sizes and seeds for a given p.
    """
    torch.manual_seed(42)
    x_cov = torch.randn(n_obs, p - 1, device=device)
    X = torch.cat([torch.ones(n_obs, 1, device=device), x_cov], dim=1)  # (n_obs, p)
    XtX = X.T @ X
    return X, XtX, X.new_empty(0)  # third slot reserved/unused, kept for clarity


def make_sampler(X, n_obs, sigma0, sigma_low, sigma_high, device):
    def sample_batch(batch_size: int):
        sigma = torch.rand(batch_size, device=device) * (sigma_high - sigma_low) + sigma_low
        beta = torch.randn(batch_size, X.shape[1], device=device) * sigma0
        y = beta @ X.T + sigma.unsqueeze(1) * torch.randn(batch_size, n_obs, device=device)
        return beta, sigma, y

    return sample_batch


def make_rb_target(X, XtX, Sigma0_inv, p, device):
    @torch.no_grad()
    def rb_target(sigma: torch.Tensor, y: torch.Tensor):
        """E[beta | sigma, y] and Cov(beta | sigma, y), vectorised over the batch.

        Cholesky solve rather than an explicit inverse -- far more stable at large p
        when sigma is small and the precision matrix is ill-conditioned.
        """
        sigma_sq = (sigma ** 2).clamp(min=1e-4)
        prec = Sigma0_inv.unsqueeze(0) + XtX.unsqueeze(0) / sigma_sq.view(-1, 1, 1)  # (B,p,p)
        L = torch.linalg.cholesky(prec)
        Xty = y @ X
        rhs = (Xty / sigma_sq.unsqueeze(1)).unsqueeze(-1)
        beta_n = torch.cholesky_solve(rhs, L).squeeze(-1)  # (B,p)
        eye = torch.eye(p, device=device).expand_as(prec)
        Sigma_n = torch.cholesky_solve(eye, L)  # (B,p,p)
        return beta_n, Sigma_n

    return rb_target


def loss_mc(phi, beta, sigma):
    theta = torch.cat([beta, sigma.unsqueeze(1)], dim=-1)
    return ((phi - theta) ** 2).mean()


def make_loss_rb(rb_target, p):
    def loss_rb(phi, sigma, y):
        beta_n, Sigma_n = rb_target(sigma, y)
        tr_Sigma_n = torch.diagonal(Sigma_n, dim1=-2, dim2=-1).sum(-1)  # constant in phi
        per_sample = ((phi[:, :p] - beta_n) ** 2).sum(-1) + tr_Sigma_n + (phi[:, p] - sigma) ** 2
        return per_sample.mean()

    return loss_rb


# --------------------------------------------------------------------------- #
# One (p, batch_size) cell, averaged over seeds
# --------------------------------------------------------------------------- #
def run_cell(p, bs, cfg, device):
    X, XtX, _ = build_design(p, cfg.n_obs, device)
    Sigma0_inv = torch.eye(p, device=device) / (cfg.sigma0 ** 2)

    sample_batch = make_sampler(X, cfg.n_obs, cfg.sigma0, cfg.sigma_low, cfg.sigma_high, device)
    rb_target = make_rb_target(X, XtX, Sigma0_inv, p, device)
    loss_rb = make_loss_rb(rb_target, p)

    n_batches = max(1, cfg.samples_per_epoch // bs)

    # common held-out test set (same across seeds for this p)
    torch.manual_seed(999 + p)
    beta_test, sigma_test, y_test = sample_batch(cfg.test_size)
    theta_test = torch.cat([beta_test, sigma_test.unsqueeze(1)], dim=-1)

    mc_curves, rb_curves, mc_b_curves, rb_b_curves = [], [], [], []

    for seed in range(cfg.n_seeds):
        torch.manual_seed(seed)
        m_mc = MLP(cfg.n_obs, p, hidden=cfg.hidden).to(device)
        m_rb = MLP(cfg.n_obs, p, hidden=cfg.hidden).to(device)
        m_rb.load_state_dict(m_mc.state_dict())  # identical init
        opt_mc = torch.optim.Adam(m_mc.parameters(), lr=cfg.lr)
        opt_rb = torch.optim.Adam(m_rb.parameters(), lr=cfg.lr)

        t_mc, t_rb, t_mc_b, t_rb_b = [], [], [], []
        for _ in range(cfg.n_epochs):
            for _ in range(n_batches):
                beta, sigma, y = sample_batch(bs)

                opt_mc.zero_grad()
                loss_mc(m_mc(y), beta, sigma).backward()
                torch.nn.utils.clip_grad_norm_(m_mc.parameters(), 10.0)
                opt_mc.step()

                opt_rb.zero_grad()
                loss_rb(m_rb(y), sigma, y).backward()
                torch.nn.utils.clip_grad_norm_(m_rb.parameters(), 10.0)
                opt_rb.step()

            with torch.no_grad():
                pmc, prb = m_mc(y_test), m_rb(y_test)
                t_mc.append(((pmc - theta_test) ** 2).mean().item())
                t_rb.append(((prb - theta_test) ** 2).mean().item())
                t_mc_b.append(((pmc[:, :p] - beta_test) ** 2).mean().item())
                t_rb_b.append(((prb[:, :p] - beta_test) ** 2).mean().item())

        mc_curves.append(t_mc); rb_curves.append(t_rb)
        mc_b_curves.append(t_mc_b); rb_b_curves.append(t_rb_b)

    return dict(
        mc=np.array(mc_curves), rb=np.array(rb_curves),
        mc_b=np.array(mc_b_curves), rb_b=np.array(rb_b_curves),
        n_batches=n_batches,
    )


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_curves(results, P_GRID, BATCH_SIZES, n_epochs, out_path):
    import matplotlib.pyplot as plt

    epochs = np.arange(1, n_epochs + 1)
    nrows, ncols = len(P_GRID), len(BATCH_SIZES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharex=True)
    for i, p in enumerate(P_GRID):
        for j, bs in enumerate(BATCH_SIZES):
            ax = axes[i][j]
            r = results[(p, bs)]
            mc_m, mc_s = r['mc'].mean(0), r['mc'].std(0)
            rb_m, rb_s = r['rb'].mean(0), r['rb'].std(0)
            ax.plot(epochs, mc_m, color='tab:blue', label='MC')
            ax.fill_between(epochs, mc_m - mc_s, mc_m + mc_s, alpha=0.2, color='tab:blue')
            ax.plot(epochs, rb_m, color='tab:orange', label='RB')
            ax.fill_between(epochs, rb_m - rb_s, rb_m + rb_s, alpha=0.2, color='tab:orange')
            ax.set_yscale('log')
            ax.grid(True, which='both', ls='--', linewidth=0.5)
            if i == 0:
                ax.set_title(f'batch_size = {bs}')
            if j == 0:
                ax.set_ylabel(f'p = {p}\nTest MSE (log)')
            if i == nrows - 1:
                ax.set_xlabel('Epoch')
            if i == 0 and j == 0:
                ax.legend()
    fig.suptitle('RB vs MC convergence (mean +/- std over seeds)', y=1.001)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_summary(results, P_GRID, BATCH_SIZES, out_path):
    import matplotlib.pyplot as plt

    P = np.array(P_GRID)
    B = np.array(BATCH_SIZES)
    final_ratio = np.zeros((len(P), len(B)))
    max_ratio = np.zeros((len(P), len(B)))
    for i, p in enumerate(P_GRID):
        for j, bs in enumerate(BATCH_SIZES):
            r = results[(p, bs)]
            final_ratio[i, j] = r['mc'][:, -1].mean() / r['rb'][:, -1].mean()
            max_ratio[i, j] = (r['mc'].mean(0) / r['rb'].mean(0)).max()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, M, title in ((axes[0], final_ratio, 'Final MC/RB ratio'),
                         (axes[1], max_ratio, 'Max MC/RB ratio along curve')):
        im = ax.imshow(M, origin='lower', aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(B))); ax.set_xticklabels(B)
        ax.set_yticks(range(len(P))); ax.set_yticklabels(P)
        ax.set_xlabel('batch_size'); ax.set_ylabel('p')
        ax.set_title(title)
        for i in range(len(P)):
            for j in range(len(B)):
                ax.text(j, i, f'{M[i, j]:.2f}', ha='center', va='center',
                        color='w' if M[i, j] < M.max() * 0.6 else 'k', fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--p', type=int, nargs='+', default=[2, 5, 10, 20, 30],
                   dest='p_grid', help='dimension grid to sweep')
    p.add_argument('--batch-sizes', type=int, nargs='+', default=[4, 16, 64, 256])
    p.add_argument('--seeds', type=int, default=3, dest='n_seeds')
    p.add_argument('--epochs', type=int, default=128, dest='n_epochs')
    p.add_argument('--samples-per-epoch', type=int, default=2048)
    p.add_argument('--n-obs', type=int, default=50)
    p.add_argument('--sigma0', type=float, default=5.0)
    p.add_argument('--sigma-low', type=float, default=0.1)
    p.add_argument('--sigma-high', type=float, default=10.0)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--test-size', type=int, default=1024)
    p.add_argument('--cpu', action='store_true', help='force CPU even if CUDA is available')
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--quick', action='store_true',
                   help='tiny config (p=[2,20], bs=[16,64], 2 seeds, 16 epochs) for a smoke test')
    cfg = p.parse_args()
    if cfg.quick:
        cfg.p_grid = [2, 20]
        cfg.batch_sizes = [16, 64]
        cfg.n_seeds = 2
        cfg.n_epochs = 16
        cfg.samples_per_epoch = 512
        cfg.test_size = 256
    return cfg


def main():
    cfg = parse_args()
    device = torch.device('cpu' if cfg.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"device: {device}")
    print(f"sweep: p={cfg.p_grid}  batch_sizes={cfg.batch_sizes}  "
          f"seeds={cfg.n_seeds}  epochs={cfg.n_epochs}  samples/epoch={cfg.samples_per_epoch}")

    results = {}
    print(f"\n{'p':>4} {'bs':>5} {'n_batch':>8} {'time(s)':>8} "
          f"{'MC final':>10} {'RB final':>10} {'ratio':>7} {'max ratio':>10}")
    print('-' * 76)
    for p in cfg.p_grid:
        for bs in cfg.batch_sizes:
            t0 = time.time()
            r = run_cell(p, bs, cfg, device)
            dt = time.time() - t0
            results[(p, bs)] = r
            mc_f = r['mc'][:, -1].mean()
            rb_f = r['rb'][:, -1].mean()
            max_ratio = (r['mc'].mean(0) / r['rb'].mean(0)).max()
            print(f"{p:>4d} {bs:>5d} {r['n_batches']:>8d} {dt:>8.1f} "
                  f"{mc_f:>10.4f} {rb_f:>10.4f} {mc_f / rb_f:>6.2f}x {max_ratio:>9.2f}x")

    # ----- save raw curves -----
    flat = {}
    for (p, bs), r in results.items():
        for k in ('mc', 'rb', 'mc_b', 'rb_b'):
            flat[f'p{p}_bs{bs}_{k}'] = r[k]
    npz_path = HERE / 'rb_vs_mc_sweep_results.npz'
    np.savez(
        npz_path,
        p_grid=np.array(cfg.p_grid),
        batch_sizes=np.array(cfg.batch_sizes),
        n_epochs=np.array(cfg.n_epochs),
        n_seeds=np.array(cfg.n_seeds),
        **flat,
    )
    print(f"\nSaved {npz_path.name}")

    # ----- plots -----
    if not cfg.no_plots:
        try:
            plot_curves(results, cfg.p_grid, cfg.batch_sizes, cfg.n_epochs,
                        HERE / 'rb_vs_mc_sweep_curves.png')
            plot_summary(results, cfg.p_grid, cfg.batch_sizes,
                         HERE / 'rb_vs_mc_sweep_summary.png')
            print("Saved rb_vs_mc_sweep_curves.png and rb_vs_mc_sweep_summary.png")
        except Exception as exc:  # plotting is non-essential; don't lose the .npz
            print(f"[warn] plotting failed: {exc}")


if __name__ == '__main__':
    main()
