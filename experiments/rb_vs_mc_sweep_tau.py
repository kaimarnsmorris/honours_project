"""Rao-Blackwellised vs Monte-Carlo Bayes-risk sweep -- tau-uniform noise variant.

Same 2-D (p, batch_size) sweep as ../sweeps/rb_vs_mc_sweep.py, but the noise is drawn
from a *uniform prior on the precision* rather than uniform on the std:

    beta  ~ N(0, sigma0^2 I_p)
    tau = 1/sigma^2 ~ U(tau_low, tau_high)   ->   sigma = 1/sqrt(tau)
    Y     = X beta + eps,   eps ~ N(0, sigma^2 I)

With tau ~ U(0.01, 1) this gives sigma in (1, 10) -- it cuts off the very-informative
low-noise datasets (sigma < 1) that sigma ~ U(0.1, 10) included, concentrating mass on
the noisier regime. Matches ../4_rb_vs_mc_longrun_tau_uniform.ipynb.

Everything else is identical to the sigma-uniform sweep: corrected RB methodology
(Cholesky solve + tr(Sigma_n), .sum() over beta dims), seeded fixed design, identical-init
MC/RB nets, multi-seed averaging, common held-out test set, gradient clipping.

Outputs (written next to this file):
  * rb_vs_mc_sweep_tau_results.npz  -- raw per-(p,bs) seed curves
  * rb_vs_mc_sweep_tau_curves.png   -- convergence curves
  * rb_vs_mc_sweep_tau_summary.png  -- final-loss and RB-advantage heatmaps

Usage:
  python rb_vs_mc_sweep_tau.py                 # full sweep (slow on CPU)
  python rb_vs_mc_sweep_tau.py --quick         # tiny config for a smoke test
  python rb_vs_mc_sweep_tau.py --tau-low 0.01 --tau-high 1.0
  python rb_vs_mc_sweep_tau.py --noise-prior sigma_uniform   # fall back to U(sigma)
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


def build_design(p: int, n_obs: int, device: torch.device,
                 x_seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    """Fixed (seeded) design matrix X = [1, N(0,1), ...] and cached XtX.

    Seeded by ``x_seed`` so the design is identical across batch sizes and training
    seeds for a given (p, x_seed); a different ``x_seed`` yields an independent
    design-matrix realisation (used for the robustness check in the appendix).
    """
    torch.manual_seed(x_seed)
    x_cov = torch.randn(n_obs, p - 1, device=device)
    X = torch.cat([torch.ones(n_obs, 1, device=device), x_cov], dim=1)  # (n_obs, p)
    XtX = X.T @ X
    return X, XtX


def make_sampler(X, n_obs, cfg, device):
    p = X.shape[1]

    def sample_batch(batch_size: int):
        if cfg.noise_prior == "tau_uniform":
            tau = torch.rand(batch_size, device=device) * (cfg.tau_high - cfg.tau_low) + cfg.tau_low
            sigma = 1.0 / torch.sqrt(tau)
        else:  # sigma_uniform
            sigma = torch.rand(batch_size, device=device) * (cfg.sigma_high - cfg.sigma_low) + cfg.sigma_low
        beta = torch.randn(batch_size, p, device=device) * cfg.sigma0
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
def run_cell(p, bs, cfg, device, x_seed=42):
    X, XtX = build_design(p, cfg.n_obs, device, x_seed)
    Sigma0_inv = torch.eye(p, device=device) / (cfg.sigma0 ** 2)

    sample_batch = make_sampler(X, cfg.n_obs, cfg, device)
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
    fig.suptitle(r'RB vs MC convergence, $\tau\sim U$ noise (mean $\pm$ std over seeds)', y=1.001)
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
    fig.suptitle(r'$\tau = 1/\sigma^2 \sim U(\tau_{low}, \tau_{high})$ noise variant', y=1.02)
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
    p.add_argument('--x-seeds', type=int, nargs='+', default=[42, 43], dest='x_seeds',
                   help='design-matrix seeds; one full sweep per seed (X realisations). '
                        'The first is the main-text X; the rest go to the appendix.')
    p.add_argument('--seeds', type=int, default=10, dest='n_seeds')
    p.add_argument('--epochs', type=int, default=200, dest='n_epochs')
    p.add_argument('--samples-per-epoch', type=int, default=2048)
    p.add_argument('--n-obs', type=int, default=50)
    p.add_argument('--sigma0', type=float, default=5.0)
    p.add_argument('--noise-prior', choices=['tau_uniform', 'sigma_uniform'], default='tau_uniform',
                   help='tau_uniform: tau=1/sigma^2 ~ U(tau_low,tau_high); sigma_uniform: sigma ~ U(sigma_low,sigma_high)')
    p.add_argument('--tau-low', type=float, default=0.01)
    p.add_argument('--tau-high', type=float, default=1.0)
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
    print(f"device: {device}", flush=True)
    if cfg.noise_prior == "tau_uniform":
        s_hi, s_lo = 1.0 / np.sqrt(cfg.tau_low), 1.0 / np.sqrt(cfg.tau_high)
        noise_desc = f"tau ~ U({cfg.tau_low}, {cfg.tau_high})  => sigma in ({s_lo:.2f}, {s_hi:.2f})"
    else:
        noise_desc = f"sigma ~ U({cfg.sigma_low}, {cfg.sigma_high})"
    print(f"noise: {noise_desc}", flush=True)
    print(f"sweep: p={cfg.p_grid}  batch_sizes={cfg.batch_sizes}  "
          f"seeds={cfg.n_seeds}  epochs={cfg.n_epochs}  samples/epoch={cfg.samples_per_epoch}", flush=True)

    print(f"x-seeds (design realisations): {cfg.x_seeds}", flush=True)

    all_results = {}
    n_cells = len(cfg.p_grid) * len(cfg.batch_sizes)
    for xi, x_seed in enumerate(cfg.x_seeds):
        print(f"\n===== X realisation {xi}  (design seed {x_seed}) =====", flush=True)
        print(f"{'cell':>7} {'p':>4} {'bs':>5} {'n_batch':>8} {'time(s)':>8} "
              f"{'MC final':>10} {'RB final':>10} {'ratio':>7} {'max ratio':>10}", flush=True)
        print('-' * 84, flush=True)
        k = 0
        for p in cfg.p_grid:
            for bs in cfg.batch_sizes:
                k += 1
                t0 = time.time()
                r = run_cell(p, bs, cfg, device, x_seed)
                dt = time.time() - t0
                all_results[(xi, p, bs)] = r
                mc_f = r['mc'][:, -1].mean()
                rb_f = r['rb'][:, -1].mean()
                max_ratio = (r['mc'].mean(0) / r['rb'].mean(0)).max()
                print(f"{k:>3d}/{n_cells:<3d} {p:>4d} {bs:>5d} {r['n_batches']:>8d} {dt:>8.1f} "
                      f"{mc_f:>10.4f} {rb_f:>10.4f} {mc_f / rb_f:>6.2f}x {max_ratio:>9.2f}x", flush=True)

    # ----- save raw curves (X-prefixed; X0 also under legacy keys for back-compat) -----
    flat = {}
    for (xi, p, bs), r in all_results.items():
        for key in ('mc', 'rb', 'mc_b', 'rb_b'):
            flat[f'X{xi}_p{p}_bs{bs}_{key}'] = r[key]
            if xi == 0:
                flat[f'p{p}_bs{bs}_{key}'] = r[key]
    npz_path = HERE / 'rb_vs_mc_sweep_tau_results.npz'
    np.savez(
        npz_path,
        p_grid=np.array(cfg.p_grid),
        batch_sizes=np.array(cfg.batch_sizes),
        x_seeds=np.array(cfg.x_seeds),
        n_epochs=np.array(cfg.n_epochs),
        n_seeds=np.array(cfg.n_seeds),
        n_obs=np.array(cfg.n_obs),
        sigma0=np.array(cfg.sigma0),
        test_size=np.array(cfg.test_size),
        noise_prior=np.array(cfg.noise_prior),
        tau_low=np.array(cfg.tau_low),
        tau_high=np.array(cfg.tau_high),
        **flat,
    )
    print(f"\nSaved {npz_path.name}", flush=True)

    # ----- plots: per-X summary heatmaps. The final per-X curves figures are
    #       produced by replot_curves_floor.py once the Bayes floors exist. -----
    if not cfg.no_plots:
        for xi, _ in enumerate(cfg.x_seeds):
            sub = {(p, bs): all_results[(xi, p, bs)]
                   for p in cfg.p_grid for bs in cfg.batch_sizes}
            try:
                plot_summary(sub, cfg.p_grid, cfg.batch_sizes,
                             HERE / f'rb_vs_mc_sweep_tau_summary_X{xi}.png')
            except Exception as exc:  # plotting is non-essential; don't lose the .npz
                print(f"[warn] plotting (X{xi}) failed: {exc}", flush=True)
        print("Saved per-X summary heatmaps", flush=True)


if __name__ == '__main__':
    main()
