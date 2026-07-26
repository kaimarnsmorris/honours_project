#!/usr/bin/env python3
# ============================================================================
#  extra_figures.py
#
#  Spatial diagnostics for Chapter 5, reusing the canonical run.py model and
#  training so the numbers match Tables 5.1/5.2 (A=2, T=100, GRU, Adam).
#  Trains a baseline (alpha=1) and an aggressive IS (alpha=0.3) estimator,
#  then produces:
#
#    error_map.png   : MSE(rho) and MSE(sigma) over the (rho, sigma) plane
#                      for the baseline -- shows the error lives in sigma and
#                      the |rho|->1 boundary is NOT the hard region.
#    binned_mse.png  : MSE(rho), MSE(sigma) binned by rho and by sigma,
#                      baseline vs IS -- shows IS does not change the profile.
# ============================================================================

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run   # canonical AR(1) sweep module (same dir)


def make_args():
    return argparse.Namespace(
        A=2.0, T=100, epochs=150, steps_per_epoch=100, batch=256, lr=1e-3,
        hidden=64, mlp=64, n_test=8000, seeds=1, device="cpu",
        alphas=[1.0], gibbs_sweeps=0, gibbs_burnin=0, gibbs_test=0,
        quick=False, outdir="extra_out", sigma_proposal="small",
    )


def predict(net, x_t):
    net.eval()
    with torch.no_grad():
        rho_hat, sig_hat = net(x_t)
    return rho_hat.cpu().numpy(), sig_hat.cpu().numpy()


def grid_mse(true_vals, sq_err, rho, sigma, A, nb=15):
    """Mean of sq_err over a nb x nb (rho, sigma) grid."""
    re = np.linspace(-1, 1, nb + 1)
    se = np.linspace(0, A, nb + 1)
    ri = np.clip(np.digitize(rho, re) - 1, 0, nb - 1)
    si = np.clip(np.digitize(sigma, se) - 1, 0, nb - 1)
    grid = np.full((nb, nb), np.nan)
    for r in range(nb):
        for s in range(nb):
            m = (ri == r) & (si == s)
            if m.sum() > 2:
                grid[s, r] = sq_err[m].mean()
    return grid


def binned_1d(sq_err, by, lo, hi, nb=12):
    edges = np.linspace(lo, hi, nb + 1)
    cen = (edges[:-1] + edges[1:]) / 2
    out = np.full(nb, np.nan)
    for b in range(nb):
        m = (by >= edges[b]) & (by < edges[b + 1])
        if m.sum() > 0:
            out[b] = sq_err[m].mean()
    return cen, out


def main():
    args = make_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = "cpu"

    test_pack, (rho, sigma), _ = run.make_test_set(args, dev)
    x_t, rho_t, sig_t = test_pack

    print("training baseline (alpha=1) ...", flush=True)
    net_base, _, _ = run.train_one(1.0, args, dev, 0, test_pack)
    torch.save(net_base.state_dict(), os.path.join(args.outdir, "net_base.pt"))
    print("training IS (alpha=0.3) ...", flush=True)
    net_is, _, _ = run.train_one(0.3, args, dev, 0, test_pack)
    torch.save(net_is.state_dict(), os.path.join(args.outdir, "net_is.pt"))

    rb, sb = predict(net_base, x_t)
    ri, si = predict(net_is, x_t)
    se_rho_b, se_sig_b = (rb - rho) ** 2, (sb - sigma) ** 2
    se_rho_i, se_sig_i = (ri - rho) ** 2, (si - sigma) ** 2

    # ---------- error_map.png ----------
    g_rho = grid_mse(rho, se_rho_b, rho, sigma, args.A)
    g_sig = grid_mse(sigma, se_sig_b, rho, sigma, args.A)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, g, ttl in ((axes[0], g_rho, r"MSE$(\rho)$"),
                       (axes[1], g_sig, r"MSE$(\sigma)$")):
        im = ax.imshow(g, origin="lower", aspect="auto",
                       extent=[-1, 1, 0, args.A], cmap="viridis")
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"$\sigma$")
        ax.set_title(ttl + " across the prior (baseline)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "error_map.png"), dpi=150,
                bbox_inches="tight")
    print("wrote error_map.png", flush=True)

    # ---------- binned_mse.png ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    # row 0: binned by rho
    for lbl, sr, ss, col in (("baseline", se_rho_b, se_sig_b, "C0"),
                             (r"IS $\alpha=0.3$", se_rho_i, se_sig_i, "C3")):
        c, m = binned_1d(sr, rho, -1, 1); axes[0, 0].plot(c, m, "o-", color=col, label=lbl)
        c, m = binned_1d(ss, rho, -1, 1); axes[0, 1].plot(c, m, "o-", color=col, label=lbl)
        c, m = binned_1d(sr, sigma, 0, args.A); axes[1, 0].plot(c, m, "o-", color=col, label=lbl)
        c, m = binned_1d(ss, sigma, 0, args.A); axes[1, 1].plot(c, m, "o-", color=col, label=lbl)
    axes[0, 0].set(title=r"MSE$(\rho)$ binned by $\rho$", xlabel=r"$\rho$", ylabel="MSE")
    axes[0, 1].set(title=r"MSE$(\sigma)$ binned by $\rho$", xlabel=r"$\rho$", ylabel="MSE")
    axes[1, 0].set(title=r"MSE$(\rho)$ binned by $\sigma$", xlabel=r"$\sigma$", ylabel="MSE")
    axes[1, 1].set(title=r"MSE$(\sigma)$ binned by $\sigma$", xlabel=r"$\sigma$", ylabel="MSE")
    for ax in axes.flat:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "binned_mse.png"), dpi=150,
                bbox_inches="tight")
    print("wrote binned_mse.png", flush=True)

    # quick console summary
    print(f"\nbaseline overall: MSE(rho)={se_rho_b.mean():.4f}  "
          f"MSE(sigma)={se_sig_b.mean():.4f}")
    print(f"IS a=0.3 overall: MSE(rho)={se_rho_i.mean():.4f}  "
          f"MSE(sigma)={se_sig_i.mean():.4f}")
    # rho error at boundary vs interior (the key claim)
    bnd = np.abs(rho) > 0.8
    print(f"baseline MSE(rho): boundary |rho|>0.8 = {se_rho_b[bnd].mean():.5f}  "
          f"interior = {se_rho_b[~bnd].mean():.5f}")


if __name__ == "__main__":
    main()
