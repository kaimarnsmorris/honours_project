#!/usr/bin/env python3
# ============================================================================
#  extra_figures_a2.py
#
#  Add an alpha=2.0 line to binned_mse.png.  Reuses the cached baseline
#  (alpha=1) and aggressive-boundary (alpha=0.3) nets from extra_out/ so those
#  two curves stay byte-identical to the published figure; only the new
#  centre/high-sigma proposal (alpha=2.0, Beta(2,2) on rho and Beta(2,1) on
#  sigma) is trained here.  Mirrors extra_figures.py settings exactly
#  (A=2, T=100, 150 epochs, 1 seed, n_test=8000).
# ============================================================================

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run                       # canonical AR(1) sweep module (same dir)
from extra_figures import make_args, predict, binned_1d


def main():
    args = make_args()
    os.makedirs(args.outdir, exist_ok=True)
    dev = "cpu"

    test_pack, (rho, sigma), _ = run.make_test_set(args, dev)
    x_t, rho_t, sig_t = test_pack

    # ---- reuse cached baseline (a=1) and a=0.3 nets (no retraining) ----
    net_base = run.AR1Estimator(args.hidden, args.mlp, args.A).to(dev)
    net_is = run.AR1Estimator(args.hidden, args.mlp, args.A).to(dev)
    net_base.load_state_dict(torch.load(os.path.join(args.outdir, "net_base.pt"),
                                        map_location=dev))
    net_is.load_state_dict(torch.load(os.path.join(args.outdir, "net_is.pt"),
                                      map_location=dev))
    print("loaded cached net_base.pt (a=1) and net_is.pt (a=0.3)", flush=True)

    # ---- train the new alpha=2.0 net (centre-rho / high-sigma proposal) ----
    print("training IS (alpha=2.0) ...", flush=True)
    net_a2, _, _ = run.train_one(2.0, args, dev, 0, test_pack)
    torch.save(net_a2.state_dict(), os.path.join(args.outdir, "net_a2.pt"))

    rb, sb = predict(net_base, x_t)
    ri, si = predict(net_is, x_t)
    ra, sa = predict(net_a2, x_t)
    se_rho_b, se_sig_b = (rb - rho) ** 2, (sb - sigma) ** 2
    se_rho_i, se_sig_i = (ri - rho) ** 2, (si - sigma) ** 2
    se_rho_a, se_sig_a = (ra - rho) ** 2, (sa - sigma) ** 2

    # ---------- binned_mse.png (3 lines) ----------
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    series = (
        ("baseline",          se_rho_b, se_sig_b, "C0"),
        (r"IS $\alpha=0.3$",  se_rho_i, se_sig_i, "C3"),
        (r"IS $\alpha=2.0$",  se_rho_a, se_sig_a, "C2"),
    )
    for lbl, sr, ss, col in series:
        c, m = binned_1d(sr, rho, -1, 1);        axes[0, 0].plot(c, m, "o-", color=col, label=lbl)
        c, m = binned_1d(ss, rho, -1, 1);        axes[0, 1].plot(c, m, "o-", color=col, label=lbl)
        c, m = binned_1d(sr, sigma, 0, args.A);  axes[1, 0].plot(c, m, "o-", color=col, label=lbl)
        c, m = binned_1d(ss, sigma, 0, args.A);  axes[1, 1].plot(c, m, "o-", color=col, label=lbl)
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

    # ---- console summary (for the chapter / table) ----
    bnd = np.abs(rho) > 0.8
    lo_s = sigma < 0.25 * args.A
    hi_s = sigma > 0.75 * args.A
    for lbl, sr, ss in (("baseline a=1", se_rho_b, se_sig_b),
                        ("IS a=0.3", se_rho_i, se_sig_i),
                        ("IS a=2.0", se_rho_a, se_sig_a)):
        tot = sr + ss
        print(f"\n{lbl}: overall MSE={tot.mean():.4f}  "
              f"MSE(rho)={sr.mean():.4f}  MSE(sigma)={ss.mean():.4f}")
        print(f"   regions  all={tot.mean():.4f}  |rho|>0.8={tot[bnd].mean():.4f}  "
              f"low_sig={tot[lo_s].mean():.4f}  high_sig={tot[hi_s].mean():.4f}")


if __name__ == "__main__":
    main()
