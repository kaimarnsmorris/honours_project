"""Overlay RB/MC convergence curves across batch sizes vs TOTAL SAMPLES SEEN.

The per-(p, batch_size) sweeps hold samples_per_epoch fixed (2048), so epoch e
corresponds to e * samples_per_epoch samples for *every* batch size. Re-plotting on
a samples-seen x-axis therefore puts all batch sizes on a common, directly comparable
scale: at equal data budget, which batch size (and which estimator) has converged
furthest? RB's per-sample variance reduction should show up most for small batches
(many noisy steps) and wash out for large batches (each step already self-averaged).

Unlike the existing *_curves.png (one subplot per (p, bs)), this overlays the batch
sizes on shared axes, one subplot per p. MC = dashed, RB = solid, colour = batch size.

Reads a sweep's .npz and writes a PNG. No retraining.

Usage:
  python plot_curves_vs_samples.py --npz sweeps/rb_vs_mc_sweep_results.npz \
      --out sweeps/rb_vs_mc_sweep_vs_samples.png --title "sigma ~ U(0.1, 10)"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def load(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    p_grid = [int(x) for x in d["p_grid"]]
    batch_sizes = [int(x) for x in d["batch_sizes"]]
    n_epochs = int(d["n_epochs"])
    curves = {}
    for p in p_grid:
        for bs in batch_sizes:
            curves[(p, bs)] = {k: d[f"p{p}_bs{bs}_{k}"] for k in ("mc", "rb")}
    return p_grid, batch_sizes, n_epochs, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    ap.add_argument("--samples-per-epoch", type=int, default=2048,
                    help="must match the sweep run (default 2048)")
    ap.add_argument("--batch-sizes", type=int, nargs="+", default=None,
                    help="restrict to a subset of batch sizes (default: all in the npz)")
    ap.add_argument("--ncols", type=int, default=None,
                    help="panels per row (default: auto, ~2 rows when >3 panels)")
    args = ap.parse_args()

    p_grid, batch_sizes, n_epochs, curves = load(args.npz)
    if args.batch_sizes:
        batch_sizes = [bs for bs in args.batch_sizes if bs in batch_sizes]

    samples = np.arange(1, n_epochs + 1) * args.samples_per_epoch
    cmap = plt.get_cmap("viridis")
    colors = {bs: cmap(i / max(1, len(batch_sizes) - 1)) for i, bs in enumerate(batch_sizes)}

    n = len(p_grid)
    # Wrap into ~2 rows when there are more than three panels: a single row of
    # five squashes each panel and shrinks the axis labels.
    ncols = args.ncols or (int(np.ceil(n / 2)) if n > 3 else n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 4.1 * nrows),
                             squeeze=False)
    axflat = axes.flatten()

    for idx, p in enumerate(p_grid):
        ax = axflat[idx]
        for bs in batch_sizes:
            r = curves[(p, bs)]
            mc_m = r["mc"].mean(0)
            rb_m = r["rb"].mean(0)
            ax.plot(samples, mc_m, ls="--", color=colors[bs], alpha=0.9, lw=1.8)
            ax.plot(samples, rb_m, ls="-", color=colors[bs], alpha=0.9, lw=1.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"p = {p}", fontsize=15)
        ax.tick_params(axis="both", which="major", labelsize=11)
        ax.grid(True, which="both", ls="--", linewidth=0.4)

    # Legend handles: colour = batch size, linestyle = estimator.
    bs_handles = [Line2D([0], [0], color=colors[bs], lw=2.5, label=f"batch size = {bs}")
                  for bs in batch_sizes]
    est_handles = [Line2D([0], [0], color="0.3", lw=2.5, ls="-", label="RB"),
                   Line2D([0], [0], color="0.3", lw=2.5, ls="--", label="MC")]
    handles = bs_handles + est_handles

    # Park the legend in any empty grid cell; otherwise drop it below the figure.
    empty = list(range(n, nrows * ncols))
    if empty:
        leg_ax = axflat[empty[0]]
        leg_ax.axis("off")
        leg_ax.legend(handles=handles, loc="center", fontsize=13, frameon=False)
        for idx in empty[1:]:
            axflat[idx].set_visible(False)
    else:
        fig.legend(handles=handles, loc="lower center", fontsize=12,
                   ncol=len(handles), bbox_to_anchor=(0.5, -0.02), frameon=False)

    fig.supxlabel("Total samples seen", fontsize=14)
    fig.supylabel("Test MSE (mean over seeds, log scale)", fontsize=14)

    suptitle = "RB (solid) vs MC (dashed) convergence vs samples seen"
    if args.title:
        suptitle += f"   —   {args.title}"
    fig.suptitle(suptitle, y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
