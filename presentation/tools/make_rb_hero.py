#!/usr/bin/env python3
"""Regenerate the RB-vs-MC hero figure: the p=10 row only, as a clean standalone
1x4 panel WITH its own Epoch x-axis, batch-size column titles and legend.

Reuses the saved sweep results from the honours repo (no retraining) -- the same
rb_vs_mc_sweep_tau_results.npz that produced tex/figures/rb_tau_curves.png. A single
row is drawn (i is always both first and last row) so every panel keeps its x labels,
unlike the old crop of the multi-row grid, which dropped them on interior rows.

Usage:  python tools/make_rb_hero.py            # -> figures/rb_hero.png
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

HERE = Path(__file__).resolve().parent
# honours repo lives beside the GitHub checkout: .../Documents/honours/final/sweeps2
DATA_DIR = HERE.parents[3] / "honours" / "final" / "sweeps2"
NPZ = DATA_DIR / "rb_vs_mc_sweep_tau_results.npz"
FLOORS = DATA_DIR / "floors.json"
OUT = HERE.parent / "figures" / "rb_hero.png"

P = 10
XI = 0  # design-matrix realisation X0 (matches the thesis rb_tau_curves.png)
# The sweep fixes samples/epoch across batch sizes (n_batches = SAMPLES_PER_EPOCH // bs),
# so an epoch = this many fresh simulated samples for every batch size (final run: 2048).
SAMPLES_PER_EPOCH = 2048
L_STAR_FALLBACK = {2: 0.0774, 5: 0.0952, 10: 0.1191, 20: 0.1430, 30: 0.2285}

plt.rcParams.update({
    "axes.titlesize": 17,
    "axes.labelsize": 16,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 14,
})


def curve(d, p, bs, k):
    """Fetch a curve array, preferring the X-prefixed key, falling back to legacy X0."""
    xkey = f"X{XI}_p{p}_bs{bs}_{k}"
    if xkey in d:
        return d[xkey]
    return d[f"p{p}_bs{bs}_{k}"]


def load_floor(p):
    if FLOORS.exists():
        data = json.load(open(FLOORS))
        if str(XI) in data and str(p) in data[str(XI)]:
            return float(data[str(XI)][str(p)])
    return L_STAR_FALLBACK[p]


def main():
    d = np.load(NPZ, allow_pickle=True)
    batch_sizes = [int(x) for x in d["batch_sizes"]]
    n_epochs = int(d["n_epochs"])
    epochs = np.arange(1, n_epochs + 1)
    samples = epochs * SAMPLES_PER_EPOCH   # cumulative simulated samples seen (same for every batch size)
    Lstar = load_floor(P)

    ncols = len(batch_sizes)
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4.2),
                             squeeze=False, sharex=True, sharey=True)
    for j, bs in enumerate(batch_sizes):
        ax = axes[0][j]
        mc, rb = curve(d, P, bs, "mc"), curve(d, P, bs, "rb")
        mc_m, mc_s = mc.mean(0), mc.std(0)
        rb_m, rb_s = rb.mean(0), rb.std(0)
        ax.plot(samples, mc_m, color="tab:blue", label="MC", lw=1.7)
        ax.fill_between(samples, mc_m - mc_s, mc_m + mc_s, alpha=0.2, color="tab:blue")
        ax.plot(samples, rb_m, color="tab:orange", label="RB", lw=1.7)
        ax.fill_between(samples, rb_m - rb_s, rb_m + rb_s, alpha=0.2, color="tab:orange")
        ax.axhline(Lstar, color="darkgreen", ls="--", lw=1.2, zorder=1)
        ax.set_yscale("log")
        ax.grid(True, which="both", ls="--", linewidth=0.5)
        ax.set_title(f"batch size = {bs}")
        ax.set_xlabel("# samples")
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.xaxis.set_major_formatter(FuncFormatter(
            lambda v, _: (f"{v/1000:.0f}k" if v > 0 else "0")))
    axes[0][0].set_ylabel(f"p = {P}\nTest MSE (log)")

    # Clip the shared y-axis to the leftmost (smallest-batch) panel's range so the
    # high-batch panels' large first-epoch errors do not squash the curves.
    lb = batch_sizes[0]
    lmc, lrb = curve(d, P, lb, "mc"), curve(d, P, lb, "rb")
    top = max((lmc.mean(0) + lmc.std(0)).max(), (lrb.mean(0) + lrb.std(0)).max())
    mins = [Lstar] + [curve(d, P, bs, k).mean(0).min()
                      for bs in batch_sizes for k in ("mc", "rb")]
    axes[0][0].set_ylim(max(min(mins), 1e-4) * 0.85, top * 1.08)

    handles = [Line2D([0], [0], color="tab:blue", lw=2, label="MC"),
               Line2D([0], [0], color="tab:orange", lw=2, label="RB"),
               Line2D([0], [0], color="darkgreen", lw=1.2, ls="--", label=r"Bayes floor $L^*$")]
    axes[0][0].legend(handles=handles, loc="upper right")
    plt.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT}  (p={P}, X{XI})")


if __name__ == "__main__":
    main()
