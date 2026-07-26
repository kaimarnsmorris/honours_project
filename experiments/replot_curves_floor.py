"""Replot the RB-vs-MC training-curve grid (tau-uniform) WITH the Bayes-floor line,
for a chosen design-matrix realisation X.

Reuses the saved sweep results (rb_vs_mc_sweep_tau_results.npz) -- no retraining --
and overlays a dark-green dashed line at the Bayes-optimal floor L*(p) on every panel
of row p. Floors are read from floors.json (written by floors_and_tables.py); if that
file is absent it falls back to the built-in X0 values. Each p-row shares one y-axis
(sharey='row'), clipped to the leftmost (smallest-batch) panel's range so the large
first-epoch errors of the high-batch panels do not blow up the row scale.

Usage:
  python replot_curves_floor.py                                  # X0 -> rb_tau_curves.png
  python replot_curves_floor.py --x-index 1 --out ../../tex/figures/rb_tau_curves_X2.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Larger, more legible fonts -- the 5x4 grid is dense at \textwidth, so the
# default sizes shrink to nothing.  Grid layout is unchanged; only text scales.
plt.rcParams.update({
    "axes.titlesize": 16,
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "figure.titlesize": 17,
})

HERE = Path(__file__).resolve().parent
DEFAULT_NPZ = HERE / "rb_vs_mc_sweep_tau_results.npz"
DEFAULT_OUT = HERE.parents[1] / "tex" / "figures" / "rb_tau_curves.png"
DEFAULT_FLOORS = HERE / "floors.json"

# Fallback MSE-based Bayes floors (X0) if floors.json is unavailable.
L_STAR_FALLBACK = {2: 0.0774, 5: 0.0952, 10: 0.1191, 20: 0.1430, 30: 0.2285}


def curve(d, xi, p, bs, k):
    """Fetch a curve array, preferring the X-prefixed key and falling back to the
    legacy (X0) key for back-compat with older .npz files."""
    xkey = f"X{xi}_p{p}_bs{bs}_{k}"
    if xkey in d:
        return d[xkey]
    if xi == 0:
        return d[f"p{p}_bs{bs}_{k}"]
    raise KeyError(xkey)


def load_floors(path, xi, p_grid):
    if Path(path).exists():
        with open(path) as fh:
            data = json.load(fh)
        if str(xi) in data:
            return {int(p): float(v) for p, v in data[str(xi)].items()}
    return {p: L_STAR_FALLBACK[p] for p in p_grid if p in L_STAR_FALLBACK}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--x-index", type=int, default=0, dest="xi")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--floors", default=str(DEFAULT_FLOORS))
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    p_grid = [int(x) for x in d["p_grid"]]
    batch_sizes = [int(x) for x in d["batch_sizes"]]
    n_epochs = int(d["n_epochs"])
    epochs = np.arange(1, n_epochs + 1)
    xi = args.xi
    L_star = load_floors(args.floors, xi, p_grid)

    nrows, ncols = len(p_grid), len(batch_sizes)
    # sharey='row': the four batch panels in each p-row share one y-axis and align.
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows),
                             squeeze=False, sharex=True, sharey="row")
    for i, p in enumerate(p_grid):
        for j, bs in enumerate(batch_sizes):
            ax = axes[i][j]
            mc = curve(d, xi, p, bs, "mc")
            rb = curve(d, xi, p, bs, "rb")
            mc_m, mc_s = mc.mean(0), mc.std(0)
            rb_m, rb_s = rb.mean(0), rb.std(0)
            ax.plot(epochs, mc_m, color="tab:blue", label="MC")
            ax.fill_between(epochs, mc_m - mc_s, mc_m + mc_s, alpha=0.2, color="tab:blue")
            ax.plot(epochs, rb_m, color="tab:orange", label="RB")
            ax.fill_between(epochs, rb_m - rb_s, rb_m + rb_s, alpha=0.2, color="tab:orange")
            ax.axhline(L_star[p], color="darkgreen", ls="--", lw=1.2, zorder=1)
            ax.set_yscale("log")
            ax.grid(True, which="both", ls="--", linewidth=0.5)
            if i == 0:
                ax.set_title(f"batch_size = {bs}")
            if j == 0:
                ax.set_ylabel(f"p = {p}\nTest MSE (log)")
            if i == nrows - 1:
                ax.set_xlabel("Epoch")

        # Clip each row's y-axis to the leftmost (smallest-batch) panel's range:
        # the high-batch panels' huge first-epoch errors would otherwise blow up
        # the shared row scale and squash the leftmost curves.
        lb = batch_sizes[0]
        lmc, lrb = curve(d, xi, p, lb, "mc"), curve(d, xi, p, lb, "rb")
        top = max((lmc.mean(0) + lmc.std(0)).max(), (lrb.mean(0) + lrb.std(0)).max())
        mins = [L_star[p]] + [curve(d, xi, p, bs, k).mean(0).min()
                              for bs in batch_sizes for k in ("mc", "rb")]
        axes[i][0].set_ylim(max(min(mins), 1e-4) * 0.85, top * 1.08)

    handles = [Line2D([0], [0], color="tab:blue", lw=2, label="MC"),
               Line2D([0], [0], color="tab:orange", lw=2, label="RB"),
               Line2D([0], [0], color="darkgreen", lw=1.2, ls="--",
                      label=r"Bayes floor $L^*$")]
    axes[0][0].legend(handles=handles, fontsize=12, loc="upper right")
    fig.suptitle("RB vs MC convergence", y=1.001)
    plt.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {args.out}  (X{xi})")


if __name__ == "__main__":
    main()
