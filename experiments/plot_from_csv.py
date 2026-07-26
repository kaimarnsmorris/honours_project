#!/usr/bin/env python3
# ============================================================================
#  plot_from_csv.py
#
#  Regenerate the Chapter 5 figures from an existing sweep_results.csv (and,
#  optionally, gru_vs_gibbs.txt) WITHOUT retraining.  No PyTorch needed --
#  just numpy + matplotlib.  Runs in seconds.
#
#  The training-curve figure cannot be reproduced from the CSV (the CSV stores
#  only the final per-alpha MSE, not the per-epoch trajectory).  This script
#  therefore produces the two figures that the CSV fully determines:
#      ess_vs_alpha.png        ESS as a function of alpha
#      region_breakdown.png    per-region MSE per alpha
#  and, if gru_vs_gibbs.txt is present, simply echoes it.
#
#  Usage:
#    python plot_from_csv.py                       # reads ./ar1_is_out/
#    python plot_from_csv.py --csv path/to/sweep_results.csv --outdir figs
# ============================================================================
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_csv(path):
    """Return region_table[alpha][region] = (mse_total, mse_rho, mse_sigma)
    and ess_by_alpha[alpha]."""
    region_table, ess_by_alpha = {}, {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            a = float(row["alpha"])
            region_table.setdefault(a, {})[row["region"]] = (
                float(row["mse_total"]),
                float(row["mse_rho"]),
                float(row["mse_sigma"]),
            )
            ess_by_alpha[a] = float(row["mean_ess"])
    return region_table, ess_by_alpha


def plot_ess(ess_by_alpha, outpath, batch):
    alphas = sorted(ess_by_alpha.keys(), reverse=True)
    vals = [ess_by_alpha[a] for a in alphas]
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, vals, "o-")
    if batch is not None:
        plt.axhline(batch, ls="--", c="grey", label=f"batch size ({batch})")
        plt.legend()
    plt.xlabel(r"proposal $\alpha$")
    plt.ylabel("mean ESS")
    plt.gca().invert_xaxis()
    plt.title("Effective sample size vs proposal strength")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_region_breakdown(region_table, outpath):
    alphas = sorted(region_table.keys(), reverse=True)
    regions = ["all", "high_|rho|", "low_sigma", "high_sigma"]
    regions = [r for r in regions if r in region_table[alphas[0]]]
    x = np.arange(len(regions))
    width = 0.8 / len(alphas)
    plt.figure(figsize=(8, 4.5))
    for j, a in enumerate(alphas):
        vals = [region_table[a][r][0] for r in regions]
        plt.bar(x + j * width, vals, width, label=f"$\\alpha={a}$")
    plt.xticks(x + 0.4 - width / 2, regions)
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Per-region MSE by proposal strength")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="ar1_is_out/sweep_results.csv")
    p.add_argument("--outdir", default=None,
                   help="defaults to the CSV's directory")
    p.add_argument("--batch", type=int, default=256,
                   help="batch size, for the dashed line on the ESS plot")
    args = p.parse_args()

    if not os.path.exists(args.csv):
        raise SystemExit(f"CSV not found: {args.csv}")
    outdir = args.outdir or os.path.dirname(args.csv) or "."
    os.makedirs(outdir, exist_ok=True)

    region_table, ess_by_alpha = load_csv(args.csv)

    ess_path = os.path.join(outdir, "ess_vs_alpha.png")
    reg_path = os.path.join(outdir, "region_breakdown.png")
    plot_ess(ess_by_alpha, ess_path, args.batch)
    plot_region_breakdown(region_table, reg_path)
    print(f"wrote {ess_path}")
    print(f"wrote {reg_path}")
    print("(training_curves.png needs per-epoch data, not in the CSV -- "
          "re-run the main script for that one.)")

    gibbs_txt = os.path.join(os.path.dirname(args.csv) or ".",
                             "gru_vs_gibbs.txt")
    if os.path.exists(gibbs_txt):
        print("\n--- gru_vs_gibbs.txt ---")
        print(open(gibbs_txt).read())


if __name__ == "__main__":
    main()