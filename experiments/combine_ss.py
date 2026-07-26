#!/usr/bin/env python3
# Combine the per-alpha small-sigma runs (_ss_*) into the three Chapter 5
# figures: training curves, ESS vs alpha, and the per-region MSE breakdown.
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHAS = [1.0, 0.5, 0.3, 0.1]
FIGDIR = "../../tex/figures/"
REGIONS = ["all", "high_|rho|", "low_sigma", "high_sigma"]
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ALPHAS)))

# ---- training curves ----
plt.figure(figsize=(7, 4.5))
for a, c in zip(ALPHAS, colors):
    d = np.load(f"_ss_{a}/curves.npz")
    curve = d[d.files[0]]
    plt.plot(range(1, len(curve) + 1), curve, color=c, lw=2, label=fr"$\alpha={a}$")
plt.xlabel("Epoch"); plt.ylabel("Validation MSE"); plt.yscale("log")
plt.legend(); plt.title("AR(1) importance-sampling sweep: training curves")
plt.tight_layout(); plt.savefig(FIGDIR + "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- read per-alpha region table ----
tab = {}   # tab[alpha][region] = (total, rho, sigma); ess[alpha]
ess = {}
for a in ALPHAS:
    tab[a] = {}
    with open(f"_ss_{a}/sweep_results.csv") as f:
        for row in csv.DictReader(f):
            tab[a][row["region"]] = (float(row["mse_total"]),
                                     float(row["mse_rho"]),
                                     float(row["mse_sigma"]))
            ess[a] = float(row["mean_ess"])

# ---- ESS vs alpha ----
plt.figure(figsize=(6, 4))
plt.plot(ALPHAS, [ess[a] for a in ALPHAS], "o-")
plt.axhline(256, ls="--", c="grey", label="batch size (256)")
plt.xlabel(r"proposal $\alpha$"); plt.ylabel("mean ESS")
plt.gca().invert_xaxis(); plt.legend()
plt.title("Effective sample size vs proposal strength")
plt.tight_layout(); plt.savefig(FIGDIR + "ess_vs_alpha.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- region breakdown (total MSE) ----
x = np.arange(len(REGIONS)); width = 0.8 / len(ALPHAS)
plt.figure(figsize=(8, 4.5))
for j, a in enumerate(ALPHAS):
    vals = [tab[a][r][0] for r in REGIONS]
    plt.bar(x + j * width, vals, width, color=colors[j], label=fr"$\alpha={a}$")
plt.xticks(x + 0.4 - width / 2, REGIONS)
plt.ylabel("MSE"); plt.legend()
plt.title("Per-region MSE by proposal strength")
plt.tight_layout(); plt.savefig(FIGDIR + "region_breakdown.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- print the table for the chapter ----
print(f"{'alpha':>6} {'all':>8} {'high|rho|':>9} {'low_sig':>8} {'high_sig':>9} {'ESS':>7}")
for a in ALPHAS:
    print(f"{a:>6} {tab[a]['all'][0]:>8.4f} {tab[a]['high_|rho|'][0]:>9.4f} "
          f"{tab[a]['low_sigma'][0]:>8.4f} {tab[a]['high_sigma'][0]:>9.4f} {ess[a]:>7.1f}")
print("\nrho-component (all):  ",
      "  ".join(f"a={a}:{tab[a]['all'][1]:.4f}" for a in ALPHAS))
print("sigma-component (all):",
      "  ".join(f"a={a}:{tab[a]['all'][2]:.4f}" for a in ALPHAS))
print("wrote training_curves.png, ess_vs_alpha.png, region_breakdown.png")
