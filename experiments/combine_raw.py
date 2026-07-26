#!/usr/bin/env python3
# Build the three Chapter 5 sweep figures from a single multi-alpha run
# (ar1_raw_out/, produced by run.py --alphas 1.0 0.5 0.3 0.1 ...).
# Mirrors combine_ss.py's styling but reads one directory instead of _ss_*.
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = "ar1_raw_out"
ALPHAS = [1.0, 0.5, 0.3, 0.1]
FIGDIR = "../../tex/figures/"
REGIONS = ["all", "high_|rho|", "low_sigma", "high_sigma"]
BATCH = 256
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ALPHAS)))

# ---- training curves ----
d = np.load(f"{SRC}/curves.npz")
plt.figure(figsize=(7, 4.5))
for a, c in zip(ALPHAS, colors):
    curve = d[f"alpha_{a}"]
    plt.plot(range(1, len(curve) + 1), curve, color=c, lw=2, label=fr"$\alpha={a}$")
plt.xlabel("Epoch"); plt.ylabel("Validation MSE"); plt.yscale("log")
plt.legend(); plt.title("AR(1) importance-sampling sweep: training curves")
plt.tight_layout(); plt.savefig(FIGDIR + "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- read per-alpha region table ----
tab = {a: {} for a in ALPHAS}
ess = {}
with open(f"{SRC}/sweep_results.csv") as f:
    for row in csv.DictReader(f):
        a = float(row["alpha"])
        tab[a][row["region"]] = (float(row["mse_total"]),
                                 float(row["mse_rho"]),
                                 float(row["mse_sigma"]))
        ess[a] = float(row["mean_ess"])

# ---- ESS vs alpha ----
plt.figure(figsize=(6, 4))
plt.plot(ALPHAS, [ess[a] for a in ALPHAS], "o-")
plt.axhline(BATCH, ls="--", c="grey", label=f"batch size ({BATCH})")
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

# ---- print the table for the chapter (total, and rho/sigma components) ----
hdr = f"{'alpha':>6} {'all':>8} {'high|rho|':>9} {'low_sig':>8} {'high_sig':>9} {'ESS':>7}"
print(hdr)
for a in ALPHAS:
    print(f"{a:>6} {tab[a]['all'][0]:>8.4f} {tab[a]['high_|rho|'][0]:>9.4f} "
          f"{tab[a]['low_sigma'][0]:>8.4f} {tab[a]['high_sigma'][0]:>9.4f} {ess[a]:>7.1f}")
print("\nrho-component:  ", "  ".join(f"a={a}:{tab[a]['all'][1]:.4f}" for a in ALPHAS))
print("sigma-component:", "  ".join(f"a={a}:{tab[a]['all'][2]:.4f}" for a in ALPHAS))
print("\nper-region rho/sigma split at alpha=1.0:")
for r in REGIONS:
    print(f"  {r:>11}: total {tab[1.0][r][0]:.4f}  rho {tab[1.0][r][1]:.4f}  sigma {tab[1.0][r][2]:.4f}")
print("\nwrote training_curves.png, ess_vs_alpha.png, region_breakdown.png")
