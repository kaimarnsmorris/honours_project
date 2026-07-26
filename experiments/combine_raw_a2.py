#!/usr/bin/env python3
# Rebuild the three Chapter 5 sweep figures with alpha=2.0 added.
# Merges the existing four-alpha run (ar1_raw_out/, from run.py --alphas
# 1.0 0.5 0.3 0.1) with the new single-alpha run (ar1_a2_out/, --alphas 2.0).
# Colours/ordering match plot_proposals.py (viridis over the 5 alphas,
# darkest = 2.0).  Non-destructive: reads both dirs, writes the figures.
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC_MAIN = "ar1_raw_out"     # alphas 1.0 0.5 0.3 0.1
SRC_A2 = "ar1_a2_out"        # alpha 2.0
ALPHAS = [2.0, 1.0, 0.5, 0.3, 0.1]
FIGDIR = "../../tex/figures/"
REGIONS = ["all", "high_|rho|", "low_sigma", "high_sigma"]
BATCH = 256
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(ALPHAS)))

# ---- load seed-averaged validation curves from both runs ----
npz = [np.load(f"{SRC_MAIN}/curves.npz"), np.load(f"{SRC_A2}/curves.npz")]
curves = {}
for a in ALPHAS:
    key = f"alpha_{a}"
    for d in npz:
        if key in d:
            curves[a] = d[key]
            break
    if a not in curves:
        raise SystemExit(f"missing curve for alpha={a}")

# ---- training curves ----
plt.figure(figsize=(7, 4.5))
for a, c in zip(ALPHAS, colors):
    curve = curves[a]
    plt.plot(range(1, len(curve) + 1), curve, color=c, lw=2, label=fr"$\alpha={a}$")
plt.xlabel("Epoch"); plt.ylabel("Validation MSE"); plt.yscale("log")
plt.legend(); plt.title("AR(1) importance-sampling sweep: training curves")
plt.tight_layout(); plt.savefig(FIGDIR + "training_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# ---- read per-alpha region table + ESS from both csvs ----
tab = {a: {} for a in ALPHAS}
ess = {}
for src in (SRC_MAIN, SRC_A2):
    with open(f"{src}/sweep_results.csv") as f:
        for row in csv.DictReader(f):
            a = float(row["alpha"])
            if a not in ALPHAS:
                continue
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

# ---- print the chapter table (LaTeX-ready rows) ----
hdr = f"{'alpha':>6} {'all':>8} {'high|rho|':>9} {'low_sig':>8} {'high_sig':>9} {'ESS':>7}"
print(hdr)
for a in ALPHAS:
    print(f"{a:>6} {tab[a]['all'][0]:>8.4f} {tab[a]['high_|rho|'][0]:>9.4f} "
          f"{tab[a]['low_sigma'][0]:>8.4f} {tab[a]['high_sigma'][0]:>9.4f} {ess[a]:>7.1f}")
print("\nLaTeX rows (all | high|rho| | low_sig | high_sig | ESS):")
for a in ALPHAS:
    t = tab[a]
    print(fr"    ${a}$ & ${t['all'][0]:.4f}$ & ${t['high_|rho|'][0]:.4f}$ & "
          fr"${t['low_sigma'][0]:.4f}$ & ${t['high_sigma'][0]:.4f}$ & ${ess[a]:.1f}$ \\")
print("\nrho-component:  ", "  ".join(f"a={a}:{tab[a]['all'][1]:.4f}" for a in ALPHAS))
print("sigma-component:", "  ".join(f"a={a}:{tab[a]['all'][2]:.4f}" for a in ALPHAS))
print("\nwrote training_curves.png, ess_vs_alpha.png, region_breakdown.png")
