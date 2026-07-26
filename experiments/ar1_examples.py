#!/usr/bin/env python3
# Example AR(1) realisations across the (rho, sigma) parameter space, to
# motivate Chapter 5: the edges of the prior produce qualitatively different,
# harder-looking series, which is what tempts one to oversample them.
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

A = 2.0          # sigma ~ U(0, A), matching the experiment
T = 100
RHOS = [0.0, 0.5, 0.9, 0.99]
SIGMAS = [1.8, 1.0, 0.2]     # top row large noise, bottom row small noise


def simulate_ar1(rho, sigma, T, rng):
    x = np.empty(T)
    x[0] = rng.normal(0.0, sigma / np.sqrt(1.0 - rho ** 2))
    for t in range(1, T):
        x[t] = rho * x[t - 1] + rng.normal(0.0, sigma)
    return x


fig, axes = plt.subplots(len(SIGMAS), len(RHOS), figsize=(12.5, 6.0),
                         sharex=True)
rng = np.random.default_rng(1)
for i, s in enumerate(SIGMAS):
    for j, r in enumerate(RHOS):
        x = simulate_ar1(r, s, T, rng)
        ax = axes[i, j]
        ax.plot(x, lw=1.0, color="C0")
        ax.axhline(0, color="grey", lw=0.5, ls=":")
        ax.set_title(rf"$\rho={r}$, $\sigma={s}$", fontsize=10)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.set_ylabel("$x_t$", fontsize=9)
        if i == len(SIGMAS) - 1:
            ax.set_xlabel("$t$", fontsize=9)

fig.suptitle("Example AR(1) realisations across the prior "
             r"($T=100$, stationary start)", y=0.99, fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = "../../tex/figures/ar1_examples.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
