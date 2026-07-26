#!/usr/bin/env python3
# Figure 5.1 (breakdown): GRU vs Gibbs estimates against the truth, with the
# worst-MSE cases highlighted in red and one of them shown as an example
# realisation. Reuses the trained baseline net (extra_out/net_base.pt) and the
# AR(1) Gibbs sampler from run.py.
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import run

A, T, N = 2.0, 100, 400
SWEEPS, BURNIN = 1500, 500
WORST_FRAC = 0.05


def main():
    net = run.AR1Estimator(64, 64, A)
    net.load_state_dict(torch.load("extra_out/net_base.pt", map_location="cpu"))
    net.eval()

    rng = np.random.default_rng(123)
    rho = rng.uniform(-1.0, 1.0, N)
    sigma = rng.uniform(0.0, A, N)
    x = run.simulate_ar1(rho, sigma, T, rng)                      # (N, T) raw

    xs = torch.as_tensor(x, dtype=torch.float32)   # raw series, no standardisation
    with torch.no_grad():
        r_hat, s_hat = net(xs)
    r_hat, s_hat = r_hat.numpy(), s_hat.numpy()

    import os
    cache = "extra_out/gibbs_vs_nn_cache.npz"
    if os.path.exists(cache):
        d = np.load(cache)
        g_rho, g_sig = d["g_rho"], d["g_sig"]
        print("loaded gibbs cache")
    else:
        g_rho = np.empty(N)
        g_sig = np.empty(N)
        grng = np.random.default_rng(7)
        for i in range(N):
            g_rho[i], g_sig[i] = run.gibbs_posterior_mean(x[i], A, SWEEPS, BURNIN, grng)
            if (i + 1) % 100 == 0:
                print(f"  gibbs {i+1}/{N}", flush=True)
        np.savez(cache, g_rho=g_rho, g_sig=g_sig)

    mse_nn = (r_hat - rho) ** 2 + (s_hat - sigma) ** 2
    nworst = max(1, int(WORST_FRAC * N))
    worst = np.argsort(mse_nn)[-nworst:]
    is_worst = np.zeros(N, dtype=bool)
    is_worst[worst] = True
    # Feature an extreme near-non-stationary case: worst overall MSE among the
    # |rho|>0.9 series (excluding the degenerate near-zero-sigma series, where
    # the process is ~0 and nothing is identifiable). These dramatic, wandering
    # series are the ones the eye flags as hard; the figure shows the GRU still
    # tracks the Gibbs reference, and where both miss the error is irreducible.
    eligible = (np.abs(rho) > 0.9) & (sigma > 0.25 * A)
    idx = np.where(eligible)[0]
    feat = int(idx[np.argmax(mse_nn[idx])])
    print(f"FEATURED case: true sigma={sigma[feat]:.2f}  GRU sig={s_hat[feat]:.2f}  "
          f"Gibbs sig={g_sig[feat]:.2f}  true rho={rho[feat]:.2f}  "
          f"GRU rho={r_hat[feat]:.2f}  Gibbs rho={g_rho[feat]:.2f}")

    print(f"\noverall  MSE(rho): NN {np.mean((r_hat-rho)**2):.4f}  "
          f"Gibbs {np.mean((g_rho-rho)**2):.4f}")
    print(f"overall  MSE(sigma): NN {np.mean((s_hat-sigma)**2):.4f}  "
          f"Gibbs {np.mean((g_sig-sigma)**2):.4f}")

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4))

    def scatter(a, true, nn, gibbs, lo, hi, label):
        a.scatter(true, gibbs, s=10, alpha=0.35, color="0.55", label="Gibbs")
        a.scatter(true[~is_worst], nn[~is_worst], s=10, alpha=0.5,
                  color="C0", label="GRU")
        a.scatter(true[is_worst], nn[is_worst], s=14, alpha=0.8,
                  color="crimson", label="GRU, worst 5\\% MSE")
        a.scatter([true[feat]], [nn[feat]], s=120, marker="*",
                  color="crimson", edgecolors="k", zorder=5,
                  label="highlighted example")
        a.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        a.set_xlim(lo, hi); a.set_ylim(lo, hi)
        a.set_xlabel(f"true ${label}$"); a.set_ylabel(f"estimated ${label}$")
        a.set_title(f"(${label}$) estimate vs.\\ truth")

    scatter(ax[0], rho, r_hat, g_rho, -1, 1, r"\rho")
    ax[0].legend(fontsize=7, loc="upper left")
    scatter(ax[1], sigma, s_hat, g_sig, 0, A, r"\sigma")

    ax[2].plot(x[feat], lw=1.0, color="crimson")
    ax[2].axhline(0, color="grey", lw=0.5, ls=":")
    ax[2].set_xlabel("$t$"); ax[2].set_ylabel("$x_t$")
    ax[2].set_title("highlighted example realisation")
    txt = (rf"true $\sigma={sigma[feat]:.2f}$" "\n"
           rf"GRU $\hat\sigma={s_hat[feat]:.2f}$" "\n"
           rf"Gibbs $\hat\sigma={g_sig[feat]:.2f}$" "\n"
           rf"($\rho={rho[feat]:.2f}$)")
    ax[2].text(0.03, 0.97, txt, transform=ax[2].transAxes, va="top",
               fontsize=8, bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    fig.tight_layout()
    out = "../../tex/figures/gibbs_vs_nn.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
