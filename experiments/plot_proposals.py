#!/usr/bin/env python3
# ============================================================================
#  plot_proposals.py
#
#  The "diagram of the importance samples" for Chapter 5: how the
#  Beta(alpha, alpha) proposal family reshapes the sampling distribution,
#  what importance weights it induces, and how the effective sample size
#  collapses as alpha shrinks.  Pure scipy/numpy -- no training needed.
#
#  Both parameters use the same shape on their mapped coordinate:
#     rho   in (-1, 1):  rho_tilde   = (rho + 1)/2 ~ Beta(alpha, alpha)
#     sigma in (0,  A):  sigma_tilde =  sigma / A   ~ Beta(alpha, alpha)
#  so a single Beta(alpha, alpha) density on [0,1] describes both, and the
#  per-coordinate weight is w = (uniform density) / (proposal density)
#  = 1 / Beta(theta_tilde; alpha, alpha).  The joint weight is the product.
#
#  Output: proposals.png  (into tex/figures by default)
# ============================================================================

import argparse
import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def effective_sample_size(w):
    s1, s2 = w.sum(), (w ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../../tex/figures/proposals.png")
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[1.0, 0.5, 0.3, 0.1])
    ap.add_argument("--batch", type=int, default=256,
                    help="batch size for the mean-ESS annotation (matches the sweep)")
    ap.add_argument("--A", type=float, default=2.0, help="sigma ~ U(0, A)")
    ap.add_argument("--scatter_alpha", type=float, default=0.3,
                    help="proposal alpha used for the (rho,sigma) scatter panel")
    ap.add_argument("--scatter_n", type=int, default=4000,
                    help="number of draws in the scatter panel")
    args = ap.parse_args()

    alphas = args.alphas
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(alphas)))

    A = args.A
    # rho uses a SYMMETRIC Beta(a,a) (oversamples both ends |rho|->1);
    # sigma uses an ASYMMETRIC Beta(a,1) (oversamples SMALL sigma only).
    rho_grid = np.linspace(-0.995, 0.995, 800)
    sig_grid = np.linspace(1e-3, A - 1e-3, 800)
    rt_grid = (rho_grid + 1.0) / 2.0
    st_grid = sig_grid / A

    # ======================================================================
    #  Figure 1: proposal densities and importance weights, by parameter
    #  (rho: symmetric Beta(a,a);  sigma: small-sigma Beta(a,1))  -- 2 x 2
    # ======================================================================
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for a, c in zip(alphas, colors):
        lab = r"$\alpha=%g$" % a + (" (uniform)" if a == 1.0 else "")
        qr = 0.5 * beta_dist.pdf(rt_grid, a, a)
        qs = (1.0 / A) * beta_dist.pdf(st_grid, a, 1.0)
        ax[0, 0].plot(rho_grid, qr, color=c, lw=2, label=lab)
        ax[0, 1].plot(sig_grid, qs, color=c, lw=2, label=lab)
        ax[1, 0].plot(rho_grid, 0.5 / np.clip(qr, 1e-8, None), color=c, lw=2)
        ax[1, 1].plot(sig_grid, (1.0 / A) / np.clip(qs, 1e-8, None), color=c, lw=2)
    ax[0, 0].set(title=r"(a) Proposal for $\rho$: Beta$(\alpha,\alpha)$ (symmetric)",
                 xlabel=r"$\rho$", ylabel=r"proposal density $q(\rho)$")
    ax[0, 0].set_ylim(0, 3.0); ax[0, 0].legend(fontsize=8, loc="upper center")
    ax[0, 1].set(title=r"(b) Proposal for $\sigma$: Beta$(\alpha,1)$ (small-$\sigma$)",
                 xlabel=r"$\sigma$", ylabel=r"proposal density $q(\sigma)$")
    ax[0, 1].set_ylim(0, 3.0); ax[0, 1].legend(fontsize=8, loc="upper right")
    ax[1, 0].set(title=r"(c) Weight $w(\rho)=p/q$", xlabel=r"$\rho$",
                 ylabel=r"importance weight"); ax[1, 0].set_yscale("log")
    ax[1, 1].set(title=r"(d) Weight $w(\sigma)=p/q$", xlabel=r"$\sigma$",
                 ylabel=r"importance weight"); ax[1, 1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print("wrote", args.out)

    # ======================================================================
    #  Figure 2: sampled (rho, sigma) draws coloured by importance weight,
    #  uniform baseline vs an aggressive proposal -- shows the clustering.
    # ======================================================================
    rng = np.random.default_rng(0)
    n = args.scatter_n
    panels = [(1.0, "uniform prior"),
              (args.scatter_alpha, r"$\alpha=%g$ proposal" % args.scatter_alpha)]

    def draw(a):
        rt = rng.beta(a, a, n)        # rho: symmetric
        st = rng.beta(a, 1.0, n)      # sigma: small-sigma only
        rho = 2.0 * rt - 1.0
        sigma = A * st
        w = 1.0 / (np.clip(beta_dist.pdf(rt, a, a), 1e-8, None)
                   * np.clip(beta_dist.pdf(st, a, 1.0), 1e-8, None))
        return rho, sigma, np.clip(w, 1e-9, None)

    data = [(a, lab) + draw(a) for a, lab in panels]
    # colour scale clipped to the bulk of the aggressive proposal's weights,
    # so the gradient is visible instead of being swamped by rare extremes
    lw_agg = np.log10(data[-1][4])
    vmin, vmax = np.percentile(lw_agg, [3, 97])

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
    sc = None
    for k, (a, lab, rho, sigma, w) in enumerate(data):
        order = np.argsort(-w)   # plot low-weight (dense boundary) points last
        sc = ax2[k].scatter(rho[order], sigma[order], c=np.log10(w[order]),
                            s=16, alpha=0.6, cmap="viridis", vmin=vmin, vmax=vmax,
                            edgecolors="none")
        ax2[k].set_xlim(-1, 1); ax2[k].set_ylim(0, A)
        ax2[k].set_xlabel(r"$\rho$")
        ax2[k].set_title(f"({chr(97+k)}) {lab}")
    ax2[0].set_ylabel(r"$\sigma$")
    cb = fig2.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04, extend="both")
    cb.set_label(r"$\log_{10}$ importance weight $w(\rho,\sigma)$")
    out2 = args.out.replace("proposals.png", "samples_weighted.png")
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print("wrote", out2)


if __name__ == "__main__":
    main()
