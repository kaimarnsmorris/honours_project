#!/usr/bin/env python3
# ============================================================================
#  ar1_is_sweep.py
#
#  Importance-sampling sweep for amortised neural Bayes estimation on an
#  AR(1) model, plus a Gibbs-sampler reference.  Standalone: implements the
#  model, the GRU estimator, the SNIS training objective, the Beta-proposal
#  alpha-sweep, per-region error breakdowns, ESS logging, and the plots.
#
#  Matches the design described in Chapter 5:
#    x_t = rho x_{t-1} + eps_t,  eps_t ~ N(0, sigma^2)
#    rho ~ U(-1, 1),  sigma ~ U(0, A),  x_1 from the stationary dist.
#    Estimator: GRU encoder -> mean-pool -> MLP -> (tanh rho-head,
#               softplus sigma-head).
#    Proposal:  rho drawn via Beta(alpha, alpha) on (0,1) mapped to (-1,1);
#               sigma drawn via Beta(alpha, alpha) on (0,1) mapped to (0,A).
#               Importance weight w = p/q, used self-normalised (SNIS).
#
#  Requires: torch, numpy, scipy, matplotlib.  Runs on CPU or GPU.
#
#  Usage:
#    python ar1_is_sweep.py                  # full run, default settings
#    python ar1_is_sweep.py --quick          # fast smoke test
#    python ar1_is_sweep.py --epochs 200 --seeds 5
#
#  Outputs (into --outdir, default ./ar1_is_out):
#    sweep_results.csv        per-alpha, per-region MSE + ESS
#    gru_vs_gibbs.txt         GRU-vs-Gibbs MSE comparison (baseline alpha=1)
#    training_curves.png      validation-MSE curves, one line per alpha
#    region_breakdown.png     grouped bars: per-region MSE per alpha
#    ess_vs_alpha.png         ESS as a function of alpha
# ============================================================================

import argparse
import os
import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import beta as beta_dist
from scipy.special import gammaln
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="ar1_is_out")
    p.add_argument("--T", type=int, default=100, help="series length")
    p.add_argument("--A", type=float, default=2.0, help="sigma ~ U(0, A)")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[1.0, 0.5, 0.3, 0.1])
    p.add_argument("--sigma-proposal", choices=["small", "symmetric"],
                   default="small", dest="sigma_proposal",
                   help="small: sigma~Beta(a,1) oversamples small sigma only; "
                        "symmetric: sigma~Beta(a,a) oversamples both tails")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--steps_per_epoch", type=int, default=100)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64, help="GRU hidden width")
    p.add_argument("--mlp", type=int, default=64, help="head MLP width")
    p.add_argument("--n_test", type=int, default=2000, help="test datasets")
    p.add_argument("--seeds", type=int, default=3, help="training seeds")
    p.add_argument("--gibbs_sweeps", type=int, default=2000)
    p.add_argument("--gibbs_burnin", type=int, default=500)
    p.add_argument("--gibbs_test", type=int, default=400,
                   help="# test sets for the (slower) Gibbs comparison")
    p.add_argument("--quick", action="store_true",
                   help="tiny settings for a smoke test")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------------
#  AR(1) simulation
# ----------------------------------------------------------------------------
def simulate_ar1(rho, sigma, T, rng):
    """Simulate AR(1) series, initialised from the stationary distribution.

    rho, sigma : (B,) arrays.  Returns x of shape (B, T).
    """
    B = rho.shape[0]
    x = np.empty((B, T), dtype=np.float64)
    stat_sd = sigma / np.sqrt(np.clip(1.0 - rho**2, 1e-6, None))
    x[:, 0] = rng.normal(0.0, stat_sd)
    for t in range(1, T):
        eps = rng.normal(0.0, sigma)
        x[:, t] = rho * x[:, t - 1] + eps
    return x


# ----------------------------------------------------------------------------
#  Proposal + importance weights
# ----------------------------------------------------------------------------
def draw_params(alpha, B, A, rng, sigma_proposal="small"):
    """Draw (rho, sigma) from the Beta proposal and return the parameters
    together with their (unnormalised) importance weights w = p/q.

    rho:   rho_t ~ Beta(a, a) on (0,1),  rho = 2 rho_t - 1  in (-1, 1)
           -- symmetric, oversamples both ends |rho| -> 1.
    sigma: sig_t ~ Beta(a, b_sig) on (0,1),  sigma = A sig_t  in (0, A), with
           sigma_proposal="small"     -> b_sig = 1  : oversamples SMALL sigma
                                          only (the genuinely hard, low-noise end);
           sigma_proposal="symmetric" -> b_sig = a  : oversamples both sigma tails.

    Priors are uniform, so p(rho) = 1/2 and p(sigma) = 1/A are constant; the
    weight is the product of the two per-coordinate ratios.  At alpha = 1 every
    Beta density is 1 and w == 1 (uniform training) for either sigma proposal.
    """
    b_sig = 1.0 if sigma_proposal == "small" else alpha
    rho_t = rng.beta(alpha, alpha, size=B)
    sig_t = rng.beta(alpha, b_sig, size=B)
    rho = 2.0 * rho_t - 1.0
    sigma = A * sig_t

    # Proposal densities on the transformed parameters (constants cancel in
    # self-normalisation, so we keep only the alpha-dependent Beta factors).
    q_rho = np.clip(beta_dist.pdf(rho_t, alpha, alpha), 1e-12, None)
    q_sig = np.clip(beta_dist.pdf(sig_t, alpha, b_sig), 1e-12, None)
    w = (1.0 / q_rho) * (1.0 / q_sig)
    return rho, sigma, w


def effective_sample_size(w):
    """ESS = (sum w)^2 / sum w^2  for a batch of (unnormalised) weights."""
    w = np.asarray(w, dtype=np.float64)
    s1 = w.sum()
    s2 = (w**2).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


# ----------------------------------------------------------------------------
#  Estimator: GRU encoder -> mean-pool -> MLP -> (rho, sigma) heads
# ----------------------------------------------------------------------------
class AR1Estimator(nn.Module):
    def __init__(self, hidden=64, mlp=64, A=2.0):
        super().__init__()
        self.A = A
        self.gru = nn.GRU(input_size=1, hidden_size=hidden,
                          batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, mlp), nn.ReLU(),
            nn.Linear(mlp, mlp), nn.ReLU(),
        )
        self.rho_head = nn.Linear(mlp, 1)
        self.sig_head = nn.Linear(mlp, 1)

    def forward(self, x):
        # x: (B, T) -> (B, T, 1)
        h, _ = self.gru(x.unsqueeze(-1))
        pooled = h.mean(dim=1)            # mean-pool hidden states
        z = self.head(pooled)
        rho = torch.tanh(self.rho_head(z)).squeeze(-1)          # (-1, 1)
        sig = torch.nn.functional.softplus(
            self.sig_head(z)).squeeze(-1)                       # (0, inf)
        return rho, sig


# ----------------------------------------------------------------------------
#  NOTE ON INPUT SCALING.  The network is fed the RAW series, with no
#  per-series standardisation.  Dividing by the sample SD is exactly
#  scale-invariant (sigma -> c*sigma => x -> c*x => x/SD(x) unchanged) and so
#  erases the amplitude that identifies sigma; an earlier version did this and
#  made sigma unrecoverable.  See verify_no_standardise.py for the comparison.
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
#  Training one estimator at a given alpha
# ----------------------------------------------------------------------------
def train_one(alpha, args, device, seed, val_pack):
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(2000 + seed)

    net = AR1Estimator(args.hidden, args.mlp, args.A).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    val_x, val_rho, val_sigma = val_pack
    val_curve = []
    ess_running = []

    for epoch in range(args.epochs):
        net.train()
        for _ in range(args.steps_per_epoch):
            rho, sigma, w = draw_params(alpha, args.batch, args.A, rng,
                                        args.sigma_proposal)
            x = simulate_ar1(rho, sigma, args.T, rng)
            x = torch.as_tensor(x, dtype=torch.float32, device=device)
            rho_t = torch.as_tensor(rho, dtype=torch.float32, device=device)
            sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
            w_t = torch.as_tensor(w, dtype=torch.float32, device=device)

            rho_hat, sig_hat = net(x)
            per_sample = (rho_hat - rho_t) ** 2 + (sig_hat - sig_t) ** 2

            # Self-normalised IS objective: weights normalised within batch.
            w_norm = w_t / w_t.sum().clamp_min(1e-12)
            loss = (w_norm * per_sample).sum()

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            ess_running.append(effective_sample_size(w))

        # Validation MSE (unweighted: measures performance under the prior).
        net.eval()
        with torch.no_grad():
            rho_hat, sig_hat = net(val_x)
            vmse = (((rho_hat - val_rho) ** 2
                     + (sig_hat - val_sigma) ** 2).mean().item())
        val_curve.append(vmse)

    mean_ess = float(np.mean(ess_running))
    return net, val_curve, mean_ess


# ----------------------------------------------------------------------------
#  Test set + region masks
# ----------------------------------------------------------------------------
def make_test_set(args, device, seed=999):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(-1.0, 1.0, size=args.n_test)
    sigma = rng.uniform(0.0, args.A, size=args.n_test)
    x = simulate_ar1(rho, sigma, args.T, rng)
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    rho_t = torch.as_tensor(rho, dtype=torch.float32, device=device)
    sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
    # Region masks (numpy, for breakdowns)
    regions = {
        "all": np.ones(args.n_test, dtype=bool),
        "high_|rho|": np.abs(rho) > 0.8,
        "low_sigma": sigma < 0.25 * args.A,
        "high_sigma": sigma > 0.75 * args.A,
    }
    return (x_t, rho_t, sig_t), (rho, sigma), regions


def per_region_mse(net, test_pack, regions):
    x_t, rho_t, sig_t = test_pack
    net.eval()
    with torch.no_grad():
        rho_hat, sig_hat = net(x_t)
        se = ((rho_hat - rho_t) ** 2 + (sig_hat - sig_t) ** 2).cpu().numpy()
        se_rho = ((rho_hat - rho_t) ** 2).cpu().numpy()
        se_sig = ((sig_hat - sig_t) ** 2).cpu().numpy()
    out = {}
    for name, mask in regions.items():
        if mask.sum() == 0:
            out[name] = (float("nan"), float("nan"), float("nan"))
        else:
            out[name] = (float(se[mask].mean()),
                         float(se_rho[mask].mean()),
                         float(se_sig[mask].mean()))
    return out


# ----------------------------------------------------------------------------
#  Gibbs sampler for the AR(1) posterior (reference)
# ----------------------------------------------------------------------------
def truncated_normal_sample(mean, sd, lo, hi, rng):
    from scipy.stats import norm
    a = norm.cdf((lo - mean) / sd)
    b = norm.cdf((hi - mean) / sd)
    u = rng.uniform(a, b)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    return mean + sd * norm.ppf(u)


def truncated_invgamma_sample(shape, scale, lo2, hi2, rng):
    """Sample sigma^2 ~ Inv-Gamma(shape, scale) truncated to (lo2, hi2).
    Inverse-CDF via the Gamma CDF of the precision 1/sigma^2."""
    from scipy.stats import gamma as gamma_dist
    # sigma^2 ~ InvGamma(a, b)  <=>  1/sigma^2 ~ Gamma(a, rate=b)
    # truncate sigma^2 in (lo2, hi2)  <=>  precision in (1/hi2, 1/lo2)
    prec_lo = 1.0 / hi2
    prec_hi = 1.0 / lo2 if lo2 > 0 else np.inf
    a, b = shape, scale
    cdf_lo = gamma_dist.cdf(prec_lo, a, scale=1.0 / b)
    cdf_hi = gamma_dist.cdf(prec_hi, a, scale=1.0 / b) if np.isfinite(prec_hi) else 1.0
    u = rng.uniform(cdf_lo, cdf_hi)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    prec = gamma_dist.ppf(u, a, scale=1.0 / b)
    prec = np.clip(prec, 1e-12, None)
    return 1.0 / prec


def gibbs_posterior_mean(x, A, sweeps, burnin, rng):
    """Run the AR(1) Gibbs sampler for one series x (length T); return the
    posterior-mean estimates of (rho, sigma)."""
    T = x.shape[0]
    S1 = np.sum(x[:-1] ** 2)
    S01 = np.sum(x[1:] * x[:-1])
    Sxx = np.sum(x[1:] ** 2)

    # init
    rho = np.clip(S01 / max(S1, 1e-8), -0.99, 0.99)
    ssr = Sxx - 2 * rho * S01 + rho**2 * S1
    sigma2 = max(ssr / max(T - 1, 1), 1e-4)

    rs, ss = [], []
    for it in range(sweeps):
        # rho | sigma^2, x  ~  N(S01/S1, sigma2/S1) truncated to (-1,1)
        m = S01 / max(S1, 1e-8)
        sd = math.sqrt(sigma2 / max(S1, 1e-8))
        rho = float(truncated_normal_sample(m, sd, -1.0, 1.0, rng))
        # sigma^2 | rho, x ~ Inv-Gamma(T/2 - 1, SSR/2) truncated to (0, A^2)
        ssr = Sxx - 2 * rho * S01 + rho**2 * S1
        shape = max(T / 2.0 - 1.0, 1e-3)
        scale = max(ssr / 2.0, 1e-8)
        sigma2 = float(truncated_invgamma_sample(shape, scale, 1e-8, A**2, rng))
        if it >= burnin:
            rs.append(rho)
            ss.append(math.sqrt(sigma2))
    return float(np.mean(rs)), float(np.mean(ss))


def gru_vs_gibbs(net, args, device, seed=4242):
    """Compare the trained GRU (alpha=1 baseline) against the Gibbs posterior
    mean on a shared set of test series."""
    rng = np.random.default_rng(seed)
    n = args.gibbs_test
    rho = rng.uniform(-1.0, 1.0, size=n)
    sigma = rng.uniform(0.0, args.A, size=n)
    x = simulate_ar1(rho, sigma, args.T, rng)

    # Gibbs estimates
    g_rho = np.empty(n)
    g_sig = np.empty(n)
    grng = np.random.default_rng(seed + 1)
    for i in range(n):
        gr, gs = gibbs_posterior_mean(x[i], args.A,
                                      args.gibbs_sweeps, args.gibbs_burnin,
                                      grng)
        g_rho[i], g_sig[i] = gr, gs

    # GRU estimates
    x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        r_hat, s_hat = net(x_t)
    r_hat = r_hat.cpu().numpy()
    s_hat = s_hat.cpu().numpy()

    mse_rho_gibbs = float(np.mean((g_rho - rho) ** 2))
    mse_rho_gru = float(np.mean((r_hat - rho) ** 2))
    mse_sig_gibbs = float(np.mean((g_sig - sigma) ** 2))
    mse_sig_gru = float(np.mean((s_hat - sigma) ** 2))
    return {
        "mse_rho_gibbs": mse_rho_gibbs, "mse_rho_gru": mse_rho_gru,
        "mse_sig_gibbs": mse_sig_gibbs, "mse_sig_gru": mse_sig_gru,
    }


# ----------------------------------------------------------------------------
#  Plotting
# ----------------------------------------------------------------------------
def plot_training_curves(curves_by_alpha, outpath):
    plt.figure(figsize=(7, 4.5))
    for alpha, curve in curves_by_alpha.items():
        plt.plot(range(1, len(curve) + 1), curve, label=f"$\\alpha={alpha}$")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.yscale("log")
    plt.legend()
    plt.title("AR(1) importance-sampling sweep: training curves")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_region_breakdown(region_table, alphas, outpath):
    regions = [r for r in ["all", "high_|rho|", "low_sigma", "high_sigma"]]
    x = np.arange(len(regions))
    width = 0.8 / len(alphas)
    plt.figure(figsize=(8, 4.5))
    for j, alpha in enumerate(alphas):
        vals = [region_table[alpha][r][0] for r in regions]
        plt.bar(x + j * width, vals, width, label=f"$\\alpha={alpha}$")
    plt.xticks(x + 0.4 - width / 2, regions)
    plt.ylabel("MSE")
    plt.legend()
    plt.title("Per-region MSE by proposal strength")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_ess(ess_by_alpha, outpath, batch):
    alphas = sorted(ess_by_alpha.keys(), reverse=True)
    vals = [ess_by_alpha[a] for a in alphas]
    plt.figure(figsize=(6, 4))
    plt.plot(alphas, vals, "o-")
    plt.axhline(batch, ls="--", c="grey", label=f"batch size ({batch})")
    plt.xlabel(r"proposal $\alpha$")
    plt.ylabel("mean ESS")
    plt.gca().invert_xaxis()
    plt.legend()
    plt.title("Effective sample size vs proposal strength")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    if args.quick:
        args.epochs = 8
        args.steps_per_epoch = 20
        args.batch = 128
        args.n_test = 400
        args.seeds = 1
        args.gibbs_sweeps = 300
        args.gibbs_burnin = 100
        args.gibbs_test = 40

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    print(f"device = {device}")
    t0 = time.time()

    # Shared validation + test set (same across alphas for fair comparison).
    test_pack, _, regions = make_test_set(args, device)
    val_pack = test_pack  # reuse held-out set for the validation curve

    curves_by_alpha = {}
    ess_by_alpha = {}
    region_table = {}
    best_nets = {}

    for alpha in args.alphas:
        print(f"\n=== alpha = {alpha} ===")
        seed_curves = []
        seed_ess = []
        seed_region = []
        best_net, best_vmse = None, float("inf")
        for s in range(args.seeds):
            net, curve, mean_ess = train_one(alpha, args, device, s, val_pack)
            seed_curves.append(curve)
            seed_ess.append(mean_ess)
            seed_region.append(per_region_mse(net, test_pack, regions))
            if curve[-1] < best_vmse:
                best_vmse, best_net = curve[-1], net
            print(f"  seed {s}: final val MSE = {curve[-1]:.6f}, "
                  f"mean ESS = {mean_ess:.1f}")
        # average curve across seeds
        curves_by_alpha[alpha] = list(np.mean(np.array(seed_curves), axis=0))
        ess_by_alpha[alpha] = float(np.mean(seed_ess))
        # average per-region MSE across seeds
        avg_region = {}
        for r in regions:
            tot = np.array([sr[r][0] for sr in seed_region])
            tot_rho = np.array([sr[r][1] for sr in seed_region])
            tot_sig = np.array([sr[r][2] for sr in seed_region])
            avg_region[r] = (float(np.nanmean(tot)),
                             float(np.nanmean(tot_rho)),
                             float(np.nanmean(tot_sig)))
        region_table[alpha] = avg_region
        best_nets[alpha] = best_net

    # ---- write sweep CSV ----
    csv_path = os.path.join(args.outdir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["alpha", "region", "mse_total", "mse_rho", "mse_sigma",
                     "mean_ess", "final_val_mse"])
        for alpha in args.alphas:
            for r in ["all", "high_|rho|", "low_sigma", "high_sigma"]:
                mt, mr, msig = region_table[alpha][r]
                wr.writerow([alpha, r, f"{mt:.6f}", f"{mr:.6f}",
                             f"{msig:.6f}", f"{ess_by_alpha[alpha]:.2f}",
                             f"{curves_by_alpha[alpha][-1]:.6f}"])
    print(f"\nwrote {csv_path}")

    # ---- cache the seed-averaged validation curves (so the curves figure can
    #      be regenerated/combined without retraining) ----
    np.savez(os.path.join(args.outdir, "curves.npz"),
             **{f"alpha_{a}": np.asarray(curves_by_alpha[a]) for a in args.alphas})

    # ---- GRU vs Gibbs on the alpha=1 baseline ----
    baseline_alpha = 1.0 if 1.0 in best_nets else args.alphas[0]
    print(f"\nrunning GRU-vs-Gibbs comparison (alpha={baseline_alpha}) ...")
    cmp = gru_vs_gibbs(best_nets[baseline_alpha], args, device)
    imp_rho = 100.0 * (cmp["mse_rho_gibbs"] - cmp["mse_rho_gru"]) / cmp["mse_rho_gibbs"]
    imp_sig = 100.0 * (cmp["mse_sig_gibbs"] - cmp["mse_sig_gru"]) / cmp["mse_sig_gibbs"]
    lines = [
        "=" * 60,
        "PERFORMANCE COMPARISON",
        "=" * 60,
        f"MSE rho (Gibbs):  {cmp['mse_rho_gibbs']:.6f}",
        f"MSE rho (GRU):    {cmp['mse_rho_gru']:.6f}",
        f"Improvement:      {imp_rho:.2f}%",
        "-" * 60,
        f"MSE sigma (Gibbs): {cmp['mse_sig_gibbs']:.6f}",
        f"MSE sigma (GRU):   {cmp['mse_sig_gru']:.6f}",
        f"Improvement:       {imp_sig:.2f}%",
    ]
    txt = "\n".join(lines)
    print("\n" + txt)
    with open(os.path.join(args.outdir, "gru_vs_gibbs.txt"), "w") as f:
        f.write(txt + "\n")

    # ---- plots ----
    plot_training_curves(curves_by_alpha,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_region_breakdown(region_table, args.alphas,
                          os.path.join(args.outdir, "region_breakdown.png"))
    plot_ess(ess_by_alpha,
             os.path.join(args.outdir, "ess_vs_alpha.png"), args.batch)
    print(f"\nwrote plots to {args.outdir}/")
    print(f"total time {time.time() - t0:.1f}s")

    # ---- console summary table ----
    print("\n" + "=" * 64)
    print("SWEEP SUMMARY (MSE total, averaged over seeds)")
    print("=" * 64)
    header = f"{'alpha':>6} | {'all':>9} {'high|rho|':>10} {'low_sig':>9} {'high_sig':>9} | {'ESS':>8}"
    print(header)
    print("-" * len(header))
    for alpha in args.alphas:
        rt = region_table[alpha]
        print(f"{alpha:>6} | {rt['all'][0]:>9.5f} {rt['high_|rho|'][0]:>10.5f} "
              f"{rt['low_sigma'][0]:>9.5f} {rt['high_sigma'][0]:>9.5f} | "
              f"{ess_by_alpha[alpha]:>8.1f}")


if __name__ == "__main__":
    main()