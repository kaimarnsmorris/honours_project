#!/usr/bin/env python3
# ============================================================================
#  grad_variance.py
#
#  Direct measurement of the per-minibatch gradient-estimator variance for the
#  AR(1) importance-sampling sweep of Chapter 5.  The chapter argues IS only
#  added noise here from the downstream test error and the ESS collapse; this
#  script measures the quantity SGD actually consumes -- the variance of the
#  self-normalised gradient estimator -- and reports it relative to the uniform
#  (alpha = 1) Monte Carlo baseline.
#
#  For a FROZEN network gamma and proposal q_alpha, one minibatch gives the
#  self-normalised gradient exactly as in training (run.py:206-208):
#
#      g = grad_gamma [ sum_i w~_i * l_i ],   w~_i = w_i / sum_j w_j,
#
#  with l_i = (rho_hat_i - rho_i)^2 + (sig_hat_i - sigma_i)^2 and w = p/q.
#  Over M independent minibatches (each of size B, so every alpha spends the
#  same simulation budget per step) we estimate the TOTAL gradient variance
#
#      V(alpha) = tr Cov(g) = (1/M) sum_m || g^(m) - gbar ||^2 ,
#
#  and report the ratio V(alpha)/V(1).  Below 1 = a genuine variance reduction;
#  above 1 = IS added noise.  Measured at a fresh random initialisation -- the
#  start of training, where gradient noise matters most.
#
#  Reuses the model, simulator and proposal from run.py -- no logic is
#  duplicated, and nothing in the existing pipeline is touched.
#
#  Usage:
#    python grad_variance.py                 # full run, paper settings
#    python grad_variance.py --quick         # fast smoke test
#    python grad_variance.py --plot-only     # re-render the figure from the CSV
#
#  Outputs (into --outdir, default ./ar1_is_out):
#    grad_variance.csv   per-alpha ESS and V(alpha)/V(1) at random init
#    grad_variance.png   bar chart of the variance ratio by proposal strength
# ============================================================================

import argparse
import csv
import os
import sys
import time

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the model, simulator and proposal from the main sweep script (same
# directory).  Importing run.py is side-effect-free: it only defines functions
# and does its work behind an __main__ guard.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run import (
    AR1Estimator,
    draw_params,
    simulate_ar1,
    effective_sample_size,
)


# ----------------------------------------------------------------------------
#  Config
# ----------------------------------------------------------------------------
def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="ar1_is_out")
    p.add_argument("--T", type=int, default=100, help="series length")
    p.add_argument("--A", type=float, default=2.0, help="sigma ~ U(0, A)")
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[2.0, 1.5, 1.0, 0.5, 0.3, 0.1])
    p.add_argument("--sigma-proposal", choices=["small", "symmetric"],
                   default="small", dest="sigma_proposal")
    p.add_argument("--batch", type=int, default=256,
                   help="minibatch size B (same for every alpha)")
    p.add_argument("--M", type=int, default=300,
                   help="# minibatches used to estimate tr Cov(g)")
    p.add_argument("--meas-seeds", type=int, default=3, dest="meas_seeds",
                   help="independent measurement seeds (for error bars)")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--mlp", type=int, default=64)
    p.add_argument("--init_seed", type=int, default=0,
                   help="seed for the frozen random-init reference net")
    p.add_argument("--quick", action="store_true",
                   help="tiny settings for a smoke test")
    p.add_argument("--plot-only", action="store_true", dest="plot_only",
                   help="skip measurement; re-render the figure from the CSV")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


# ----------------------------------------------------------------------------
#  Core measurement
# ----------------------------------------------------------------------------
def _flat_grad(net):
    """Flatten the current .grad of every parameter into one 1-D numpy vector."""
    return torch.cat([p.grad.detach().reshape(-1) for p in net.parameters()
                      if p.grad is not None]).cpu().numpy()


def grad_samples(net, alpha, args, device, M, rng):
    """Draw M independent minibatches from q_alpha and return, for each, the
    frozen-weight self-normalised gradient (stacked, shape (M, P)) together with
    that batch's effective sample size.

    Weights are frozen: we compute .grad but never call opt.step(), so the
    network is identical across every alpha -- the only thing that changes is
    the proposal.
    """
    grads = np.empty((M, sum(p.numel() for p in net.parameters())),
                     dtype=np.float64)
    ess = np.empty(M, dtype=np.float64)
    net.train()  # no dropout/BatchNorm in this model, so grads are deterministic
    for m in range(M):
        rho, sigma, w = draw_params(alpha, args.batch, args.A, rng,
                                    args.sigma_proposal)
        x = simulate_ar1(rho, sigma, args.T, rng)
        x = torch.as_tensor(x, dtype=torch.float32, device=device)
        rho_t = torch.as_tensor(rho, dtype=torch.float32, device=device)
        sig_t = torch.as_tensor(sigma, dtype=torch.float32, device=device)
        w_t = torch.as_tensor(w, dtype=torch.float32, device=device)

        rho_hat, sig_hat = net(x)
        per_sample = (rho_hat - rho_t) ** 2 + (sig_hat - sig_t) ** 2
        # Self-normalised IS gradient -- identical to the training objective.
        w_norm = w_t / w_t.sum().clamp_min(1e-12)
        loss = (w_norm * per_sample).sum()

        net.zero_grad(set_to_none=False)
        loss.backward()
        grads[m] = _flat_grad(net)
        ess[m] = effective_sample_size(w)
    return grads, ess


def total_grad_variance(grads):
    """tr Cov(g) = mean over batches of the squared deviation from the mean
    gradient = (1/M) sum_m || g_m - gbar ||^2."""
    gbar = grads.mean(axis=0, keepdims=True)
    return float(((grads - gbar) ** 2).sum(axis=1).mean())


def measure_net(net, args, device, label):
    """For one frozen reference net, measure V(alpha)/V(1) across alphas over
    several measurement seeds.  Returns a dict keyed by alpha with V_mean,
    V_std, ratio_mean, ratio_std, ess_mean.  Ratios are paired within each seed
    (V_s(alpha)/V_s(1)) so the baseline noise cancels, giving cleaner error
    bars."""
    alphas = list(args.alphas)
    if 1.0 not in alphas:
        raise SystemExit("alpha=1.0 (the MC baseline) must be in --alphas")

    V_by_seed = {a: [] for a in alphas}        # V_s(alpha) per seed
    ratio_by_seed = {a: [] for a in alphas}    # V_s(alpha)/V_s(1) per seed
    ess_acc = {a: [] for a in alphas}

    for s in range(args.meas_seeds):
        Vs = {}
        for a in alphas:
            rng = np.random.default_rng(70000 + 1000 * s + int(round(a * 100)))
            grads, ess = grad_samples(net, a, args, device, args.M, rng)
            Vs[a] = total_grad_variance(grads)
            ess_acc[a].append(float(ess.mean()))
        base = Vs[1.0]
        for a in alphas:
            V_by_seed[a].append(Vs[a])
            ratio_by_seed[a].append(Vs[a] / base if base > 0 else float("nan"))
        print(f"  [{label}] seed {s}: "
              + ", ".join(f"a={a}:{ratio_by_seed[a][-1]:.2f}x" for a in alphas))

    out = {}
    for a in alphas:
        out[a] = {
            "V_mean": float(np.mean(V_by_seed[a])),
            "V_std": float(np.std(V_by_seed[a])),
            "ratio_mean": float(np.mean(ratio_by_seed[a])),
            "ratio_std": float(np.std(ratio_by_seed[a])),
            "ess_mean": float(np.mean(ess_acc[a])),
        }
    return out


# ----------------------------------------------------------------------------
#  CSV I/O
# ----------------------------------------------------------------------------
def write_csv(res, alphas, batch, path):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["alpha", "mean_ess", "ess_penalty",
                     "V_init_mean", "V_init_std",
                     "ratio_init_mean", "ratio_init_std"])
        for a in alphas:
            ess = res[a]["ess_mean"]
            wr.writerow([
                a, f"{ess:.2f}", f"{batch / max(ess, 1e-9):.3f}",
                f"{res[a]['V_mean']:.6e}", f"{res[a]['V_std']:.6e}",
                f"{res[a]['ratio_mean']:.4f}", f"{res[a]['ratio_std']:.4f}",
            ])


def read_csv(path):
    """Load grad_variance.csv into (res, alphas) for re-plotting.  Tolerates the
    older wide schema (extra converged columns) by keying off the init columns."""
    res, alphas = {}, []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            a = float(row["alpha"])
            alphas.append(a)
            res[a] = {
                "ratio_mean": float(row["ratio_init_mean"]),
                "ratio_std": float(row["ratio_init_std"]),
                "ess_mean": float(row["mean_ess"]),
            }
    return res, alphas


# ----------------------------------------------------------------------------
#  Plot: bar chart of the gradient-variance ratio by proposal strength
# ----------------------------------------------------------------------------
def plot_grad_variance(res, alphas, outpath):
    order = list(alphas)                                  # 2.0, 1.5, 1.0, 0.5, 0.3, 0.1
    ratios = [res[a]["ratio_mean"] for a in order]
    errs = [res[a]["ratio_std"] for a in order]
    x = np.arange(len(order))

    plt.figure(figsize=(6.5, 4.2))
    plt.bar(x, ratios, yerr=errs, capsize=4,
            color="#4C72B0", edgecolor="black", linewidth=0.6)
    plt.axhline(1.0, color="grey", ls="--", lw=1.0)      # uniform baseline
    plt.xticks(x, [f"{a:g}" for a in order])
    plt.xlabel(r"proposal $\alpha$")
    plt.ylabel(r"gradient variance ratio $V(\alpha)/V(1)$")
    plt.title("Per-minibatch gradient variance by proposal strength")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


# ----------------------------------------------------------------------------
#  Main
# ----------------------------------------------------------------------------
def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)
    csv_path = os.path.join(args.outdir, "grad_variance.csv")
    png_path = os.path.join(args.outdir, "grad_variance.png")

    # ---- re-render only (no measurement) ----
    if args.plot_only:
        if not os.path.exists(csv_path):
            raise SystemExit(f"CSV not found: {csv_path}")
        res, alphas = read_csv(csv_path)
        plot_grad_variance(res, alphas, png_path)
        print(f"re-rendered {png_path} from {csv_path}")
        return

    if args.quick:
        args.batch = 128
        args.M = 40
        args.meas_seeds = 1

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")
    print(f"alphas = {args.alphas}, B = {args.batch}, M = {args.M}, "
          f"meas_seeds = {args.meas_seeds}")
    t0 = time.time()

    # ---- frozen reference net: random init ----
    torch.manual_seed(args.init_seed)
    init_net = AR1Estimator(args.hidden, args.mlp, args.A).to(device)
    for p in init_net.parameters():
        p.requires_grad_(True)

    print("\nmeasuring gradient variance at the random-init net ...")
    res = measure_net(init_net, args, device, "init")

    write_csv(res, args.alphas, args.batch, csv_path)
    print(f"\nwrote {csv_path}")
    plot_grad_variance(res, args.alphas, png_path)
    print(f"wrote {png_path}")

    # ---- console summary ----
    print("\n" + "=" * 60)
    print("GRADIENT-VARIANCE SUMMARY  (ratio V(alpha)/V(1); <1 = reduced)")
    print("=" * 60)
    header = f"{'alpha':>6} | {'ESS':>7} {'B/ESS':>7} | {'ratio':>16}"
    print(header)
    print("-" * len(header))
    for a in args.alphas:
        ess = res[a]["ess_mean"]
        print(f"{a:>6} | {ess:>7.1f} {args.batch / max(ess, 1e-9):>7.2f} | "
              f"{res[a]['ratio_mean']:>7.2f} +/- {res[a]['ratio_std']:<6.2f}")
    print(f"\ntotal time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
