#!/usr/bin/env python3
# ============================================================================
#  sigma_is_demo.py
#
#  The "when importance sampling helps" companion to the AR(1) null result.
#  A deliberately IS-friendly NBE toy: estimate the scale sigma of an iid
#  Gaussian sample, where loss- and gradient-heteroscedasticity are ALIGNED.
#
#    log sigma ~ U(log 0.1, log 10),   y_1..y_n ~ N(0, sigma^2) iid,  n = N_OBS
#    estimator input  : sufficient statistic s = sqrt(mean(y_i^2))  (scales w/ sigma)
#    estimator         : small MLP s -> sigma_hat,  MSE loss on sigma
#    proposal (adaptive): q ~ p * sqrt(loss)  -- the gradient-optimal tilt for MSE
#
#  Why this is the success mode (contrast with the AR(1) failure):
#    Bayes MSE ~ sigma^2 / n  varies ~10^4x across the prior (loss heteroscedastic),
#    and because the input scales with sigma the gradient norm scales too, so the
#    two heteroscedasticities line up and q ~ p*sqrt(loss) genuinely cuts the
#    gradient-estimation variance.  Reweighting then lowers the SGD noise floor
#    (excess stationary loss ~ lr * Var[grad]) and reaches a target sooner.
#
#  Usage:
#    python sigma_is_demo.py --lr 1e-3 --epochs 4000 --seeds 20
#    python sigma_is_demo.py --sweep_lr           # locate a clean regime
# ============================================================================

import argparse
import csv
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOW, HIGH = 0.1, 10.0
LOG_LOW, LOG_HIGH = float(np.log(LOW)), float(np.log(HIGH))
LOG_RANGE = LOG_HIGH - LOG_LOW


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=str, default="sigma_is_out")
    p.add_argument("--n_obs", type=int, default=10)
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--weight_clip", type=float, default=20.0)
    p.add_argument("--n_bins", type=int, default=20)
    p.add_argument("--defensive", type=float, default=0.2)
    p.add_argument("--sweep_lr", action="store_true")
    return p.parse_args()


def sample_prior_log(B):
    return torch.rand(B, 1) * LOG_RANGE + LOG_LOW


def sufficient_stat(log_sigma, n):
    sigma = torch.exp(log_sigma)
    y = torch.randn(log_sigma.shape[0], n) * sigma
    return torch.sqrt(torch.mean(y ** 2, dim=1, keepdim=True))


class SigmaEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.ReLU(),
            nn.Linear(32, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, s):
        return self.net(s)


class AdaptiveIS:
    """Bin log(sigma); track EMA per-bin loss; sample bin ~ sqrt(bin_loss).
    q ~ p * sqrt(loss) is the gradient-optimal proposal for an MSE objective."""

    def __init__(self, n_bins=20, defensive=0.2, ema_alpha=0.1):
        self.n_bins = n_bins
        self.defensive = defensive
        self.ema_alpha = ema_alpha
        self.bin_width = LOG_RANGE / n_bins
        self.bin_edges = torch.linspace(LOG_LOW, LOG_HIGH, n_bins + 1)
        self.bin_loss = torch.ones(n_bins)
        self.p_density = 1.0 / LOG_RANGE

    def _bin_of(self, log_sigma):
        idx = ((log_sigma.squeeze(-1) - LOG_LOW) / self.bin_width).long()
        return idx.clamp(0, self.n_bins - 1)

    def _bin_probs(self):
        sqrt_loss = torch.sqrt(self.bin_loss.clamp(min=1e-12))
        return sqrt_loss / sqrt_loss.sum()

    def sample_q(self, B):
        use_uniform = torch.rand(B) < self.defensive
        u_uniform = torch.rand(B) * LOG_RANGE + LOG_LOW
        bin_idx = torch.multinomial(self._bin_probs(), B, replacement=True)
        u_adaptive = self.bin_edges[bin_idx] + torch.rand(B) * self.bin_width
        return torch.where(use_uniform, u_uniform, u_adaptive).unsqueeze(1)

    def get_weights(self, log_sigma):
        bin_idx = self._bin_of(log_sigma)
        q_adaptive = self._bin_probs()[bin_idx] / self.bin_width
        q = (1 - self.defensive) * q_adaptive + self.defensive * self.p_density
        return (self.p_density / (q + 1e-8)).unsqueeze(-1)

    def update(self, log_sigma, per_sample_loss):
        bin_idx = self._bin_of(log_sigma)
        loss_flat = per_sample_loss.detach().squeeze(-1)
        sum_b = torch.zeros(self.n_bins).scatter_add_(0, bin_idx, loss_flat)
        cnt_b = torch.zeros(self.n_bins).scatter_add_(0, bin_idx, torch.ones_like(loss_flat))
        mean_b = sum_b / cnt_b.clamp(min=1)
        upd = (1 - self.ema_alpha) * self.bin_loss + self.ema_alpha * mean_b
        self.bin_loss = torch.where(cnt_b > 0, upd, self.bin_loss)


def train(mode, args, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    sampler = AdaptiveIS(args.n_bins, args.defensive) if mode == "adaptive_is" else None

    model = SigmaEstimator()
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.0)
    crit = nn.MSELoss(reduction="none")

    val_log = sample_prior_log(2000)
    val_sigma = torch.exp(val_log)
    val_s = sufficient_stat(val_log, args.n_obs)

    history = []
    for _ in range(args.epochs):
        model.train()
        if mode == "adaptive_is":
            log_sigma = sampler.sample_q(args.batch)
            weights = sampler.get_weights(log_sigma)
        else:
            log_sigma = sample_prior_log(args.batch)
            weights = torch.ones(args.batch, 1)

        sigma = torch.exp(log_sigma)
        s = sufficient_stat(log_sigma, args.n_obs)

        opt.zero_grad()
        raw_loss = crit(model(s), sigma)
        w = weights / (weights.mean() + 1e-8)
        w = torch.clamp(w, max=args.weight_clip)
        (raw_loss * w).mean().backward()
        opt.step()
        if mode == "adaptive_is":
            sampler.update(log_sigma, raw_loss)

        model.eval()
        with torch.no_grad():
            history.append(nn.MSELoss()(model(val_s), val_sigma).item())
    return np.array(history)


def run_modes(args, seeds):
    modes = ["vanilla", "adaptive_is"]
    res = {m: [] for m in modes}
    for seed in range(seeds):
        for m in modes:
            res[m].append(train(m, args, seed))
        print(f"  seed {seed+1}/{seeds} done", flush=True)
    return {m: np.array(v) for m, v in res.items()}


def first_crossing(curve, thr):
    idx = np.where(curve <= thr)[0]
    return int(idx[0]) if len(idx) else np.inf


def summarise(res, tail=300):
    """Plateau (mean of last `tail` steps) per mode and the speedup-at-threshold."""
    out = {}
    for m, curves in res.items():
        plateau = curves[:, -tail:].mean(axis=1)
        out[m] = dict(plateau_mean=float(plateau.mean()),
                      plateau_se=float(plateau.std() / np.sqrt(len(plateau))))
    van_floor = res["vanilla"][:, -tail:].mean()
    is_floor = res["adaptive_is"][:, -tail:].mean()
    # Speedup: steps for vanilla vs IS to first reach IS's eventual plateau * 1.1
    target = is_floor * 1.1
    v_steps = np.median([first_crossing(c, target) for c in res["vanilla"]])
    i_steps = np.median([first_crossing(c, target) for c in res["adaptive_is"]])
    speedup = (v_steps / i_steps) if (np.isfinite(v_steps) and np.isfinite(i_steps) and i_steps > 0) else np.nan
    return out, dict(target=float(target), v_steps=float(v_steps),
                     i_steps=float(i_steps), speedup=float(speedup),
                     floor_ratio=float(is_floor / van_floor))


def main():
    args = get_args()
    os.makedirs(args.outdir, exist_ok=True)

    if args.sweep_lr:
        print("=== LR sweep (5 seeds each) to locate a clean speedup regime ===")
        for lr in [3e-3, 1e-3, 3e-4, 1e-4]:
            args.lr = lr
            res = run_modes(args, seeds=5)
            _, sp = summarise(res)
            print(f"lr={lr:.0e}: vanilla_floor={res['vanilla'][:,-300:].mean():.4f}  "
                  f"is_floor={res['adaptive_is'][:,-300:].mean():.4f}  "
                  f"floor_ratio={sp['floor_ratio']:.3f}  speedup~{sp['speedup']:.2f}x")
        return

    print(f"=== sigma-IS demo: lr={args.lr:.0e}, batch={args.batch}, "
          f"epochs={args.epochs}, seeds={args.seeds} ===")
    res = run_modes(args, args.seeds)
    perm, sp = summarise(res)
    print("\nplateau (mean val MSE over last 300 steps):")
    for m, d in perm.items():
        print(f"  {m:12s}: {d['plateau_mean']:.4f} +/- {d['plateau_se']:.4f}")
    print(f"\nfloor ratio (IS/vanilla): {sp['floor_ratio']:.3f}")
    print(f"steps to reach {sp['target']:.4f}:  vanilla={sp['v_steps']:.0f}  "
          f"IS={sp['i_steps']:.0f}  -> speedup {sp['speedup']:.2f}x")

    # --- save curve plot ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"vanilla": "gray", "adaptive_is": "crimson"}
    labels = {"vanilla": r"Uniform prior $p(\log\sigma)$",
              "adaptive_is": r"Adaptive IS $q\propto p\sqrt{\ell}$"}
    for m, curves in res.items():
        log_mean = np.mean(np.log(curves), axis=0)
        se = np.std(np.log(curves), axis=0) / np.sqrt(curves.shape[0])
        mean = np.exp(log_mean)
        steps = np.arange(mean.shape[0])
        ax.plot(steps, mean, color=colors[m], lw=2, label=labels[m])
        ax.fill_between(steps, np.exp(log_mean - se), np.exp(log_mean + se),
                        color=colors[m], alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.set_ylabel(r"Validation MSE on $\sigma$ (log scale)")
    ax.set_title(f"$\\sigma$-estimation: IS vs uniform (lr={args.lr:.0e}, "
                 f"batch={args.batch}, {args.seeds} seeds)")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "sigma_is_curves.png"), dpi=130,
                bbox_inches="tight")
    print(f"\nsaved {os.path.join(args.outdir, 'sigma_is_curves.png')}")

    with open(os.path.join(args.outdir, "sigma_is_results.csv"), "w", newline="") as f:
        wtr = csv.writer(f)
        wtr.writerow(["mode", "plateau_mean", "plateau_se"])
        for m, d in perm.items():
            wtr.writerow([m, d["plateau_mean"], d["plateau_se"]])
        wtr.writerow(["floor_ratio", sp["floor_ratio"], ""])
        wtr.writerow(["speedup", sp["speedup"], ""])


if __name__ == "__main__":
    main()
