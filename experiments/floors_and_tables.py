#!/usr/bin/env python3
# ============================================================================
#  floors_and_tables.py
#
#  Bayes-optimal floors + final-loss tables for the tau-uniform (sweeps2)
#  linear-regression experiment. Driven entirely by the saved .npz metadata, so
#  it adapts to whatever grid / X-realisations / seeds the sweep was run with.
#
#  For each design-matrix realisation X{xi} it reproduces the held-out test set
#  (same build_design seed + same 999+p draw seed as run_cell), runs a vectorised
#  Gibbs sampler for the LR posterior, and reads the trained MC/RB curves from
#  rb_vs_mc_sweep_tau_results.npz. Writes floors.json (consumed by
#  replot_curves_floor.py) and prints per-X tables + a combined floor table.
# ============================================================================

import json

import numpy as np
import torch
import scipy.stats as st

import rb_vs_mc_sweep_tau as S

TAIL = 10            # average final loss over the last TAIL epochs
GIBBS_SWEEPS, GIBBS_BURNIN = 1500, 500
NPZ = "rb_vs_mc_sweep_tau_results.npz"
FLOORS_JSON = "floors.json"


class Cfg:
    """Problem config reconstructed from the .npz metadata, so the Gibbs test set
    matches the one the networks were evaluated on."""

    def __init__(self, npz):
        self.noise_prior = str(npz["noise_prior"])
        self.tau_low = float(npz["tau_low"])
        self.tau_high = float(npz["tau_high"])
        self.sigma0 = float(npz["sigma0"])
        self.n_obs = int(npz["n_obs"])
        self.test_size = int(npz["test_size"])
        self.sigma_low, self.sigma_high = 0.1, 10.0   # sigma_uniform fallback (unused)


def gibbs_floor(p, cfg, x_seed, device="cpu"):
    """Bayes floor for dimension p under design seed x_seed: reproduce the test
    set, run Gibbs, return the MSE-based and variance-based floor estimates
    (matching the network test-loss definition: mean over datasets AND the p+1
    components)."""
    X, XtX = S.build_design(p, cfg.n_obs, device, x_seed)
    Sigma0_inv = torch.eye(p, device=device) / (cfg.sigma0 ** 2)
    sample_batch = S.make_sampler(X, cfg.n_obs, cfg, device)

    torch.manual_seed(999 + p)                 # same seed as run_cell's test set
    beta_t, sigma_t, y = sample_batch(cfg.test_size)
    theta_t = torch.cat([beta_t, sigma_t.unsqueeze(1)], dim=-1)   # (B, p+1)
    B = cfg.test_size
    Xty = y @ X                                                  # (B, p)

    g = torch.Generator(device=device).manual_seed(7)
    rng = np.random.default_rng(7)
    tau = torch.full((B,), 0.5 * (cfg.tau_low + cfg.tau_high), device=device)
    eye = torch.eye(p, device=device)
    beta_draws, sig_draws = [], []
    shape = cfg.n_obs / 2.0 + 1.0

    for it in range(GIBBS_SWEEPS):
        # beta | tau, y  ~ N(beta_n, Sigma_n)
        sigma_sq = (1.0 / tau).clamp(min=1e-4)
        prec = Sigma0_inv.unsqueeze(0) + XtX.unsqueeze(0) / sigma_sq.view(-1, 1, 1)
        L = torch.linalg.cholesky(prec)
        rhs = (Xty / sigma_sq.unsqueeze(1)).unsqueeze(-1)
        beta_n = torch.cholesky_solve(rhs, L).squeeze(-1)              # (B, p)
        Sigma_n = torch.cholesky_solve(eye.expand_as(prec), L)         # (B, p, p)
        Lc = torch.linalg.cholesky(Sigma_n)
        z = torch.randn(B, p, generator=g, device=device)
        beta = beta_n + torch.bmm(Lc, z.unsqueeze(-1)).squeeze(-1)

        # tau | beta, y  ~ Gamma(n/2+1, rate) truncated to (tau_low, tau_high),
        # sampled by inverse-CDF through scipy's Gamma (scale = 1/rate)
        rate = 0.5 * ((y - beta @ X.T) ** 2).sum(-1).clamp(min=1e-8)   # (B,)
        scale = (1.0 / rate).cpu().numpy()
        clo = st.gamma.cdf(cfg.tau_low, shape, scale=scale)
        chi = st.gamma.cdf(cfg.tau_high, shape, scale=scale)
        u = np.clip(clo + (chi - clo) * rng.random(B), 1e-9, 1 - 1e-9)
        tau_np = np.clip(st.gamma.ppf(u, shape, scale=scale),
                         cfg.tau_low, cfg.tau_high)
        tau = torch.tensor(tau_np, dtype=torch.float32, device=device)

        if it >= GIBBS_BURNIN:
            beta_draws.append(beta)
            sig_draws.append(1.0 / torch.sqrt(tau))

    beta_draws = torch.stack(beta_draws)        # (S, B, p)
    sig_draws = torch.stack(sig_draws)          # (S, B)
    post_mean = torch.cat([beta_draws.mean(0), sig_draws.mean(0).unsqueeze(1)], -1)
    floor_mse = ((post_mean - theta_t) ** 2).mean().item()
    # variance-based: mean over datasets+components of within-chain variance
    var_theta = torch.cat([beta_draws.var(0), sig_draws.var(0).unsqueeze(1)], -1)
    floor_var = var_theta.mean().item()
    return floor_mse, floor_var


def final_loss(arr):
    """Mean over seeds of the mean of the last TAIL epochs."""
    return float(np.mean(arr[:, -TAIL:]))


def main():
    npz = np.load(NPZ, allow_pickle=True)
    cfg = Cfg(npz)
    p_grid = [int(x) for x in npz["p_grid"]]
    batch_sizes = [int(x) for x in npz["batch_sizes"]]
    x_seeds = [int(x) for x in npz["x_seeds"]]
    n_seeds = int(npz["n_seeds"])

    # CRITICAL: the floor regenerates the held-out test set from (x_seed, 999+p),
    # so it must run on the SAME device the sweep used. torch's CPU and CUDA RNGs
    # produce different draws from the same seed, so a device mismatch would put
    # the floor and the network on different test samples (and could make the
    # network's loss fall below the "floor"). The sweep selects cuda-if-available;
    # match that here.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"floor device: {device}  (must match the sweep's device)", flush=True)

    floors_all, floors_var_all = {}, {}
    for xi, x_seed in enumerate(x_seeds):
        print(f"\n########## X realisation {xi}  (design seed {x_seed}) ##########")
        floors, floors_var = {}, {}
        print(f"=== Bayes floors (n={cfg.n_obs}, tau~U({cfg.tau_low},{cfg.tau_high})) ===")
        print(f"{'p':>3} {'floor(MSE)':>11} {'floor(var)':>11}")
        for p in p_grid:
            fm, fv = gibbs_floor(p, cfg, x_seed, device)
            floors[p], floors_var[p] = fm, fv
            print(f"{p:>3} {fm:>11.4f} {fv:>11.4f}", flush=True)
        floors_all[xi], floors_var_all[xi] = floors, floors_var

        print(f"\n=== Main grid X{xi}: final test loss "
              f"(mean last {TAIL} epochs, {n_seeds} seeds) ===")
        print(f"{'p':>3} {'B':>4} {'MC':>9} {'RB':>9} {'floor':>9} {'ratio':>7} {'excess':>7}")
        print(f"--- (LaTeX rows below) ---")
        latex = []
        for p in p_grid:
            for B in batch_sizes:
                mc = final_loss(npz[f"X{xi}_p{p}_bs{B}_mc"])
                rb = final_loss(npz[f"X{xi}_p{p}_bs{B}_rb"])
                fl = floors[p]
                ratio = mc / rb
                excess = (mc - fl) / (rb - fl) if (rb - fl) > 1e-9 else float("nan")
                print(f"{p:>3} {B:>4} {mc:>9.4f} {rb:>9.4f} {fl:>9.4f} "
                      f"{ratio:>7.3f} {excess:>7.2f}")
                latex.append(f"    {p} & {B} & {mc:.4f} & {rb:.4f} & {fl:.4f} "
                             f"& {ratio:.3f} & {excess:.2f} \\\\")
        print(f"\n--- LaTeX main-grid rows (X{xi}) ---")
        print("\n".join(latex))

    # headline + batch sweep for the main-text realisation (X0)
    f0 = floors_all[0]
    if 20 in f0 and 16 in batch_sizes:
        mc = final_loss(npz["X0_p20_bs16_mc"]); rb = final_loss(npz["X0_p20_bs16_rb"])
        fl = f0[20]
        print(f"\n=== Headline X0 (p=20, B=16) ===")
        print(f"MC plateau {mc:.3f} ({100*(mc-fl)/fl:.0f}% over floor), "
              f"RB plateau {rb:.3f} ({100*(rb-fl)/fl:.0f}% over floor), floor {fl:.3f}")
        print(f"RB closes {100*(1-(rb-fl)/(mc-fl)):.0f}% of the MC-to-floor gap")
        print("\n=== Batch sweep X0 at p=20 (plateau = mean last 10 epochs) ===")
        for B in batch_sizes:
            mc = final_loss(npz[f"X0_p20_bs{B}_mc"]); rb = final_loss(npz[f"X0_p20_bs{B}_rb"])
            print(f"B={B:>4}: MC {mc:.3f} (excess {mc-fl:.3f}), "
                  f"RB {rb:.3f} (excess {rb-fl:.3f})")

    # combined floor table (all X side by side -- for the extended tab:floors)
    print("\n=== Combined Bayes floors (MSE) across X realisations ===")
    print("  p " + " ".join(f"X{xi}(s{s})".rjust(12) for xi, s in enumerate(x_seeds)))
    for p in p_grid:
        print(f"{p:>3} " + " ".join(f"{floors_all[xi][p]:>12.4f}"
                                    for xi in range(len(x_seeds))))

    # persist floors for the curves plotter
    with open(FLOORS_JSON, "w") as fh:
        json.dump({str(xi): {str(p): floors_all[xi][p] for p in p_grid}
                   for xi in range(len(x_seeds))}, fh, indent=2)
    print(f"\nWrote {FLOORS_JSON}")


if __name__ == "__main__":
    main()
