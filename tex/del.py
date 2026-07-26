"""
Importance sampling the prior for neural Bayes estimation  --  SGD-only, corrected metrics
==========================================================================================

Consolidated replacement for the earlier scripts. Assumes SGD+momentum is the optimizer
(under which the optimal-proposal theory q* ~ p*sqrt(s) holds exactly, no preconditioning
caveats). Answers, honestly: how much does importance-sampling the prior actually buy?

WHAT WE LEARNED, BAKED IN
-------------------------
1. The TRUSTWORTHY measure of the benefit is the gradient-variance ceiling
       ratio = (E_p[sqrt(ell)])^2 / E_p[ell]   in (0,1],   max speedup ~ 1/ratio,
   computed from a short uniform pilot. Under SGD a variance reduction of factor 1/ratio
   yields at most ~1/ratio faster convergence -- so this bounds the real benefit.
   (ell is the cheap forward-only per-parameter risk; it is a proxy for the exact
    gradient 2nd moment s, justified by sqrt(s) ~ sqrt(ell) under Jacobian conditioning.
    It is mildly OPTIMISTIC since it counts irreducible posterior variance.)

2. Curve-read speedups are FRAGILE and routinely LIE:
     - "budget to reach X * final risk" degenerates when the target is easy (ratio -> 1),
     - "budget to match uniform's final risk" inflates on converged/flat tails, where a
       tiny risk gap maps to a huge apparent budget gap (we saw a bogus "70% saving").
   This script computes a curve-based speedup too, but CROSS-CHECKS it against the
   ceiling and prints a warning when it exceeds what the variance reduction can justify.

3. The honest empirical signal is the RISK REDUCTION AT EQUAL BUDGET (it cannot be gamed
   by target/flat-tail choices). Report that, with seed spread.

TOY PROBLEM (swap the two marked functions to test YOUR model)
--------------------------------------------------------------
  theta ~ Uniform([-1,1]^d)
  encoder  G(theta) = R1 @ theta + A * c(theta) * sin(R2 @ theta) in R^m, m=2d
           c(theta) = 0.5*(1+tanh(3*theta_1))   (reducible difficulty tied to theta_1;
                                                  does NOT wash out as d grows)
  observe  n_obs replicates; net sees the sample mean xbar (approx sufficient);
           estimate theta under squared-error loss, small sigma => error is REDUCIBLE.

IS proposal: adaptive over the difficulty feature f=theta_1, via a prior reservoir indexed
by f-bins so f~proposal and theta|f~prior exactly; weights = p_bin/pi_bin, ESS-floored.
Uniform baseline = same machinery with lam=0 (weights identically 1). Equal calls/step.

Run:  python is_nbe_sgd.py        (numpy + matplotlib only)
"""
from dataclasses import dataclass
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    dims: tuple = (5, 10, 20)        # set to a single value e.g. (10,) for one problem
    n_obs: int = 20
    sigma: float = 0.05
    A_nl: float = 2.0                # nonlinear difficulty amplitude
    batch_size: int = 256
    total_budget: int = 300_000      # simulator calls per run
    eval_every: int = 25
    n_test: int = 4000
    n_seeds: int = 4
    hidden: int = 64
    sgd_lr: float = 0.12             # SGD step (tune for your problem; clipping helps)
    momentum: float = 0.0
    grad_clip: float = 10.0
    n_bins: int = 30                 # proposal resolution over the difficulty feature
    ema: float = 0.1
    eps_floor: float = 0.10          # min proposal mass per bin -> caps IS weights at 1/eps
    lam: float = 1.0                 # q ~ p * ell^(lam/2); lam=1 -> the sqrt(ell) surrogate
    pilot_budget: int = 150_000      # for the ceiling diagnostic
    reservoir: int = 200_000


# ====================== SWAP THESE TWO FOR YOUR OWN MODEL ======================
def make_problem(d, cfg, seed=0):
    rng = np.random.default_rng(seed)
    m = 2 * d
    return {"d": d, "m": m,
            "R1": rng.standard_normal((m, d)) / np.sqrt(d),
            "R2": rng.standard_normal((m, d)) * 1.5}

def simulate_xbar(theta, prob, cfg, rng):
    """theta: (B,d) drawn from the prior -> net input xbar: (B,m).
    Replace with your simulator: draw data given theta and return a (B, n_features) summary."""
    c = 0.5 * (1 + np.tanh(3 * theta[:, [0]]))                 # difficulty feature = theta_1
    g = theta @ prob["R1"].T + cfg.A_nl * c * np.sin(theta @ prob["R2"].T)
    return g + (cfg.sigma / np.sqrt(cfg.n_obs)) * rng.standard_normal(g.shape)

def sample_prior(n, d, rng):
    return rng.uniform(-1, 1, (n, d))
# ===============================================================================


# ---- tiny MLP (vectorised forward/backward) ----
def init_params(n_in, h, d_out, rng):
    he = lambda s: rng.standard_normal(s) * np.sqrt(2.0 / s[0])
    return {"W1": he((n_in, h)), "b1": np.zeros(h),
            "W2": he((h, h)),    "b2": np.zeros(h),
            "W3": np.zeros((h, d_out)), "b3": np.zeros(d_out)}

def forward(P, x):
    h1 = np.tanh(x @ P["W1"] + P["b1"])
    h2 = np.tanh(h1 @ P["W2"] + P["b2"])
    return h2 @ P["W3"] + P["b3"], (x, h1, h2)

def backward(P, cache, dout):           # dout: (B,d) already = (1/B)*w*dL/dout
    x, h1, h2 = cache
    g = {"W3": h2.T @ dout, "b3": dout.sum(0)}
    dz2 = (dout @ P["W3"].T) * (1 - h2**2)
    g["W2"] = h1.T @ dz2; g["b2"] = dz2.sum(0)
    dz1 = (dz2 @ P["W2"].T) * (1 - h1**2)
    g["W1"] = x.T @ dz1;  g["b1"] = dz1.sum(0)
    return g

class SGDm:
    def __init__(self, P, lr, mom):
        self.lr, self.mom = lr, mom
        self.v = {k: np.zeros_like(v) for k, v in P.items()}
    def step(self, P, g, clip):
        gn = np.sqrt(sum(np.sum(gi**2) for gi in g.values()))
        sc = min(1.0, clip / (gn + 1e-12))
        for k in P:
            self.v[k] = self.mom * self.v[k] + sc * g[k]
            P[k] -= self.lr * self.v[k]


# ---- adaptive proposal over the difficulty feature, via a prior reservoir ----
class Reservoir:
    def __init__(self, prob, cfg, rng):
        d = prob["d"]
        self.theta = sample_prior(cfg.reservoir, d, rng)
        f = self.theta[:, 0]                                 # difficulty feature = theta_1
        self.edges = np.linspace(f.min(), f.max() + 1e-9, cfg.n_bins + 1)
        b = np.clip(np.digitize(f, self.edges) - 1, 0, cfg.n_bins - 1)
        self.idx = [np.where(b == k)[0] for k in range(cfg.n_bins)]
        counts = np.array([len(ix) for ix in self.idx], dtype=float)
        self.p_bin = counts / counts.sum()
        self.nonempty = counts > 0
    def draw(self, n, pi, rng):
        bins = rng.choice(len(pi), size=n, p=pi)
        theta = np.empty((n, self.theta.shape[1]))
        for k in np.unique(bins):
            sel = bins == k
            theta[sel] = self.theta[rng.choice(self.idx[k], size=sel.sum())]
        return theta, bins

def proposal_pi(ell, p_bin, nonempty, lam, eps):
    raw = np.where(nonempty, p_bin * (ell ** (lam / 2.0)), 0.0)
    pl = p_bin.copy() if raw.sum() == 0 else raw / raw.sum()
    pi = (1 - eps) * pl + eps * p_bin
    pi /= pi.sum()
    w_bin = np.where(pi > 0, p_bin / np.maximum(pi, 1e-12), 0.0)
    return pi, w_bin


# ---- one training run ----
def train_run(prob, cfg, lam, seed, tt, txb):
    rng = np.random.default_rng(seed); d, m = prob["d"], prob["m"]
    P = init_params(m, cfg.hidden, d, rng); opt = SGDm(P, cfg.sgd_lr, cfg.momentum)
    res = Reservoir(prob, cfg, rng); ell = np.ones(cfg.n_bins)
    pi, w_bin = proposal_pi(ell, res.p_bin, res.nonempty, lam, cfg.eps_floor)
    nsteps = cfg.total_budget // cfg.batch_size
    bud, risk, essf = [], [], []
    for step in range(nsteps):
        theta, bins = res.draw(cfg.batch_size, pi, rng); w = w_bin[bins]
        xbar = simulate_xbar(theta, prob, cfg, rng)
        out, cache = forward(P, xbar); diff = out - theta
        per = np.sum(diff**2, axis=1)
        opt.step(P, backward(P, cache, (w[:, None] * 2.0 * diff) / cfg.batch_size), cfg.grad_clip)
        sums = np.bincount(bins, weights=per, minlength=cfg.n_bins)
        cnts = np.bincount(bins, minlength=cfg.n_bins); seen = cnts > 0
        bm = np.where(seen, sums / np.maximum(cnts, 1), ell)
        ell = np.where(seen, (1 - cfg.ema) * ell + cfg.ema * bm, ell)
        pi, w_bin = proposal_pi(ell, res.p_bin, res.nonempty, lam, cfg.eps_floor)
        if step % cfg.eval_every == 0 or step == nsteps - 1:
            te, _ = forward(P, txb)
            bud.append((step + 1) * cfg.batch_size)
            risk.append(float(np.mean(np.sum((te - tt)**2, axis=1))))
            essf.append((w.sum()**2) / np.sum(w**2) / cfg.batch_size)
    return np.array(bud), np.array(risk), float(np.mean(essf))


# ---- ceiling diagnostic (the trustworthy benefit estimate) ----
def ceiling(prob, cfg, seed=3):
    rng = np.random.default_rng(seed)
    P = init_params(prob["m"], cfg.hidden, prob["d"], rng); opt = SGDm(P, cfg.sgd_lr, cfg.momentum)
    nb = cfg.n_bins; edges = np.linspace(-1, 1, nb + 1)
    la = np.zeros(nb); cc = np.zeros(nb)
    for step in range(cfg.pilot_budget // cfg.batch_size):
        th = sample_prior(cfg.batch_size, prob["d"], rng)
        xb = simulate_xbar(th, prob, cfg, rng)
        out, cache = forward(P, xb); diff = out - th
        opt.step(P, backward(P, cache, (2.0 * diff) / cfg.batch_size), cfg.grad_clip)
        b = np.clip(np.digitize(th[:, 0], edges) - 1, 0, nb - 1)
        la += np.bincount(b, weights=np.sum(diff**2, 1), minlength=nb)
        cc += np.bincount(b, minlength=nb)
    ell = (la / np.maximum(cc, 1))[cc > 0]
    ratio = (np.mean(np.sqrt(ell))**2) / np.mean(ell)
    hetero = np.mean(np.sort(ell)[-3:]) / np.mean(np.sort(ell)[:3])
    return 1.0 / ratio, hetero


def budget_to_reach(b, r, tgt):
    below = np.where(r <= tgt)[0]
    if len(below) == 0: return np.inf
    i = below[0]
    if i == 0: return float(b[0])
    r0, r1 = r[i-1], r[i]
    return float(b[i]) if r0 == r1 else float(b[i-1] + (r0-tgt)/(r0-r1)*(b[i]-b[i-1]))


def run_dim(d, cfg):
    prob = make_problem(d, cfg, seed=d)
    rng = np.random.default_rng(777)
    tt = sample_prior(cfg.n_test, d, rng); txb = simulate_xbar(tt, prob, cfg, rng)
    out = {}
    for name, lam in [("uniform", 0.0), ("IS", cfg.lam)]:
        R, B, E = [], None, []
        for s in range(cfg.n_seeds):
            b, r, e = train_run(prob, cfg, lam, 4000 + s, tt, txb); R.append(r); B = b; E.append(e)
        out[name] = {"b": B, "R": np.array(R), "ess": float(np.mean(E))}
    out["ceiling"], out["hetero"] = ceiling(prob, cfg)
    return out


def report(cfg, results):
    print("\n" + "=" * 74)
    print("RESULTS  (TRUSTWORTHY = ceiling & equal-budget; curve speedup is cross-checked)")
    print("=" * 74)
    for d in cfg.dims:
        o = results[d]; uni, isr = o["uniform"], o["IS"]
        uf = uni["R"][:, -1].mean(); isf = isr["R"][:, -1].mean()
        eq_red = (uf - isf) / uf * 100
        eq_sd = np.std([(uni["R"][s, -1] - isr["R"][s, -1]) / uni["R"][s, -1] * 100
                        for s in range(cfg.n_seeds)])
        half = len(uni["b"]) // 2
        conv = (uni["R"][:, half].mean() - uf) / uni["R"][:, half].mean() * 100
        # fragile curve-based speedup: budget for IS to match uniform's final risk
        sp = [cfg.total_budget / budget_to_reach(isr["b"], isr["R"][s], uf)
              for s in range(cfg.n_seeds) if np.isfinite(budget_to_reach(isr["b"], isr["R"][s], uf))]
        curve_sp = np.mean(sp) if sp else np.nan
        print(f"\n d = {d}")
        print(f"   loss heterogeneity (hard/easy theta_1):  {o['hetero']:.1f}x")
        print(f"   gradient-variance ceiling  [TRUST]:      <= {o['ceiling']:.2f}x")
        print(f"   risk reduction @ equal budget [TRUST]:   {eq_red:.1f}% (sd {eq_sd:.1f}%)")
        print(f"   IS effective-sample-size fraction:       {isr['ess']:.2f}")
        print(f"   uniform converged:                       "
              f"{'yes' if conv < 5 else f'no ({conv:.0f}% further drop)'}")
        flag = ""
        if np.isfinite(curve_sp) and curve_sp > 1.3 * o["ceiling"]:
            flag = "  <-- EXCEEDS CEILING: artifact (flat tail/target), do NOT trust this"
        print(f"   curve-based speedup [FRAGILE]:           {curve_sp:.2f}x{flag}")

    print("\n" + "-" * 74)
    ceilings = [results[d]["ceiling"] for d in cfg.dims]
    print("VERDICT (SGD)")
    print(f"  Trustworthy benefit (ceiling) across d={cfg.dims}: "
          f"{', '.join(f'{c:.2f}x' for c in ceilings)}")
    if max(ceilings) < 1.3:
        print(f"  -> Benefit is small and bounded (~{np.mean(ceilings):.2f}x) and does not blow up")
        print("     with dimension. Importance sampling is a marginal optimisation here.")
    else:
        print(f"  -> Ceiling reaches {max(ceilings):.2f}x; IS may be worth it in this regime.")
    print("  The theory's ceiling (E[sqrt(ell)])^2/E[ell] is the number to report -- curve-read")
    print("  speedups are unreliable (we auto-flag any that exceed the ceiling above).")
    print("  To test YOUR model: edit make_problem / simulate_xbar / sample_prior and rerun.")


def make_plot(cfg, results, path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(cfg.dims)))
    for d, col in zip(cfg.dims, cols):
        o = results[d]
        for name, ls in [("uniform", ":"), ("IS", "-")]:
            r = o[name]["R"].mean(0)
            axL.plot(o[name]["b"], r / r[0], ls, color=col, lw=1.8,
                     label=f"d={d} {name}")
    axL.set_yscale("log"); axL.set_xlabel("simulator calls")
    axL.set_ylabel("Bayes risk (normalised to start)")
    axL.set_title("IS (solid) vs uniform (dotted), SGD")
    axL.legend(frameon=False, fontsize=7, ncol=len(cfg.dims)); axL.grid(alpha=0.3, which="both")
    axR.plot(list(cfg.dims), [results[d]["ceiling"] for d in cfg.dims],
             "o-", color="#d62728", lw=2, label="ceiling (trustworthy)")
    axR.axhline(1.0, color="#888", ls=":")
    axR.set_xlabel("parameter dimension d"); axR.set_ylabel("max plausible IS speedup")
    axR.set_title("Trustworthy benefit vs dimension"); axR.grid(alpha=0.3); axR.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130)
    print(f"\nSaved figure -> {path}")


def main():
    cfg = Cfg()
    print(f"SGD-only IS experiment | dims={cfg.dims} | {cfg.n_seeds} seeds | "
          f"{cfg.total_budget:,} calls | lr={cfg.sgd_lr}")
    results = {d: run_dim(d, cfg) for d in cfg.dims}
    report(cfg, results)
    make_plot(cfg, results, "is_nbe_sgd_result.png")


if __name__ == "__main__":
    main()