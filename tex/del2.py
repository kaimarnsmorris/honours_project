"""
del2.py -- does importance sampling the prior help NBE training on a 1-D scale problem?
=======================================================================================
Implements the positive-contrast example from Chapter 5 (Section "When importance
sampling can help"):

    log sigma ~ Uniform(log 0.1, log 10)          # prior  p(sigma) ~ 1/sigma
    y_1, ..., y_n ~ N(0, sigma^2)
    estimator fed the sample RMS  s = sqrt(mean(y^2))     (scales with sigma)
    trained under squared error to recover sigma (the posterior mean).

The Bayes risk grows like sigma^2, so the per-draw loss -- and hence the gradient --
is dominated by the large-sigma datasets, which a log-uniform prior draws rarely.
The gradient-optimal proposal is  q ~ p*sqrt(L) ~ p*sigma ~ const, i.e. UNIFORM on
sigma, with importance weight  w = p/q ~ 1/sigma.  It draws the high-gradient
large-sigma datasets more often and down-weights them, flattening the per-draw
gradient contribution and (in principle) cutting the gradient variance.

Two runs at EQUAL simulator budget:
    PRIOR (baseline):  sigma ~ prior (log-uniform), unweighted loss.
    IS:                sigma ~ proposal (uniform on sigma), self-normalised
                       weights w ~ 1/sigma.

Plots seed-averaged test MSE (Bayes risk under the prior) vs simulator calls, and
prints the predicted gradient-variance ceiling alongside the realised improvement.

Everything is rescaled by sigma_hi so the network sees O(1) inputs/targets; the
reported MSE is in raw sigma units. SGD+momentum (the optimiser under which the
sqrt(L) proposal theory is clean). numpy + matplotlib only.

Run:  python del2.py
"""
from dataclasses import dataclass
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass
class Cfg:
    sigma_lo: float = 0.1
    sigma_hi: float = 10.0
    n_obs: int = 20               # observations per simulated dataset
    batch: int = 16               # small batch -> gradient-noise-limited training
    total_budget: int = 300_000   # datasets simulated per run
    eval_every: int = 40          # steps between evaluations
    n_test: int = 8000
    n_seeds: int = 8
    hidden: int = 64
    lr: float = 0.004
    momentum: float = 0.0
    grad_clip: float = 5.0
    tilt: float = 0.0             # proposal q(sigma) ~ sigma^tilt on [lo,hi];
                                  # tilt=0 is the sqrt(L)-optimal choice (uniform on sigma)


# ----------------------------- problem -----------------------------
def draw_sigma(n, cfg, rng, proposal):
    """Return (sigma, unnormalised IS weight w = p/q).
    prior     p(sigma) ~ 1/sigma            (log-uniform on [lo,hi])
    proposal  q(sigma) ~ sigma^tilt         (tilt=0 -> uniform on sigma)."""
    lo, hi = cfg.sigma_lo, cfg.sigma_hi
    if not proposal:
        sigma = np.exp(rng.uniform(np.log(lo), np.log(hi), n))   # draw from the prior
        return sigma, np.ones(n)
    a = cfg.tilt
    if a == 0.0:
        sigma = rng.uniform(lo, hi, n)                           # q uniform on sigma
    else:
        u = rng.uniform(0.0, 1.0, n)                             # inverse-CDF, q ~ sigma^a
        p1 = a + 1.0
        sigma = (lo**p1 + u * (hi**p1 - lo**p1)) ** (1.0 / p1)
    w = sigma ** (-(1.0 + a))     # w = p/q ~ sigma^-1 / sigma^a   (constants cancel in SNIS)
    return sigma, w


def statistic(sigma, cfg, rng):
    """Simulate n_obs ~ N(0, sigma^2) per row; return s = sqrt(mean(y^2)), shape (B,1)."""
    y = rng.standard_normal((sigma.shape[0], cfg.n_obs)) * sigma[:, None]
    return np.sqrt(np.mean(y * y, axis=1))[:, None]


# ------------------- tiny MLP (numpy), 1 -> H -> H -> 1 -------------------
def init_params(h, rng):
    he = lambda s: rng.standard_normal(s) * np.sqrt(2.0 / s[0])
    return {"W1": he((1, h)), "b1": np.zeros(h),
            "W2": he((h, h)), "b2": np.zeros(h),
            "W3": np.zeros((h, 1)), "b3": np.zeros(1)}

def forward(P, x):
    h1 = np.tanh(x @ P["W1"] + P["b1"])
    h2 = np.tanh(h1 @ P["W2"] + P["b2"])
    return h2 @ P["W3"] + P["b3"], (x, h1, h2)

def backward(P, cache, dout):
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


# ----------------------------- training -----------------------------
def train_run(cfg, seed, proposal, test_x, test_sigma):
    rng = np.random.default_rng(seed)
    P = init_params(cfg.hidden, rng); opt = SGDm(P, cfg.lr, cfg.momentum)
    H = cfg.sigma_hi
    nsteps = cfg.total_budget // cfg.batch
    bud, risk = [], []
    for step in range(nsteps):
        sigma, w = draw_sigma(cfg.batch, cfg, rng, proposal)
        x = statistic(sigma, cfg, rng) / H               # rescaled input
        out, cache = forward(P, x)
        diff = out - (sigma[:, None] / H)                # rescaled residual
        wn = w / w.sum()                                 # self-normalised IS weights
        opt.step(P, backward(P, cache, wn[:, None] * 2.0 * diff), cfg.grad_clip)
        if step % cfg.eval_every == 0 or step == nsteps - 1:
            te, _ = forward(P, test_x)
            sig_hat = te[:, 0] * H
            bud.append((step + 1) * cfg.batch)
            risk.append(float(np.mean((sig_hat - test_sigma) ** 2)))
    return np.array(bud), np.array(risk)


def ceiling(cfg, seed=12345):
    """Gradient-variance ceiling (E_p[sqrt(L)])^2 / E_p[L] from a short prior pilot;
    1/ratio bounds the plausible speedup of the sqrt(L) proposal."""
    rng = np.random.default_rng(seed)
    P = init_params(cfg.hidden, rng); opt = SGDm(P, cfg.lr, cfg.momentum)
    H = cfg.sigma_hi
    nb = 24; edges = np.linspace(np.log(cfg.sigma_lo), np.log(cfg.sigma_hi), nb + 1)
    la = np.zeros(nb); cc = np.zeros(nb)
    for _ in range((cfg.total_budget // 2) // cfg.batch):
        sigma, _ = draw_sigma(cfg.batch, cfg, rng, proposal=False)
        x = statistic(sigma, cfg, rng) / H
        out, cache = forward(P, x); diff = out - (sigma[:, None] / H)
        opt.step(P, backward(P, cache, (2.0 * diff) / cfg.batch), cfg.grad_clip)
        per = diff[:, 0] ** 2
        b = np.clip(np.digitize(np.log(sigma), edges) - 1, 0, nb - 1)
        la += np.bincount(b, weights=per, minlength=nb)
        cc += np.bincount(b, minlength=nb)
    ell = (la / np.maximum(cc, 1))[cc > 0]
    ratio = (np.mean(np.sqrt(ell)) ** 2) / np.mean(ell)
    return 1.0 / ratio


def main():
    cfg = Cfg()
    rng = np.random.default_rng(0)
    test_sigma = np.exp(rng.uniform(np.log(cfg.sigma_lo), np.log(cfg.sigma_hi), cfg.n_test))
    test_x = statistic(test_sigma, cfg, rng) / cfg.sigma_hi
    print(f"1-D scale estimation | sigma log-uniform [{cfg.sigma_lo}, {cfg.sigma_hi}] | "
          f"n_obs={cfg.n_obs} | {cfg.n_seeds} seeds | {cfg.total_budget:,} datasets/run")

    curves = {}
    for name, proposal in [("prior", False), ("IS", True)]:
        R, B = [], None
        for sd in range(cfg.n_seeds):
            B, r = train_run(cfg, 100 + sd, proposal, test_x, test_sigma)
            R.append(r)
        curves[name] = (B, np.array(R))
        print(f"  {name:>5}: final test MSE = {np.array(R)[:, -1].mean():.5f}")

    cap = ceiling(cfg)
    uf = curves["prior"][1][:, -1].mean()
    isf = curves["IS"][1][:, -1].mean()
    print(f"\npredicted gradient-variance ceiling (max speedup): {cap:.2f}x")
    print(f"risk reduction at equal budget: {100 * (uf - isf) / uf:.1f}%")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, col in [("prior", "crimson"), ("IS", "dodgerblue")]:
        B, R = curves[name]
        ax.plot(B, R.mean(0), color=col, lw=2,
                label=("prior (baseline)" if name == "prior" else r"IS ($q\propto p\sqrt{L}$)"))
        ax.fill_between(B, R.min(0), R.max(0), color=col, alpha=0.15)
    ax.set_yscale("log"); ax.set_xscale("log"); ax.set_xlabel("simulator calls (datasets)")
    ax.set_ylabel(r"test MSE in $\sigma$ (Bayes risk under the prior)")
    ax.set_title("1-D scale estimation: importance sampling vs prior (SGD)")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout(); fig.savefig("del2_scale_result.png", dpi=140)
    print("saved del2_scale_result.png")


if __name__ == "__main__":
    main()
