import json, copy, os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "4_linear_regression.ipynb")

with open(SRC, "r", encoding="utf-8") as f:
    base = json.load(f)


def get_src(cell):
    s = cell["source"]
    return s if isinstance(s, str) else "".join(s)


def set_src(cell, text):
    cell["source"] = text


def clear_outputs(nb):
    for c in nb["cells"]:
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None


# ---- shared replacement pieces -------------------------------------------

MLP_AND_DESIGN = '''import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Flat-y MLP: (B, n_obs) -> (B, p+1). Last col is softplus-σ head with 1e-4 floor."""

    def __init__(self, n_obs: int, p: int, hidden: int = 128, depth: int = 3):
        super().__init__()
        self.p = p
        layers: list[nn.Module] = [nn.Linear(n_obs, hidden), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, p + 1))
        self.net = nn.Sequential(*layers)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        out = self.net(y)
        sigma_head = nn.functional.softplus(out[:, self.p:]) + 1e-4
        return torch.cat([out[:, :self.p], sigma_head], dim=-1)


def sample_design_matrix(n: int, p: int, device: torch.device = device) -> torch.Tensor:
    """Sample a design matrix of size (n, p) with first column of 1s and rest N(0, 1)."""
    X = torch.randn(n, p - 1, device=device)
    return torch.cat([torch.ones(n, 1, device=device), X], dim=1)

'''

SAMPLE_PRIOR_UNIFORM = '''def sample_prior(
                    n: int,
                    p: int,
                    mu_beta: float = 0.0,
                    sigma_beta: float = 5.0,
                    tau_low: float = 0.01,
                    tau_high: float = 1.0,
                    device: torch.device | str = "cpu",
                ) -> tuple[torch.Tensor, torch.Tensor]:

      beta = mu_beta + sigma_beta * torch.randn(n, p, device=device)
      tau = torch.rand(n, device=device) * (tau_high - tau_low) + tau_low
      sigma = 1.0 / torch.sqrt(tau)   # tau ~ U(tau_low, tau_high), sigma = 1/sqrt(tau)
      return beta, sigma'''

SAMPLE_PRIOR_GAMMA = '''def sample_prior(
                    n: int,
                    p: int,
                    mu_beta: float = 0.0,
                    sigma_beta: float = 5.0,
                    gamma_shape: float = 3.0,
                    gamma_rate: float = 6.0,
                    device: torch.device | str = "cpu",
                ) -> tuple[torch.Tensor, torch.Tensor]:

      beta = mu_beta + sigma_beta * torch.randn(n, p, device=device)
      # tau ~ Gamma(shape, rate); moments matched to U(0.01, 1): mean 0.5, sd 0.289
      tau = torch.distributions.Gamma(gamma_shape, gamma_rate).sample((n,)).to(device)
      sigma = 1.0 / torch.sqrt(tau)
      return beta, sigma'''

MD_UNIFORM = '''We are interested in looking at our Rao-Blackwellized loss estimator applied to Bayesian linear regression.
$$
\\begin{aligned}
\\beta &\\sim \\mathcal{N}(\\mu_\\beta, \\Sigma_\\beta)  \\\\
 \\tau = \\frac{1}{\\sigma^2} &\\sim \\mathcal{U}(0.01, 1) \\\\
Y = X\\beta + \\epsilon, \\quad \\epsilon &\\sim \\mathcal{N}(0, \\sigma^2 I)
\\end{aligned}
$$

where here X is a "design matrix" of size $n \\times p$ and $Y$ is a vector of size $n$. We will choose the design matrix as $X=[1, N(0, 1), N(0, 1), \\ldots]$

**Note:** this notebook actually samples $\\tau \\sim \\mathcal{U}(0.01, 1)$ and sets $\\sigma = 1/\\sqrt{\\tau}$ (so $\\sigma \\in (1, 10)$), unlike the original which sampled $\\sigma \\sim \\mathcal{U}(0, 10)$ directly.'''

MD_GAMMA = '''We are interested in looking at our Rao-Blackwellized loss estimator applied to Bayesian linear regression, now with a **conjugate Gamma prior on the precision** $\\tau$.
$$
\\begin{aligned}
\\beta &\\sim \\mathcal{N}(\\mu_\\beta, \\Sigma_\\beta)  \\\\
 \\tau = \\frac{1}{\\sigma^2} &\\sim \\text{Gamma}(\\alpha=3, \\beta=6) \\\\
Y = X\\beta + \\epsilon, \\quad \\epsilon &\\sim \\mathcal{N}(0, \\sigma^2 I)
\\end{aligned}
$$

where here X is a "design matrix" of size $n \\times p$ and $Y$ is a vector of size $n$. We will choose the design matrix as $X=[1, N(0, 1), N(0, 1), \\ldots]$

**Why Gamma(3, 6)?** It is the conjugate prior on $\\tau$ whose first two moments match $\\tau \\sim \\mathcal{U}(0.01, 1)$ (mean $0.5$ vs $0.505$, sd $0.289$ vs $0.286$). This lets us check whether swapping the (non-conjugate) uniform prior for a realistic conjugate Gamma reproduces the same Rao-Blackwellized vs. Monte Carlo comparison.'''


def build(variant):
    nb = copy.deepcopy(base)
    clear_outputs(nb)
    cells = nb["cells"]

    # cell 0: markdown intro
    set_src(cells[0], MD_UNIFORM if variant == "uniform" else MD_GAMMA)

    # cell 1: imports + MLP + sample_prior
    sp = SAMPLE_PRIOR_UNIFORM if variant == "uniform" else SAMPLE_PRIOR_GAMMA
    set_src(cells[1], MLP_AND_DESIGN + sp)

    if variant == "gamma":
        # swap tau_low/tau_high plumbing for gamma_shape/gamma_rate everywhere
        for c in cells:
            if c.get("cell_type") != "code":
                continue
            t = get_src(c)
            t = t.replace("tau_low, tau_high = 0.01, 1", "gamma_shape, gamma_rate = 3.0, 6.0")
            t = t.replace("tau_low=tau_low, tau_high=tau_high",
                          "gamma_shape=gamma_shape, gamma_rate=gamma_rate")
            set_src(c, t)

    out = os.path.join(HERE, f"4_linear_regression_tau_{variant}.ipynb")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("wrote", out)


build("uniform")
build("gamma")
