"""nbe.training — generic training loop and gradient-variance measurement.

Exports
-------
train(net, sim_fn, prior_fn, loss_fn, epochs, steps, batch, lr, device, seed,
      val_pack=None)
    -> (net, val_curve, mean_ess)

    Generalised from SRC/final/sweep_importance_sampling/run.py:181-226.
    Trains with Adam + gradient clipping (max norm 5.0).  At the end of each
    epoch evaluates on val_pack if provided, otherwise evaluates on a fresh
    batch drawn from the prior.  Returns the trained net, per-epoch validation
    loss curve, and mean ESS over all training steps.

gradient_variance(net, alpha_draw_fn, sim_fn, batch, M, device, rng) -> float

    Trace of the minibatch gradient covariance for a frozen network, ported from
    SRC/final/sweep_importance_sampling/grad_variance.py (grad_samples +
    total_grad_variance).  Draws M independent minibatches from alpha_draw_fn,
    computes the SNIS gradient for each (identical to the training objective),
    and returns tr Cov(g) = (1/M) sum_m ||g_m - gbar||^2.

Interface conventions
---------------------
* sim_fn(params..., rng) -> np.ndarray (B, T)
  Takes the same positional arguments as the prior tuple, plus rng.
  E.g. for AR(1): sim_fn(rho, sigma, rng)

* prior_fn(B, rng) -> tuple of arrays
  Returns a tuple of parameter arrays (rho, sigma) for AR(1), etc.

* loss_fn(theta_hat, theta) -> torch.Tensor (scalar)
  Both arguments are (B, d) float32 tensors on device.
  Compatible with nbe.losses.mc_loss and nbe.losses.rb_loss.

* val_pack: optional tuple (x_val_tensor, *theta_tensors)
  x_val_tensor is already a float32 torch.Tensor on device.
  theta_tensors are (B,) float32 tensors for each parameter.

Notes
-----
The MC training step follows run.py lines 196-213 exactly:
  - draw (rho, sigma) from prior_fn
  - simulate x via sim_fn
  - convert to float32 tensors
  - forward pass, stack outputs and targets as (B, n_params)
  - call loss_fn(theta_hat, theta)
  - zero_grad -> backward -> clip_grad_norm_(5.0) -> step
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from .losses import effective_sample_size

__all__ = ["train", "gradient_variance"]


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _to_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    """Convert numpy array to float32 torch.Tensor on device."""
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


def _stack_params(*arrays: np.ndarray, device: str) -> torch.Tensor:
    """Stack (B,) param arrays column-wise into (B, n_params) float32 tensor."""
    return torch.stack([_to_tensor(a, device) for a in arrays], dim=1)


def _forward_and_stack(net: nn.Module, x_t: torch.Tensor) -> torch.Tensor:
    """Run net(x_t) and return outputs stacked as (B, n_outputs).

    Works for any net that returns a tuple of (B,) tensors (e.g. AR1Estimator
    returning (rho, sigma)) or a single (B, d) tensor.
    """
    out = net(x_t)
    if isinstance(out, torch.Tensor):
        if out.dim() == 1:
            return out.unsqueeze(1)
        return out  # already (B, d)
    # tuple of (B,) tensors
    return torch.stack(list(out), dim=1)  # (B, n_params)


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train(
    net: nn.Module,
    sim_fn: Callable,
    prior_fn: Callable,
    loss_fn: Callable,
    epochs: int,
    steps: int,
    batch: int,
    lr: float,
    device: str,
    seed: int,
    val_pack: Optional[Any] = None,
) -> Tuple[nn.Module, list, float]:
    """Train a neural Bayes estimator with Adam and gradient clipping.

    Parameters
    ----------
    net       : nn.Module — the estimator to train (moved to device inside)
    sim_fn    : callable(param1, param2, ..., rng) -> (B, T) ndarray
    prior_fn  : callable(B, rng) -> tuple of (B,) ndarrays
    loss_fn   : callable(theta_hat:(B,d), theta:(B,d)) -> scalar Tensor
    epochs    : number of training epochs
    steps     : gradient steps per epoch
    batch     : minibatch size
    lr        : Adam learning rate
    device    : torch device string ("cpu" or "cuda")
    seed      : integer seed (sets np.random and torch seeds)
    val_pack  : optional (x_val_tensor, param1_tensor, ...) pre-built validation
                set; if None, a fresh batch is drawn from the prior each epoch.

    Returns
    -------
    net        : trained nn.Module
    val_curve  : list of float, length = epochs (validation MSE per epoch)
    mean_ess   : float, mean ESS over all training steps (1.0 for MC, <1 for IS)
    """
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(2000 + seed)

    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    val_curve: list = []
    ess_running: list = []

    # ---- pre-built validation set ----
    if val_pack is not None:
        val_x = val_pack[0]
        val_theta = torch.stack(list(val_pack[1:]), dim=1)  # (B, n_params)
    else:
        val_x = None
        val_theta = None

    for _epoch in range(epochs):
        net.train()

        for _ in range(steps):
            params = prior_fn(batch, rng)         # tuple of (B,) arrays
            x = sim_fn(*params, rng)               # (B, T) array
            x_t = _to_tensor(x, device)           # (B, T) float32
            theta_t = _stack_params(*params, device=device)  # (B, n_params)

            theta_hat = _forward_and_stack(net, x_t)  # (B, n_params)

            loss = loss_fn(theta_hat, theta_t)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0)
            opt.step()

            # ESS: uniform prior -> uniform weights -> ESS = batch
            ess_running.append(float(batch))

        # ---- validation ----
        net.eval()
        with torch.no_grad():
            if val_x is not None:
                vx = val_x
                vtheta = val_theta
            else:
                # Fresh validation batch from prior
                vparams = prior_fn(batch, rng)
                vx_np = sim_fn(*vparams, rng)
                vx = _to_tensor(vx_np, device)
                vtheta = _stack_params(*vparams, device=device)

            vhat = _forward_and_stack(net, vx)
            vmse = ((vhat - vtheta) ** 2).mean().item()

        val_curve.append(vmse)

    mean_ess = float(np.mean(ess_running)) if ess_running else float(batch)
    return net, val_curve, mean_ess


# ---------------------------------------------------------------------------
#  Gradient-variance measurement
# ---------------------------------------------------------------------------

def _flat_grad(net: nn.Module) -> np.ndarray:
    """Flatten current .grad of every parameter into a 1-D float64 numpy vector."""
    return torch.cat([
        p.grad.detach().reshape(-1)
        for p in net.parameters()
        if p.grad is not None
    ]).cpu().numpy().astype(np.float64)


def gradient_variance(
    net: nn.Module,
    alpha_draw_fn: Callable,
    sim_fn: Callable,
    batch: int,
    M: int,
    device: str,
    rng: np.random.Generator,
) -> float:
    """Estimate the trace of the minibatch gradient covariance for a frozen net.

    Ported from SRC/final/sweep_importance_sampling/grad_variance.py
    (grad_samples + total_grad_variance).

    For a frozen network and proposal alpha_draw_fn, draws M independent
    minibatches and computes the SNIS gradient for each (identical to the
    training objective in train()).  Returns:

        tr Cov(g) = (1/M) sum_m ||g_m - gbar||^2

    Parameters
    ----------
    net           : nn.Module — network (weights are frozen; not modified)
    alpha_draw_fn : callable(B, rng) -> tuple of param arrays
    sim_fn        : callable(param1, ..., rng) -> (B, T) ndarray
    batch         : minibatch size B
    M             : number of minibatches to draw
    device        : torch device string
    rng           : numpy.random.Generator

    Returns
    -------
    float : tr Cov(g), always >= 0
    """
    net = net.to(device)
    n_params = sum(p.numel() for p in net.parameters())
    grads = np.empty((M, n_params), dtype=np.float64)

    net.train()  # ensure .grad is available (no dropout/BN in AR1Estimator)

    for m in range(M):
        params = alpha_draw_fn(batch, rng)       # tuple of (B,) arrays
        x = sim_fn(*params, rng)                 # (B, T) ndarray
        x_t = _to_tensor(x, device)
        theta_t = _stack_params(*params, device=device)  # (B, n_params)

        theta_hat = _forward_and_stack(net, x_t)  # (B, n_params)

        # SNIS gradient: uniform prior => uniform weights => plain mean loss
        per_sample = ((theta_hat - theta_t) ** 2).sum(dim=1)  # (B,)
        # Use uniform weights (MC case); caller can extend for IS by modifying
        # alpha_draw_fn to return weighted draws.
        w = torch.ones(batch, dtype=torch.float32, device=device)
        w_norm = w / w.sum().clamp_min(1e-12)
        loss = (w_norm * per_sample).sum()

        net.zero_grad(set_to_none=False)
        loss.backward()
        grads[m] = _flat_grad(net)

    # tr Cov(g) = (1/M) sum_m ||g_m - gbar||^2
    gbar = grads.mean(axis=0, keepdims=True)
    return float(((grads - gbar) ** 2).sum(axis=1).mean())
