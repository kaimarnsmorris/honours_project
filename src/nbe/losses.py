"""Loss functions for neural Bayes estimator training.

Provides:
    effective_sample_size(w) -> float
        ESS = (sum w)^2 / sum w^2  for a batch of (unnormalised) importance weights.
        Operates on numpy arrays.

    snis_loss(per_sample, w) -> torch.Tensor
        Self-normalised importance-sampling objective: normalise weights within
        the batch, then return the weighted mean of per-sample losses.

    mc_loss(theta_hat, theta) -> torch.Tensor
        Monte-Carlo loss: mean squared error between the estimate and the true
        parameter drawn from the prior.

    rb_loss(theta_hat, posterior_mean) -> torch.Tensor
        Rao-Blackwellised loss: mean squared error between the estimate and the
        conjugate posterior mean (the variance-reduced target).  The posterior
        mean is computed externally (e.g. via a Cholesky solve) and passed in.
"""

from __future__ import annotations

import numpy as np
import torch


def effective_sample_size(w) -> float:
    """ESS = (sum w)^2 / sum w^2  for a batch of (unnormalised) weights.

    Parameters
    ----------
    w : array-like of float
        Unnormalised importance weights.

    Returns
    -------
    float
        Effective sample size.  Returns 0.0 if all weights are zero.
    """
    w = np.asarray(w, dtype=np.float64)
    s1 = w.sum()
    s2 = (w ** 2).sum()
    if s2 <= 0:
        return 0.0
    return float(s1 * s1 / s2)


def snis_loss(per_sample: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Self-normalised IS objective: weights normalised within the batch.

    Parameters
    ----------
    per_sample : torch.Tensor, shape (B,)
        Per-sample loss values (e.g. squared errors).
    w : torch.Tensor, shape (B,)
        Unnormalised importance weights for the same batch.

    Returns
    -------
    torch.Tensor
        Scalar: (w_norm * per_sample).sum()  where  w_norm = w / w.sum().
    """
    w_norm = w / w.sum().clamp_min(1e-12)
    return (w_norm * per_sample).sum()


def mc_loss(theta_hat: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Monte-Carlo Bayes-risk estimate: MSE to the true parameter.

    Parameters
    ----------
    theta_hat : torch.Tensor, shape (B, d) or (B,)
        Network output (estimator).
    theta : torch.Tensor, shape (B, d) or (B,)
        True parameter values sampled from the prior.

    Returns
    -------
    torch.Tensor
        Scalar mean squared error.
    """
    return ((theta_hat - theta) ** 2).mean()


def rb_loss(theta_hat: torch.Tensor, posterior_mean: torch.Tensor) -> torch.Tensor:
    """Rao-Blackwellised loss: MSE to the conjugate posterior mean.

    Replacing the noisy Monte-Carlo target ``theta`` with the posterior mean
    E[theta | data] reduces gradient variance (Rao-Blackwell theorem).

    Parameters
    ----------
    theta_hat : torch.Tensor, shape (B, d) or (B,)
        Network output (estimator).
    posterior_mean : torch.Tensor, shape (B, d) or (B,)
        Conjugate posterior mean, computed externally (e.g. via Cholesky solve).

    Returns
    -------
    torch.Tensor
        Scalar mean squared error between the estimate and the posterior mean.
    """
    return ((theta_hat - posterior_mean) ** 2).mean()
