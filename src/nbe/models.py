"""nbe.models — neural estimator architectures for the honours thesis.

Exports
-------
AR1Estimator      GRU-based estimator for AR(1) data; forward(x:(B,T)) -> (rho:(B,), sigma:(B,))
DeepSetsEstimator Permutation-invariant estimator; forward(y:(B,n,in_dim)) -> theta:(B,out_dim)
NormalNBE         DeepSetsEstimator pre-configured for Normal(mu, sigma) with softplus on sigma
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["AR1Estimator", "DeepSetsEstimator", "NormalNBE"]


# ---------------------------------------------------------------------------
#  AR(1) Estimator — ported verbatim from
#  SRC/final/sweep_importance_sampling/run.py lines 145-166
# ---------------------------------------------------------------------------
class AR1Estimator(nn.Module):
    """GRU encoder -> mean-pool -> MLP -> (rho, sigma) heads.

    Parameters
    ----------
    hidden : int   GRU hidden size
    mlp    : int   MLP hidden size
    A      : float unused amplitude parameter kept for API compatibility
    """

    def __init__(self, hidden: int = 64, mlp: int = 64, A: float = 2.0):
        super().__init__()
        self.A = A
        self.gru = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden, mlp), nn.ReLU(),
            nn.Linear(mlp, mlp), nn.ReLU(),
        )
        self.rho_head = nn.Linear(mlp, 1)
        self.sig_head = nn.Linear(mlp, 1)

    def forward(self, x: torch.Tensor):
        """
        Parameters
        ----------
        x : (B, T) raw AR(1) series (no standardisation — see NOTE in source)

        Returns
        -------
        rho : (B,)  in (-1, 1) via tanh
        sig : (B,)  in (0, inf) via softplus
        """
        # x: (B, T) -> (B, T, 1)
        h, _ = self.gru(x.unsqueeze(-1))
        pooled = h.mean(dim=1)            # mean-pool hidden states
        z = self.head(pooled)
        rho = torch.tanh(self.rho_head(z)).squeeze(-1)          # (-1, 1)
        sig = F.softplus(self.sig_head(z)).squeeze(-1)          # (0, inf)
        return rho, sig


# ---------------------------------------------------------------------------
#  DeepSets Estimator — permutation-invariant architecture
# ---------------------------------------------------------------------------
def _make_mlp(in_dim: int, hidden: int, out_dim: int, depth: int) -> nn.Sequential:
    """Build a depth-layer MLP: in_dim -> hidden (x depth-1) -> out_dim."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.ReLU()]
    for _ in range(depth - 1):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*layers)


class DeepSetsEstimator(nn.Module):
    """Permutation-invariant estimator via the DeepSets architecture.

    Architecture
    ------------
    1. Per-element MLP phi applied independently to each element of the set.
    2. Mean-pool over the set (sample) axis — this is what guarantees
       permutation invariance.
    3. MLP head rho applied to the pooled representation.

    Parameters
    ----------
    in_dim      : int  dimension of each set element
    out_dim     : int  dimension of the output (parameter vector)
    phi_hidden  : int  hidden width of the per-element MLP
    phi_depth   : int  number of hidden layers in phi (excluding I/O)
    rho_hidden  : int  hidden width of the aggregation MLP head
    rho_depth   : int  number of hidden layers in rho (excluding I/O)
    latent_dim  : int  dimension of the per-element embedding (phi output)
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        phi_hidden: int = 64,
        phi_depth: int = 2,
        rho_hidden: int = 64,
        rho_depth: int = 2,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.phi = _make_mlp(in_dim, phi_hidden, latent_dim, phi_depth)
        self.rho = _make_mlp(latent_dim, rho_hidden, out_dim, rho_depth)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y : (B, n, in_dim)  batch of sets, each of n elements

        Returns
        -------
        theta : (B, out_dim)
        """
        # Apply phi element-wise: (B, n, in_dim) -> (B, n, latent_dim)
        z = self.phi(y)
        # Mean-pool over the set axis (dim=1): (B, n, latent_dim) -> (B, latent_dim)
        # This is the operation that enforces permutation invariance.
        pooled = z.mean(dim=1)
        # Apply rho to the aggregated representation: (B, latent_dim) -> (B, out_dim)
        return self.rho(pooled)


# ---------------------------------------------------------------------------
#  NormalNBE — DeepSets configured for Normal(mu, sigma) estimation
# ---------------------------------------------------------------------------
class NormalNBE(DeepSetsEstimator):
    """DeepSets estimator for Normal(mu, sigma) from i.i.d. scalar observations.

    Input  : y of shape (B, n, 1)  — batch of n scalar observations
    Output : (B, 2) where column 0 = mu (unconstrained),
                                column 1 = sigma (positive, via softplus)

    The softplus on sigma is applied in forward() so the raw network output
    for the sigma head is unconstrained, and positivity is enforced post-hoc.
    """

    def __init__(self, **kwargs):
        # Force in_dim=1 (scalar observations), out_dim=2 (mu + sigma)
        kwargs.setdefault("in_dim", 1)
        kwargs.setdefault("out_dim", 2)
        if kwargs["in_dim"] != 1 or kwargs["out_dim"] != 2:
            raise ValueError("NormalNBE requires in_dim=1, out_dim=2")
        super().__init__(**kwargs)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        y : (B, n, 1)

        Returns
        -------
        out : (B, 2)  — [mu, sigma] where sigma > 0
        """
        raw = super().forward(y)                      # (B, 2)
        mu = raw[:, 0:1]                              # (B, 1) unconstrained
        sigma = F.softplus(raw[:, 1:2])               # (B, 1) positive
        return torch.cat([mu, sigma], dim=1)          # (B, 2)
