import torch


def simulate_ar1_batch(rho, sigma, T, device='cpu'):
    """Vectorised AR(1) simulation with stationary initialisation.

    x_t = rho * x_{t-1} + eps_t,  eps_t ~ N(0, sigma^2)
    x_0 ~ N(0, sigma^2 / (1 - rho^2))

    Args:
        rho: tensor (batch_size,)
        sigma: tensor (batch_size,)
        T: int, time series length
        device: torch device

    Returns:
        x: tensor (batch_size, T)
    """
    batch_size = rho.shape[0]
    eps = torch.randn(batch_size, T, device=device) * sigma[:, None]
    rho_safe = torch.clamp(rho, -0.9999, 0.9999)
    x0 = torch.randn(batch_size, device=device) * sigma / torch.sqrt(1 - rho_safe ** 2 + 1e-8)
    x = torch.zeros(batch_size, T, device=device)
    x[:, 0] = rho * x0 + eps[:, 0]
    for t in range(1, T):
        x[:, t] = rho * x[:, t - 1] + eps[:, t]
    return x
