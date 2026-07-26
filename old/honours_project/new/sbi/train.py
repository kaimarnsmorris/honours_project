import numpy as np
import torch
import torch.nn as nn


def train(model, sample_fn, n_epochs=100, n_batches=100, batch_size=128,
          lr=1e-3, test_set=None, test_size=2000, clip_grad=1.0, seed=42,
          verbose=True, normalize_weights=True):
    """Unified training loop. Auto-detects IS when sample_fn returns 3 values.

    normalize_weights: if True, use self-normalized IS: sum(w*L)/sum(w).
                       if False, use unnormalized IS: mean(w*L).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Detect IS mode
    probe = sample_fn(2)
    is_mode = len(probe) == 3

    # Build test set from uniform samples (strip weights if IS)
    if test_set is None:
        test_sample = sample_fn(test_size)
        t_test = _to_tensor(test_sample[0]).to(device)
        x_test = _to_tensor(test_sample[1]).to(device)
    else:
        t_test = _to_tensor(test_set[0]).to(device)
        x_test = _to_tensor(test_set[1]).to(device)

    loss_fn = nn.MSELoss()
    test_loss_history = []
    print_every = max(1, n_epochs // 8)

    if verbose:
        print(f'Training — device: {device}, IS: {is_mode}, epochs: {n_epochs}')

    for epoch in range(n_epochs):
        model.train()
        for _ in range(n_batches):
            sample = sample_fn(batch_size)
            theta_b = _to_tensor(sample[0]).to(device)
            x_b = _to_tensor(sample[1]).to(device)

            optimizer.zero_grad()
            pred = model(x_b)

            if is_mode:
                weights = _to_tensor(sample[2]).to(device)
                sq_err = ((pred - theta_b) ** 2).sum(dim=1)
                if normalize_weights:
                    loss = (weights * sq_err).sum() / weights.sum()
                else:
                    loss = (weights * sq_err).mean()
            else:
                loss = loss_fn(pred, theta_b)

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            if clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            test_loss = loss_fn(model(x_test), t_test).item()
            test_loss_history.append(test_loss)

        if verbose and (epoch + 1) % print_every == 0:
            print(f'  Epoch {epoch+1}/{n_epochs}, Test Loss: {test_loss:.4f}')

    return model, test_loss_history


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float()
    return torch.tensor(x, dtype=torch.float32)
