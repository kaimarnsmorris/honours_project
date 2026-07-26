import numpy as np
import matplotlib.pyplot as plt
import torch


def evaluate(model, sample_fn, history, param_names, n_test=2000):
    """Single-model evaluation: learning curve + scatter plots + MSE."""
    device = next(model.parameters()).device
    n_params = len(param_names)

    model.eval()
    with torch.no_grad():
        sample = sample_fn(n_test)
        theta_test = _to_device(sample[0], device)
        x_test = _to_device(sample[1], device)
        pred = model(x_test)

    theta_true = theta_test.cpu().numpy()
    theta_pred = pred.cpu().numpy()

    fig, axes = plt.subplots(1, 1 + n_params, figsize=(5 * (1 + n_params), 4))

    # Learning curve
    axes[0].plot(history)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Test Loss')
    axes[0].set_title('Test Loss over Epochs')
    axes[0].set_yscale('log')

    # Scatter plots
    for i, name in enumerate(param_names):
        ax = axes[1 + i]
        ax.scatter(theta_true[:, i], theta_pred[:, i], alpha=0.3, s=10)
        lo = min(theta_true[:, i].min(), theta_pred[:, i].min())
        hi = max(theta_true[:, i].max(), theta_pred[:, i].max())
        ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Predicted {name}')
        ax.set_title(f'{name}: Predicted vs True')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()

    mse = np.mean((theta_true - theta_pred) ** 2, axis=0)
    for i, name in enumerate(param_names):
        print(f'  MSE({name}): {mse[i]:.4f}')
    print(f'  Total MSE: {mse.sum():.4f}')


def compare_models(models, histories, test_set, param_names, param_ranges=None):
    """Multi-model comparison: learning curves, MSE table, binned MSE.

    Args:
        models: dict {name: nn.Module}
        histories: dict {name: [losses]}
        test_set: (theta_tensor, x_tensor) — shared uniform test data
        param_names: list of str
        param_ranges: list of (lo, hi) per param, or None to infer
    """
    device = next(next(iter(models.values())).parameters()).device
    theta_test = _to_device(test_set[0], device)
    x_test = _to_device(test_set[1], device)
    theta_true = theta_test.cpu().numpy()
    n_params = len(param_names)

    # Get predictions
    predictions = {}
    for name, model in models.items():
        model.eval()
        with torch.no_grad():
            predictions[name] = model(x_test).cpu().numpy()

    # 1. Overlaid learning curves
    plt.figure(figsize=(8, 5))
    for name, history in histories.items():
        plt.plot(history, label=name)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Test MSE (log)')
    plt.title('Test Loss Comparison')
    plt.legend()
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.show()

    # 2. MSE table
    print(f"{'Model':25s}", end='')
    for name in param_names:
        print(f" {'MSE(' + name + ')':>12s}", end='')
    print(f" {'Total':>10s}")
    print('-' * (25 + 12 * n_params + 10))

    for name, pred in predictions.items():
        mse_per = np.mean((pred - theta_true) ** 2, axis=0)
        print(f"{name:25s}", end='')
        for m in mse_per:
            print(f" {m:12.4f}", end='')
        print(f" {mse_per.sum():10.4f}")

    # 3. Binned MSE plots
    if param_ranges is None:
        param_ranges = [
            (theta_true[:, i].min(), theta_true[:, i].max())
            for i in range(n_params)
        ]

    n_bins = 10
    fig, axes = plt.subplots(n_params, n_params, figsize=(6 * n_params, 5 * n_params),
                             squeeze=False)

    for bin_by in range(n_params):
        lo, hi = param_ranges[bin_by]
        edges = np.linspace(lo, hi, n_bins + 1)
        centres = (edges[:-1] + edges[1:]) / 2

        for param_idx in range(n_params):
            ax = axes[param_idx, bin_by]
            for name, pred in predictions.items():
                mse_bins = np.zeros(n_bins)
                for b in range(n_bins):
                    mask = (theta_true[:, bin_by] >= edges[b]) & \
                           (theta_true[:, bin_by] < edges[b + 1])
                    if mask.sum() > 0:
                        mse_bins[b] = np.mean(
                            (pred[mask, param_idx] - theta_true[mask, param_idx]) ** 2
                        )
                ax.plot(centres, mse_bins, 'o-', label=name, markersize=4)
            ax.set_xlabel(f'True {param_names[bin_by]}')
            ax.set_ylabel(f'MSE({param_names[param_idx]})')
            ax.set_title(f'MSE({param_names[param_idx]}) binned by {param_names[bin_by]}')
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def binned_mse(theta_true, predictions, param_names, bin_by=0, n_bins=10):
    """Lower-level binned MSE for specific analysis.

    Args:
        theta_true: numpy array (N, n_params)
        predictions: dict {name: numpy array (N, n_params)}
        param_names: list of str
        bin_by: param index to bin by
        n_bins: number of bins
    """
    n_params = len(param_names)
    lo, hi = theta_true[:, bin_by].min(), theta_true[:, bin_by].max()
    edges = np.linspace(lo, hi, n_bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 4), squeeze=False)

    for param_idx in range(n_params):
        ax = axes[0, param_idx]
        for name, pred in predictions.items():
            mse_bins = np.zeros(n_bins)
            for b in range(n_bins):
                mask = (theta_true[:, bin_by] >= edges[b]) & \
                       (theta_true[:, bin_by] < edges[b + 1])
                if mask.sum() > 0:
                    mse_bins[b] = np.mean(
                        (pred[mask, param_idx] - theta_true[mask, param_idx]) ** 2
                    )
            ax.plot(centres, mse_bins, 'o-', label=name, markersize=4)
        ax.set_xlabel(f'True {param_names[bin_by]}')
        ax.set_ylabel(f'MSE({param_names[param_idx]})')
        ax.set_title(f'MSE({param_names[param_idx]}) binned by {param_names[bin_by]}')
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def heatmap_2d(theta_true, predictions, param_names, param_ranges, n_bins=15):
    """2D MSE heatmap over parameter space.

    Args:
        theta_true: numpy array (N, 2)
        predictions: dict {name: numpy array (N, 2)}
        param_names: list of 2 str
        param_ranges: [(lo, hi), (lo, hi)]
        n_bins: grid resolution
    """
    n_models = len(predictions)
    edges_0 = np.linspace(param_ranges[0][0], param_ranges[0][1], n_bins + 1)
    edges_1 = np.linspace(param_ranges[1][0], param_ranges[1][1], n_bins + 1)

    ri = np.clip(np.digitize(theta_true[:, 0], edges_0) - 1, 0, n_bins - 1)
    si = np.clip(np.digitize(theta_true[:, 1], edges_1) - 1, 0, n_bins - 1)

    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for idx, (name, pred) in enumerate(predictions.items()):
        total_sq_err = ((pred - theta_true) ** 2).sum(axis=1)
        grid = np.full((n_bins, n_bins), np.nan)
        for r in range(n_bins):
            for s in range(n_bins):
                mask = (ri == r) & (si == s)
                if mask.sum() > 2:
                    grid[s, r] = np.mean(total_sq_err[mask])

        im = axes[idx].imshow(
            grid, origin='lower', aspect='auto',
            extent=[param_ranges[0][0], param_ranges[0][1],
                    param_ranges[1][0], param_ranges[1][1]],
            cmap='viridis'
        )
        axes[idx].set_xlabel(param_names[0])
        axes[idx].set_ylabel(param_names[1])
        axes[idx].set_title(name)
        plt.colorbar(im, ax=axes[idx], label='Total MSE')

    plt.suptitle(f'MSE across ({param_names[0]}, {param_names[1]}) space', fontsize=14)
    plt.tight_layout()
    plt.show()


def _to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.float().to(device)
    return torch.tensor(x, dtype=torch.float32).to(device)
