"""nbe.plotting — house-style matplotlib helpers for the honours thesis.

Each function returns a ``matplotlib.figure.Figure`` and, if *out* is given,
saves it to that path (PNG at 150 dpi).

Colour palette mirrors the thesis sweep figures:
  - graded colours: ``plt.cm.viridis(np.linspace(0.1, 0.85, n))``
  - MC baseline: ``tab:blue``
  - RB method: ``tab:orange``
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.figure

__all__ = ["est_vs_true", "graded_bars", "convergence_curves"]

_DPI = 150


def _savefig(fig: matplotlib.figure.Figure, out: Optional[str]) -> None:
    """Save *fig* to *out* if a path is given; close nothing (caller decides)."""
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=_DPI, bbox_inches="tight")


# ---------------------------------------------------------------------------
#  est_vs_true
# ---------------------------------------------------------------------------

def est_vs_true(
    true: np.ndarray,
    est_dict: Dict[str, np.ndarray],
    names: List[str],
    out: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Scatter estimated vs true parameter values with a y = x reference line.

    Parameters
    ----------
    true:
        1-D array of ground-truth values (x-axis).
    est_dict:
        Mapping from estimator name to a 1-D array of estimates (same length
        as *true*).
    names:
        Ordered list of keys from *est_dict* to plot (controls legend order).
    out:
        Optional file path.  If given the figure is saved there.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(names)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(n, 1)))

    fig, ax = plt.subplots(figsize=(5.0, 4.5))

    vmin, vmax = float(true.min()), float(true.max())
    pad = (vmax - vmin) * 0.05
    lim = (vmin - pad, vmax + pad)

    # y = x identity line
    ax.plot(lim, lim, color="grey", linestyle="--", linewidth=1.0,
            zorder=0, label="_nolegend_")

    for i, name in enumerate(names):
        est = est_dict[name]
        ax.scatter(true, est, s=18, alpha=0.7, color=colors[i],
                   edgecolors="none", label=name, zorder=2)

    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("True")
    ax.set_ylabel("Estimated")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    _savefig(fig, out)
    return fig


# ---------------------------------------------------------------------------
#  graded_bars
# ---------------------------------------------------------------------------

def graded_bars(
    labels: Sequence[str],
    values: Sequence[float],
    errs: Optional[Sequence[float]] = None,
    out: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Bar chart with viridis-graded colours (one colour per bar).

    Parameters
    ----------
    labels:
        Category labels for the x-axis.
    values:
        Bar heights.
    errs:
        Optional error bar half-widths (same length as *values*).
    out:
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(labels)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(n, 1)))
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(4.0, 0.9 * n + 1.5), 4.2))

    ax.bar(
        x,
        values,
        yerr=errs,
        capsize=4 if errs is not None else 0,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    _savefig(fig, out)
    return fig


# ---------------------------------------------------------------------------
#  convergence_curves
# ---------------------------------------------------------------------------

def convergence_curves(
    samples: np.ndarray,
    curves_by_key: Dict[str, np.ndarray],
    out: Optional[str] = None,
) -> matplotlib.figure.Figure:
    """Log-log convergence plot: error vs number of samples.

    Parameters
    ----------
    samples:
        1-D array of sample counts (x-axis, log-scaled).
    curves_by_key:
        Mapping from method name to a 1-D array of error values (same length
        as *samples*).
    out:
        Optional file path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    keys = list(curves_by_key)
    n = len(keys)
    colors = plt.cm.viridis(np.linspace(0.1, 0.85, max(n, 1)))

    fig, ax = plt.subplots(figsize=(6.0, 4.5))

    for i, key in enumerate(keys):
        ax.plot(samples, curves_by_key[key], color=colors[i],
                linewidth=1.8, marker="o", markersize=4, label=key)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Samples")
    ax.set_ylabel("Error")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.tight_layout()

    _savefig(fig, out)
    return fig
