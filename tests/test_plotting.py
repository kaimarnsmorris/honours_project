"""Smoke tests for nbe.plotting — house-style matplotlib helpers."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import sys
sys.path.insert(0, "src")
import matplotlib.figure

from nbe import plotting


def test_plot_helpers_return_figures(tmp_path):
    t = np.linspace(0, 1, 50)
    f1 = plotting.est_vs_true(t, {"MH": t + 0.01}, ["MH"])
    f2 = plotting.graded_bars(["a", "b", "c"], [1, 2, 3])
    assert f1 is not None and f2 is not None


def test_est_vs_true_returns_figure():
    true = np.random.default_rng(0).uniform(0, 1, 30)
    est_dict = {"Method A": true + 0.05, "Method B": true - 0.03}
    names = ["Method A", "Method B"]
    fig = plotting.est_vs_true(true, est_dict, names)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_graded_bars_returns_figure():
    labels = ["x", "y", "z"]
    values = [1.0, 2.5, 0.8]
    fig = plotting.graded_bars(labels, values)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_graded_bars_with_errors():
    labels = ["x", "y", "z"]
    values = [1.0, 2.5, 0.8]
    errs = [0.1, 0.2, 0.05]
    fig = plotting.graded_bars(labels, values, errs=errs)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_convergence_curves_returns_figure():
    samples = np.array([10, 50, 100, 500, 1000])
    curves = {
        "RB": np.array([0.5, 0.3, 0.2, 0.1, 0.05]),
        "MC": np.array([0.6, 0.4, 0.25, 0.12, 0.07]),
    }
    fig = plotting.convergence_curves(samples, curves)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_est_vs_true_saves_file(tmp_path):
    true = np.linspace(0, 1, 20)
    est_dict = {"Est": true + 0.01}
    out = tmp_path / "scatter.png"
    fig = plotting.est_vs_true(true, est_dict, ["Est"], out=str(out))
    assert out.exists()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_graded_bars_saves_file(tmp_path):
    out = tmp_path / "bars.png"
    fig = plotting.graded_bars(["a", "b"], [1.0, 2.0], out=str(out))
    assert out.exists()
    assert isinstance(fig, matplotlib.figure.Figure)


def test_convergence_curves_saves_file(tmp_path):
    out = tmp_path / "curves.png"
    samples = np.array([10, 100, 1000])
    curves = {"A": np.array([0.5, 0.2, 0.1])}
    fig = plotting.convergence_curves(samples, curves, out=str(out))
    assert out.exists()
    assert isinstance(fig, matplotlib.figure.Figure)
