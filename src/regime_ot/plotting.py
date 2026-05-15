"""Matplotlib visualizations for regime clustering outputs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


def _as_label_series(labels: np.ndarray, dates: pd.Index | list) -> pd.Series:
    if len(labels) != len(dates):
        raise ValueError("labels and dates must have the same length")
    return pd.Series(np.asarray(labels, dtype=int), index=pd.Index(dates))


def _shade_regimes(ax, label_series: pd.Series, alpha: float = 0.16) -> None:
    cmap = plt.get_cmap("tab10")
    labels = label_series.to_numpy()
    dates = label_series.index
    for i, label in enumerate(labels):
        start = dates[i - 1] if i > 0 else dates[i]
        end = dates[i]
        ax.axvspan(start, end, color=cmap(int(label) % 10), alpha=alpha, linewidth=0)


def _legend_for_labels(labels: np.ndarray) -> list[Patch]:
    cmap = plt.get_cmap("tab10")
    return [
        Patch(color=cmap(int(label) % 10), alpha=0.35, label=f"Regime {int(label)}")
        for label in np.unique(labels)
    ]


def plot_price_with_regimes(
    price_series: pd.Series,
    labels: np.ndarray,
    dates: pd.Index | list,
):
    """Plot price with regime-colored background spans."""
    label_series = _as_label_series(labels, dates)
    fig, ax = plt.subplots(figsize=(12, 5))
    price_series.sort_index().plot(ax=ax, color="black", linewidth=1.2)
    _shade_regimes(ax, label_series)
    ax.set_title(f"{price_series.name or 'Price'} with Wasserstein regimes")
    ax.set_ylabel("Adjusted close")
    ax.legend(handles=_legend_for_labels(labels), loc="upper left")
    fig.tight_layout()
    return fig, ax


def plot_returns_with_regimes(
    return_series: pd.Series,
    labels: np.ndarray,
    dates: pd.Index | list,
):
    """Plot returns with regime-colored background spans."""
    label_series = _as_label_series(labels, dates)
    fig, ax = plt.subplots(figsize=(12, 4))
    return_series.sort_index().plot(ax=ax, color="black", linewidth=0.8)
    _shade_regimes(ax, label_series)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_title(f"{return_series.name or 'Returns'} with regimes")
    ax.set_ylabel("Return")
    fig.tight_layout()
    return fig, ax


def plot_centroid_distributions(centroids: np.ndarray):
    """Plot centroid quantile functions."""
    centroid_arr = np.asarray(centroids, dtype=float)
    if centroid_arr.ndim != 2:
        raise ValueError("centroids must be a two-dimensional array")
    fig, ax = plt.subplots(figsize=(8, 5))
    quantile_levels = np.linspace(0.0, 1.0, centroid_arr.shape[1])
    for idx, centroid in enumerate(centroid_arr):
        ax.plot(quantile_levels, np.sort(centroid), label=f"Regime {idx}")
    ax.set_title("Wasserstein centroid quantile functions")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("Return")
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_regime_transition_matrix(matrix: pd.DataFrame | np.ndarray):
    """Plot a regime transition matrix heatmap with matplotlib."""
    values = matrix.to_numpy() if isinstance(matrix, pd.DataFrame) else np.asarray(matrix)
    labels = list(matrix.index) if isinstance(matrix, pd.DataFrame) else list(range(values.shape[0]))
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(values, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title("Regime transition matrix")
    ax.set_xlabel("Next regime")
    ax.set_ylabel("Current regime")
    ax.set_xticks(range(len(labels)), labels=labels)
    ax.set_yticks(range(len(labels)), labels=labels)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, f"{values[row, col]:.2f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_baseline_comparison(results: dict):
    """Plot comparable scalar metrics from clustering result dictionaries."""
    names = []
    silhouettes = []
    for name, result in results.items():
        if "silhouette" in result:
            names.append(name)
            silhouettes.append(result["silhouette"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, silhouettes, color="#4C78A8")
    ax.set_title("Clustering silhouette comparison")
    ax.set_ylabel("Silhouette")
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    return fig, ax
