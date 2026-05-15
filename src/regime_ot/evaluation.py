"""Regime diagnostics and validation metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from regime_ot.wasserstein import pairwise_wasserstein_distance


def _label_series(labels: np.ndarray, dates: pd.Index | list | None = None) -> pd.Series:
    labels_arr = np.asarray(labels, dtype=int)
    if dates is None:
        return pd.Series(labels_arr, name="label")
    if len(dates) != len(labels_arr):
        raise ValueError("dates and labels must have the same length")
    return pd.Series(labels_arr, index=pd.Index(dates), name="label")


def summarize_regimes(
    labels: np.ndarray,
    returns: pd.Series,
    dates: pd.Index | list,
) -> pd.DataFrame:
    """Summarize realized return behavior at labeled window end dates."""
    label_series = _label_series(labels, dates)
    aligned_returns = returns.reindex(label_series.index)
    frame = pd.DataFrame({"label": label_series, "return": aligned_returns}).dropna()
    grouped = frame.groupby("label")["return"]
    summary = grouped.agg(["count", "mean", "std", "min", "max"])
    summary["annualized_vol"] = summary["std"] * np.sqrt(252)
    summary["start"] = frame.groupby("label").apply(lambda x: x.index.min())
    summary["end"] = frame.groupby("label").apply(lambda x: x.index.max())
    return summary


def compute_cluster_stats(windows: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Compute distributional cluster diagnostics for fitted labels."""
    arr = np.asarray(windows, dtype=float)
    labels_arr = np.asarray(labels, dtype=int)
    if arr.ndim != 2:
        raise ValueError("windows must be a two-dimensional array")
    if len(arr) != len(labels_arr):
        raise ValueError("windows and labels must have the same length")

    rows = []
    for label in np.unique(labels_arr):
        cluster_windows = arr[labels_arr == label]
        flattened = cluster_windows.ravel()
        if len(cluster_windows) > 1:
            distances = pairwise_wasserstein_distance(cluster_windows)
            upper = distances[np.triu_indices_from(distances, k=1)]
            mean_within = float(np.mean(upper))
        else:
            mean_within = 0.0
        rows.append(
            {
                "label": int(label),
                "count": int(len(cluster_windows)),
                "window_mean": float(np.mean(flattened)),
                "window_std": float(np.std(flattened, ddof=1)),
                "q05": float(np.quantile(flattened, 0.05)),
                "q95": float(np.quantile(flattened, 0.95)),
                "mean_within_wasserstein": mean_within,
            }
        )
    return pd.DataFrame(rows).set_index("label")


def silhouette_from_distance_matrix(distance_matrix: np.ndarray, labels: np.ndarray) -> float:
    """Compute silhouette score from a precomputed distance matrix."""
    labels_arr = np.asarray(labels, dtype=int)
    unique = np.unique(labels_arr)
    if len(unique) < 2 or len(unique) >= len(labels_arr):
        return float("nan")
    return float(silhouette_score(distance_matrix, labels_arr, metric="precomputed"))


def transition_matrix(labels: np.ndarray) -> pd.DataFrame:
    """Estimate a row-normalized transition matrix between consecutive labels."""
    labels_arr = np.asarray(labels, dtype=int)
    unique = np.unique(labels_arr)
    counts = pd.DataFrame(0.0, index=unique, columns=unique)
    for current, nxt in zip(labels_arr[:-1], labels_arr[1:]):
        counts.loc[current, nxt] += 1.0
    row_sums = counts.sum(axis=1).replace(0.0, np.nan)
    return counts.div(row_sums, axis=0).fillna(0.0)


def compare_regime_future_returns(
    labels: np.ndarray,
    forward_returns: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """Compare next-period returns by regime label without same-period leakage.

    The label at position t is paired with the return at t+1. This makes the
    function safe when callers pass realized returns indexed like the labels.
    If callers already precomputed forward returns, they should pass a series
    whose value at t is the return realized after t and set the last value to
    missing before calling.
    """
    labels_arr = np.asarray(labels, dtype=int)
    returns = pd.Series(forward_returns).reset_index(drop=True)
    if len(returns) != len(labels_arr):
        raise ValueError("forward_returns and labels must have the same length")

    frame = pd.DataFrame(
        {
            "label": labels_arr[:-1],
            "future_return": returns.shift(-1).iloc[:-1].to_numpy(),
        }
    ).dropna()
    grouped = frame.groupby("label")["future_return"]
    return grouped.agg(["count", "mean", "std", "min", "max"])
