"""One-dimensional Wasserstein distance utilities.

For one-dimensional empirical distributions with equal sample counts, the
optimal transport matching is obtained by sorting both samples and pairing
equal quantile ranks. The p-Wasserstein distance is therefore the Lp distance
between sorted samples:

    W_p(x, y) = (mean(|sort(x) - sort(y)| ** p)) ** (1 / p)

The 1D barycenter under this representation is the average quantile function,
implemented here by averaging sorted samples across distributions.
"""

from __future__ import annotations

import numpy as np


def _as_1d_finite(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _as_2d_finite(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def _validate_p(p: int | float) -> float:
    p_float = float(p)
    if p_float < 1:
        raise ValueError("p must be at least 1")
    return p_float


def wasserstein_1d(x: np.ndarray, y: np.ndarray, p: int | float = 2) -> float:
    """Compute the 1D empirical p-Wasserstein distance.

    Parameters
    ----------
    x, y:
        Equal-length samples from two empirical distributions.
    p:
        Order of the Wasserstein distance. For example, p=1 gives the average
        absolute sorted-sample gap, and p=2 gives the square root of the mean
        squared sorted-sample gap.
    """
    x_arr = _as_1d_finite(x, "x")
    y_arr = _as_1d_finite(y, "y")
    p_float = _validate_p(p)
    if x_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("x and y must have the same number of samples")

    diff = np.abs(np.sort(x_arr) - np.sort(y_arr))
    return float(np.mean(diff**p_float) ** (1.0 / p_float))


def pairwise_wasserstein_distance(
    windows: np.ndarray,
    p: int | float = 2,
) -> np.ndarray:
    """Compute a symmetric pairwise Wasserstein distance matrix.

    Rows are treated as empirical distributions with equal sample counts.
    Sorting each row once gives the quantile representation used for all
    pairwise distances.
    """
    arr = _as_2d_finite(windows, "windows")
    p_float = _validate_p(p)
    sorted_windows = np.sort(arr, axis=1)
    n_windows = sorted_windows.shape[0]
    distances = np.zeros((n_windows, n_windows), dtype=float)

    for i in range(n_windows):
        diff = np.abs(sorted_windows[i + 1 :] - sorted_windows[i])
        if diff.size == 0:
            continue
        row_distances = np.mean(diff**p_float, axis=1) ** (1.0 / p_float)
        distances[i, i + 1 :] = row_distances
        distances[i + 1 :, i] = row_distances

    return distances


def wasserstein_barycenter_1d(windows: np.ndarray) -> np.ndarray:
    """Approximate a 1D Wasserstein barycenter by averaging quantiles.

    In one dimension, the Wasserstein barycenter has a quantile function equal
    to the weighted average of member quantile functions. With equally weighted
    equal-length empirical windows, this reduces to sorting every window and
    averaging each quantile rank.
    """
    arr = _as_2d_finite(windows, "windows")
    return np.mean(np.sort(arr, axis=1), axis=0)


def quantile_grid(distribution: np.ndarray, n_quantiles: int) -> tuple[np.ndarray, np.ndarray]:
    """Return evenly spaced quantile levels and values for a distribution."""
    if n_quantiles <= 1:
        raise ValueError("n_quantiles must be greater than 1")
    arr = _as_1d_finite(distribution, "distribution")
    levels = np.linspace(0.0, 1.0, n_quantiles)
    return levels, np.quantile(arr, levels)
