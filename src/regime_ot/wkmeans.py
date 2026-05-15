"""Wasserstein k-means for equal-length 1D empirical distributions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from regime_ot.wasserstein import wasserstein_barycenter_1d


@dataclass
class _FitResult:
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    n_iter: int


class WassersteinKMeans:
    """K-means clustering where points and centroids are 1D distributions.

    Each input row is a return-window empirical distribution. Distances are
    p-Wasserstein distances between sorted samples, and centroid updates use
    the 1D Wasserstein barycenter, i.e. the average sorted quantile function.
    """

    def __init__(
        self,
        n_clusters: int = 3,
        p: int | float = 2,
        max_iter: int = 100,
        tol: float = 1e-4,
        random_state: int | None = None,
        n_init: int = 10,
    ) -> None:
        if n_clusters <= 0:
            raise ValueError("n_clusters must be positive")
        if p < 1:
            raise ValueError("p must be at least 1")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol < 0:
            raise ValueError("tol must be non-negative")
        if n_init <= 0:
            raise ValueError("n_init must be positive")

        self.n_clusters = int(n_clusters)
        self.p = float(p)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.random_state = random_state
        self.n_init = int(n_init)

        self.inertia_: float | None = None
        self.labels_: np.ndarray | None = None
        self.centroids_: np.ndarray | None = None
        self.n_iter_: int | None = None

    def fit(self, windows: np.ndarray) -> "WassersteinKMeans":
        """Fit the Wasserstein k-means model."""
        sorted_windows = self._validate_windows(windows)
        n_samples = sorted_windows.shape[0]
        if self.n_clusters > n_samples:
            raise ValueError("n_clusters cannot exceed the number of windows")

        rng = np.random.default_rng(self.random_state)
        best: _FitResult | None = None
        for _ in range(self.n_init):
            result = self._fit_once(sorted_windows, rng)
            if best is None or result.inertia < best.inertia:
                best = result

        if best is None:
            raise RuntimeError("failed to fit WassersteinKMeans")

        self.labels_ = best.labels
        self.centroids_ = best.centroids
        self.inertia_ = best.inertia
        self.n_iter_ = best.n_iter
        return self

    def predict(self, windows: np.ndarray) -> np.ndarray:
        """Assign windows to the nearest fitted Wasserstein centroid."""
        if self.centroids_ is None:
            raise ValueError("WassersteinKMeans must be fitted before predict")
        sorted_windows = self._validate_windows(windows)
        distances = self._distances_to_centroids(sorted_windows, self.centroids_)
        return np.argmin(distances, axis=1)

    def fit_predict(self, windows: np.ndarray) -> np.ndarray:
        """Fit the model and return cluster labels."""
        return self.fit(windows).labels_.copy()

    def _fit_once(self, sorted_windows: np.ndarray, rng: np.random.Generator) -> _FitResult:
        indices = rng.choice(sorted_windows.shape[0], size=self.n_clusters, replace=False)
        centroids = sorted_windows[indices].copy()
        labels = np.full(sorted_windows.shape[0], -1, dtype=int)
        inertia = np.inf

        for iteration in range(1, self.max_iter + 1):
            distances = self._distances_to_centroids(sorted_windows, centroids)
            new_labels = np.argmin(distances, axis=1)
            new_inertia = float(np.sum(np.min(distances, axis=1) ** 2))

            new_centroids = np.empty_like(centroids)
            for cluster in range(self.n_clusters):
                members = sorted_windows[new_labels == cluster]
                if len(members) == 0:
                    new_centroids[cluster] = sorted_windows[
                        rng.integers(0, sorted_windows.shape[0])
                    ]
                else:
                    new_centroids[cluster] = wasserstein_barycenter_1d(members)

            centroid_shift = float(
                np.max(self._paired_centroid_distance(centroids, new_centroids))
            )
            labels_changed = not np.array_equal(labels, new_labels)
            labels = new_labels
            centroids = new_centroids
            inertia = new_inertia

            if not labels_changed or centroid_shift <= self.tol:
                break

        return _FitResult(labels=labels, centroids=centroids, inertia=inertia, n_iter=iteration)

    def _distances_to_centroids(
        self,
        sorted_windows: np.ndarray,
        sorted_centroids: np.ndarray,
    ) -> np.ndarray:
        diff = np.abs(sorted_windows[:, None, :] - sorted_centroids[None, :, :])
        return np.mean(diff**self.p, axis=2) ** (1.0 / self.p)

    def _paired_centroid_distance(
        self,
        old_centroids: np.ndarray,
        new_centroids: np.ndarray,
    ) -> np.ndarray:
        diff = np.abs(old_centroids - new_centroids)
        return np.mean(diff**self.p, axis=1) ** (1.0 / self.p)

    @staticmethod
    def _validate_windows(windows: np.ndarray) -> np.ndarray:
        arr = np.asarray(windows, dtype=float)
        if arr.ndim != 2:
            raise ValueError("windows must be a two-dimensional array")
        if arr.shape[0] == 0 or arr.shape[1] == 0:
            raise ValueError("windows must not be empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError("windows contains non-finite values")
        return np.sort(arr, axis=1)
