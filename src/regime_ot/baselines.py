"""Feature-based baseline models for comparison."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "mean",
    "std",
    "skew",
    "kurtosis",
    "min",
    "max",
    "q05",
    "q95",
]


def extract_moment_features(windows: np.ndarray) -> pd.DataFrame:
    """Extract summary statistics for classical feature-based baselines."""
    arr = np.asarray(windows, dtype=float)
    if arr.ndim != 2:
        raise ValueError("windows must be a two-dimensional array")
    if not np.all(np.isfinite(arr)):
        raise ValueError("windows contains non-finite values")

    features = np.column_stack(
        [
            np.mean(arr, axis=1),
            np.std(arr, axis=1, ddof=1),
            skew(arr, axis=1, bias=False),
            kurtosis(arr, axis=1, fisher=True, bias=False),
            np.min(arr, axis=1),
            np.max(arr, axis=1),
            np.quantile(arr, 0.05, axis=1),
            np.quantile(arr, 0.95, axis=1),
        ]
    )
    return pd.DataFrame(features, columns=FEATURE_NAMES)


def run_standard_kmeans(
    features: pd.DataFrame | np.ndarray,
    n_clusters: int,
    random_state: int | None = 42,
) -> dict:
    """Run sklearn KMeans on moment features as a baseline."""
    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10),
    )
    labels = model.fit_predict(features)
    return {
        "name": "moment_kmeans",
        "model": model,
        "labels": labels,
        "feature_names": list(getattr(features, "columns", FEATURE_NAMES)),
    }


def run_gmm(
    features: pd.DataFrame | np.ndarray,
    n_clusters: int,
    random_state: int | None = 42,
) -> dict:
    """Run a Gaussian mixture model on standardized moment features."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = GaussianMixture(n_components=n_clusters, random_state=random_state)
    labels = model.fit_predict(scaled)
    return {
        "name": "moment_gmm",
        "model": model,
        "scaler": scaler,
        "labels": labels,
        "feature_names": list(getattr(features, "columns", FEATURE_NAMES)),
    }
