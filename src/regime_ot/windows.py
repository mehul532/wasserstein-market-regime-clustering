"""Rolling-window construction for empirical return distributions."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def make_rolling_windows(
    return_series: pd.Series,
    window_size: int = 63,
    step: int = 5,
) -> Tuple[np.ndarray, pd.Index]:
    """Create overlapping return windows for distributional clustering.

    Each row in the returned array is one empirical return distribution. The
    corresponding date is the end date of that rolling window, which is the
    timestamp at which the distribution first becomes observable.
    """
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step <= 0:
        raise ValueError("step must be positive")
    if not isinstance(return_series, pd.Series):
        raise TypeError("return_series must be a pandas Series")

    cleaned = return_series.dropna()
    if len(cleaned) < window_size:
        raise ValueError("return_series is shorter than window_size")

    values = cleaned.to_numpy(dtype=float)
    windows = []
    end_positions = []
    for start in range(0, len(values) - window_size + 1, step):
        end = start + window_size
        windows.append(values[start:end])
        end_positions.append(end - 1)

    return np.asarray(windows, dtype=float), cleaned.index[end_positions]
