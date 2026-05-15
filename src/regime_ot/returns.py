"""Return calculation helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_log_returns(price_df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute log returns from prices."""
    prices = price_df.sort_index()
    return np.log(prices / prices.shift(1))


def compute_simple_returns(price_df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Compute simple percentage returns from prices."""
    return price_df.sort_index().pct_change()


def clean_returns(returns_df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Remove non-finite and missing return observations."""
    cleaned = returns_df.replace([np.inf, -np.inf], np.nan)
    return cleaned.dropna(how="all")
