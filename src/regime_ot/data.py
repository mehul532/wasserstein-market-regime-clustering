"""Market data loading and caching utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalize_tickers(tickers: str | Iterable[str]) -> list[str]:
    if isinstance(tickers, str):
        return [tickers.upper()]
    normalized = [ticker.upper() for ticker in tickers]
    if not normalized:
        raise ValueError("at least one ticker is required")
    return normalized


def download_prices(
    tickers: str | Iterable[str],
    start: str,
    end: str | None = None,
    source: str = "yfinance",
) -> pd.DataFrame:
    """Download adjusted close prices for one or more tickers.

    The default yfinance path uses adjusted prices so returns include splits
    and dividends when those adjustments are available from Yahoo Finance.
    """
    normalized = _normalize_tickers(tickers)
    if source != "yfinance":
        raise ValueError("only source='yfinance' is currently supported")

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for download_prices; install requirements.txt "
            "or create the conda environment from environment.yml"
        ) from exc

    raw = yf.download(
        normalized,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("no price data returned")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("downloaded data does not contain Close prices")
        prices = raw["Close"].copy()
    else:
        if "Close" not in raw:
            raise ValueError("downloaded data does not contain Close prices")
        prices = raw[["Close"]].copy()
        prices.columns = normalized

    prices = prices.sort_index()
    prices.index.name = "Date"
    return prices.dropna(how="all")


def load_cached_prices(path: str | Path) -> pd.DataFrame:
    """Load cached prices from a CSV or Parquet file."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, index_col=0, parse_dates=True)


def save_prices(df: pd.DataFrame, path: str | Path) -> None:
    """Save prices to CSV or Parquet, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path)
    else:
        df.to_csv(path)
