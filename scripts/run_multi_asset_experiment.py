"""Run the Wasserstein clustering workflow across multiple tickers."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", nargs="+", default=["SPY", "QQQ", "TLT", "GLD"])
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cache", default="data/raw/multi_asset_prices.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from regime_ot.data import download_prices, load_cached_prices, save_prices
    from regime_ot.evaluation import silhouette_from_distance_matrix
    from regime_ot.returns import clean_returns, compute_log_returns
    from regime_ot.wasserstein import pairwise_wasserstein_distance
    from regime_ot.windows import make_rolling_windows
    from regime_ot.wkmeans import WassersteinKMeans

    cache = REPO_ROOT / args.cache
    if cache.exists():
        prices = load_cached_prices(cache)
    else:
        prices = download_prices(args.tickers, start=args.start, end=args.end)
        save_prices(prices, cache)

    for ticker in args.tickers:
        if ticker not in prices:
            print(f"Skipping {ticker}: not found in price data")
            continue
        returns = clean_returns(compute_log_returns(prices[ticker])).dropna()
        windows, _ = make_rolling_windows(returns, window_size=args.window, step=args.step)
        model = WassersteinKMeans(
            n_clusters=args.clusters,
            random_state=42,
            n_init=10,
        ).fit(windows)
        distances = pairwise_wasserstein_distance(windows)
        silhouette = silhouette_from_distance_matrix(distances, model.labels_)
        print(
            f"{ticker}: windows={len(windows)} "
            f"inertia={model.inertia_:.8f} silhouette={silhouette:.4f}"
        )


if __name__ == "__main__":
    main()
