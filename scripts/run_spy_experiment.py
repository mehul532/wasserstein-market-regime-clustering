"""Run the primary SPY Wasserstein regime clustering experiment."""

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
    parser.add_argument("--ticker", default="SPY")
    parser.add_argument("--start", default="2000-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--clusters", type=int, default=3)
    parser.add_argument("--cache", default=None)
    parser.add_argument("--figures-dir", default="reports/figures")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from regime_ot.backtest import regime_strategy_backtest
    from regime_ot.baselines import extract_moment_features, run_standard_kmeans
    from regime_ot.data import download_prices, load_cached_prices, save_prices
    from regime_ot.evaluation import silhouette_from_distance_matrix, transition_matrix
    from regime_ot.plotting import (
        plot_centroid_distributions,
        plot_price_with_regimes,
        plot_regime_transition_matrix,
    )
    from regime_ot.returns import clean_returns, compute_log_returns
    from regime_ot.wasserstein import pairwise_wasserstein_distance
    from regime_ot.windows import make_rolling_windows
    from regime_ot.wkmeans import WassersteinKMeans

    cache = Path(args.cache) if args.cache else REPO_ROOT / "data" / "raw" / f"{args.ticker}.csv"
    if cache.exists():
        prices = load_cached_prices(cache)
    else:
        prices = download_prices([args.ticker], start=args.start, end=args.end)
        save_prices(prices, cache)

    price_series = prices[args.ticker] if args.ticker in prices else prices.iloc[:, 0]
    returns = clean_returns(compute_log_returns(price_series)).dropna()
    windows, dates = make_rolling_windows(returns, window_size=args.window, step=args.step)

    model = WassersteinKMeans(
        n_clusters=args.clusters,
        random_state=42,
        n_init=10,
    ).fit(windows)
    distance_matrix = pairwise_wasserstein_distance(windows)
    silhouette = silhouette_from_distance_matrix(distance_matrix, model.labels_)

    features = extract_moment_features(windows)
    baseline = run_standard_kmeans(features, n_clusters=args.clusters)
    transitions = transition_matrix(model.labels_)
    backtest, metrics = regime_strategy_backtest(price_series, model.labels_, dates)

    figures_dir = REPO_ROOT / args.figures_dir
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, _ = plot_price_with_regimes(price_series, model.labels_, dates)
    fig.savefig(figures_dir / f"{args.ticker}_regimes.png", dpi=150)
    fig, _ = plot_centroid_distributions(model.centroids_)
    fig.savefig(figures_dir / f"{args.ticker}_centroids.png", dpi=150)
    fig, _ = plot_regime_transition_matrix(transitions)
    fig.savefig(figures_dir / f"{args.ticker}_transitions.png", dpi=150)

    print(f"Wasserstein silhouette: {silhouette:.4f}")
    print(f"Wasserstein inertia: {model.inertia_:.8f}")
    print(f"Moment KMeans regimes: {sorted(set(baseline['labels']))}")
    print(f"Backtest final equity: {backtest['equity_curve'].iloc[-1]:.4f}")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
