# Wasserstein Market Regime Clustering

**K-means on full return distributions, not just mean and volatility.**

This project is a Python research-reproduction workspace for market regime
clustering with Wasserstein distance and optimal transport ideas. The core
model treats each rolling window of returns as an empirical probability
distribution, then clusters those distributions directly with a custom
Wasserstein k-means implementation.

The project is designed as a polished quant/data science portfolio repo:
readable math, tested implementation, baseline comparisons, diagnostic plots,
and an illustrative regime-aware backtest with explicit look-ahead controls.

## Research Background

The implementation is motivated by:

1. Horvath, Issa, and Muguruza, *Clustering Market Regimes using the Wasserstein Distance*
2. Zhuang, Chen, and Yang, *Wasserstein K-means for clustering probability distributions*
3. Related optimal transport methods for distributional time-series clustering

Traditional regime models often reduce each window to summary statistics such
as mean, volatility, skew, or drawdown. Those features are useful baselines, but
they discard distribution shape. Two windows can share similar volatility while
having very different downside tails, asymmetry, or quantile structure.

This repo keeps the full empirical distribution. A 63-day return window is not
converted into a scalar feature vector for the main model; it remains a set of
63 observed returns.

## Methodology

For a ticker such as SPY:

1. Download adjusted close prices.
2. Compute log returns.
3. Build overlapping rolling return windows.
4. Treat each window as an empirical distribution.
5. Compare windows using 1D Wasserstein distance.
6. Cluster windows with custom Wasserstein k-means.
7. Compare against moment-feature KMeans and Gaussian mixture baselines.
8. Evaluate regimes with distance metrics, transition matrices, centroid
   quantile plots, price overlays, and an illustrative walk-forward backtest.

For equal-length 1D empirical distributions, the optimal transport matching is
simple: sort both samples and compare equal quantile ranks.

```python
W_2(x, y) = sqrt(mean((sort(x) - sort(y)) ** 2))
W_1(x, y) = mean(abs(sort(x) - sort(y)))
```

The Wasserstein barycenter is implemented as the average sorted quantile
function across all distributions assigned to a cluster. Each centroid is
therefore a sorted return distribution, not a mean/variance vector.

## Project Layout

```text
src/regime_ot/
  data.py          # download and cache adjusted close prices
  returns.py       # log/simple returns and cleaning
  windows.py       # rolling empirical distributions
  wasserstein.py   # 1D distance, pairwise matrix, barycenter
  wkmeans.py       # custom WassersteinKMeans
  baselines.py     # moment KMeans and GMM baselines
  evaluation.py    # cluster stats, silhouettes, transitions
  plotting.py      # matplotlib regime diagnostics
  backtest.py      # illustrative walk-forward regime strategy
  cli.py           # command-line workflows
```

## Setup

Python 3.11+ is the intended runtime.

```bash
conda env create -f environment.yml
conda activate regime-ot
pip install -e .
```

If you prefer pip only:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## How To Run

Download market data:

```bash
python -m regime_ot.cli download --tickers SPY QQQ TLT GLD --start 2000-01-01
```

Run the primary SPY Wasserstein clustering workflow:

```bash
python -m regime_ot.cli run-wkmeans --ticker SPY --window 63 --clusters 3
```

Run the script version:

```bash
python scripts/run_spy_experiment.py --ticker SPY --window 63 --clusters 3
```

Run tests:

```bash
pytest
```

## Results And Outputs

The SPY experiment writes figures to `reports/figures/`, including:

- SPY price with regime-colored background spans
- Wasserstein centroid quantile functions
- Regime transition matrix

The notebook `notebooks/02_wasserstein_regime_clustering.ipynb` runs k=2, k=3,
and k=4, compares silhouette scores, plots regimes over price, visualizes
centroid distributions, and compares the Wasserstein model against KMeans on
moment features.

## Backtest Caveat

The backtest is illustrative, not predictive. Regime labels are unsupervised and
do not guarantee future returns. The implementation shifts positions so labels
can affect only subsequent returns, and the default high-risk regime selection
is walk-forward: it uses only volatility observed up to each decision date.

This project is not investment advice.

## Limitations

- The first implementation focuses on single-asset SPY workflows.
- Multi-asset scripts are a lightweight extension, not a full cross-sectional
  allocation model.
- 1D Wasserstein distance is appropriate for univariate return windows; richer
  multivariate transport would require additional modeling choices.
- Clustering is unsupervised, so regime names such as "low risk" or "high risk"
  are interpretations based on diagnostics, not labels learned from future
  outcomes.
- Data quality depends on Yahoo Finance through `yfinance`.

## Git Workflow

- Work on `main` unless a separate feature branch is needed.
- Commit after coherent milestones and passing checks.
- Do not commit secrets, virtual environments, caches, raw market data, or large
  generated artifacts.
- Do not push unless explicitly requested.
