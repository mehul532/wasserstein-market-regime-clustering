# Wasserstein Market Regime Clustering Summary

This report is the landing page for generated experiment outputs.

Run the primary SPY workflow:

```bash
python scripts/run_spy_experiment.py --ticker SPY --window 63 --clusters 3
```

Expected generated figures:

- `reports/figures/SPY_regimes.png`
- `reports/figures/SPY_centroids.png`
- `reports/figures/SPY_transitions.png`

The regime figures are exploratory, in-sample diagnostics. The script's printed
backtest metrics use the walk-forward helper: centroids are fit only on windows
available at each decision date, the high-risk regime is selected from centroid
dispersion, and positions are shifted so signals affect only subsequent returns.

Regime labels are unsupervised and should not be interpreted as forecasts.

Research basis:

- Horvath, Issa, and Muguruza, *Clustering Market Regimes using the Wasserstein Distance*, 2021.
- Zhuang, Chen, and Yang, *Wasserstein K-means for clustering probability distributions*, NeurIPS 2022.
