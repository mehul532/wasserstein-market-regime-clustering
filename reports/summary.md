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

The backtest is illustrative and uses shifted positions so regime labels affect
only subsequent returns. Regime labels are unsupervised and should not be
interpreted as forecasts.
