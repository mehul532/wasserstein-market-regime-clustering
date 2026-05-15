import numpy as np
import pandas as pd

from regime_ot.backtest import performance_metrics, regime_strategy_backtest


def test_regime_strategy_backtest_shifts_position_after_label():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    label_dates = dates[1:]
    labels = np.array([0, 1, 1, 0])

    results, metrics = regime_strategy_backtest(
        prices,
        labels,
        label_dates,
        high_risk_regime=1,
    )

    assert results.loc[dates[2], "raw_position"] == 0.0
    assert results.loc[dates[2], "position"] == 1.0
    assert results.loc[dates[3], "position"] == 0.0
    assert metrics["turnover"] > 0.0


def test_performance_metrics_returns_expected_keys():
    equity = pd.Series([1.0, 1.01, 1.02, 1.01])

    metrics = performance_metrics(equity)

    assert set(metrics) == {
        "CAGR",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "turnover",
    }
    assert metrics["max_drawdown"] < 0.0
