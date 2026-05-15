"""Illustrative regime-aware backtesting utilities.

These helpers are intentionally conservative about timing: regime labels are
treated as known only at their window end date, and positions are shifted so a
label can affect only the next return observation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _make_label_series(labels: np.ndarray, dates: pd.Index | list) -> pd.Series:
    labels_arr = np.asarray(labels, dtype=int)
    if len(labels_arr) != len(dates):
        raise ValueError("labels and dates must have the same length")
    return pd.Series(labels_arr, index=pd.Index(dates), name="label").sort_index()


def _walk_forward_high_risk_regime(
    returns: pd.Series,
    label_series: pd.Series,
) -> pd.Series:
    high_risk = []
    for date in label_series.index:
        historical_labels = label_series.loc[:date]
        historical_returns = returns.loc[:date]
        daily_labels = historical_labels.reindex(historical_returns.index, method="ffill")
        frame = pd.DataFrame(
            {"return": historical_returns, "label": daily_labels}
        ).dropna()
        vol_by_label = frame.groupby("label")["return"].std().dropna()
        if vol_by_label.empty:
            high_risk.append(int(historical_labels.iloc[-1]))
        else:
            high_risk.append(int(vol_by_label.idxmax()))
    return pd.Series(high_risk, index=label_series.index, name="high_risk_regime")


def walk_forward_regime_strategy_backtest(
    price_series: pd.Series,
    windows: np.ndarray,
    dates: pd.Index | list,
    n_clusters: int = 3,
    min_train_windows: int = 60,
    refit_every: int = 20,
    p: int | float = 2,
    max_iter: int = 100,
    n_init: int = 3,
    random_state: int | None = 42,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run a stricter walk-forward long/cash regime strategy.

    At each decision date, the model is fit only on windows whose end dates are
    not after that decision date. The current observable window is then assigned
    to the nearest historical Wasserstein centroid. The high-risk regime is the
    fitted centroid with the largest return dispersion, and the resulting
    position is shifted so it can affect only subsequent daily returns.

    This is still illustrative and unsupervised, but it avoids the stronger
    in-sample assumption of fitting one clustering model on the full history
    before computing backtest signals.
    """
    from regime_ot.wkmeans import WassersteinKMeans

    prices = price_series.dropna().sort_index().astype(float)
    if prices.empty:
        raise ValueError("price_series must not be empty")

    window_arr = np.asarray(windows, dtype=float)
    if window_arr.ndim != 2:
        raise ValueError("windows must be a two-dimensional array")
    if not np.all(np.isfinite(window_arr)):
        raise ValueError("windows contains non-finite values")

    decision_dates = pd.Index(dates)
    if len(decision_dates) != len(window_arr):
        raise ValueError("dates and windows must have the same length")
    if n_clusters <= 0:
        raise ValueError("n_clusters must be positive")
    if refit_every <= 0:
        raise ValueError("refit_every must be positive")

    min_required = max(int(min_train_windows), int(n_clusters))
    raw_position = pd.Series(0.0, index=decision_dates, name="raw_position")
    online_label = pd.Series(np.nan, index=decision_dates, name="online_label")
    high_risk = pd.Series(np.nan, index=decision_dates, name="high_risk_regime")
    centroid_vol = pd.Series(np.nan, index=decision_dates, name="high_risk_centroid_std")

    model: WassersteinKMeans | None = None
    last_fit_at = -1
    for i, date in enumerate(decision_dates):
        observed_count = i + 1
        if observed_count < min_required:
            continue
        if model is None or observed_count - last_fit_at >= refit_every:
            seed = None if random_state is None else int(random_state) + i
            model = WassersteinKMeans(
                n_clusters=n_clusters,
                p=p,
                max_iter=max_iter,
                n_init=n_init,
                random_state=seed,
            ).fit(window_arr[:observed_count])
            last_fit_at = observed_count

        label = int(model.predict(window_arr[i : i + 1])[0])
        centroid_std = np.std(model.centroids_, axis=1, ddof=1)
        high_risk_label = int(np.nanargmax(centroid_std))
        online_label.loc[date] = label
        high_risk.loc[date] = high_risk_label
        centroid_vol.loc[date] = float(centroid_std[high_risk_label])
        raw_position.loc[date] = 0.0 if label == high_risk_label else 1.0

    asset_returns = prices.pct_change().fillna(0.0)
    daily_raw_position = raw_position.reindex(prices.index, method="ffill").fillna(0.0)
    daily_label = online_label.reindex(prices.index, method="ffill")
    daily_high_risk = high_risk.reindex(prices.index, method="ffill")
    daily_centroid_vol = centroid_vol.reindex(prices.index, method="ffill")

    position = daily_raw_position.shift(1).fillna(0.0).rename("position")
    strategy_returns = (position * asset_returns).rename("strategy_return")
    equity_curve = (1.0 + strategy_returns).cumprod().rename("equity_curve")
    turnover = position.diff().abs().fillna(position.abs()).rename("turnover")

    results = pd.DataFrame(
        {
            "price": prices,
            "asset_return": asset_returns,
            "online_label": daily_label,
            "high_risk_regime": daily_high_risk,
            "high_risk_centroid_std": daily_centroid_vol,
            "raw_position": daily_raw_position,
            "position": position,
            "strategy_return": strategy_returns,
            "equity_curve": equity_curve,
            "turnover": turnover,
        }
    )
    metrics = performance_metrics(equity_curve, position=position)
    return results, metrics


def regime_strategy_backtest(
    price_series: pd.Series,
    labels: np.ndarray,
    dates: pd.Index | list,
    high_risk_regime: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run an illustrative long/cash regime strategy.

    The strategy holds the asset outside the high-risk regime and moves to cash
    during the high-risk regime. If ``high_risk_regime`` is omitted, the regime
    with the largest realized volatility is selected walk-forward using only
    data available up to each label date.

    This function assumes labels were supplied by the caller. If those labels
    come from a model fit on the full sample, the trading rule is timing-safe
    but the labels are still in-sample. Use
    ``walk_forward_regime_strategy_backtest`` for a stricter no-future-data
    experiment.
    """
    prices = price_series.dropna().sort_index().astype(float)
    if prices.empty:
        raise ValueError("price_series must not be empty")
    label_series = _make_label_series(labels, dates)

    asset_returns = prices.pct_change().fillna(0.0)
    daily_labels = label_series.reindex(prices.index, method="ffill")

    if high_risk_regime is None:
        high_risk_series = _walk_forward_high_risk_regime(asset_returns, label_series)
    else:
        high_risk_series = pd.Series(
            int(high_risk_regime),
            index=label_series.index,
            name="high_risk_regime",
        )

    daily_high_risk = high_risk_series.reindex(prices.index, method="ffill")
    raw_position = pd.Series(
        np.where(daily_labels.isna(), 0.0, np.where(daily_labels == daily_high_risk, 0.0, 1.0)),
        index=prices.index,
        name="raw_position",
    )
    position = raw_position.shift(1).fillna(0.0).rename("position")
    strategy_returns = (position * asset_returns).rename("strategy_return")
    equity_curve = (1.0 + strategy_returns).cumprod().rename("equity_curve")
    turnover = position.diff().abs().fillna(position.abs()).rename("turnover")

    results = pd.DataFrame(
        {
            "price": prices,
            "asset_return": asset_returns,
            "label": daily_labels,
            "high_risk_regime": daily_high_risk,
            "raw_position": raw_position,
            "position": position,
            "strategy_return": strategy_returns,
            "equity_curve": equity_curve,
            "turnover": turnover,
        }
    )
    metrics = performance_metrics(equity_curve, position=position)
    return results, metrics


def performance_metrics(
    equity_curve: pd.Series,
    position: pd.Series | None = None,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute standard performance metrics from an equity curve."""
    equity = equity_curve.dropna().astype(float)
    if len(equity) < 2:
        raise ValueError("equity_curve must contain at least two observations")
    if (equity <= 0).any():
        raise ValueError("equity_curve must be strictly positive")

    returns = equity.pct_change().fillna(0.0)
    years = max((len(equity) - 1) / periods_per_year, 1.0 / periods_per_year)
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)
    annual_vol = float(returns.std(ddof=1) * np.sqrt(periods_per_year))
    sharpe = float(np.sqrt(periods_per_year) * returns.mean() / returns.std(ddof=1))
    if not np.isfinite(sharpe):
        sharpe = 0.0
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min())
    turnover = float("nan")
    if position is not None:
        aligned_position = position.reindex(equity.index).fillna(0.0)
        turnover = float(aligned_position.diff().abs().fillna(aligned_position.abs()).sum())

    return {
        "CAGR": cagr,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "turnover": turnover,
    }
