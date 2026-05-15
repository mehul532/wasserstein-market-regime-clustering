import numpy as np
import pandas as pd
import pytest

from regime_ot.windows import make_rolling_windows


def test_make_rolling_windows_returns_expected_shape_and_dates():
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    returns = pd.Series(np.arange(8, dtype=float), index=dates)

    windows, end_dates = make_rolling_windows(returns, window_size=3, step=2)

    assert windows.shape == (3, 3)
    assert np.allclose(windows, np.array([[0, 1, 2], [2, 3, 4], [4, 5, 6]]))
    assert list(end_dates) == [dates[2], dates[4], dates[6]]


def test_make_rolling_windows_drops_missing_values_before_windowing():
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    returns = pd.Series([1.0, np.nan, 2.0, 3.0, 4.0], index=dates)

    windows, end_dates = make_rolling_windows(returns, window_size=2, step=1)

    assert np.allclose(windows, np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]))
    assert list(end_dates) == [dates[2], dates[3], dates[4]]


def test_make_rolling_windows_rejects_short_series():
    returns = pd.Series([0.01, 0.02])

    with pytest.raises(ValueError, match="shorter"):
        make_rolling_windows(returns, window_size=3)
