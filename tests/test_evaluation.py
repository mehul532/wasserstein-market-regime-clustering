import numpy as np
import pandas as pd

from regime_ot.evaluation import (
    compare_regime_future_returns,
    silhouette_from_distance_matrix,
    transition_matrix,
)


def test_transition_matrix_is_row_normalized():
    matrix = transition_matrix(np.array([0, 0, 1, 1, 0]))

    assert matrix.loc[0, 0] == 0.5
    assert matrix.loc[0, 1] == 0.5
    assert matrix.loc[1, 0] == 0.5
    assert matrix.loc[1, 1] == 0.5


def test_silhouette_from_distance_matrix_handles_degenerate_labels():
    distances = np.zeros((3, 3))

    assert np.isnan(silhouette_from_distance_matrix(distances, np.array([1, 1, 1])))


def test_compare_regime_future_returns_uses_next_period_return():
    labels = np.array([0, 1, 0])
    returns = pd.Series([0.10, 0.20, -0.30])

    comparison = compare_regime_future_returns(labels, returns)

    assert comparison.loc[0, "mean"] == 0.20
    assert comparison.loc[1, "mean"] == -0.30


def test_compare_regime_future_returns_can_use_pre_shifted_returns():
    labels = np.array([0, 1, 0])
    forward_returns = pd.Series([0.20, -0.30, np.nan])

    comparison = compare_regime_future_returns(
        labels,
        forward_returns,
        returns_are_forward=True,
    )

    assert comparison.loc[0, "mean"] == 0.20
    assert comparison.loc[1, "mean"] == -0.30
