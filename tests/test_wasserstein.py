import numpy as np

from regime_ot.wasserstein import (
    pairwise_wasserstein_distance,
    quantile_grid,
    wasserstein_1d,
    wasserstein_barycenter_1d,
)


def test_wasserstein_1d_p1_uses_sorted_samples():
    x = np.array([3.0, 1.0, 2.0])
    y = np.array([2.0, 5.0, 4.0])

    assert wasserstein_1d(x, y, p=1) == 5.0 / 3.0


def test_wasserstein_1d_p2_matches_hand_calculation():
    x = np.array([0.0, 2.0, 4.0])
    y = np.array([1.0, 3.0, 7.0])

    expected = np.sqrt((1.0 + 1.0 + 9.0) / 3.0)
    assert wasserstein_1d(x, y, p=2) == expected


def test_pairwise_wasserstein_distance_is_symmetric():
    windows = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [10.0, 11.0, 12.0],
        ]
    )

    distances = pairwise_wasserstein_distance(windows, p=2)

    assert distances.shape == (3, 3)
    assert np.allclose(distances, distances.T)
    assert np.allclose(np.diag(distances), 0.0)
    assert distances[0, 1] == 1.0


def test_wasserstein_barycenter_averages_sorted_quantiles():
    windows = np.array(
        [
            [3.0, 1.0, 2.0],
            [6.0, 4.0, 8.0],
        ]
    )

    barycenter = wasserstein_barycenter_1d(windows)

    assert np.allclose(barycenter, np.array([2.5, 4.0, 5.5]))


def test_quantile_grid_returns_levels_and_values():
    levels, values = quantile_grid(np.array([0.0, 10.0]), n_quantiles=3)

    assert np.allclose(levels, np.array([0.0, 0.5, 1.0]))
    assert np.allclose(values, np.array([0.0, 5.0, 10.0]))
