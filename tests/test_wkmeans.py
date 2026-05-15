import numpy as np
import pytest

from regime_ot.wkmeans import WassersteinKMeans


def test_wasserstein_kmeans_separates_synthetic_distributions():
    rng = np.random.default_rng(7)
    low_risk = rng.normal(0.001, 0.005, size=(20, 30))
    high_risk = rng.normal(-0.002, 0.04, size=(20, 30))
    windows = np.vstack([low_risk, high_risk])

    model = WassersteinKMeans(
        n_clusters=2,
        random_state=42,
        n_init=5,
        max_iter=50,
    )
    labels = model.fit_predict(windows)

    low_majority = np.bincount(labels[:20]).argmax()
    high_majority = np.bincount(labels[20:]).argmax()
    assert low_majority != high_majority
    assert model.centroids_.shape == (2, 30)
    assert np.isfinite(model.inertia_)
    assert model.n_iter_ >= 1


def test_wasserstein_kmeans_predict_returns_labels_for_new_windows():
    windows = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.01, 0.0, 0.09],
            [1.0, 1.1, 0.9],
            [1.05, 0.95, 1.0],
        ]
    )

    model = WassersteinKMeans(n_clusters=2, random_state=0, n_init=3).fit(windows)
    predictions = model.predict(windows[:2])

    assert predictions.shape == (2,)
    assert np.all((predictions >= 0) & (predictions < 2))


def test_wasserstein_kmeans_rejects_too_many_clusters():
    model = WassersteinKMeans(n_clusters=3)

    with pytest.raises(ValueError, match="cannot exceed"):
        model.fit(np.array([[0.0, 1.0], [2.0, 3.0]]))
