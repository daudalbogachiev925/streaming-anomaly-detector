"""Tests for anomaly detectors."""
import numpy as np
from src.detectors.isolation_forest import IForestDetector
from src.detectors.half_space_trees import HSTDetector


def test_iforest_detects_outlier():
    detector = IForestDetector(contamination=0.1, window=200, refit_every=50)
    rng = np.random.default_rng(0)

    # Feed normal data
    for _ in range(300):
        detector.update(rng.normal(0, 1, size=(1, 4)))

    # Score a clear outlier
    outlier = np.array([[10.0, 10.0, 10.0, 10.0]])
    score_out = detector.score(outlier)[0]

    # Score a normal point
    normal = np.array([[0.1, -0.2, 0.0, 0.3]])
    score_norm = detector.score(normal)[0]

    assert score_out > score_norm


def test_hst_updates():
    detector = HSTDetector()
    rng = np.random.default_rng(1)

    for _ in range(100):
        detector.update(rng.normal(0, 1, size=(1, 4)))

    scores = detector.score(rng.normal(0, 1, size=(10, 4)))
    assert scores.shape == (10,)
