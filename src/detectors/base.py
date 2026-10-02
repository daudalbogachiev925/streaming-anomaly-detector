"""Base detector interface."""
import numpy as np


class BaseDetector:
    """All detectors implement update/score/is_anomaly."""

    def update(self, X: np.ndarray):
        raise NotImplementedError

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
