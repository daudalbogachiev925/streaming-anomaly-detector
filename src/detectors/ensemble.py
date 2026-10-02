"""Ensemble of multiple detectors."""
import numpy as np
from src.detectors.base import BaseDetector


class DetectorEnsemble(BaseDetector):
    """Average scores from multiple detectors. Anomaly if majority vote."""

    def __init__(self, detectors: list[BaseDetector], weights: list[float] | None = None):
        self.detectors = detectors
        self.weights = weights or [1.0] * len(detectors)

    def update(self, X: np.ndarray):
        for d in self.detectors:
            d.update(X)

    def score(self, X: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(X))
        total_w = sum(self.weights)
        for d, w in zip(self.detectors, self.weights):
            # Normalize scores to [0, 1] range
            s = d.score(X)
            if s.max() > s.min():
                s = (s - s.min()) / (s.max() - s.min() + 1e-8)
            scores += w * s
        return scores / total_w

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        votes = np.zeros(len(X))
        for d in self.detectors:
            votes += d.is_anomaly(X).astype(float)
        return votes >= len(self.detectors) / 2
