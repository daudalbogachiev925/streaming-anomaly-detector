"""Isolation Forest detector."""
import numpy as np
from sklearn.ensemble import IsolationForest
from src.detectors.base import BaseDetector


class IForestDetector(BaseDetector):
    """Wraps sklearn IsolationForest with online refit."""

    def __init__(self, contamination: float = 0.05, window: int = 500, refit_every: int = 200):
        self.contamination = contamination
        self.window = window
        self.refit_every = refit_every
        self.model = None
        self.buffer = []
        self.seen = 0

    def update(self, X: np.ndarray):
        for row in X:
            self.buffer.append(row)
        if len(self.buffer) > self.window:
            self.buffer = self.buffer[-self.window:]

        self.seen += 1
        if self.seen % self.refit_every == 0 or self.model is None:
            self._refit()

    def _refit(self):
        if len(self.buffer) < 50:
            return
        data = np.array(self.buffer)
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(data)

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X))
        return -self.model.score_samples(X)

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.zeros(len(X), dtype=bool)
        return self.model.predict(X) == -1
