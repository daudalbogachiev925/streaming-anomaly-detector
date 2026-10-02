"""Half-Space Trees detector using river."""
import numpy as np
from src.detectors.base import BaseDetector


class HSTDetector(BaseDetector):
    """
    Half-Space Trees — designed for streaming, doesn't need refit.
    Uses river library which is sklearn-compatible but for streams.
    """

    def __init__(self, n_trees: int = 25, height: int = 8, window_size: int = 250):
        from river import anomaly

        self.model = anomaly.HalfSpaceTrees(
            n_trees=n_trees,
            height=height,
            window_size=window_size,
            seed=42,
        )
        self._feature_names = [f"f{i}" for i in range(10)]

    def update(self, X: np.ndarray):
        for row in X:
            d = {name: float(v) for name, v in zip(self._feature_names, row)}
            self.model.learn_one(d)

    def score(self, X: np.ndarray) -> np.ndarray:
        out = []
        for row in X:
            d = {name: float(v) for name, v in zip(self._feature_names, row)}
            out.append(self.model.score_one(d))
        return np.array(out)

    def is_anomaly(self, X: np.ndarray) -> np.ndarray:
        scores = self.score(X)
        return scores > 0.7   # threshold tuned empirically
