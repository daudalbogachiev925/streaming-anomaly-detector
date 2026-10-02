"""Simulated event stream producer.

In production this would be a Kafka producer reading from a real topic.
For this project we simulate a stream with regime changes to test drift detection.
"""
import time
import numpy as np
from typing import Generator
from src.config import SEED


class EventStream:
    """
    Generates a synthetic stream with drift.

    Phases:
      1-1000:   normal (mean=0, std=1)
      1001-2000: mean shift (+2)
      2001-3000: variance increase (std=3)
      3001+:     anomaly injection (5% outliers)
    """

    def __init__(self, seed: int = SEED):
        self.rng = np.random.default_rng(seed)
        self.t = 0

    def stream(self, rate: int = 100, max_events: int | None = None) -> Generator[dict, None, None]:
        interval = 1.0 / rate
        while max_events is None or self.t < max_events:
            yield self._next_event()
            self.t += 1
            time.sleep(interval)

    def fast_stream(self, max_events: int = 5000):
        """No sleep — for testing."""
        for _ in range(max_events):
            yield self._next_event()
            self.t += 1

    def _next_event(self) -> dict:
        t = self.t
        if t < 1000:
            x = self.rng.normal(0, 1, size=4)
        elif t < 2000:
            x = self.rng.normal(2, 1, size=4)
        elif t < 3000:
            x = self.rng.normal(0, 3, size=4)
        else:
            if self.rng.random() < 0.05:
                x = self.rng.normal(10, 1, size=4)
            else:
                x = self.rng.normal(0, 1, size=4)
        return {"t": t, "features": x.tolist()}
