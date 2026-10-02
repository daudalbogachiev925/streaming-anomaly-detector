"""Streaming consumer with batching and detector integration."""
from collections import deque
from typing import Callable
import numpy as np


class StreamConsumer:
    """Consumes events, batches them, calls detectors."""

    def __init__(self, detector, buffer_size: int = 1000, batch_size: int = 100):
        self.detector = detector
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.n_processed = 0
        self.n_anomalies = 0

    def process(self, event: dict) -> dict:
        x = np.array(event["features"]).reshape(1, -1)
        self.buffer.append(x[0])

        # Update detector
        self.detector.update(x)

        # Score
        score = self.detector.score(x)[0]
        is_anomaly = self.detector.is_anomaly(x)[0]

        self.n_processed += 1
        if is_anomaly:
            self.n_anomalies += 1

        return {
            "t": event["t"],
            "score": float(score),
            "is_anomaly": bool(is_anomaly),
        }

    def run(self, stream, max_events: int = 5000, callback: Callable | None = None):
        """Consume stream, optionally call callback(result) each event."""
        results = []
        for event in stream:
            r = self.process(event)
            results.append(r)
            if callback is not None:
                callback(r)
            if len(results) >= max_events:
                break
        return results
