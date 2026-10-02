"""ADWIN drift detector using river."""
import numpy as np


class ADWINDetector:
    """
    Adaptive Windowing drift detector.

    Signals drift when the mean of a recent window differs
    significantly from the historical mean.
    """

    def __init__(self, delta: float = 0.002):
        from river import drift

        self.model = drift.ADWIN(delta=delta)
        self.drift_events = []
        self.t = 0

    def update(self, value: float) -> bool:
        """Feed one value. Returns True if drift detected."""
        self.model.update(value)
        self.t += 1
        if self.model.drift_detected:
            self.drift_events.append(self.t)
            return True
        return False

    def n_drifts(self) -> int:
        return len(self.drift_events)
