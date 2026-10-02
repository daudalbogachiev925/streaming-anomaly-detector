"""Page-Hinkley drift detector using river."""


class PageHinkleyDetector:
    """
    Page-Hinkley test: cumulative difference between values and running mean.
    Detects upward or downward shift in mean.
    """

    def __init__(self, min_instances: int = 30, delta: float = 0.005, threshold: float = 50):
        from river import drift

        self.model = drift.PageHinkley(
            min_instances=min_instances,
            delta=delta,
            threshold=threshold,
        )
        self.drift_events = []
        self.t = 0

    def update(self, value: float) -> bool:
        self.model.update(value)
        self.t += 1
        if self.model.drift_detected:
            self.drift_events.append(self.t)
            return True
        return False

    def n_drifts(self) -> int:
        return len(self.drift_events)
