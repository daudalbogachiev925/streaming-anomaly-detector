"""Prometheus metrics for stream monitoring."""
from prometheus_client import Counter, Gauge, Histogram


events_processed = Counter(
    "stream_events_processed_total",
    "Total events processed",
)

anomalies_detected = Counter(
    "stream_anomalies_detected_total",
    "Total anomalies detected",
)

current_anomaly_rate = Gauge(
    "stream_anomaly_rate",
    "Rolling anomaly rate",
)

detector_latency = Histogram(
    "stream_detector_latency_ms",
    "Detector latency per event",
    buckets=[0.1, 0.5, 1, 5, 10, 50, 100],
)

drift_events_total = Counter(
    "stream_drift_events_total",
    "Total drift events detected",
    labelnames=["detector"],
)
