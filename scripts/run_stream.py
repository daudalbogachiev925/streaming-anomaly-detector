"""Run the streaming pipeline end-to-end."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from src.stream.producer import EventStream
from src.stream.consumer import StreamConsumer
from src.detectors.isolation_forest import IForestDetector
from src.detectors.half_space_trees import HSTDetector
from src.detectors.ensemble import DetectorEnsemble
from src.drift.adwin import ADWINDetector
from src.drift.page_hinkley import PageHinkleyDetector


def main():
    print("Starting stream simulation...")

    detector = DetectorEnsemble([
        IForestDetector(contamination=0.05),
        HSTDetector(),
    ])
    consumer = StreamConsumer(detector, batch_size=1)

    adwin = ADWINDetector()
    ph = PageHinkleyDetector()

    stream = EventStream()
    events = stream.fast_stream(max_events=5000)

    t0 = time.time()
    results = []
    for i, event in enumerate(events):
        r = consumer.process(event)
        results.append(r)

        # Drift detection on the anomaly score
        adwin.update(r["score"])
        ph.update(r["score"])

        if (i + 1) % 500 == 0:
            recent = results[-500:]
            rate = np.mean([x["is_anomaly"] for x in recent])
            print(f"  t={i+1}, recent anomaly rate={rate:.3f}, "
                  f"ADWIN drifts={adwin.n_drifts()}, PH drifts={ph.n_drifts()}")

    elapsed = time.time() - t0
    total_anom = sum(r["is_anomaly"] for r in results)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Events processed: {len(results)}")
    print(f"Anomalies detected: {total_anom} ({total_anom / len(results):.3%})")
    print(f"ADWIN drift events: {adwin.n_drifts()}")
    print(f"Page-Hinkley drift events: {ph.n_drifts()}")


if __name__ == "__main__":
    main()
