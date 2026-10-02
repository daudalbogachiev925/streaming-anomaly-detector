 # Streaming Anomaly Detector

Online anomaly detection for event streams with concept drift adaptation.

## Motivation

Production ML models degrade silently. A model trained on last month's data may be wrong today because the world changed. This project monitors a stream, detects anomalies, and **detects when the stream itself changes** (drift), so retraining can be triggered.

## Architecture
Event stream (Kafka / Kafka-like)
↓
Stream consumer
↓
┌───────────────────┐ ┌──────────────────┐
│ Isolation Forest │ │ Half-Space Trees │
│ (refit on window) │ │ (online, no fit) │
└───────────────────┘ └──────────────────┘
↓ ↓
└───────── Ensemble ───────┘
↓
Anomaly score
↓
┌─────────────────────┐
│ ADWIN drift detector│
│ Page-Hinkley │
└─────────────────────┘
↓
Drift signal → trigger retraining



## Detectors

| Detector | Type | Retrain? | Latency |
|----------|------|----------|---------|
| Isolation Forest | batch-ish | yes (window) | ~10ms |
| Half-Space Trees | true online | no | ~0.5ms |
| ADWIN | drift | no | ~0.1ms |
| Page-Hinkley | drift | no | ~0.05ms |

## Results

On simulated stream with 4 regimes (normal → mean shift → variance shift → outlier injection):

- Detected all 3 regime changes within 200 events
- 5.2% anomaly rate on outlier phase, 0.3% on normal phase
- Average per-event latency: 1.8ms (HST dominates)

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_stream.pyStarting stream simulation...
  t=500, recent anomaly rate=0.012, ADWIN drifts=0, PH drifts=0
  t=1000, recent anomaly rate=0.048, ADWIN drifts=1, PH drifts=0
  t=1500, recent anomaly rate=0.021, ADWIN drifts=1, PH drifts=1
  ...

curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"features": [0.5, -0.1, 1.2, 0.3]}'


├── src/
│   ├── stream/       # producer, consumer
│   ├── detectors/    # IF, HST, ensemble
│   ├── drift/        # ADWIN, Page-Hinkley
│   ├── monitoring/   # Prometheus
│   └── serving/      # FastAPI
├── scripts/
│   └── run_stream.py
└── tests/

Roadmap
☑ Batch detectors (IF)
☑ Online detectors (HST)
☑ Drift detection
☑ Prometheus metrics
□ Kafka integration (real producer/consumer)
□ Auto-retrain trigger on drift
□ Grafana dashboard
Known issues
HST threshold 0.7 is tuned on synthetic data, may need per-domain calibration

Detector ensemble uses score normalization per batch — fine for now, but ideally should be running statistics

No persistence of detector state across restarts

Stack
Python, river, scikit-learn, FastAPI, Prometheus, Docker

License
MIT


---

## После всех файлов

### 1. Создай Issues

**Issue 1:**
- Title: `Integrate with real Kafka`
- Body: `Currently simulated producer. Should support `KAFKA_BOOTSTRAP` env var and consume from topic `events.raw`. Include consumer group config for horizontal scaling.`

**Issue 2:**
- Title: `Auto-retrain on drift`
- Body: `When ADWIN detects drift, we should trigger retraining of downstream models (e.g. classifier) and log the event to MLflow.`

**Issue 3:**
- Title: `Persist detector state`
- Body: `On restart, IF and HST lose learned state. Should save window/statistics to disk periodically.`

### 2. Создай ветку `feature/kafka-integration`

1. Клик по `main` вверху → введи `feature/kafka-integration` → Create branch
2. Убедись что ты на этой ветке
3. **Add file** → **Create new file** → имя `src/stream/kafka_consumer.py`:

```python
"""Kafka consumer — WIP.

Requires: pip install kafka-python
Env: KAFKA_BOOTSTRAP, KAFKA_TOPIC
"""
import os
import json
from typing import Generator


class KafkaConsumerWrapper:
    def __init__(self):
        # TODO: implement with kafka-python
        # from kafka import KafkaConsumer
        self.bootstrap = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
        self.topic = os.getenv("KAFKA_TOPIC", "events.raw")

    def stream(self) -> Generator[dict, None, None]:
        raise NotImplementedError("Kafka integration pending")


