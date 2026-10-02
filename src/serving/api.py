"""FastAPI endpoint for anomaly detection."""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
from src.detectors.isolation_forest import IForestDetector
from src.detectors.half_space_trees import HSTDetector
from src.detectors.ensemble import DetectorEnsemble

app = FastAPI(title="Anomaly Detection API", version="0.1.0")

# Global detector state
ensemble = DetectorEnsemble([
    IForestDetector(contamination=0.05),
    HSTDetector(),
])
n_seen = 0


class EventRequest(BaseModel):
    features: List[float]


class EventResponse(BaseModel):
    score: float
    is_anomaly: bool
    n_events_seen: int


@app.get("/health")
def health():
    return {"status": "ok", "n_seen": n_seen}


@app.post("/detect", response_model=EventResponse)
def detect(req: EventRequest):
    global n_seen
    x = np.array(req.features).reshape(1, -1)
    ensemble.update(x)
    score = float(ensemble.score(x)[0])
    is_anom = bool(ensemble.is_anomaly(x)[0])
    n_seen += 1
    return EventResponse(score=score, is_anomaly=is_anom, n_events_seen=n_seen)
