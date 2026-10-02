"""Project configuration."""
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"

SEED = 42

# Stream
STREAM_RATE_PER_SEC = 100
BUFFER_SIZE = 1000
WINDOW_SIZE = 500

# Detector
CONTAMINATION = 0.05
DRIFT_THRESHOLD = 0.01
