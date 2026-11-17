"""Runtime settings read from environment variables, never hard-coded per call."""

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_URI = "models:/road-severity-model@production"
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"


def _float(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else fallback


def _int(name: str, fallback: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else fallback


@dataclass(frozen=True)
class Settings:
    """Everything the service layer needs from the outside world."""

    model_uri: str
    tracking_uri: str
    timeout_seconds: float
    top_features: int
    max_participants: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_uri=os.environ.get("MODEL_URI", DEFAULT_MODEL_URI),
            tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI),
            timeout_seconds=_float("PREDICT_TIMEOUT_SECONDS", 10.0),
            top_features=_int("EXPLAIN_TOP_FEATURES", 8),
            max_participants=_int("MAX_PARTICIPANTS_PER_REQUEST", 20),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
