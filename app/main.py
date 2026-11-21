"""FastAPI service for predicting hostel mess food waste."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Application and model setup
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

FEATURES = [
    "students_registered",
    "attendance_pct",
    "menu_rating",
    "temp_c",
    "cooked_kg",
    "plate_size_cm",
    "meal",
    "hostel",
]

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model was not found at {MODEL_PATH}. "
        "Make sure models/model.joblib is present."
    )

# The serialized artifact is a complete sklearn Pipeline, including its
# preprocessing steps, so the API passes it the original named features.
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Hostel Mess Food Waste Prediction API",
    description="Predicts post-meal food waste in kilograms for hostel messes.",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# JSON logging to stdout (one JSON record per successful prediction)
# ---------------------------------------------------------------------------
class JsonLogFormatter(logging.Formatter):
    """Format log records as one-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        event = getattr(record, "event", None)
        if event is not None:
            payload["event"] = event
        prediction_kg = getattr(record, "prediction_kg", None)
        if prediction_kg is not None:
            payload["prediction_kg"] = prediction_kg
        return json.dumps(payload, ensure_ascii=False)


logger = logging.getLogger("mp053.api")
logger.setLevel(logging.INFO)
logger.propagate = False

# Avoid adding duplicate handlers when the module is reloaded by development tools.
if not any(getattr(handler, "_mp053_json_handler", False) for handler in logger.handlers):
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(JsonLogFormatter())
    stdout_handler._mp053_json_handler = True  # type: ignore[attr-defined]
    logger.addHandler(stdout_handler)


# ---------------------------------------------------------------------------
# Request schema and endpoints
# ---------------------------------------------------------------------------
class PredictionRequest(BaseModel):
    """The exact eight input features accepted by POST /predict."""

    model_config = ConfigDict(extra="forbid")

    students_registered: int = Field(..., le=2500)
    attendance_pct: float
    menu_rating: float
    temp_c: float = Field(..., ge=4.0)
    cooked_kg: float
    plate_size_cm: float
    meal: str
    hostel: str


@app.get("/health")
def health() -> dict[str, str]:
    """Return the service health status."""
    return {"status": "ok"}


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, float]:
    """Validate hostel meal data and return predicted waste in kilograms."""
    input_data = pd.DataFrame([request.model_dump()], columns=FEATURES)
    prediction = float(model.predict(input_data)[0])

    # Do not log request contents; only log the event and prediction result.
    logger.info(
        "Food-waste prediction generated",
        extra={"event": "prediction_completed", "prediction_kg": prediction},
    )
    return {"prediction": prediction}
