
import json
import logging
import sys
from datetime import datetime, timezone


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "event": record.getMessage()
        }

        if hasattr(record, "prediction_kg"):
            log_data["prediction_kg"] = record.prediction_kg

        return json.dumps(log_data)


def get_prediction_logger():
    logger = logging.getLogger("mp053.predictions")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)

    return logger


def log_prediction(prediction_kg):
    logger = get_prediction_logger()
    logger.info(
        "prediction",
        extra={"prediction_kg": float(prediction_kg)}
    )
