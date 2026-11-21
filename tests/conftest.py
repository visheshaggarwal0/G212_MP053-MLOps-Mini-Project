"""Pytest fixtures for model and test suite."""

from pathlib import Path
import pytest
from src.mess_waste.model import load_model


@pytest.fixture(scope="session")
def trained_champion_model():
    """Load the champion model saved in models/model.joblib."""
    model_path = Path("models/model.joblib")
    if model_path.exists():
        return load_model(model_path)
    return None
