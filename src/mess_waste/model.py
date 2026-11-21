"""Model building, training, evaluation, and persistence."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline

from .features import build_preprocessor


def get_estimator(model_type: str = "linear", **kwargs):
    """Instantiate the regressor based on model type."""
    if model_type == "linear":
        return LinearRegression(**kwargs)
    elif model_type == "ridge":
        return Ridge(**kwargs)
    elif model_type == "rf":
        return RandomForestRegressor(random_state=42, **kwargs)
    elif model_type == "gb":
        return GradientBoostingRegressor(random_state=42, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def build_pipeline(model_type: str = "linear", **kwargs) -> Pipeline:
    """Combine preprocessor and regressor into a single pipeline."""
    preprocessor = build_preprocessor()
    estimator = get_estimator(model_type=model_type, **kwargs)
    return Pipeline([("preprocessor", preprocessor), ("regressor", estimator)])


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str = "linear", **kwargs) -> Pipeline:
    """Train pipeline model on feature matrix X and target y."""
    pipeline = build_pipeline(model_type=model_type, **kwargs)
    pipeline.fit(X, y)
    return pipeline


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
    """Compute Mean Absolute Error (MAE) in kg."""
    preds = model.predict(X)
    return float(mean_absolute_error(y, preds))


def save_model(model: Pipeline, path: str | Path) -> None:
    """Save trained pipeline model to disk."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)


def load_model(path: str | Path) -> Pipeline:
    """Load serialized pipeline model from disk."""
    return joblib.load(path)
