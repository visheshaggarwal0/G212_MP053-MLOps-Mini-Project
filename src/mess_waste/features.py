"""Feature definitions and preprocessing pipelines."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "students_registered",
    "attendance_pct",
    "menu_rating",
    "temp_c",
    "cooked_kg",
    "plate_size_cm",
]

CATEGORICAL_FEATURES = [
    "meal",
    "hostel",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "waste_kg"


def build_preprocessor() -> ColumnTransformer:
    """Construct standardized preprocessing transformer for numeric and categorical features."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
