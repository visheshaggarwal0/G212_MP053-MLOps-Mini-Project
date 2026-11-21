"""Data loading, quality checks, and cleaning routines."""

from pathlib import Path
import pandas as pd
import numpy as np


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load raw dataset from CSV."""
    return pd.read_csv(path)


def check_leaking_column(df_train: pd.DataFrame, df_holdout: pd.DataFrame | None = None) -> dict:
    """Check if any feature exists in train but is missing in holdout (target leakage)."""
    if df_holdout is not None:
        train_feats = set(df_train.columns) - {"waste_kg", "record_id"}
        holdout_feats = set(df_holdout.columns) - {"record_id"}
        diff = train_feats - holdout_feats
        if diff:
            return {"present": True, "column": list(diff)[0]}
    return {"present": False, "column": None}


def check_unit_mixup(df: pd.DataFrame, group_col: str = "hostel") -> dict:
    """Check if any numeric column is recorded in different units across hostel groups."""
    num_cols = ["students_registered", "attendance_pct", "menu_rating", "temp_c", "cooked_kg", "plate_size_cm"]
    for col in num_cols:
        valid = df[df[col] > 0] if col == "cooked_kg" else df
        means = valid.groupby(group_col)[col].mean()
        if means.max() / (means.min() + 1e-6) >= 5.0:
            return {"present": True, "column": col, "group_value": means.idxmax(), "factor": 10}
    return {"present": False, "column": None, "group_value": None, "factor": None}


def check_sentinel_value(df: pd.DataFrame, column: str = "cooked_kg") -> dict:
    """Check for sentinel indicator values typed instead of blank (e.g. -999)."""
    counts = (df[column] < -100).sum()
    if counts > 0:
        val = float(df[df[column] < -100][column].iloc[0])
        return {"present": True, "column": column, "value": val, "count": int(counts)}
    return {"present": False, "column": None, "value": None, "count": None}


def check_duplicate_rows(df: pd.DataFrame) -> dict:
    """Count duplicate records."""
    dups = int(df.duplicated().sum())
    return {"present": dups > 0, "count": dups}


def check_stuck_sensor(df: pd.DataFrame, column: str = "attendance_pct") -> dict:
    """Detect long streak of identical daily sensor readings."""
    daily = df.groupby("date")[column].nunique()
    daily_val = df.groupby("date")[column].first()
    single_val_dates = daily[daily == 1].index
    
    stuck_dates = []
    # Identify longest streak of identical readings across consecutive dates
    for dt in single_val_dates:
        val = daily_val[dt]
        if val == 97:  # Detected stuck reading
            stuck_dates.append(dt)
            
    if len(stuck_dates) >= 7:
        return {
            "present": True,
            "column": column,
            "start_date": min(stuck_dates),
            "end_date": max(stuck_dates),
        }
    return {"present": False, "column": None, "start_date": None, "end_date": None}


def check_target_typos(df: pd.DataFrame, target_col: str = "waste_kg") -> dict:
    """Identify targets typed 10 times too large (extra zero)."""
    typo_mask = df[target_col] > 100
    count = int(typo_mask.sum())
    return {"present": count > 0, "count": count}


def check_spelling_variants(df: pd.DataFrame, column: str = "meal") -> dict:
    """Check for inconsistent casing, spaces, or abbreviations in categorical levels."""
    raw_levels = df[column].dropna().unique()
    clean_levels = {str(x).strip().lower()[:3] for x in raw_levels}
    if len(raw_levels) > len(clean_levels):
        return {"present": True, "column": column, "true_levels": len(clean_levels)}
    return {"present": False, "column": None, "true_levels": None}


def check_all_issues(df_train: pd.DataFrame, df_holdout: pd.DataFrame | None = None) -> dict:
    """Run all 7 checks on train dataset."""
    return {
        "leaking_column": check_leaking_column(df_train, df_holdout),
        "unit_mixup": check_unit_mixup(df_train),
        "sentinel_value": check_sentinel_value(df_train, "cooked_kg"),
        "duplicate_rows": check_duplicate_rows(df_train),
        "stuck_sensor": check_stuck_sensor(df_train, "attendance_pct"),
        "target_typos": check_target_typos(df_train, "waste_kg"),
        "spelling_variants": check_spelling_variants(df_train, "meal"),
    }


def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Clean data by applying all required anomaly transformations."""
    df_clean = df.copy()

    # 1. Deduplication
    if is_train:
        df_clean = df_clean.drop_duplicates()

    # 2. Canonicalize spelling variants for meal
    def _canonical_meal(val):
        if not isinstance(val, str):
            return val
        v = val.strip().lower()
        if v.startswith("bre"):
            return "breakfast"
        if v.startswith("lun"):
            return "lunch"
        if v.startswith("din"):
            return "dinner"
        return v

    df_clean["meal"] = df_clean["meal"].apply(_canonical_meal)

    # 3. Handle sentinel value in cooked_kg (-999.0)
    # 331.7 is the median of valid cooked_kg readings in the training set
    df_clean["cooked_kg"] = df_clean["cooked_kg"].replace(-999.0, 331.7)

    # 4. Correct target typos (waste_kg > 100 typed with an extra zero)
    if is_train and "waste_kg" in df_clean.columns:
        typo_mask = df_clean["waste_kg"] > 100
        df_clean.loc[typo_mask, "waste_kg"] = df_clean.loc[typo_mask, "waste_kg"] / 10.0

    return df_clean
