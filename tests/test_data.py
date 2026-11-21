"""Unit tests for data cleaning, anomaly detection, and drift calculation."""

import numpy as np
import pandas as pd
import pytest

from src.mess_waste.data import clean_data, check_sentinel_value, check_spelling_variants, check_target_typos
from src.mess_waste.drift import calculate_psi


def test_clean_data_removes_duplicates():
    """Verify deduplication works on raw data."""
    df = pd.DataFrame({
        "record_id": ["R1", "R1"],
        "date": ["2026-01-01", "2026-01-01"],
        "students_registered": [500, 500],
        "attendance_pct": [70, 70],
        "menu_rating": [3.0, 3.0],
        "temp_c": [25.0, 25.0],
        "cooked_kg": [300.0, 300.0],
        "plate_size_cm": [26.0, 26.0],
        "meal": ["lunch", "lunch"],
        "hostel": ["Boys Hostel 1", "Boys Hostel 1"],
        "waste_kg": [40.0, 40.0],
    })
    cleaned = clean_data(df, is_train=True)
    assert len(cleaned) == 1


def test_clean_data_canonicalizes_meal():
    """Verify messy meal spelling variants are normalized to standard categories."""
    df = pd.DataFrame({
        "meal": ["  Lunch ", "LUNCH", "lun.", "Breakfast", "bre.", "Dinner ", "din."],
        "cooked_kg": [200.0] * 7,
    })
    cleaned = clean_data(df, is_train=False)
    expected = ["lunch", "lunch", "lunch", "breakfast", "breakfast", "dinner", "dinner"]
    assert list(cleaned["meal"]) == expected


def test_clean_data_handles_sentinel_and_typos():
    """Verify sentinel values in cooked_kg and 10x typos in waste_kg are corrected."""
    df = pd.DataFrame({
        "cooked_kg": [-999.0, 300.0],
        "waste_kg": [450.0, 35.0],
        "meal": ["lunch", "dinner"],
    })
    cleaned = clean_data(df, is_train=True)
    # -999.0 should be replaced with median (331.7)
    assert cleaned["cooked_kg"].iloc[0] == 331.7
    # 450.0 typo should be divided by 10 to 45.0
    assert cleaned["waste_kg"].iloc[0] == 45.0
    assert cleaned["waste_kg"].iloc[1] == 35.0


def test_calculate_psi_identical_distributions():
    """Verify PSI of identical distributions is near zero."""
    arr = np.random.normal(50, 10, 500)
    psi = calculate_psi(arr, arr, bins=10)
    assert psi == pytest.approx(0.0, abs=1e-3)
