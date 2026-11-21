"""Monitoring, drift detection (PSI), and subgroup slice evaluation."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def calculate_psi(ref: pd.Series | np.ndarray, comp: pd.Series | np.ndarray, bins: int = 10) -> float:
    """Calculate Population Stability Index (PSI) using 10 quantile bins based on reference data."""
    ref_arr = np.asarray(ref)
    comp_arr = np.asarray(comp)

    # 10 quantile deciles from reference distribution
    percentiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref_arr, percentiles)

    # Adjust boundary edges slightly to include min and max
    bin_edges[0] -= 1e-5
    bin_edges[-1] += 1e-5

    # Fallback to equal-width if identical percentiles occur due to discrete values
    if len(np.unique(bin_edges)) < len(bin_edges):
        min_val = min(ref_arr.min(), comp_arr.min())
        max_val = max(ref_arr.max(), comp_arr.max())
        bin_edges = np.linspace(min_val - 1e-5, max_val + 1e-5, bins + 1)

    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
    comp_counts, _ = np.histogram(comp_arr, bins=bin_edges)

    ref_props = ref_counts / len(ref_arr)
    comp_props = comp_counts / len(comp_arr)

    # Epsilon to handle zero frequencies safely without division-by-zero
    eps = 1e-4
    ref_props = np.where(ref_props == 0, eps, ref_props)
    comp_props = np.where(comp_props == 0, eps, comp_props)

    # Normalize proportions
    ref_props /= ref_props.sum()
    comp_props /= comp_props.sum()

    psi_val = np.sum((comp_props - ref_props) * np.log(comp_props / ref_props))
    return float(round(psi_val, 4))


def evaluate_mae_by_group(
    model,
    df: pd.DataFrame,
    features: list[str],
    group_col: str = "hostel",
    target_col: str = "waste_kg",
) -> dict[str, float]:
    """Calculate model MAE sliced per subgroup category."""
    results = {}
    for group_val in sorted(df[group_col].unique()):
        sub_df = df[df[group_col] == group_val]
        preds = model.predict(sub_df[features])
        mae = mean_absolute_error(sub_df[target_col], preds)
        results[str(group_val)] = float(round(mae, 4))
    return results
