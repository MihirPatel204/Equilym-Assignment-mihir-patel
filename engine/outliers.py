"""
Outlier Detection Module.

Supports:
1. IQR (Interquartile Range / Tukey Fences)
2. Leave-One-Out Z-Score (robust against small-sample self-masking)
3. Standard Z-Score
"""

from typing import Dict, List, Any
import pandas as pd
import numpy as np


def detect_outliers_iqr(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    indicator_cols: List[str],
    iqr_multiplier: float = 1.5,
) -> List[Dict[str, Any]]:
    """
    Detects outliers using Tukey's IQR rule:
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
    """
    outliers: List[Dict[str, Any]] = []

    for ind in indicator_cols:
        series = df[ind].dropna()
        if len(series) < 4:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        median = float(series.median())

        if iqr == 0:
            # Constant or zero-spread data: cannot use standard IQR fences
            continue

        lower_bound = q1 - (iqr_multiplier * iqr)
        upper_bound = q3 + (iqr_multiplier * iqr)

        for _, row in df.iterrows():
            val = row[ind]
            if pd.isna(val):
                continue

            val_float = float(val)
            is_low = val_float < lower_bound
            is_high = val_float > upper_bound

            if is_low or is_high:
                distance_iqr = abs(val_float - median) / iqr
                bound_violated = "below lower fence" if is_low else "above upper fence"
                fence_val = lower_bound if is_low else upper_bound

                outliers.append({
                    "entity": row[entity_col],
                    "indicator": ind,
                    "period": row[time_col],
                    "value": round(val_float, 2),
                    "method": "IQR",
                    "benchmark_type": "median",
                    "benchmark_value": round(median, 2),
                    "fence_value": round(fence_val, 2),
                    "distance_metric": round(distance_iqr, 2),
                    "distance_desc": f"{distance_iqr:.1f}x IQR from median",
                    "bound_violated": bound_violated,
                    "threshold_applied": iqr_multiplier,
                })

    return outliers


def detect_outliers_zscore(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    indicator_cols: List[str],
    z_threshold: float = 3.0,
    leave_one_out: bool = True,
) -> List[Dict[str, Any]]:
    """
    Detects outliers using Z-score.
    
    If leave_one_out is True (recommended for small samples like N=12):
        Calculates mean and standard deviation of all OTHER rows excluding the point itself.
        This prevents an extreme point (like Mehsana's 42) from inflating the std dev
        and masking itself.
    """
    outliers: List[Dict[str, Any]] = []

    for ind in indicator_cols:
        series = df[ind].dropna()
        n = len(series)
        if n < 3:
            continue

        global_std = float(series.std(ddof=1))
        if global_std == 0:
            continue  # Constant column, skip

        for idx, row in df.iterrows():
            val = row[ind]
            if pd.isna(val):
                continue

            val_float = float(val)

            if leave_one_out and n > 3:
                # Compute reference mean & std excluding this specific index
                other_values = series.drop(index=idx)
                ref_mean = float(other_values.mean())
                ref_std = float(other_values.std(ddof=1))
                if ref_std == 0:
                    continue
                z = (val_float - ref_mean) / ref_std
                benchmark_mean = ref_mean
            else:
                ref_mean = float(series.mean())
                ref_std = global_std
                z = (val_float - ref_mean) / ref_std
                benchmark_mean = ref_mean

            if abs(z) >= z_threshold:
                direction = "below" if z < 0 else "above"
                outliers.append({
                    "entity": row[entity_col],
                    "indicator": ind,
                    "period": row[time_col],
                    "value": round(val_float, 2),
                    "method": "Leave-One-Out Z-Score" if leave_one_out else "Z-Score",
                    "benchmark_type": "state mean",
                    "benchmark_value": round(benchmark_mean, 2),
                    "distance_metric": round(abs(z), 2),
                    "distance_desc": f"{abs(z):.1f}σ {direction} the mean",
                    "bound_violated": f"{abs(z):.1f}σ {direction} reference mean ({benchmark_mean:.1f})",
                    "threshold_applied": z_threshold,
                })

    return outliers


def detect_outliers(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    indicator_cols: List[str],
    method: str = "iqr",
    iqr_multiplier: float = 1.5,
    z_threshold: float = 3.0,
    leave_one_out: bool = True,
) -> List[Dict[str, Any]]:
    """
    Unified router for outlier detection.
    """
    if method.lower() == "zscore" or method.lower() == "z-score":
        return detect_outliers_zscore(
            df, time_col, entity_col, indicator_cols, z_threshold, leave_one_out
        )
    return detect_outliers_iqr(
        df, time_col, entity_col, indicator_cols, iqr_multiplier
    )
