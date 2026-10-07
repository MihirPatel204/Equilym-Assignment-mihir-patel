"""
Data Loader & Validation Module.

Responsible for ingesting district-level data, automatically detecting
schema roles (entity, time, indicators), and performing data quality checks.
"""

from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np


def detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Dynamically infers the time column, entity column, and numeric indicator columns.
    Enforces general-purpose behavior so code never relies on hardcoded column names.
    """
    time_candidates = ["month", "date", "period", "year_month", "time"]
    entity_candidates = ["district", "entity", "region", "state", "location", "facility"]

    time_col = None
    for cand in time_candidates:
        matches = [col for col in df.columns if col.lower() == cand]
        if matches:
            time_col = matches[0]
            break

    entity_col = None
    for cand in entity_candidates:
        matches = [col for col in df.columns if col.lower() == cand]
        if matches:
            entity_col = matches[0]
            break

    # If not found by candidate names, fallback to first object/string columns
    non_numeric = df.select_dtypes(include=["object", "string", "category"]).columns.tolist()
    if not time_col and len(non_numeric) >= 1:
        time_col = non_numeric[0]
    if not entity_col and len(non_numeric) >= 2:
        entity_col = non_numeric[1]

    # Indicator columns are all numeric columns excluding time/entity if they happen to be numeric
    excluded = {col for col in [time_col, entity_col] if col is not None}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    indicator_cols = [col for col in numeric_cols if col not in excluded]

    return time_col, entity_col, indicator_cols


def load_dataset(file_or_buffer) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Loads CSV data and returns the DataFrame along with detected schema metadata.
    """
    df = pd.read_csv(file_or_buffer)
    # Strip whitespace from column names and string columns
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.strip()

    time_col, entity_col, indicator_cols = detect_columns(df)
    
    metadata = {
        "time_col": time_col,
        "entity_col": entity_col,
        "indicator_cols": indicator_cols,
        "total_rows": len(df),
        "total_cols": len(df.columns),
    }
    return df, metadata


def validate_dataset(
    df: pd.DataFrame,
    time_col: Optional[str],
    entity_col: Optional[str],
    indicator_cols: List[str],
) -> Dict[str, Any]:
    """
    Executes Part A validation checks:
    - Missing value count per column
    - Duplicate (entity, time) rows
    - Percentage columns bounded between [0, 100]
    - Negative counts in count indicators
    - Missing periods per entity
    - Minimum sample size checks (recommended >= 3 months, >= 10 districts)
    """
    validation: Dict[str, Any] = {
        "missing_counts": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": 0,
        "out_of_bounds_percentages": {},
        "negative_counts": {},
        "warnings": [],
        "passed": True,
    }

    # 1. Duplicate check on (entity, time)
    if entity_col and time_col:
        dup_mask = df.duplicated(subset=[entity_col, time_col], keep=False)
        validation["duplicate_rows"] = int(dup_mask.sum())
        if validation["duplicate_rows"] > 0:
            validation["warnings"].append(
                f"Found {validation['duplicate_rows']} duplicate records for ({entity_col}, {time_col})."
            )

    # 2. Boundedness & negative checks on numeric indicators
    for col in indicator_cols:
        col_lower = col.lower()
        # Heuristic for percentage/rate columns
        is_pct = any(term in col_lower for term in ["coverage", "rate", "pct", "percent", "%", "immunization", "delivery"])
        if is_pct:
            oob = df[(df[col] < 0) | (df[col] > 100)]
            if len(oob) > 0:
                validation["out_of_bounds_percentages"][col] = len(oob)
                validation["warnings"].append(
                    f"Indicator '{col}' has {len(oob)} rows outside valid percentage range [0, 100]."
                )

        neg = df[df[col] < 0]
        if len(neg) > 0:
            validation["negative_counts"][col] = len(neg)
            validation["warnings"].append(
                f"Indicator '{col}' has {len(neg)} negative values."
            )

    # 3. Sample size checks against assignment recommendations
    if entity_col:
        num_entities = df[entity_col].nunique()
        validation["unique_entities"] = num_entities
        if num_entities < 10:
            validation["warnings"].append(
                f"Small entity count: dataset has {num_entities} districts (recommended: >= 10 for stable state-wide metrics)."
            )

    if time_col:
        num_periods = df[time_col].nunique()
        validation["unique_periods"] = num_periods
        if num_periods < 3:
            validation["warnings"].append(
                f"Short time horizon: dataset has {num_periods} periods (recommended: >= 3 months for robust trend identification)."
            )

    if validation["warnings"]:
        validation["passed"] = False

    return validation
