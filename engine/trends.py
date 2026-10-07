"""
Trend Detection Module.

Calculates consecutive period-over-period percentage changes per entity
and flags statistically significant shifts based on a user-defined threshold.
"""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np


def detect_trends(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    indicator_cols: List[str],
    threshold_pct: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Computes period-over-period percentage change for each (entity, indicator) pair.
    
    Returns a list of trend dictionaries for shifts where:
        abs(pct_change) >= threshold_pct
    """
    trends: List[Dict[str, Any]] = []

    if not time_col or not entity_col or not indicator_cols:
        return trends

    # Sort deterministically by entity and chronological order
    sorted_df = df.sort_values(by=[entity_col, time_col]).copy()

    # Group by district/entity
    for entity_val, group in sorted_df.groupby(entity_col):
        if len(group) < 2:
            continue  # Need at least two consecutive periods to measure trend

        for ind in indicator_cols:
            prev_row = None
            for _, curr_row in group.iterrows():
                if prev_row is not None:
                    prev_period = prev_row[time_col]
                    curr_period = curr_row[time_col]
                    prev_val = prev_row[ind]
                    curr_val = curr_row[ind]

                    # Check for valid numeric values
                    if pd.notna(prev_val) and pd.notna(curr_val):
                        # Guard against divide-by-zero
                        if prev_val == 0:
                            pct_change = None
                            is_significant = False
                            note = "Baseline value was 0 (division by zero skipped)"
                        else:
                            pct_change = ((curr_val - prev_val) / abs(prev_val)) * 100.0
                            is_significant = abs(pct_change) >= threshold_pct
                            note = ""

                        abs_change = curr_val - prev_val

                        # Small base check: if absolute values are small (< 10), note it
                        if abs(prev_val) < 10 and abs(curr_val) < 10:
                            note = "Small baseline count; percentage change may be noisy" if not note else note

                        if is_significant and pct_change is not None:
                            trends.append({
                                "entity": entity_val,
                                "indicator": ind,
                                "period": curr_period,
                                "prev_period": prev_period,
                                "prev_value": round(float(prev_val), 2),
                                "value": round(float(curr_val), 2),
                                "change_pct": round(float(pct_change), 2),
                                "abs_change": round(float(abs_change), 2),
                                "threshold_pct": threshold_pct,
                                "note": note,
                            })

                prev_row = curr_row

    return trends
