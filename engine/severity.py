"""
Severity Assessment & Escalation Module.

Calculates data-driven severity tiers (Low, Medium, High) based on:
1. Ratio of observed magnitude to the user-selected threshold.
2. Healthcare polarity (beneficial changes are capped at Low).
3. Corroboration escalation (multiple anomalies in the same district/month bump severity).
"""

from typing import Optional
from engine.config import INDICATOR_POLARITY


SEVERITY_ORDER = ["Low", "Medium", "High"]


def map_ratio_to_severity(
    ratio: float,
    ratio_medium: float = 1.25,
    ratio_high: float = 1.75,
) -> str:
    """
    Directly maps magnitude/threshold ratio to baseline severity:
        ratio < 1.25        -> Low
        1.25 <= ratio < 1.75 -> Medium
        ratio >= 1.75       -> High
    """
    if ratio >= ratio_high:
        return "High"
    elif ratio >= ratio_medium:
        return "Medium"
    return "Low"


def escalate_severity(current_severity: str, step_count: int = 1) -> str:
    """
    Bumps severity level up by step_count (e.g., Low -> Medium -> High).
    """
    curr_idx = SEVERITY_ORDER.index(current_severity)
    new_idx = min(len(SEVERITY_ORDER) - 1, curr_idx + step_count)
    return SEVERITY_ORDER[new_idx]


def determine_severity(
    insight_type: str,
    magnitude: float,
    threshold: float,
    indicator: Optional[str] = None,
    change_value: Optional[float] = None,
    corroborating_count: int = 0,
    ratio_medium: float = 1.25,
    ratio_high: float = 1.75,
) -> str:
    """
    Computes data-driven severity.
    
    - magnitude: observed percentage change, z-score distance, or correlation r
    - threshold: active threshold set by user
    - indicator: used to look up polarity ('positive' vs 'negative')
    - change_value: direction of change (+ or -)
    - corroborating_count: number of other anomalies for the same entity and month
    """
    if threshold <= 0:
        ratio = 1.0
    else:
        ratio = abs(magnitude) / threshold

    base_severity = map_ratio_to_severity(ratio, ratio_medium, ratio_high)

    # Polarity check:
    # If the indicator improved (e.g. coverage rose, or high risk cases decreased),
    # it is positive progress. We cap the alert at 'Low'.
    if indicator and change_value is not None:
        polarity = INDICATOR_POLARITY.get(indicator, "neutral")
        is_favorable = False
        if polarity == "positive" and change_value > 0:
            is_favorable = True
        elif polarity == "negative" and change_value < 0:
            is_favorable = True

        if is_favorable:
            return "Low"  # An improvement is celebrated/noted, not treated as a crisis

    # Corroboration escalation:
    # If 2 or more anomalies strike the same entity in the same period, escalate severity.
    if corroborating_count >= 2:
        return escalate_severity(base_severity, step_count=1)

    return base_severity
