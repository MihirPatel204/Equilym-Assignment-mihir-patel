"""
Configuration and domain metadata for the Auto-Analytics Engine.

Keeps indicator polarities and readable labels in one configurable place,
ensuring the rest of the engine remains 100% general-purpose with ZERO
hardcoded entity or indicator references.
"""

# Indicator Polarity:
# 'positive': higher value is desirable (e.g., coverage, immunization)
#             -> drop is bad (high alert), increase is good (low alert)
# 'negative': higher value is undesirable (e.g., high-risk cases, mortality)
#             -> increase is bad (high alert), drop is good (low alert)
INDICATOR_POLARITY = {
    "anc_coverage": "positive",
    "institutional_delivery": "positive",
    "immunization": "positive",
    "high_risk_cases": "negative",
}

# User-friendly display names for standard indicators
INDICATOR_LABELS = {
    "anc_coverage": "ANC Coverage",
    "institutional_delivery": "Institutional Delivery",
    "immunization": "Immunization Coverage",
    "high_risk_cases": "High-Risk Cases",
}

# Default engine thresholds (can be overridden dynamically via UI sliders)
DEFAULT_CONFIG = {
    "trend_threshold_pct": 10.0,       # Flag trend if |% change| >= 10%
    "outlier_method": "iqr",           # 'iqr' or 'zscore'
    "iqr_multiplier": 1.5,             # Standard Tukey fence
    "zscore_threshold": 3.0,           # Standard 3-sigma rule
    "leave_one_out": True,             # Leave-one-out z-score for small sample stability
    "correlation_threshold_r": 0.70,   # Flag if |r| >= 0.70
    "severity_ratio_medium": 1.25,     # Ratio of magnitude / threshold for Medium
    "severity_ratio_high": 1.75,       # Ratio of magnitude / threshold for High
}
