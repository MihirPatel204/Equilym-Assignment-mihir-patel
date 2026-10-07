"""
Auto-Analytics Engine Package.
"""

from engine.loader import load_dataset, validate_dataset, detect_columns
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import compute_correlation_matrix, detect_correlations
from engine.insights import build_insights_dataframe, generate_executive_summary
from engine.config import DEFAULT_CONFIG, INDICATOR_POLARITY, INDICATOR_LABELS

__all__ = [
    "load_dataset",
    "validate_dataset",
    "detect_columns",
    "detect_trends",
    "detect_outliers",
    "compute_correlation_matrix",
    "detect_correlations",
    "build_insights_dataframe",
    "generate_executive_summary",
    "DEFAULT_CONFIG",
    "INDICATOR_POLARITY",
    "INDICATOR_LABELS",
]
