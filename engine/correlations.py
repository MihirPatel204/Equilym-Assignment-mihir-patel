"""
Correlation Detection Module.

Calculates Pearson correlation matrix across numeric indicators, flags strong
associations (|r| >= threshold), computes p-values, performs sensitivity checks
(recomputing r without outliers), and warns about small-sample fragility.
"""

from typing import Dict, List, Tuple, Any, Optional
import itertools
import pandas as pd
import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def compute_correlation_matrix(df: pd.DataFrame, indicator_cols: List[str]) -> pd.DataFrame:
    """
    Computes standard Pearson correlation matrix across numeric indicators.
    """
    if not indicator_cols or len(indicator_cols) < 2:
        return pd.DataFrame()
    return df[indicator_cols].corr(method="pearson").round(3)


def calculate_pearson_p_value(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Computes Pearson r and two-tailed p-value.
    Uses scipy.stats.pearsonr if available, else numpy with Student's t distribution approximation.
    """
    valid = (~np.isnan(x)) & (~np.isnan(y))
    x_clean = x[valid]
    y_clean = y[valid]
    n = len(x_clean)

    if n < 3:
        return 0.0, 1.0

    if SCIPY_AVAILABLE:
        res = stats.pearsonr(x_clean, y_clean)
        return float(res.statistic), float(res.pvalue)

    # Fallback without scipy
    r = float(np.corrcoef(x_clean, y_clean)[0, 1])
    if np.isnan(r):
        return 0.0, 1.0
    if abs(r) >= 1.0:
        return r, 0.0

    # t = r * sqrt(n - 2) / sqrt(1 - r^2)
    df_deg = n - 2
    t_stat = r * np.sqrt(df_deg / (1.0 - r**2))
    # Approximate normal tail for p-value fallback
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + np.math.erf(abs(t_stat) / np.sqrt(2.0))))
    return r, float(p_val)


def detect_correlations(
    df: pd.DataFrame,
    indicator_cols: List[str],
    threshold_r: float = 0.70,
    outlier_rows: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Evaluates pairwise correlations and flags pairs with |r| >= threshold_r.
    Includes p-value, sample size caveat, and sensitivity check against outliers.
    """
    flagged: List[Dict[str, Any]] = []
    if len(indicator_cols) < 2:
        return flagged

    pairs = list(itertools.combinations(indicator_cols, 2))

    for ind1, ind2 in pairs:
        s1 = df[ind1].values.astype(float)
        s2 = df[ind2].values.astype(float)
        valid_mask = (~np.isnan(s1)) & (~np.isnan(s2))
        n_rows = int(valid_mask.sum())

        if n_rows < 3:
            continue

        r, p_val = calculate_pearson_p_value(s1, s2)
        if np.isnan(r):
            continue

        if abs(r) >= threshold_r:
            direction = "positive" if r > 0 else "negative"

            # Sample size caveat
            if n_rows < 15:
                sample_note = f"Warning: Very small sample (N={n_rows}); r is highly volatile."
            elif n_rows < 30:
                sample_note = f"Caution: Limited sample size (N={n_rows}); interpret correlation with care."
            else:
                sample_note = f"Sample size N={n_rows}."

            # Sensitivity Check: Remove influential points/outliers if provided
            sensitivity_note = ""
            r_without_outliers = None
            if outlier_rows and len(outlier_rows) > 0 and (n_rows - len(outlier_rows)) >= 3:
                clean_indices = [i for i in range(len(df)) if i not in outlier_rows]
                s1_clean = df.iloc[clean_indices][ind1].values.astype(float)
                s2_clean = df.iloc[clean_indices][ind2].values.astype(float)
                r_clean, _ = calculate_pearson_p_value(s1_clean, s2_clean)
                r_without_outliers = round(r_clean, 2)

                # Check if correlation was driven primarily by the outlier
                if abs(r) - abs(r_clean) > 0.30:
                    sensitivity_note = (
                        f"Sensitivity alert: Without extreme outliers, r shifts from {r:.2f} to {r_clean:.2f}. "
                        "The association is heavily driven by leverage points."
                    )
                else:
                    sensitivity_note = f"Robustness: r remains stable ({r_clean:.2f}) when excluding outliers."

            flagged.append({
                "indicator_1": ind1,
                "indicator_2": ind2,
                "indicator": f"{ind1} : {ind2}",
                "r": round(float(r), 2),
                "abs_r": round(float(abs(r)), 2),
                "p_value": round(float(p_val), 4),
                "n": n_rows,
                "direction": direction,
                "threshold_applied": threshold_r,
                "sample_note": sample_note,
                "sensitivity_note": sensitivity_note,
                "r_without_outliers": r_without_outliers,
            })

    return flagged
