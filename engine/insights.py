"""
Automated Insight Generation Module.

Transforms raw trends, outliers, correlations, and threshold breaches into
standardized, human-readable insights with dynamic templates and severity ratings.
"""

from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np

from engine.config import INDICATOR_LABELS, DEFAULT_CONFIG
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations
from engine.severity import determine_severity


def get_indicator_label(col: str) -> str:
    """Returns friendly indicator label if configured, otherwise capitalized column name."""
    if col in INDICATOR_LABELS:
        return INDICATOR_LABELS[col]
    return col.replace("_", " ").title()


def format_trend_explanation(trend: Dict[str, Any]) -> str:
    """Generates dynamic English explanation for a trend alert."""
    label = get_indicator_label(trend["indicator"])
    entity = trend["entity"]
    change_pct = trend["change_pct"]
    direction = "rose" if change_pct > 0 else "dropped"
    abs_pct = abs(change_pct)
    prev = trend["prev_value"]
    curr = trend["value"]
    thresh = trend["threshold_pct"]
    period = trend["period"]

    base = (
        f"{label} in {entity} {direction} by {abs_pct:.1f}% compared to the previous month "
        f"({prev} → {curr}) in {period}, exceeding the {thresh:.0f}% significant-change threshold."
    )
    if trend.get("note"):
        base += f" ({trend['note']})"
    return base


def format_outlier_explanation(outlier: Dict[str, Any]) -> str:
    """Generates dynamic English explanation for an outlier alert."""
    label = get_indicator_label(outlier["indicator"])
    entity = outlier["entity"]
    val = outlier["value"]
    desc = outlier["distance_desc"]
    period = outlier["period"]
    method = outlier["method"]
    bench_type = outlier["benchmark_type"]
    bench_val = outlier["benchmark_value"]

    return (
        f"{entity}'s {label} of {val} is {desc} ({bench_type} {bench_val}) in {period}, "
        f"flagged as a statistical anomaly via {method}."
    )


def format_correlation_explanation(corr: Dict[str, Any]) -> str:
    """Generates dynamic English explanation for a correlation alert."""
    label1 = get_indicator_label(corr["indicator_1"])
    label2 = get_indicator_label(corr["indicator_2"])
    r = corr["r"]
    p = corr["p_value"]
    n = corr["n"]
    direction = corr["direction"]
    sample_note = corr.get("sample_note", "")
    sensitivity_note = corr.get("sensitivity_note", "")

    explanation = (
        f"{label1} and {label2} exhibit a strong {direction} correlation "
        f"(r={r:+.2f}, p={p:.4f}, n={n}). {sample_note}"
    )
    if sensitivity_note:
        explanation += f" {sensitivity_note}"
    return explanation


def build_insights_dataframe(
    df: pd.DataFrame,
    time_col: str,
    entity_col: str,
    indicator_cols: List[str],
    trend_threshold_pct: float = 10.0,
    outlier_method: str = "iqr",
    iqr_multiplier: float = 1.5,
    zscore_threshold: float = 3.0,
    correlation_threshold_r: float = 0.70,
    ratio_medium: float = 1.25,
    ratio_high: float = 1.75,
) -> pd.DataFrame:
    """
    Orchestrates detection across all four analytical dimensions, computes
    corroboration counts per (entity, period), applies severity rules,
    and returns a standardized Pandas DataFrame of insights.
    """
    raw_insights: List[Dict[str, Any]] = []

    # 1. Detect Trends
    trends = detect_trends(
        df, time_col, entity_col, indicator_cols, threshold_pct=trend_threshold_pct
    )
    for t in trends:
        raw_insights.append({
            "type": "trend",
            "entity": t["entity"],
            "period": t["period"],
            "indicator": t["indicator"],
            "value": t["value"],
            "prev_value": t["prev_value"],
            "change_pct": t["change_pct"],
            "magnitude": abs(t["change_pct"]),
            "threshold": trend_threshold_pct,
            "raw": t,
        })

    # 2. Detect Outliers
    outliers = detect_outliers(
        df,
        time_col,
        entity_col,
        indicator_cols,
        method=outlier_method,
        iqr_multiplier=iqr_multiplier,
        z_threshold=zscore_threshold,
        leave_one_out=True,
    )
    # Collect row indices of outliers for correlation sensitivity testing
    outlier_rows = []
    for o in outliers:
        # Match row index in df
        match = df[(df[entity_col] == o["entity"]) & (df[time_col] == o["period"])].index
        if len(match) > 0:
            outlier_rows.append(match[0])

        active_thresh = zscore_threshold if "Z-Score" in o["method"] else iqr_multiplier
        raw_insights.append({
            "type": "outlier",
            "entity": o["entity"],
            "period": o["period"],
            "indicator": o["indicator"],
            "value": o["value"],
            "prev_value": np.nan,
            "change_pct": np.nan,
            "magnitude": o["distance_metric"],
            "threshold": active_thresh,
            "raw": o,
        })

    # 3. Detect Correlations (run with outlier rows passed for sensitivity check)
    correlations = detect_correlations(
        df,
        indicator_cols,
        threshold_r=correlation_threshold_r,
        outlier_rows=outlier_rows,
    )
    for c in correlations:
        raw_insights.append({
            "type": "correlation",
            "entity": "State-wide",
            "period": "All Periods",
            "indicator": c["indicator"],
            "value": c["r"],
            "prev_value": np.nan,
            "change_pct": np.nan,
            "magnitude": c["abs_r"],
            "threshold": correlation_threshold_r,
            "raw": c,
        })

    # Count corroborating anomalies per (entity, period)
    anomaly_counts: Dict[str, int] = {}
    for item in raw_insights:
        if item["type"] in ["trend", "outlier"]:
            key = f"{item['entity']}::{item['period']}"
            anomaly_counts[key] = anomaly_counts.get(key, 0) + 1

    # Standardize final structured records
    formatted_records: List[Dict[str, Any]] = []
    for idx, item in enumerate(raw_insights, start=1):
        ins_id = f"INS-{idx:04d}"
        itype = item["type"]
        entity = item["entity"]
        period = item["period"]
        key = f"{entity}::{period}"
        corrob = anomaly_counts.get(key, 0)

        # Calculate severity
        if itype == "trend":
            severity = determine_severity(
                itype,
                item["magnitude"],
                item["threshold"],
                indicator=item["indicator"],
                change_value=item["change_pct"],
                corroborating_count=corrob,
                ratio_medium=ratio_medium,
                ratio_high=ratio_high,
            )
            explanation = format_trend_explanation(item["raw"])
        elif itype == "outlier":
            severity = determine_severity(
                itype,
                item["magnitude"],
                item["threshold"],
                indicator=item["indicator"],
                change_value=-1 if "below" in item["raw"]["distance_desc"] else 1,
                corroborating_count=corrob,
                ratio_medium=ratio_medium,
                ratio_high=ratio_high,
            )
            explanation = format_outlier_explanation(item["raw"])
        else:  # Correlation
            severity = determine_severity(
                itype,
                item["magnitude"],
                item["threshold"],
                ratio_medium=ratio_medium,
                ratio_high=ratio_high,
            )
            explanation = format_correlation_explanation(item["raw"])

        formatted_records.append({
            "insight_id": ins_id,
            "type": itype,
            "indicator": item["indicator"],
            "entity": entity,
            "period": period,
            "value": item["value"],
            "prev_value": item["prev_value"],
            "change_pct": item["change_pct"],
            "severity": severity,
            "explanation": explanation,
        })

    if not formatted_records:
        return pd.DataFrame(columns=[
            "insight_id", "type", "indicator", "entity", "period",
            "value", "prev_value", "change_pct", "severity", "explanation"
        ])

    res_df = pd.DataFrame(formatted_records)
    # Order by severity priority: High first, then Medium, then Low
    severity_rank = {"High": 0, "Medium": 1, "Low": 2}
    res_df["_rank"] = res_df["severity"].map(severity_rank)
    res_df = res_df.sort_values(by=["_rank", "insight_id"]).drop(columns=["_rank"]).reset_index(drop=True)

    return res_df


def generate_executive_summary(insights_df: pd.DataFrame) -> str:
    """
    Produces a crisp 2-3 sentence executive takeaway based on top high-severity findings.
    """
    if insights_df.empty:
        return "No statistical anomalies or significant trends detected with the current thresholds."

    high_items = insights_df[insights_df["severity"] == "High"]
    med_items = insights_df[insights_df["severity"] == "Medium"]

    total = len(insights_df)
    high_count = len(high_items)
    med_count = len(med_items)

    lines = [
        f"**Automated Summary**: Detected **{total} actionable insights** across the dataset "
        f"({high_count} High, {med_count} Medium, {total - high_count - med_count} Low priority)."
    ]

    # Highlight top 2 high alerts
    if not high_items.empty:
        top_alerts = [f"• {row['explanation']}" for _, row in high_items.head(2).iterrows()]
        lines.append("\n**Primary Concerns for Leadership Review**:\n" + "\n".join(top_alerts))

    return "\n\n".join(lines)
