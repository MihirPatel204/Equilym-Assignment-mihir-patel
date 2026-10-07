"""
Unit Tests for Auto-Analytics Engine.

Validates:
1. Ahmedabad ANC Coverage trend detection (-18.8%)
2. Mehsana ANC Coverage outlier detection (value 42)
3. Mehsana High-Risk Cases trend detection (+154.5%)
4. Strict absence of hardcoded district names in core engine code
5. Compliance with required output schema
"""

import os
import sys
import unittest
import pandas as pd

# Add root directory to path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from engine.loader import load_dataset, validate_dataset
from engine.trends import detect_trends
from engine.outliers import detect_outliers
from engine.correlations import detect_correlations
from engine.insights import build_insights_dataframe


class TestAutoAnalyticsEngine(unittest.TestCase):

    def setUp(self):
        self.sample_path = os.path.join(ROOT_DIR, "data", "sample.csv")
        self.df, self.meta = load_dataset(self.sample_path)
        self.time_col = self.meta["time_col"]
        self.entity_col = self.meta["entity_col"]
        self.indicator_cols = self.meta["indicator_cols"]

    def test_ahmedabad_anc_trend(self):
        """Verify Ahmedabad ANC drop from 85 to 69 (-18.8%) is flagged at 10% threshold."""
        trends = detect_trends(
            self.df, self.time_col, self.entity_col, self.indicator_cols, threshold_pct=10.0
        )
        ahmedabad_anc = [
            t for t in trends if t["entity"] == "Ahmedabad" and t["indicator"] == "anc_coverage"
        ]
        self.assertTrue(len(ahmedabad_anc) > 0, "Ahmedabad ANC trend was not flagged.")
        record = ahmedabad_anc[0]
        self.assertEqual(record["prev_value"], 85.0)
        self.assertEqual(record["value"], 69.0)
        self.assertAlmostEqual(record["change_pct"], -18.82, places=1)

    def test_mehsana_anc_outlier(self):
        """Verify Mehsana ANC = 42 is detected as an outlier using IQR or Leave-One-Out Z-Score."""
        outliers = detect_outliers(
            self.df,
            self.time_col,
            self.entity_col,
            self.indicator_cols,
            method="zscore",
            z_threshold=2.5,
            leave_one_out=True,
        )
        mehsana_anc = [
            o for o in outliers if o["entity"] == "Mehsana" and o["indicator"] == "anc_coverage"
        ]
        self.assertTrue(len(mehsana_anc) > 0, "Mehsana ANC outlier of 42 was not flagged.")
        self.assertEqual(mehsana_anc[0]["value"], 42.0)

    def test_mehsana_high_risk_spike(self):
        """Verify Mehsana high_risk_cases surge from 11 to 28 (+154.5%) is flagged."""
        trends = detect_trends(
            self.df, self.time_col, self.entity_col, self.indicator_cols, threshold_pct=10.0
        )
        mehsana_hr = [
            t for t in trends if t["entity"] == "Mehsana" and t["indicator"] == "high_risk_cases"
        ]
        self.assertTrue(len(mehsana_hr) > 0, "Mehsana high-risk spike was not flagged.")
        self.assertAlmostEqual(mehsana_hr[0]["change_pct"], 154.55, places=1)

    def test_no_hardcoded_entities_in_engine(self):
        """Verify no district names are hardcoded in the engine modules."""
        forbidden_terms = ["Ahmedabad", "Mehsana", "Surat", "Vadodara", "Rajkot", "Bhavnagar"]
        engine_dir = os.path.join(ROOT_DIR, "engine")
        for filename in os.listdir(engine_dir):
            if filename.endswith(".py"):
                filepath = os.path.join(engine_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                    for term in forbidden_terms:
                        self.assertNotIn(
                            term,
                            content,
                            f"Hardcoded district '{term}' found in {filename}!",
                        )

    def test_insights_schema(self):
        """Verify generated insights DataFrame conforms exactly to specification."""
        insights_df = build_insights_dataframe(
            self.df,
            self.time_col,
            self.entity_col,
            self.indicator_cols,
            trend_threshold_pct=10.0,
        )
        required_columns = [
            "insight_id",
            "type",
            "indicator",
            "entity",
            "period",
            "value",
            "prev_value",
            "change_pct",
            "severity",
            "explanation",
        ]
        for col in required_columns:
            self.assertIn(col, insights_df.columns, f"Missing required column: {col}")

        # Check severity values are strictly valid
        valid_severities = {"Low", "Medium", "High"}
        self.assertTrue(
            set(insights_df["severity"].unique()).issubset(valid_severities),
            "Invalid severity value found in insights.",
        )


if __name__ == "__main__":
    unittest.main()
