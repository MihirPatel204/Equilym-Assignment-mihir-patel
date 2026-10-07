"""
Auto-Analytics Engine - Streamlit Web Application.

Provides a clean, interactive UI for program managers and evaluators to:
- Ingest district-level monthly health indicators
- Interactively tune analytical thresholds
- Filter by district, month, and indicator
- View ranked, data-driven insights and interactive visualizations
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from engine.loader import load_dataset, validate_dataset
from engine.correlations import compute_correlation_matrix
from engine.insights import build_insights_dataframe, generate_executive_summary
from engine.config import DEFAULT_CONFIG, INDICATOR_LABELS

# Page configuration
st.set_page_config(
    page_title="District Health Auto-Analytics Engine",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for clean aesthetics
st.markdown(
    """
    <style>
    .metric-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 16px 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge-high {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-medium {
        background-color: #fef3c7;
        color: #92400e;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-low {
        background-color: #dbeafe;
        color: #1e40af;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def main():
    st.title("🏥 District Health Auto-Analytics Engine")
    st.caption("Automated detection of trends, outliers, correlations, and data quality flags for district healthcare data.")

    # -------------------------------------------------------------
    # 1. SIDEBAR: DATA INGESTION & CONFIGURATION
    # -------------------------------------------------------------
    st.sidebar.header("📁 Data Source")
    data_source = st.sidebar.radio(
        "Select Dataset:",
        ["Bundled Benchmark Sample", "Upload Custom CSV"],
        index=0,
    )

    sample_path = os.path.join(os.path.dirname(__file__), "data", "sample.csv")

    if data_source == "Upload Custom CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df_raw, meta = load_dataset(uploaded_file)
        else:
            st.sidebar.info("Upload a CSV file or switch to the bundled sample.")
            if os.path.exists(sample_path):
                df_raw, meta = load_dataset(sample_path)
            else:
                st.error("No dataset available.")
                return
    else:
        if os.path.exists(sample_path):
            df_raw, meta = load_dataset(sample_path)
        else:
            st.error(f"Sample file not found at {sample_path}")
            return

    time_col = meta["time_col"]
    entity_col = meta["entity_col"]
    indicator_cols = meta["indicator_cols"]

    # -------------------------------------------------------------
    # 2. SIDEBAR: FILTERS
    # -------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("🔍 Filters")

    # Entity filter
    if entity_col and entity_col in df_raw.columns:
        all_entities = sorted(df_raw[entity_col].unique().tolist())
        selected_entities = st.sidebar.multiselect(
            f"Filter {entity_col.title()}s:",
            options=all_entities,
            default=all_entities,
        )
    else:
        selected_entities = []

    # Time filter
    if time_col and time_col in df_raw.columns:
        all_periods = sorted(df_raw[time_col].unique().tolist())
        selected_periods = st.sidebar.multiselect(
            f"Filter {time_col.title()}s:",
            options=all_periods,
            default=all_periods,
        )
    else:
        selected_periods = []

    # Indicator filter
    selected_indicators = st.sidebar.multiselect(
        "Filter Indicators:",
        options=indicator_cols,
        default=indicator_cols,
        format_func=lambda x: INDICATOR_LABELS.get(x, x.replace("_", " ").title()),
    )

    # -------------------------------------------------------------
    # 3. SIDEBAR: THRESHOLD SLIDERS (CONFIGURABLE PARAMETERS)
    # -------------------------------------------------------------
    st.sidebar.markdown("---")
    st.sidebar.header("⚙️ Detection Thresholds")

    trend_threshold = st.sidebar.slider(
        "Trend Significance Threshold (%)",
        min_value=5.0,
        max_value=50.0,
        value=float(DEFAULT_CONFIG["trend_threshold_pct"]),
        step=1.0,
        help="Flag month-over-month percentage changes greater than or equal to this threshold.",
    )

    outlier_method = st.sidebar.selectbox(
        "Outlier Detection Method",
        ["IQR", "Z-Score (Leave-One-Out)"],
        index=0,
    )

    if outlier_method == "IQR":
        iqr_mult = st.sidebar.slider(
            "IQR Multiplier (Tukey Fence)",
            min_value=1.0,
            max_value=3.0,
            value=float(DEFAULT_CONFIG["iqr_multiplier"]),
            step=0.1,
            help="Flag values outside [Q1 - k*IQR, Q3 + k*IQR]. Default 1.5.",
        )
        zscore_thresh = float(DEFAULT_CONFIG["zscore_threshold"])
    else:
        zscore_thresh = st.sidebar.slider(
            "Z-Score Threshold (σ)",
            min_value=1.5,
            max_value=4.0,
            value=float(DEFAULT_CONFIG["zscore_threshold"]),
            step=0.25,
            help="Flag values exceeding this many standard deviations from the leave-one-out reference mean.",
        )
        iqr_mult = float(DEFAULT_CONFIG["iqr_multiplier"])

    corr_threshold = st.sidebar.slider(
        "Correlation Flag Threshold (|r|)",
        min_value=0.50,
        max_value=0.99,
        value=float(DEFAULT_CONFIG["correlation_threshold_r"]),
        step=0.05,
        help="Flag indicator pairs where Pearson correlation |r| meets or exceeds this value.",
    )

    # -------------------------------------------------------------
    # 4. FILTER EXECUTION & ANALYSIS PIPELINE
    # -------------------------------------------------------------
    # Apply filters to working dataframe
    filtered_df = df_raw.copy()
    if entity_col and selected_entities:
        filtered_df = filtered_df[filtered_df[entity_col].isin(selected_entities)]
    if time_col and selected_periods:
        filtered_df = filtered_df[filtered_df[time_col].isin(selected_periods)]

    active_indicators = selected_indicators if selected_indicators else indicator_cols

    # Run Data Validation
    validation_report = validate_dataset(
        filtered_df, time_col, entity_col, active_indicators
    )

    # Run Automated Insights Engine
    insights_df = build_insights_dataframe(
        filtered_df,
        time_col=time_col,
        entity_col=entity_col,
        indicator_cols=active_indicators,
        trend_threshold_pct=trend_threshold,
        outlier_method=outlier_method.lower(),
        iqr_multiplier=iqr_mult,
        zscore_threshold=zscore_thresh,
        correlation_threshold_r=corr_threshold,
    )

    # -------------------------------------------------------------
    # 5. TOP METRICS DASHBOARD
    # -------------------------------------------------------------
    total_insights = len(insights_df)
    high_count = len(insights_df[insights_df["severity"] == "High"])
    med_count = len(insights_df[insights_df["severity"] == "Medium"])
    low_count = len(insights_df[insights_df["severity"] == "Low"])

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Insights", total_insights)
    with col2:
        st.metric("🔴 High Severity", high_count)
    with col3:
        st.metric("🟡 Medium Severity", med_count)
    with col4:
        st.metric("🔵 Low Severity", low_count)

    # Executive Summary Card
    exec_summary = generate_executive_summary(insights_df)
    st.info(exec_summary)

    # -------------------------------------------------------------
    # 6. TABS NAVIGATION
    # -------------------------------------------------------------
    tab_insights, tab_viz, tab_quality = st.tabs([
        "📋 Actionable Insights",
        "📊 Visualizations",
        "🛡️ Data Quality & Schema",
    ])

    # -------------------------------------------------------------
    # TAB 1: ACTIONABLE INSIGHTS FEED
    # -------------------------------------------------------------
    with tab_insights:
        st.subheader("Generated Insights Feed")

        # Sub-filters
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            type_options = ["All"] + sorted(insights_df["type"].unique().tolist()) if not insights_df.empty else ["All"]
            selected_type = st.selectbox("Filter by Insight Type:", type_options)
        with fcol2:
            sev_options = ["All", "High", "Medium", "Low"]
            selected_sev = st.selectbox("Filter by Severity:", sev_options)

        display_df = insights_df.copy()
        if selected_type != "All":
            display_df = display_df[display_df["type"] == selected_type]
        if selected_sev != "All":
            display_df = display_df[display_df["severity"] == selected_sev]

        if display_df.empty:
            st.warning("No insights match the current filters and threshold settings.")
        else:
            # Render individual insight cards for high readability
            for _, row in display_df.iterrows():
                badge_class = f"badge-{row['severity'].lower()}"
                st.markdown(
                    f"""
                    <div style="background:#f8fafc; border-left: 5px solid {'#ef4444' if row['severity']=='High' else '#f59e0b' if row['severity']=='Medium' else '#3b82f6'}; padding: 12px 16px; border-radius: 4px; margin-bottom: 10px;">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <strong>{row['insight_id']} • {row['type'].upper()} • {row['entity']} ({row['period']})</strong>
                            <span class="{badge_class}">{row['severity']}</span>
                        </div>
                        <p style="margin-top: 6px; margin-bottom: 0px; font-size: 0.95rem; color: #1e293b;">{row['explanation']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Tabular view expander
            with st.expander("View Structured Insights Table"):
                st.dataframe(display_df, use_container_width=True)

        # Download Buttons
        st.markdown("---")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            csv_data = insights_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download insights.csv",
                data=csv_data,
                file_name="insights.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dcol2:
            json_data = insights_df.to_json(orient="records", indent=2)
            st.download_button(
                "📥 Download insights.json",
                data=json_data,
                file_name="insights.json",
                mime="application/json",
                use_container_width=True,
            )

    # -------------------------------------------------------------
    # TAB 2: VISUALIZATIONS
    # -------------------------------------------------------------
    with tab_viz:
        st.subheader("Interactive Visual Analytics")

        vcol1, vcol2 = st.columns(2)

        # Visualization 1: Severity Counts Bar Chart
        with vcol1:
            st.markdown("#### Severity Distribution")
            sev_counts = (
                insights_df["severity"]
                .value_counts()
                .reindex(["High", "Medium", "Low"], fill_value=0)
                .reset_index()
            )
            sev_counts.columns = ["Severity", "Count"]

            fig_bar = px.bar(
                sev_counts,
                x="Severity",
                y="Count",
                color="Severity",
                color_discrete_map={"High": "#ef4444", "Medium": "#f59e0b", "Low": "#3b82f6"},
                text="Count",
            )
            fig_bar.update_layout(height=350, showlegend=False, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_bar, use_container_width=True)

        # Visualization 2: Correlation Heatmap
        with vcol2:
            st.markdown("#### Indicator Correlation Matrix")
            corr_matrix = compute_correlation_matrix(filtered_df, active_indicators)

            if not corr_matrix.empty:
                fig_heat = px.imshow(
                    corr_matrix,
                    text_auto=True,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1,
                    labels=dict(color="Pearson r"),
                )
                fig_heat.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_heat, use_container_width=True)
            else:
                st.info("Select at least 2 indicators to display correlation heatmap.")

        # Visualization 3: Per-District Indicator Trend Lines
        st.markdown("---")
        st.markdown("#### Per-District Indicator Trajectories")
        plot_ind = st.selectbox(
            "Select Indicator to Plot over Time:",
            options=active_indicators,
            format_func=lambda x: INDICATOR_LABELS.get(x, x.replace("_", " ").title()),
        )

        if time_col and entity_col and plot_ind:
            fig_line = px.line(
                filtered_df.sort_values(by=time_col),
                x=time_col,
                y=plot_ind,
                color=entity_col,
                markers=True,
                title=f"{INDICATOR_LABELS.get(plot_ind, plot_ind)} Trajectory by {entity_col.title()}",
            )
            fig_line.update_layout(height=400, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_line, use_container_width=True)

        # Download Correlation Matrix CSV
        if not corr_matrix.empty:
            corr_csv = corr_matrix.to_csv().encode("utf-8")
            st.download_button(
                "📥 Download correlation_matrix.csv",
                data=corr_csv,
                file_name="correlation_matrix.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------
    # TAB 3: DATA QUALITY & VALIDATION
    # -------------------------------------------------------------
    with tab_quality:
        st.subheader("Data Quality & Integrity Report")

        qcol1, qcol2 = st.columns(2)
        with qcol1:
            st.markdown("#### Validation Status")
            if validation_report["passed"]:
                st.success("✅ All data quality integrity checks passed.")
            else:
                st.warning("⚠️ Some data quality warnings were detected:")
                for warn in validation_report["warnings"]:
                    st.write(f"- {warn}")

            st.write(f"**Duplicate ({entity_col}, {time_col}) Rows:** {validation_report['duplicate_rows']}")
            st.write(f"**Total Missing Values:** {validation_report['total_missing']}")
            st.write(f"**Unique {entity_col.title()}s:** {validation_report.get('unique_entities', 'N/A')}")
            st.write(f"**Unique {time_col.title()}s:** {validation_report.get('unique_periods', 'N/A')}")

        with qcol2:
            st.markdown("#### Missing Values per Column")
            missing_df = pd.DataFrame(
                list(validation_report["missing_counts"].items()),
                columns=["Column", "Missing Count"],
            )
            st.dataframe(missing_df, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Raw Dataset Preview")
        st.dataframe(filtered_df, use_container_width=True)

if __name__ == "__main__":
    main()
