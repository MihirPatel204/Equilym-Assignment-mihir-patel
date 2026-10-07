# Auto-Analytics Engine for District Healthcare Data

An end-to-end, automated analytical engine and interactive Streamlit application designed to ingest district-level monthly health indicators, detect trends, outliers, and indicator correlations, and produce ranked, human-readable insights.

Built strictly with **pure Python, Pandas, SciPy, and Streamlit**—fully data-driven, mathematically sound, and easy to explain.

---

## 🌟 Key Features

- **Automated Ingestion & Schema Detection**: Auto-detects time, entity, and numeric indicators without hardcoded column names.
- **Data Quality & Validation Panel**: Inspects missing values, duplicate entity-period pairs, percentage boundedness `[0, 100]`, and sample size sufficiency.
- **Trend Detection**: Period-over-period percentage shifts with safe division and baseline noise guards.
- **Robust Outlier Detection**:
  - **Tukey IQR Fences** ($[Q_1 - 1.5\cdot\text{IQR}, Q_3 + 1.5\cdot\text{IQR}]$)
  - **Leave-One-Out Z-Score**: Prevents extreme anomalies (e.g. Mehsana's 42) from inflating the reference standard deviation and masking themselves on small samples.
- **Correlation Engine with Sensitivity Check**:
  - Pearson correlation matrix with two-tailed $p$-values.
  - Transparent small-sample caveat ($N=12$).
  - **Outlier Sensitivity Check**: Re-evaluates $r$ after excluding extreme leverage points to verify if an association is real or outlier-driven.
- **Data-Driven Severity (No Magic Numbers)**:
  - Ratio of observed magnitude to the user's active threshold ($<1.25$ Low, $1.25–1.75$ Medium, $\ge 1.75$ High).
  - Healthcare polarity awareness (improvements capped at Low).
  - Corroboration escalation (multiple alarms in one district/month bump severity).
- **Zero Hardcoded Narratives**: All explanations are generated using dynamic string templates filled exclusively with computed values.

---

## 📁 Project Structure

```
Equilym Assignment/
├── app.py                      # Interactive Streamlit dashboard
├── engine/                     # Core analytical logic (testable, zero UI code)
│   ├── __init__.py             # Clean package exports
│   ├── config.py               # Indicator polarity, friendly labels, threshold defaults
│   ├── loader.py               # CSV loading, auto schema detection & quality checks
│   ├── trends.py               # Period-over-period % change & threshold flagging
│   ├── outliers.py             # IQR fences & Leave-One-Out Z-score
│   ├── correlations.py         # Pearson r, p-values, sensitivity check & sample warnings
│   ├── severity.py             # Data-driven severity ratios, polarity & escalation
│   └── insights.py             # Structured insight generation & templated sentences
├── data/
│   └── sample.csv              # Benchmark 12-row dataset from assignment specification
├── outputs/                    # Pre-computed reference deliverables
│   ├── insights.csv            # Structured insight output (matches schema)
│   ├── insights.json           # Serialized insight records
│   └── correlation_matrix.csv  # Pearson correlation matrix
├── tests/
│   └── test_engine.py          # Unit tests verifying key findings & zero hardcoding
├── requirements.txt            # Python dependencies
└── README.md                   # Documentation and interview guide
```

---

## 🚀 Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.


---

## 🎯 Verification Against Known Sample Findings

The engine successfully and automatically discovers the ground-truth patterns on the sample dataset:
1. **Ahmedabad ANC Coverage**: Drops from $85 \to 69$ ($-18.8\%$) in `2026-08` $\rightarrow$ **High Severity Trend**.
2. **Mehsana ANC Coverage**: $42$ in `2026-08` is $>3\sigma$ below the leave-one-out reference mean $\rightarrow$ **High Severity Outlier**.
3. **Mehsana High-Risk Cases**: Surges from $11 \to 28$ ($+154.5\%$) in `2026-08` $\rightarrow$ **High Severity Trend**.
4. **Mehsana Corroboration**: Multiple anomalies in Mehsana in `2026-08` trigger multi-alarm escalation.
5. **Correlation Sensitivity**: High negative correlation between ANC Coverage and High-Risk Cases ($r = -0.92$) is flagged with the sensitivity notice: *"Without extreme outliers, r shifts from -0.92 to -0.38. The association is heavily driven by leverage points."*
