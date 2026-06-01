"""
Gradio Frontend for AI Data Quality Copilot.
Three tabs: Dataset Profiling, Data Drift, Analysis History.
Features rich Plotly visualizations and a premium dark-themed UI.
"""
import io
import json
import tempfile
import os

import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import hashlib
import requests

from backend.services.profiler import profile_dataset
from backend.services.outliers import detect_outliers
from backend.services.drift import detect_drift
from backend.services.scorer import calculate_quality_score
from backend.services.ai import get_ai_recommendations
from backend.services.report import generate_pdf_report
from backend.database import SessionLocal
from backend.models import AnalysisHistory, User


# ═══════════════════════════════════════════════════════════
#  Color Palette & Theme
# ═══════════════════════════════════════════════════════════
COLORS = {
    "bg_dark": "#0f172a",
    "bg_card": "#1e293b",
    "accent_blue": "#3b82f6",
    "accent_purple": "#8b5cf6",
    "accent_cyan": "#06b6d4",
    "accent_green": "#22c55e",
    "accent_yellow": "#eab308",
    "accent_orange": "#f97316",
    "accent_red": "#ef4444",
    "text_primary": "#f1f5f9",
    "text_secondary": "#94a3b8",
    "gradient_start": "#3b82f6",
    "gradient_end": "#8b5cf6",
}

PLOTLY_TEMPLATE = {
    "paper_bgcolor": "rgba(15, 23, 42, 0)",
    "plot_bgcolor": "rgba(30, 41, 59, 0.5)",
    "font": {"color": "#f1f5f9", "family": "Inter, sans-serif"},
    "colorway": [
        "#3b82f6", "#8b5cf6", "#06b6d4", "#22c55e",
        "#eab308", "#f97316", "#ef4444", "#ec4899",
    ],
}


# ═══════════════════════════════════════════════════════════
#  Plotly Chart Builders
# ═══════════════════════════════════════════════════════════

def _build_quality_gauge(score: float, risk: str) -> go.Figure:
    """Create an animated gauge chart for the quality score."""
    if score >= 90:
        bar_color = COLORS["accent_green"]
    elif score >= 75:
        bar_color = COLORS["accent_yellow"]
    elif score >= 50:
        bar_color = COLORS["accent_orange"]
    else:
        bar_color = COLORS["accent_red"]

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={"suffix": "/100", "font": {"size": 42, "color": COLORS["text_primary"]}},
        title={"text": f"Risk Level: {risk}", "font": {"size": 16, "color": COLORS["text_secondary"]}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["text_secondary"]},
            "bar": {"color": bar_color, "thickness": 0.7},
            "bgcolor": COLORS["bg_card"],
            "borderwidth": 0,
            "steps": [
                {"range": [0, 50], "color": "rgba(239,68,68,0.15)"},
                {"range": [50, 75], "color": "rgba(249,115,22,0.15)"},
                {"range": [75, 90], "color": "rgba(234,179,8,0.15)"},
                {"range": [90, 100], "color": "rgba(34,197,94,0.15)"},
            ],
            "threshold": {
                "line": {"color": COLORS["text_primary"], "width": 3},
                "thickness": 0.8,
                "value": score,
            },
        },
    ))
    fig.update_layout(
        height=300,
        margin=dict(t=40, b=10, l=30, r=30),
        **PLOTLY_TEMPLATE,
    )
    return fig


def _build_score_breakdown_radar(breakdown: dict) -> go.Figure:
    """Radar chart showing the score breakdown across dimensions."""
    categories = ["Missing\nValues", "Duplicates", "Outliers", "Type\nValidation", "Data\nDrift"]
    values = [
        breakdown.get("missing_score", 0),
        breakdown.get("duplicate_score", 0),
        breakdown.get("outlier_score", 0),
        breakdown.get("type_score", 0),
        breakdown.get("drift_score", 0),
    ]
    values.append(values[0])  # close the polygon
    categories.append(categories[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill="toself",
        fillcolor="rgba(59,130,246,0.2)",
        line=dict(color=COLORS["accent_blue"], width=2),
        marker=dict(size=6, color=COLORS["accent_cyan"]),
        name="Score",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor="rgba(30,41,59,0.5)",
            radialaxis=dict(
                visible=True, range=[0, 100],
                gridcolor="rgba(148,163,184,0.2)",
                tickfont=dict(size=9, color=COLORS["text_secondary"]),
            ),
            angularaxis=dict(
                gridcolor="rgba(148,163,184,0.2)",
                tickfont=dict(size=10, color=COLORS["text_primary"]),
            ),
        ),
        showlegend=False,
        height=350,
        margin=dict(t=30, b=30, l=60, r=60),
        **PLOTLY_TEMPLATE,
    )
    return fig


def _build_missing_heatmap(profile: dict) -> go.Figure:
    """Horizontal bar chart of missing values per column."""
    missing_pct = profile["missing"]["per_column_pct"]
    sorted_cols = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:20]

    if not sorted_cols or all(v == 0 for _, v in sorted_cols):
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No missing values detected!",
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=18, color=COLORS["accent_green"]),
            showarrow=False,
        )
        fig.update_layout(height=300, **PLOTLY_TEMPLATE)
        return fig

    cols = [c for c, _ in sorted_cols if _ > 0]
    vals = [v for _, v in sorted_cols if v > 0]

    colors = [
        COLORS["accent_green"] if v < 5
        else COLORS["accent_yellow"] if v < 20
        else COLORS["accent_orange"] if v < 50
        else COLORS["accent_red"]
        for v in vals
    ]

    fig = go.Figure(go.Bar(
        y=cols, x=vals,
        orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{v}%" for v in vals],
        textposition="outside",
        textfont=dict(color=COLORS["text_primary"], size=10),
    ))
    fig.update_layout(
        title=dict(text="Missing Values by Column", font=dict(size=14)),
        xaxis_title="Missing %",
        height=max(250, len(cols) * 28 + 80),
        margin=dict(t=40, b=40, l=120, r=40),
        xaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        yaxis=dict(autorange="reversed"),
        **PLOTLY_TEMPLATE,
    )
    return fig


def _build_outlier_chart(outlier_data: dict) -> go.Figure:
    """Grouped bar chart comparing outlier counts across methods per column."""
    per_col = outlier_data.get("per_column", {})
    if not per_col:
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No outliers detected!",
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=18, color=COLORS["accent_green"]),
            showarrow=False,
        )
        fig.update_layout(height=300, **PLOTLY_TEMPLATE)
        return fig

    # Sort by combined count
    sorted_cols = sorted(per_col.items(), key=lambda x: x[1]["combined_count"], reverse=True)[:15]
    sorted_cols = [(c, v) for c, v in sorted_cols if v["combined_count"] > 0]

    if not sorted_cols:
        fig = go.Figure()
        fig.add_annotation(
            text="✅ No outliers detected!",
            xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=18, color=COLORS["accent_green"]),
            showarrow=False,
        )
        fig.update_layout(height=300, **PLOTLY_TEMPLATE)
        return fig

    cols = [c for c, _ in sorted_cols]
    iqr_vals = [v["iqr"]["count"] for _, v in sorted_cols]
    zscore_vals = [v["zscore"]["count"] for _, v in sorted_cols]
    iso_vals = [v["isolation_forest"]["count"] for _, v in sorted_cols]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="IQR", x=cols, y=iqr_vals, marker_color=COLORS["accent_blue"]))
    fig.add_trace(go.Bar(name="Z-Score", x=cols, y=zscore_vals, marker_color=COLORS["accent_purple"]))
    fig.add_trace(go.Bar(name="Isolation Forest", x=cols, y=iso_vals, marker_color=COLORS["accent_cyan"]))

    fig.update_layout(
        barmode="group",
        title=dict(text="Outliers by Detection Method", font=dict(size=14)),
        yaxis_title="Count",
        height=350,
        margin=dict(t=40, b=40, l=50, r=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1, font=dict(size=10),
        ),
        xaxis=dict(tickangle=-45),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        **PLOTLY_TEMPLATE,
    )
    return fig


def _build_dtype_pie(profile: dict) -> go.Figure:
    """Pie chart of data types distribution."""
    dtypes = profile.get("dtypes", {})
    type_counts = {}
    for _, dtype in dtypes.items():
        label = str(dtype)
        if "int" in label:
            label = "Integer"
        elif "float" in label:
            label = "Float"
        elif "object" in label:
            label = "String/Object"
        elif "bool" in label:
            label = "Boolean"
        elif "datetime" in label:
            label = "DateTime"
        else:
            label = "Other"
        type_counts[label] = type_counts.get(label, 0) + 1

    fig = go.Figure(go.Pie(
        labels=list(type_counts.keys()),
        values=list(type_counts.values()),
        hole=0.55,
        textinfo="label+percent",
        textfont=dict(size=11),
        marker=dict(
            colors=[COLORS["accent_blue"], COLORS["accent_purple"], COLORS["accent_cyan"],
                    COLORS["accent_green"], COLORS["accent_yellow"], COLORS["accent_orange"]],
            line=dict(color=COLORS["bg_dark"], width=2),
        ),
    ))
    fig.update_layout(
        title=dict(text="Column Data Types", font=dict(size=14)),
        height=300,
        margin=dict(t=40, b=20, l=20, r=20),
        showlegend=True,
        legend=dict(font=dict(size=10)),
        **PLOTLY_TEMPLATE,
    )
    return fig


def _build_drift_comparison(drift_result: dict) -> go.Figure:
    """Bar chart showing drifted vs stable columns."""
    total = drift_result.get("total_columns_compared", 0)
    drifted = drift_result.get("total_drifted", 0)
    stable = total - drifted

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Stable", "Drifted"],
        y=[stable, drifted],
        marker=dict(
            color=[COLORS["accent_green"], COLORS["accent_red"]],
            line=dict(width=0),
        ),
        text=[stable, drifted],
        textposition="outside",
        textfont=dict(size=14, color=COLORS["text_primary"]),
    ))
    fig.update_layout(
        title=dict(text="Column Drift Status", font=dict(size=14)),
        yaxis_title="Number of Columns",
        height=300,
        margin=dict(t=40, b=40, l=50, r=30),
        yaxis=dict(gridcolor="rgba(148,163,184,0.15)"),
        **PLOTLY_TEMPLATE,
    )
    return fig


def _build_drift_detail_table(drift_result: dict) -> pd.DataFrame:
    """Build a DataFrame summarizing drift per column."""
    rows = []
    for col, details in drift_result.get("drift_summary", {}).items():
        row = {
            "Column": col,
            "Type": details.get("drift_type", "N/A"),
            "Drifted": "⚠️ Yes" if details.get("is_drifted") else "✅ No",
            "Missing Δ": f"{details.get('details', {}).get('missing_value_change', {}).get('change', 0)}%",
        }
        if details.get("drift_type") == "numeric":
            mc = details.get("details", {}).get("mean_change", {})
            row["Mean Δ"] = str(mc.get("absolute_change", "N/A"))
            dt = details.get("details", {}).get("distribution_test", {})
            row["KS p-value"] = str(dt.get("p_value", "N/A"))
        else:
            cc = details.get("details", {}).get("category_changes", {})
            row["Mean Δ"] = "N/A (categorical)"
            si = details.get("details", {}).get("stability_index", {})
            row["KS p-value"] = f"PSI: {si.get('psi', 'N/A')}"
        rows.append(row)

    if not rows:
        return pd.DataFrame({"Message": ["No common columns found for comparison"]})

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════
#  Tab 1: Dataset Profiling Handler
# ═══════════════════════════════════════════════════════════

def run_full_analysis(file, user_state=None, request: gr.Request = None):
    """Main handler for the profiling tab."""
    if file is None:
        raise gr.Error("Please upload a CSV or Excel file.")

    username = user_state if user_state else (request.username if request else None)

    try:
        filename = os.path.basename(file.name) if hasattr(file, 'name') else "uploaded_file"
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name if hasattr(file, 'name') else file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name if hasattr(file, 'name') else file)
        else:
            raise gr.Error("Unsupported file format. Use CSV or Excel.")

        # Run pipeline
        profile = profile_dataset(df)
        outliers = detect_outliers(df)
        score_result = calculate_quality_score(
            missing_pct=profile["missing"]["total_pct"],
            duplicate_pct=profile["duplicates"]["pct"],
            outlier_pct=outliers["total_outlier_pct"],
            invalid_type_pct=profile["type_issue_pct"],
        )
        ai_recs = get_ai_recommendations(profile, outliers, score_result)

        # Save to DB
        try:
            db = SessionLocal()
            entry = AnalysisHistory(
                filename=filename,
                username=username,
                quality_score=score_result["quality_score"],
                risk_level=score_result["risk_level"],
                row_count=profile["row_count"],
                col_count=profile["col_count"],
                missing_pct=profile["missing"]["total_pct"],
                duplicate_pct=profile["duplicates"]["pct"],
                outlier_pct=outliers["total_outlier_pct"],
                invalid_type_pct=profile["type_issue_pct"],
                ai_recommendations=ai_recs,
                summary_json={"score_breakdown": score_result["breakdown"]},
            )
            db.add(entry)
            db.commit()
            db.close()
        except Exception as db_err:
            print(f"DB save warning: {db_err}")

        # Build charts
        gauge = _build_quality_gauge(score_result["quality_score"], score_result["risk_level"])
        radar = _build_score_breakdown_radar(score_result["breakdown"])
        missing_chart = _build_missing_heatmap(profile)
        outlier_chart = _build_outlier_chart(outliers)
        dtype_chart = _build_dtype_pie(profile)

        # Build overview markdown
        overview_md = f"""
## 📊 Dataset Overview
| Metric | Value |
|--------|-------|
| **Rows** | {profile['row_count']:,} |
| **Columns** | {profile['col_count']} |
| **Total Missing** | {profile['missing']['total']} ({profile['missing']['total_pct']}%) |
| **Duplicate Rows** | {profile['duplicates']['count']} ({profile['duplicates']['pct']}%) |
| **Outlier Rows** | {outliers.get('total_outlier_rows', 0)} ({outliers['total_outlier_pct']}%) |
| **Type Issues** | {profile['type_issue_pct']}% |
"""
        # Column stats table
        col_stats_df = pd.DataFrame(profile["column_stats"]).T.reset_index()
        col_stats_df.rename(columns={"index": "Column"}, inplace=True)

        return (
            gauge,           # quality gauge
            radar,           # radar breakdown
            overview_md,     # overview markdown
            missing_chart,   # missing values chart
            outlier_chart,   # outlier chart
            dtype_chart,     # data type pie
            ai_recs,         # AI recommendations
            col_stats_df,    # column stats table
        )
    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Analysis failed: {str(e)}")


# ═══════════════════════════════════════════════════════════
#  PDF Export Handler
# ═══════════════════════════════════════════════════════════

def export_pdf(file):
    """Generate and return a PDF report file."""
    if file is None:
        raise gr.Error("Please upload a file first.")

    try:
        filename = os.path.basename(file.name) if hasattr(file, 'name') else "uploaded_file"
        if filename.endswith(".csv"):
            df = pd.read_csv(file.name if hasattr(file, 'name') else file)
        elif filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file.name if hasattr(file, 'name') else file)
        else:
            raise gr.Error("Unsupported file format.")

        profile = profile_dataset(df)
        outliers = detect_outliers(df)
        score_result = calculate_quality_score(
            missing_pct=profile["missing"]["total_pct"],
            duplicate_pct=profile["duplicates"]["pct"],
            outlier_pct=outliers["total_outlier_pct"],
            invalid_type_pct=profile["type_issue_pct"],
        )
        ai_recs = get_ai_recommendations(profile, outliers, score_result)

        pdf_bytes = generate_pdf_report(
            filename=filename,
            profile_data=profile,
            outlier_data=outliers,
            score_data=score_result,
            ai_recommendations=ai_recs,
        )

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tmp.write(pdf_bytes)
        tmp.close()
        return tmp.name

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"PDF generation failed: {str(e)}")


# ═══════════════════════════════════════════════════════════
#  Tab 2: Data Drift Handler
# ═══════════════════════════════════════════════════════════

def run_drift_analysis(baseline_file, target_file):
    """Main handler for drift detection tab."""
    if baseline_file is None or target_file is None:
        raise gr.Error("Please upload both a Baseline and a Target dataset.")

    try:
        def _read(f):
            name = os.path.basename(f.name) if hasattr(f, 'name') else ""
            if name.endswith(".csv"):
                return pd.read_csv(f.name)
            elif name.endswith((".xlsx", ".xls")):
                return pd.read_excel(f.name)
            else:
                raise gr.Error("Unsupported format. Use CSV or Excel.")

        baseline_df = _read(baseline_file)
        target_df = _read(target_file)
        drift_result = detect_drift(baseline_df, target_df)

        drift_chart = _build_drift_comparison(drift_result)
        drift_table = _build_drift_detail_table(drift_result)
        drift_score = drift_result.get("overall_drift_score", 100)

        summary_md = f"""
## 🔄 Drift Analysis Summary
| Metric | Value |
|--------|-------|
| **Columns Compared** | {drift_result.get('total_columns_compared', 0)} |
| **Drifted Columns** | {drift_result.get('total_drifted', 0)} |
| **Drift Score** | {drift_score}/100 |
| **Drifted Columns List** | {', '.join(drift_result.get('drifted_columns', [])) or 'None'} |
"""
        return drift_chart, drift_table, summary_md

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Drift analysis failed: {str(e)}")


# ═══════════════════════════════════════════════════════════
#  Tab 3: History Handler
# ═══════════════════════════════════════════════════════════

def load_history_by_username(username: str):
    """Fetch analysis history from the database, filtered by username."""
    try:
        db = SessionLocal()
        query = db.query(AnalysisHistory)
        if username:
            query = query.filter(AnalysisHistory.username == username)
        else:
            query = query.filter(AnalysisHistory.username == None)
        records = (
            query.order_by(AnalysisHistory.upload_time.desc())
            .limit(50)
            .all()
        )
        db.close()

        if not records:
            return pd.DataFrame({
                "Message": ["No analysis history yet. Upload a dataset to get started!"]
            })

        rows = []
        for r in records:
            rows.append({
                "Filename": r.filename,
                "Date": r.upload_time.strftime("%Y-%m-%d %H:%M") if r.upload_time else "N/A",
                "Score": f"{r.quality_score}/100",
                "Risk": r.risk_level,
                "Rows": r.row_count,
                "Cols": r.col_count,
                "Missing %": f"{r.missing_pct}%",
                "Duplicates %": f"{r.duplicate_pct}%",
                "Outliers %": f"{r.outlier_pct}%",
            })
        return pd.DataFrame(rows)
    except Exception as e:
        return pd.DataFrame({"Error": [f"Failed to load history: {str(e)}"]})


def build_chart_by_username(username: str):
    """Build a trend line chart of quality scores over time, filtered by username."""
    try:
        db = SessionLocal()
        query = db.query(AnalysisHistory)
        if username:
            query = query.filter(AnalysisHistory.username == username)
        else:
            query = query.filter(AnalysisHistory.username == None)
        records = (
            query.order_by(AnalysisHistory.upload_time.asc())
            .limit(50)
            .all()
        )
        db.close()

        if not records:
            fig = go.Figure()
            fig.add_annotation(
                text="No history data yet",
                xref="paper", yref="paper", x=0.5, y=0.5,
                font=dict(size=16, color=COLORS["text_secondary"]),
                showarrow=False,
            )
            fig.update_layout(height=300, **PLOTLY_TEMPLATE)
            return fig

        dates = [r.upload_time for r in records]
        scores = [r.quality_score for r in records]
        filenames = [r.filename for r in records]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=dates, y=scores,
            mode="lines+markers",
            line=dict(color=COLORS["accent_blue"], width=3),
            marker=dict(size=8, color=COLORS["accent_cyan"], line=dict(width=2, color=COLORS["accent_blue"])),
            text=filenames,
            hovertemplate="<b>%{text}</b><br>Score: %{y}/100<br>Date: %{x}<extra></extra>",
        ))
        fig.add_hline(
            y=90, line_dash="dash",
            line_color=COLORS["accent_green"],
            annotation_text="Low Risk (90+)",
            annotation_font_color=COLORS["accent_green"],
        )
        fig.add_hline(
            y=75, line_dash="dash",
            line_color=COLORS["accent_yellow"],
            annotation_text="Medium Risk (75)",
            annotation_font_color=COLORS["accent_yellow"],
        )
        fig.update_layout(
            title=dict(text="Quality Score Trend", font=dict(size=14)),
            xaxis_title="Date",
            yaxis_title="Quality Score",
            yaxis=dict(range=[0, 105], gridcolor="rgba(148,163,184,0.15)"),
            height=350,
            margin=dict(t=40, b=40, l=50, r=30),
            **PLOTLY_TEMPLATE,
        )
        return fig
    except Exception:
        fig = go.Figure()
        fig.update_layout(height=300, **PLOTLY_TEMPLATE)
        return fig


def load_history(user_state=None, request: gr.Request = None):
    """Callback for loading history."""
    username = user_state if user_state else (request.username if request else None)
    return load_history_by_username(username)


def build_history_chart(user_state=None, request: gr.Request = None):
    """Callback for loading history trend chart."""
    username = user_state if user_state else (request.username if request else None)
    return build_chart_by_username(username)


# ═══════════════════════════════════════════════════════════
#  Custom CSS
# ═══════════════════════════════════════════════════════════

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
}

.gradio-container {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%) !important;
    min-height: 100vh;
}

.main-header {
    text-align: center;
    padding: 20px 0;
    background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(139,92,246,0.1));
    border-radius: 16px;
    border: 1px solid rgba(59,130,246,0.2);
    margin-bottom: 20px;
}

.main-header h1 {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    margin-bottom: 5px !important;
}

.main-header p {
    color: #94a3b8 !important;
    font-size: 1rem !important;
}

.tab-nav button {
    background: rgba(30,41,59,0.8) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 10px !important;
    padding: 10px 24px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    color: white !important;
    border-color: transparent !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.4) !important;
}

.tab-nav button:hover {
    border-color: #3b82f6 !important;
    color: #f1f5f9 !important;
}

footer { display: none !important; }

.upload-area {
    border: 2px dashed rgba(59,130,246,0.4) !important;
    border-radius: 16px !important;
    background: rgba(30,41,59,0.5) !important;
    transition: all 0.3s ease !important;
}

.upload-area:hover {
    border-color: #3b82f6 !important;
    background: rgba(59,130,246,0.1) !important;
}

button.primary {
    background: linear-gradient(135deg, #3b82f6, #8b5cf6) !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    box-shadow: 0 4px 15px rgba(59,130,246,0.3) !important;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(59,130,246,0.5) !important;
}

button.secondary {
    background: rgba(30,41,59,0.8) !important;
    border: 1px solid rgba(59,130,246,0.3) !important;
    border-radius: 12px !important;
    color: #3b82f6 !important;
    font-weight: 600 !important;
}

.block {
    background: rgba(30,41,59,0.6) !important;
    border: 1px solid rgba(148,163,184,0.1) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(10px) !important;
}

.markdown-text h2 {
    color: #f1f5f9 !important;
    font-weight: 700 !important;
}

.markdown-text table {
    border-collapse: collapse !important;
    width: 100% !important;
}

.markdown-text th, .markdown-text td {
    border: 1px solid rgba(148,163,184,0.2) !important;
    padding: 8px 12px !important;
    color: #f1f5f9 !important;
}

.markdown-text th {
    background: rgba(59,130,246,0.2) !important;
    font-weight: 600 !important;
}

.dataframe {
    border-radius: 12px !important;
    overflow: hidden !important;
}

.scrollable-table {
    max-height: 400px !important;
    overflow-y: auto !important;
}
"""


def hash_password(password: str) -> str:
    """Return SHA-256 hash of a password."""
    return hashlib.sha256(password.encode()).hexdigest()


def login_handler(username, password):
    """Verify login credentials against Neon DB."""
    if not username or not password:
        return gr.update(visible=True), gr.update(visible=False), None, "⚠️ Please enter both username and password."

    username_clean = username.strip()
    try:
        db = SessionLocal()
        user = db.query(User).filter(User.username == username_clean).first()
        db.close()

        if user and user.password_hash == hash_password(password):
            return gr.update(visible=False), gr.update(visible=True), username_clean, ""
        else:
            # Check fallback default admin in APP_USERS if DB is empty or for initial local test
            fallback_pass = os.getenv("APP_USERS", "admin:admin123")
            for item in fallback_pass.split(","):
                if ":" in item:
                    u, p = item.split(":", 1)
                    if u.strip() == username_clean and p.strip() == password:
                        # Auto-register fallback user to DB for seamless persistence
                        try:
                            db = SessionLocal()
                            # Double check
                            exists = db.query(User).filter(User.username == username_clean).first()
                            if not exists:
                                db.add(User(username=username_clean, password_hash=hash_password(password)))
                                db.commit()
                            db.close()
                        except Exception:
                            pass
                        return gr.update(visible=False), gr.update(visible=True), username_clean, ""

            return gr.update(visible=True), gr.update(visible=False), None, "❌ Invalid username or password."
    except Exception as e:
        print(f"Auth error: {e}")
        # Fallback check on db connection error
        return gr.update(visible=True), gr.update(visible=False), None, f"❌ Database error: {str(e)}"


def signup_handler(username, password):
    """Register a new user in the PostgreSQL database."""
    if not username or not password:
        return "⚠️ Please enter both username and password."
    if len(username.strip()) < 3:
        return "⚠️ Username must be at least 3 characters long."
    if len(password) < 6:
        return "⚠️ Password must be at least 6 characters long."

    username_clean = username.strip()
    try:
        db = SessionLocal()
        existing = db.query(User).filter(User.username == username_clean).first()
        if existing:
            db.close()
            return "❌ Username already exists. Please choose a different one."

        new_user = User(
            username=username_clean,
            password_hash=hash_password(password)
        )
        db.add(new_user)
        db.commit()
        db.close()
        return "✅ Account created successfully! Switch to the Login tab and log in."
    except Exception as e:
        return f"❌ Database registration error: {str(e)}"


def handle_login(username, password):
    """Securely log the user in, then load their specific historical analysis and charts."""
    auth_hide, main_show, user, err = login_handler(username, password)
    if user:
        hist_df = load_history_by_username(user)
        trend_fig = build_chart_by_username(user)
        return auth_hide, main_show, user, err, hist_df, trend_fig
    else:
        return auth_hide, main_show, None, err, gr.update(), gr.update()


def create_gradio_app() -> gr.Blocks:
    """Build and return the Gradio Blocks app."""
    with gr.Blocks(
        css=CUSTOM_CSS,
        title="AI Data Quality Copilot",
        theme=gr.themes.Base(
            primary_hue=gr.themes.Color(
                c50="#eff6ff", c100="#dbeafe", c200="#bfdbfe", c300="#93c5fd",
                c400="#60a5fa", c500="#3b82f6", c600="#2563eb", c700="#1d4ed8",
                c800="#1e40af", c900="#1e3a8a", c950="#172554",
            ),
            secondary_hue="slate",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            block_background_fill="*neutral_800",
            block_background_fill_dark="*neutral_800",
            block_border_color="*neutral_700",
            input_background_fill="*neutral_800",
            input_background_fill_dark="*neutral_900",
        ),
    ) as app:

        # ── Authentication Session States ──
        username_state = gr.State(None)

        # ── 1. Secure Custom Login / Signup Screen ──
        with gr.Column(visible=True) as auth_container:
            gr.HTML("""
            <div style="max-width: 460px; margin: 80px auto 20px auto; padding: 30px; background: rgba(30, 41, 59, 0.75); border: 1px solid rgba(59, 130, 246, 0.3); border-radius: 20px; backdrop-filter: blur(16px); box-shadow: 0 20px 40px rgba(0,0,0,0.6); text-align: center;">
                <h1 style="background: linear-gradient(135deg, #3b82f6, #8b5cf6, #06b6d4); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.2rem; font-weight: 800; margin-bottom: 8px;">🔬 Data Quality Copilot</h1>
                <p style="color: #94a3b8; font-size: 0.95rem; line-height: 1.5;">Enterprise AI-Powered Profiling & Drift Analyzer</p>
                <div style="width: 50px; height: 3px; background: linear-gradient(90deg, #3b82f6, #8b5cf6); margin: 20px auto 0 auto; border-radius: 2px;"></div>
            </div>
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    pass
                with gr.Column(scale=2):
                    with gr.Tabs() as auth_tabs:
                        
                        # Sign In Tab
                        with gr.Tab("🔑 Sign In"):
                            gr.Markdown("Enter your credentials to access your private quality reports:")
                            login_user = gr.Textbox(label="Username", placeholder="e.g. admin", interactive=True)
                            login_pass = gr.Textbox(label="Password", type="password", placeholder="e.g. admin123", interactive=True)
                            login_btn = gr.Button("🔒 Secure Log In", variant="primary", size="lg")
                            login_error = gr.Markdown("", elem_id="login-error")
                        
                        # Sign Up Tab
                        with gr.Tab("📝 Sign Up"):
                            gr.Markdown("Create a new private account on your Neon PG database:")
                            signup_user = gr.Textbox(label="Desired Username", placeholder="Minimum 3 characters", interactive=True)
                            signup_pass = gr.Textbox(label="Desired Password", type="password", placeholder="Minimum 6 characters", interactive=True)
                            signup_btn = gr.Button("🚀 Create Free Account", variant="secondary", size="lg")
                            signup_status = gr.Markdown("", elem_id="signup-status")
                            
                with gr.Column(scale=1):
                    pass

        # ── 2. Primary Dashboard Workspace ──
        with gr.Column(visible=False) as main_container:
            
            # Premium Enterprise Header
            gr.HTML("""
            <div class="main-header">
                <h1>🔬 AI Data Quality Copilot</h1>
                <p>Enterprise Automated Profiling, Outlier Tracking, Distribution Drift & Recommendations</p>
            </div>
            """)

            with gr.Tabs() as tabs:

                # ── Tab 1: Dataset Profiling ──
                with gr.Tab("📊 Dataset Profiling", id="profiling"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            upload_file = gr.File(
                                label="📁 Upload Dataset (CSV / Excel)",
                                file_types=[".csv", ".xlsx", ".xls"],
                                elem_classes=["upload-area"],
                            )
                        with gr.Column(scale=1):
                            analyze_btn = gr.Button(
                                "🚀 Run Analysis", variant="primary", size="lg",
                            )
                            export_btn = gr.Button(
                                "📄 Export PDF Report", variant="secondary", size="lg",
                            )
                            pdf_output = gr.File(label="📥 Download Report", visible=True)

                    # Score section
                    with gr.Row():
                        with gr.Column(scale=1):
                            gauge_chart = gr.Plot(label="Quality Score")
                        with gr.Column(scale=1):
                            radar_chart = gr.Plot(label="Score Breakdown")

                    # Overview
                    overview_md = gr.Markdown("*Upload a dataset to see the analysis results*")

                    # Charts
                    with gr.Row():
                        with gr.Column(scale=1):
                            missing_chart = gr.Plot(label="Missing Values")
                        with gr.Column(scale=1):
                            outlier_chart = gr.Plot(label="Outlier Detection")

                    with gr.Row():
                        with gr.Column(scale=1):
                            dtype_chart = gr.Plot(label="Data Types")
                        with gr.Column(scale=1):
                            ai_recs_md = gr.Markdown("*AI recommendations will appear here after analysis*", label="🤖 AI Recommendations")

                    # Column stats
                    col_stats_table = gr.Dataframe(
                        label="📋 Column Statistics",
                        interactive=False,
                        wrap=True,
                        elem_classes=["scrollable-table"],
                    )

                    # Wire profiling events
                    analyze_btn.click(
                        fn=run_full_analysis,
                        inputs=[upload_file, username_state],
                        outputs=[
                            gauge_chart, radar_chart, overview_md,
                            missing_chart, outlier_chart, dtype_chart,
                            ai_recs_md, col_stats_table,
                        ],
                    )
                    export_btn.click(
                        fn=export_pdf,
                        inputs=[upload_file],
                        outputs=[pdf_output],
                    )

                # ── Tab 2: Data Drift ──
                with gr.Tab("🔄 Data Drift Detection", id="drift"):
                    gr.Markdown("""
    ### Compare Baseline vs Target Dataset
    Upload your **baseline** (production) dataset and a **target** (new) dataset to detect distribution shifts,
    mean changes, missing value changes, and category distribution changes.
                    """)

                    with gr.Row():
                        with gr.Column():
                            baseline_file = gr.File(
                                label="📁 Baseline Dataset (Production)",
                                file_types=[".csv", ".xlsx", ".xls"],
                                elem_classes=["upload-area"],
                            )
                        with gr.Column():
                            target_file = gr.File(
                                label="📁 Target Dataset (New Data)",
                                file_types=[".csv", ".xlsx", ".xls"],
                                elem_classes=["upload-area"],
                            )

                    drift_btn = gr.Button("🔍 Detect Drift", variant="primary", size="lg")

                    drift_summary_md = gr.Markdown("*Upload two datasets and click 'Detect Drift'*")

                    with gr.Row():
                        drift_chart = gr.Plot(label="Drift Overview")

                    drift_table = gr.Dataframe(
                        label="📋 Drift Detail by Column",
                        interactive=False,
                        wrap=True,
                        elem_classes=["scrollable-table"],
                    )

                    drift_btn.click(
                        fn=run_drift_analysis,
                        inputs=[baseline_file, target_file],
                        outputs=[drift_chart, drift_table, drift_summary_md],
                    )

                # ── Tab 3: Analysis History ──
                with gr.Tab("📜 Analysis History", id="history"):
                    gr.Markdown("### Past Analysis Results")

                    refresh_btn = gr.Button("🔄 Refresh History", variant="secondary")

                    history_chart = gr.Plot(label="Quality Score Trend")
                    history_table = gr.Dataframe(
                        label="📋 Analysis History",
                        interactive=False,
                        wrap=True,
                        elem_classes=["scrollable-table"],
                    )

                    # Wire history events using user private state
                    refresh_btn.click(
                        fn=load_history, 
                        inputs=[username_state], 
                        outputs=[history_table]
                    )
                    refresh_btn.click(
                        fn=build_history_chart, 
                        inputs=[username_state], 
                        outputs=[history_chart]
                    )

        # ── 3. Wire Custom Login / Sign Up Event Callbacks ──
        login_btn.click(
            fn=handle_login,
            inputs=[login_user, login_pass],
            outputs=[auth_container, main_container, username_state, login_error, history_table, history_chart]
        )
        signup_btn.click(
            fn=signup_handler,
            inputs=[signup_user, signup_pass],
            outputs=[signup_status]
        )

    return app
