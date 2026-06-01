"""
PDF Report Export Service
Generates a styled textual/tabular PDF report of the analysis findings with visual charts.
"""
import io
import os
import tempfile
from datetime import datetime, timezone
from fpdf import FPDF

# Configure matplotlib for headless plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class QualityReportPDF(FPDF):
    """Custom PDF class with header/footer for the Data Quality Report."""

    def __init__(self, filename: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_filename = filename

    def header(self):
        self.set_font("Helvetica", "B", 16)
        self.set_text_color(30, 58, 138)
        self.cell(0, 10, "AI Data Quality Copilot", new_x="LMARGIN", new_y="NEXT", align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, f"Dataset: {self.report_filename}", new_x="LMARGIN", new_y="NEXT", align="C")
        self.cell(
            0, 6,
            f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            new_x="LMARGIN", new_y="NEXT", align="C",
        )
        self.ln(5)
        self.set_draw_color(30, 58, 138)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def _check_and_add_page_for_image(pdf, height_needed=80):
    """If the remaining page height is less than height_needed, add a new page."""
    # A4 height is 297mm, margin is 20mm, so printable limit is ~277mm
    if pdf.get_y() + height_needed > 277:
        pdf.add_page()


def _generate_pdf_charts(profile_data: dict, outlier_data: dict, score_data: dict) -> dict:
    """Generate clean, publication-quality light-themed charts for the PDF report."""
    charts = {}
    temp_dir = tempfile.gettempdir()

    # 1. Quality Score Breakdown Bar Chart
    try:
        breakdown = score_data.get("breakdown", {})
        categories = ["Missing Values", "Duplicates", "Outliers", "Type Issues", "Data Drift"]
        values = [
            breakdown.get("missing_score", 0),
            breakdown.get("duplicate_score", 0),
            breakdown.get("outlier_score", 0),
            breakdown.get("type_score", 0),
            breakdown.get("drift_score", 0),
        ]

        fig, ax = plt.subplots(figsize=(6.5, 2.5))
        # Color code based on performance
        colors = ['#ef4444' if v < 50 else '#f97316' if v < 75 else '#eab308' if v < 90 else '#22c55e' for v in values]
        bars = ax.bar(categories, values, color=colors, edgecolor='none', width=0.55)
        
        ax.set_ylim(0, 105)
        ax.set_title("Quality Scores by Dimension (0-100)", fontsize=11, fontweight='bold', pad=10, color='#1e3a8a')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cbd5e1')
        ax.spines['bottom'].set_color('#cbd5e1')
        ax.tick_params(colors='#475569', labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8')
        ax.set_axisbelow(True)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, height + 2, f'{int(height)}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#1e293b')

        plt.tight_layout()
        path = os.path.join(temp_dir, "pdf_breakdown.png")
        plt.savefig(path, dpi=200, transparent=True)
        plt.close()
        charts["breakdown"] = path
    except Exception as e:
        print(f"Failed to generate breakdown chart: {e}")

    # 2. Missing Values Horizontal Chart
    try:
        missing_pct = profile_data.get("missing", {}).get("per_column_pct", {})
        sorted_cols = sorted(missing_pct.items(), key=lambda x: x[1], reverse=True)[:8]
        cols = [c[:18] for c, v in sorted_cols if v > 0]
        vals = [v for _, v in sorted_cols if v > 0]

        if cols:
            fig, ax = plt.subplots(figsize=(6.5, 2.5))
            colors = ['#22c55e' if v < 5 else '#eab308' if v < 20 else '#f97316' if v < 50 else '#ef4444' for v in vals]
            bars = ax.barh(cols, vals, color=colors, height=0.5)
            ax.set_xlim(0, max(100, max(vals) + 12))
            ax.invert_yaxis()
            ax.set_title("Missing Values Percentage by Column (Top 8)", fontsize=11, fontweight='bold', pad=10, color='#1e3a8a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.tick_params(colors='#475569', labelsize=9)
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='#94a3b8')
            ax.set_axisbelow(True)

            for bar in bars:
                width = bar.get_width()
                ax.text(width + 1.5, bar.get_y() + bar.get_height()/2.0, f'{width:.1f}%', ha='left', va='center', fontsize=9, fontweight='bold', color='#1e293b')

            plt.tight_layout()
            path = os.path.join(temp_dir, "pdf_missing.png")
            plt.savefig(path, dpi=200, transparent=True)
            plt.close()
            charts["missing"] = path
    except Exception as e:
        print(f"Failed to generate missing values chart: {e}")

    # 3. Outliers Chart
    try:
        per_col = outlier_data.get("per_column", {})
        sorted_cols = sorted(per_col.items(), key=lambda x: x[1]["combined_count"], reverse=True)[:6]
        sorted_cols = [(c, v) for c, v in sorted_cols if v["combined_count"] > 0]

        if sorted_cols:
            cols = [c[:12] for c, _ in sorted_cols]
            iqr_vals = [v["iqr"]["count"] for _, v in sorted_cols]
            zscore_vals = [v["zscore"]["count"] for _, v in sorted_cols]
            iso_vals = [v["isolation_forest"]["count"] for _, v in sorted_cols]

            x_indices = range(len(cols))
            width = 0.25

            fig, ax = plt.subplots(figsize=(6.5, 2.5))
            ax.bar([pos - width for pos in x_indices], iqr_vals, width, label='IQR', color='#3b82f6')
            ax.bar(x_indices, zscore_vals, width, label='Z-Score', color='#8b5cf6')
            ax.bar([pos + width for pos in x_indices], iso_vals, width, label='Iso. Forest', color='#06b6d4')

            ax.set_xticks(x_indices)
            ax.set_xticklabels(cols, fontsize=9, color='#475569')
            ax.set_title("Outliers Count by Detection Method (Top 6)", fontsize=11, fontweight='bold', pad=10, color='#1e3a8a')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.tick_params(colors='#475569', labelsize=9)
            ax.grid(axis='y', linestyle='--', alpha=0.3, color='#94a3b8')
            ax.set_axisbelow(True)
            ax.legend(fontsize=8.5, frameon=True, facecolor='white', edgecolor='none')

            plt.tight_layout()
            path = os.path.join(temp_dir, "pdf_outliers.png")
            plt.savefig(path, dpi=200, transparent=True)
            plt.close()
            charts["outliers"] = path
    except Exception as e:
        print(f"Failed to generate outliers chart: {e}")

    # 4. Data Types Distribution Pie Chart
    try:
        dtypes = profile_data.get("dtypes", {})
        type_counts = {}
        for _, dtype in dtypes.items():
            label = str(dtype)
            if "int" in label: label = "Integer"
            elif "float" in label: label = "Float"
            elif "object" in label: label = "String/Object"
            elif "bool" in label: label = "Boolean"
            elif "datetime" in label: label = "DateTime"
            else: label = "Other"
            type_counts[label] = type_counts.get(label, 0) + 1

        fig, ax = plt.subplots(figsize=(4.5, 2.5))
        colors = ['#3b82f6', '#8b5cf6', '#06b6d4', '#22c55e', '#eab308', '#f97316']
        
        wedges, texts, autotexts = ax.pie(
            type_counts.values(), 
            labels=type_counts.keys(), 
            autopct='%1.1f%%',
            colors=colors[:len(type_counts)],
            startangle=90,
            textprops=dict(fontsize=8.5, color='#1e293b'),
            wedgeprops=dict(width=0.45, edgecolor='w', linewidth=1.5)
        )
        
        # Style percentages inside the donut slices
        for autotext in autotexts:
            autotext.set_fontsize(8)
            autotext.set_weight('bold')
            autotext.set_color('#1e293b')

        ax.set_title("Column Data Types Distribution", fontsize=11, fontweight='bold', pad=10, color='#1e3a8a')
        plt.tight_layout()
        path = os.path.join(temp_dir, "pdf_dtypes.png")
        plt.savefig(path, dpi=200, transparent=True)
        plt.close()
        charts["dtypes"] = path
    except Exception as e:
        print(f"Failed to generate dtypes chart: {e}")

    return charts


def generate_pdf_report(
    filename: str,
    profile_data: dict,
    outlier_data: dict,
    score_data: dict,
    ai_recommendations: str,
) -> bytes:
    """
    Generate a complete PDF report including high-quality visual charts.

    Returns:
        bytes of the PDF file content.
    """
    # Generate visual charts first
    charts = _generate_pdf_charts(profile_data, outlier_data, score_data)

    try:
        pdf = QualityReportPDF(filename)
        pdf.alias_nb_pages()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=20)

        # ── Quality Score Section ──
        _section_title(pdf, "Quality Score")
        score = score_data.get("quality_score", 0)
        risk = score_data.get("risk_level", "Unknown")

        if score >= 90:
            score_color = (34, 197, 94)
        elif score >= 75:
            score_color = (234, 179, 8)
        elif score >= 50:
            score_color = (249, 115, 22)
        else:
            score_color = (239, 68, 68)

        pdf.set_font("Helvetica", "B", 28)
        pdf.set_text_color(*score_color)
        pdf.cell(0, 15, f"{int(score)}/100", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 14)
        pdf.cell(0, 8, f"Risk Level: {risk}", new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_text_color(0, 0, 0)
        pdf.ln(5)

        # ── Quality Score Breakdown Chart ──
        if "breakdown" in charts:
            _check_and_add_page_for_image(pdf, 75)
            pdf.image(charts["breakdown"], x=10, y=pdf.get_y(), w=190)
            pdf.ln(72)

        # Score breakdown table
        breakdown = score_data.get("breakdown", {})
        _section_subtitle(pdf, "Score Breakdown Details")
        _add_table(pdf, [
            ["Component", "Score", "Weight"],
            ["Missing Values", str(breakdown.get("missing_score", 0)), "35%"],
            ["Duplicates", str(breakdown.get("duplicate_score", 0)), "20%"],
            ["Outliers", str(breakdown.get("outlier_score", 0)), "15%"],
            ["Type Validation", str(breakdown.get("type_score", 0)), "15%"],
            ["Data Drift", str(breakdown.get("drift_score", 0)), "15%"],
        ])

        # ── Dataset Overview & Columns Distribution ──
        pdf.add_page()
        _section_title(pdf, "Dataset Overview")
        _add_table(pdf, [
            ["Metric", "Value"],
            ["Rows", str(profile_data.get("row_count", 0))],
            ["Columns", str(profile_data.get("col_count", 0))],
            ["Total Missing %", f"{profile_data.get('missing', {}).get('total_pct', 0)}%"],
            ["Duplicate Rows", str(profile_data.get("duplicates", {}).get("count", 0))],
            ["Outlier Rows %", f"{outlier_data.get('total_outlier_pct', 0)}%"],
        ])

        if "dtypes" in charts:
            _check_and_add_page_for_image(pdf, 75)
            pdf.image(charts["dtypes"], x=40, y=pdf.get_y(), w=130)
            pdf.ln(72)

        # ── Missing Values Detail & Chart ──
        missing = profile_data.get("missing", {}).get("per_column_pct", {})
        has_missing = any(pct > 0 for pct in missing.values())
        if missing and has_missing:
            pdf.add_page()
            _section_title(pdf, "Missing Values by Column")
            
            if "missing" in charts:
                pdf.image(charts["missing"], x=10, y=pdf.get_y(), w=190)
                pdf.ln(72)

            rows = [["Column", "Missing %"]]
            for col, pct in sorted(missing.items(), key=lambda x: x[1], reverse=True):
                if pct > 0:
                    rows.append([str(col)[:30], f"{pct}%"])
            if len(rows) > 1:
                _section_subtitle(pdf, "Top Column Missing Statistics")
                _add_table(pdf, rows[:16])  # top 15 columns

        # ── Outlier Summary & Chart ──
        outlier_cols = outlier_data.get("per_column", {})
        has_outliers = any(info["combined_count"] > 0 for info in outlier_cols.values())
        if outlier_cols and has_outliers:
            pdf.add_page()
            _section_title(pdf, "Outlier Analysis")

            if "outliers" in charts:
                pdf.image(charts["outliers"], x=10, y=pdf.get_y(), w=190)
                pdf.ln(72)

            rows = [["Column", "IQR", "Z-Score", "Iso. Forest", "Combined"]]
            for col, info in sorted(outlier_cols.items(), key=lambda x: x[1]["combined_count"], reverse=True):
                if info["combined_count"] > 0:
                    rows.append([
                        str(col)[:25],
                        str(info.get("iqr", {}).get("count", 0)),
                        str(info.get("zscore", {}).get("count", 0)),
                        str(info.get("isolation_forest", {}).get("count", 0)),
                        str(info["combined_count"]),
                    ])
            if len(rows) > 1:
                _section_subtitle(pdf, "Outliers Breakdown per Column")
                _add_table(pdf, rows[:16])

        # ── AI Recommendations ──
        if ai_recommendations:
            pdf.add_page()
            _section_title(pdf, "AI-Generated Recommendations")
            pdf.set_font("Helvetica", "", 10)
            pdf.set_text_color(30, 30, 30)
            # Clean markdown formatting for PDF
            clean_text = ai_recommendations.replace("**", "").replace("##", "").replace("###", "")
            pdf.multi_cell(0, 6, clean_text)

        return pdf.output()

    finally:
        # Clean up temp files automatically to avoid storage leaks
        for path in charts.values():
            try:
                if os.path.exists(path):
                    os.remove(path)
            except Exception as clean_err:
                print(f"Error removing temp chart file {path}: {clean_err}")


def _section_title(pdf: FPDF, title: str):
    pdf.ln(3)
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_text_color(30, 58, 138)
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(30, 58, 138)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_text_color(0, 0, 0)


def _section_subtitle(pdf: FPDF, title: str):
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(60, 60, 60)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_text_color(0, 0, 0)


def _add_table(pdf: FPDF, data: list):
    """Add a simple table to the PDF."""
    if not data:
        return

    col_count = len(data[0])
    col_width = (190) / col_count

    # Header row
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_fill_color(240, 242, 255)
    for cell in data[0]:
        pdf.cell(col_width, 7, str(cell), border=1, fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 9)
    for row in data[1:]:
        for cell in row:
            pdf.cell(col_width, 6, str(cell), border=1)
        pdf.ln()
    pdf.ln(3)
