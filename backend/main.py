"""
FastAPI Backend for AI Data Quality Copilot.
Provides REST API endpoints and mounts the Gradio UI.
"""
import io
import os
import traceback
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from dotenv import load_dotenv

load_dotenv()

from backend.database import get_db, init_db
from backend.models import AnalysisHistory, User
from backend.services.profiler import profile_dataset
from backend.services.outliers import detect_outliers
from backend.services.drift import detect_drift
from backend.services.scorer import calculate_quality_score
from backend.services.ai import get_ai_recommendations
from backend.services.report import generate_pdf_report

app = FastAPI(
    title="AI Data Quality Copilot",
    description="Automated data quality analysis powered by AI",
    version="1.0.0",
)


@app.on_event("startup")
def startup():
    """Initialize database tables on startup."""
    init_db()


# ─────────────────────── Helper ───────────────────────

def _read_uploaded_file(file: UploadFile) -> pd.DataFrame:
    """Read an uploaded CSV or Excel file into a DataFrame."""
    filename = file.filename.lower() if file.filename else ""
    content = file.file.read()

    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(content))
    elif filename.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(content))
    else:
        raise HTTPException(
            status_code=400,
            detail="Unsupported file format. Please upload CSV or Excel (.xlsx/.xls) files.",
        )


# ─────────────────────── Endpoints ───────────────────────

@app.post("/api/analyze")
async def analyze_dataset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Full dataset analysis: profiling, outliers, quality score, AI recommendations.
    """
    try:
        df = _read_uploaded_file(file)

        # Run analysis pipeline
        profile = profile_dataset(df)
        outliers = detect_outliers(df)

        score_result = calculate_quality_score(
            missing_pct=profile["missing"]["total_pct"],
            duplicate_pct=profile["duplicates"]["pct"],
            outlier_pct=outliers["total_outlier_pct"],
            invalid_type_pct=profile["type_issue_pct"],
        )

        ai_recs = get_ai_recommendations(profile, outliers, score_result)

        # Save to database
        history_entry = AnalysisHistory(
            filename=file.filename,
            quality_score=score_result["quality_score"],
            risk_level=score_result["risk_level"],
            row_count=profile["row_count"],
            col_count=profile["col_count"],
            missing_pct=profile["missing"]["total_pct"],
            duplicate_pct=profile["duplicates"]["pct"],
            outlier_pct=outliers["total_outlier_pct"],
            invalid_type_pct=profile["type_issue_pct"],
            ai_recommendations=ai_recs,
            summary_json={
                "score_breakdown": score_result["breakdown"],
                "missing_top_cols": dict(
                    sorted(
                        profile["missing"]["per_column_pct"].items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:5]
                ),
            },
        )
        db.add(history_entry)
        db.commit()
        db.refresh(history_entry)

        return {
            "id": history_entry.id,
            "filename": file.filename,
            "profile": profile,
            "outliers": outliers,
            "score": score_result,
            "ai_recommendations": ai_recs,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/api/drift")
async def detect_data_drift(
    baseline: UploadFile = File(...),
    target: UploadFile = File(...),
):
    """Compare baseline and target datasets for data drift."""
    try:
        baseline_df = _read_uploaded_file(baseline)
        target_df = _read_uploaded_file(target)

        drift_result = detect_drift(baseline_df, target_df)

        return {
            "baseline_file": baseline.filename,
            "target_file": target.filename,
            "drift": drift_result,
        }

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Drift detection failed: {str(e)}")


@app.post("/api/report")
async def export_pdf_report(file: UploadFile = File(...)):
    """Generate and download a PDF quality report."""
    try:
        df = _read_uploaded_file(file)

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
            filename=file.filename,
            profile_data=profile,
            outlier_data=outliers,
            score_data=score_result,
            ai_recommendations=ai_recs,
        )

        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="quality_report_{file.filename}.pdf"'
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/history")
async def get_analysis_history(db: Session = Depends(get_db)):
    """Fetch all past analysis records, most recent first."""
    try:
        records = (
            db.query(AnalysisHistory)
            .order_by(AnalysisHistory.upload_time.desc())
            .limit(50)
            .all()
        )
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "upload_time": r.upload_time.isoformat() if r.upload_time else None,
                "quality_score": r.quality_score,
                "risk_level": r.risk_level,
                "row_count": r.row_count,
                "col_count": r.col_count,
                "missing_pct": r.missing_pct,
                "duplicate_pct": r.duplicate_pct,
                "outlier_pct": r.outlier_pct,
            }
            for r in records
        ]
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@app.delete("/api/history/{record_id}")
async def delete_history_record(record_id: str, db: Session = Depends(get_db)):
    """Delete a specific analysis history record."""
    record = db.query(AnalysisHistory).filter(AnalysisHistory.id == record_id).first()
    if not record:
        raise HTTPException(status_code=404, detail="Record not found")
    db.delete(record)
    db.commit()
    return {"message": "Record deleted successfully"}
