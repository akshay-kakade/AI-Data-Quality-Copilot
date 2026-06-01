"""
SQLAlchemy ORM models for the AI Data Quality Copilot.
"""
import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, JSON
from backend.database import Base


class AnalysisHistory(Base):
    """Stores metadata and results for each dataset analysis run."""
    __tablename__ = "analysis_history"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    filename = Column(String, nullable=False)
    upload_time = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    quality_score = Column(Float, nullable=False)
    risk_level = Column(String, nullable=False)
    row_count = Column(Integer, nullable=False)
    col_count = Column(Integer, nullable=False)
    missing_pct = Column(Float, default=0.0)
    duplicate_pct = Column(Float, default=0.0)
    outlier_pct = Column(Float, default=0.0)
    invalid_type_pct = Column(Float, default=0.0)
    ai_recommendations = Column(Text, default="")
    summary_json = Column(JSON, default=dict)
    username = Column(String, nullable=True, default=None)


class User(Base):
    """Stores user registration credentials securely with hashed passwords."""
    __tablename__ = "users"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    username = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

