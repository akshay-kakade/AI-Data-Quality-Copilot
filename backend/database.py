"""
Database connection setup for Neon PostgreSQL using SQLAlchemy.
"""
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

DATABASE_URL = os.getenv("NEON_POSTGRESS_SQL_CONNECTION")

if not DATABASE_URL:
    raise ValueError("NEON_POSTGRESS_SQL_CONNECTION environment variable is not set.")

engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5, max_overflow=10)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency that yields a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables in the database."""
    Base.metadata.create_all(bind=engine)
    # Safe migration: add username column if it doesn't exist in existing databases
    try:
        with engine.connect() as conn:
            result = conn.execute(text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='analysis_history' AND column_name='username'"
            ))
            if not result.fetchone():
                conn.execute(text("ALTER TABLE analysis_history ADD COLUMN username VARCHAR"))
                conn.commit()
                print("Database migration: successfully added 'username' column to 'analysis_history' table.")
    except Exception as e:
        print(f"Database migration warning: {e}")
