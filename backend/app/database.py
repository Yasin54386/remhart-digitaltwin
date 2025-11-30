"""
REMHART Digital Twin - Database Configuration
============================================
This module handles all database connections using SQLAlchemy ORM.
Connects to MySQL database with your existing schema.

Author: REMHART Team
Date: 2025
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from typing import Generator
import os
from dotenv import load_dotenv
from sqlalchemy import text

# Load environment variables
load_dotenv()

# Database URL from environment
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "mysql+pymysql://root:@localhost:3306/remhart_db"
)


# Create SQLAlchemy engine
# pool_pre_ping=True ensures connections are valid before using
# echo=True for development (shows SQL queries in console)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    echo=False,  # Set to True for debugging SQL queries
    pool_size=10,  # Connection pool size
    max_overflow=20  # Additional connections when pool is full
)

# Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Base class for all ORM models
Base = declarative_base()


def get_db() -> Generator:
    """
    Dependency function to get database session.
    
    Usage in FastAPI endpoints:
        @app.get("/data")
        def get_data(db: Session = Depends(get_db)):
            # Use db here
            
    Yields:
        Session: SQLAlchemy database session
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database - create all tables if they don't exist.
    Call this on application startup.
    """
    # Import all models here to ensure they're registered
    from app.models import db_models
    
    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")


def check_db_connection() -> bool:
    """
    Check if database connection is working.
    
    Returns:
        bool: True if connection successful, False otherwise
    """
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print("Database connection successful")
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False