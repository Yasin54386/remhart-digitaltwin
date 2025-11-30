#!/usr/bin/env python3
"""
Quick script to check if database has data
"""

from app.database import SessionLocal
from app.models.db_models import DateTimeTable
from sqlalchemy import func

db = SessionLocal()

# Count total records
total = db.query(func.count(DateTimeTable.id)).scalar()
print(f"Total records in database: {total}")

if total > 0:
    # Get first and last timestamps
    first = db.query(DateTimeTable).order_by(DateTimeTable.timestamp.asc()).first()
    last = db.query(DateTimeTable).order_by(DateTimeTable.timestamp.desc()).first()

    print(f"\nFirst record: {first.timestamp}")
    print(f"Last record: {last.timestamp}")

    # Check simulation vs real-time
    sim_count = db.query(func.count(DateTimeTable.id)).filter(DateTimeTable.is_simulation == True).scalar()
    real_count = db.query(func.count(DateTimeTable.id)).filter(DateTimeTable.is_simulation == False).scalar()

    print(f"\nSimulation records: {sim_count}")
    print(f"Real-time records: {real_count}")

    # Show sample records
    print("\nSample records:")
    samples = db.query(DateTimeTable).limit(5).all()
    for sample in samples:
        print(f"  ID: {sample.id}, Time: {sample.timestamp}, Simulation: {sample.is_simulation}")
else:
    print("\n⚠️  Database is empty! No records found.")
    print("\nTo add test data, run:")
    print("  curl -X GET 'http://localhost:8001/api/grid/generate?num_points=100&scenario=normal' \\")
    print("       -H 'Authorization: Bearer YOUR_TOKEN'")

db.close()
