#!/usr/bin/env python3
"""
REMHART Digital Twin - Database Seeding Script
===============================================
Populates the database with initial grid data for testing and demonstration.

This script:
1. Initializes database tables
2. Generates realistic grid data
3. Populates all measurement tables

Usage:
    python seed_database.py [--points NUM] [--scenario SCENARIO]

Arguments:
    --points NUM        Number of data points to generate (default: 1000)
    --scenario SCENARIO Scenario type: normal, voltage_sag, overcurrent,
                       frequency_drift, mixed (default: normal)
    --clear            Clear existing data before seeding

Examples:
    python seed_database.py
    python seed_database.py --points 5000 --scenario mixed
    python seed_database.py --clear --points 2000

Author: REMHART Team
Date: 2025
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

# Add app to path
sys.path.insert(0, str(Path(__file__).parent))

from app.database import engine, SessionLocal, Base, check_db_connection
from app.models.db_models import (
    DateTimeTable, VoltageTable, CurrentTable, FrequencyTable,
    ActivePowerTable, ReactivePowerTable
)
from app.services.data_generator import GridDataGenerator


def init_database():
    """Initialize database tables."""
    print("Initializing database tables...")
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully")


def clear_data(db: Session):
    """Clear all existing grid data."""
    print("\nClearing existing data...")

    # Delete in correct order (children first)
    db.query(ReactivePowerTable).delete()
    db.query(ActivePowerTable).delete()
    db.query(FrequencyTable).delete()
    db.query(CurrentTable).delete()
    db.query(VoltageTable).delete()
    db.query(DateTimeTable).delete()

    db.commit()
    print("✓ All existing data cleared")


def seed_grid_data(db: Session, num_points: int = 1000, scenario: str = "normal"):
    """
    Seed the database with grid data.

    Args:
        db: Database session
        num_points: Number of data points to generate
        scenario: Data generation scenario
    """
    print(f"\nGenerating {num_points} data points ({scenario} scenario)...")

    generator = GridDataGenerator()

    # Generate time series data
    start_time = datetime.now() - timedelta(hours=24)  # Start from 24 hours ago
    interval_seconds = 3  # 3 seconds between readings

    data_points = []
    current_time = start_time

    for i in range(num_points):
        # Determine if this should be an anomaly based on scenario
        force_anomaly = False

        if scenario == "voltage_sag" and num_points * 0.4 <= i <= num_points * 0.6:
            force_anomaly = True
        elif scenario == "overcurrent" and num_points * 0.3 <= i <= num_points * 0.7:
            force_anomaly = True
        elif scenario == "frequency_drift" and num_points * 0.2 <= i <= num_points * 0.8:
            force_anomaly = True
        elif scenario == "mixed" and i % 10 == 0:
            force_anomaly = True

        data_point = generator.generate_single_datapoint(
            timestamp=current_time,
            force_anomaly=force_anomaly
        )
        data_points.append(data_point)
        current_time += timedelta(seconds=interval_seconds)

        # Progress indicator
        if (i + 1) % 100 == 0:
            print(f"  Generated {i + 1}/{num_points} points...")

    print(f"✓ Generated {len(data_points)} data points")

    # Insert into database
    print("\nInserting data into database...")
    inserted_count = 0

    for i, data_point in enumerate(data_points):
        try:
            # Create timestamp record
            dt_record = DateTimeTable(
                timestamp=data_point["timestamp"],
                is_simulation=False,
                simulation_id=None,
                simulation_name="Initial Seed Data",
                simulation_scenario=scenario
            )
            db.add(dt_record)
            db.flush()  # Get the ID

            # Create voltage record
            voltage_record = VoltageTable(
                timestamp_id=dt_record.id,
                phaseA=data_point["voltage"]["phaseA"],
                phaseB=data_point["voltage"]["phaseB"],
                phaseC=data_point["voltage"]["phaseC"],
                average=data_point["voltage"]["average"]
            )
            db.add(voltage_record)

            # Create current record
            current_record = CurrentTable(
                timestamp_id=dt_record.id,
                phaseA=data_point["current"]["phaseA"],
                phaseB=data_point["current"]["phaseB"],
                phaseC=data_point["current"]["phaseC"],
                average=data_point["current"]["average"]
            )
            db.add(current_record)

            # Create frequency record
            frequency_record = FrequencyTable(
                timestamp_id=dt_record.id,
                frequency_value=data_point["frequency"]["frequency_value"]
            )
            db.add(frequency_record)

            # Create active power record
            active_power_record = ActivePowerTable(
                timestamp_id=dt_record.id,
                phaseA=data_point["active_power"]["phaseA"],
                phaseB=data_point["active_power"]["phaseB"],
                phaseC=data_point["active_power"]["phaseC"],
                total=data_point["active_power"]["total"]
            )
            db.add(active_power_record)

            # Create reactive power record
            reactive_power_record = ReactivePowerTable(
                timestamp_id=dt_record.id,
                phaseA=data_point["reactive_power"]["phaseA"],
                phaseB=data_point["reactive_power"]["phaseB"],
                phaseC=data_point["reactive_power"]["phaseC"],
                total=data_point["reactive_power"]["total"]
            )
            db.add(reactive_power_record)

            inserted_count += 1

            # Commit in batches for better performance
            if inserted_count % 100 == 0:
                db.commit()
                print(f"  Inserted {inserted_count}/{num_points} records...")

        except Exception as e:
            print(f"Error inserting record {i}: {e}")
            db.rollback()
            continue

    # Final commit
    db.commit()
    print(f"✓ Successfully inserted {inserted_count} complete data records")


def verify_data(db: Session):
    """Verify that data was inserted correctly."""
    print("\nVerifying data...")

    dt_count = db.query(DateTimeTable).count()
    voltage_count = db.query(VoltageTable).count()
    current_count = db.query(CurrentTable).count()
    frequency_count = db.query(FrequencyTable).count()
    active_power_count = db.query(ActivePowerTable).count()
    reactive_power_count = db.query(ReactivePowerTable).count()

    print(f"  Timestamps:      {dt_count}")
    print(f"  Voltage records: {voltage_count}")
    print(f"  Current records: {current_count}")
    print(f"  Frequency records: {frequency_count}")
    print(f"  Active power:    {active_power_count}")
    print(f"  Reactive power:  {reactive_power_count}")

    if dt_count == voltage_count == current_count == frequency_count == active_power_count == reactive_power_count:
        print("✓ Data integrity verified - all counts match!")
        return True
    else:
        print("✗ Data integrity issue - counts don't match!")
        return False


def get_sample_data(db: Session, limit: int = 5):
    """Display sample data from the database."""
    print(f"\nSample data (first {limit} records):")
    print("-" * 80)

    records = db.query(DateTimeTable).order_by(DateTimeTable.timestamp).limit(limit).all()

    for record in records:
        voltage = record.voltage[0] if record.voltage else None
        current = record.current[0] if record.current else None
        frequency = record.frequency[0] if record.frequency else None

        print(f"\nTimestamp: {record.timestamp}")
        if voltage:
            print(f"  Voltage: A={voltage.phaseA}V, B={voltage.phaseB}V, C={voltage.phaseC}V")
        if current:
            print(f"  Current: A={current.phaseA}A, B={current.phaseB}A, C={current.phaseC}A")
        if frequency:
            print(f"  Frequency: {frequency.frequency_value}Hz")


def main():
    """Main seeding function."""
    parser = argparse.ArgumentParser(
        description="Seed REMHART database with grid data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Default: 1000 points, normal scenario
  %(prog)s --points 5000                # 5000 points
  %(prog)s --scenario mixed             # Mixed anomaly scenario
  %(prog)s --clear --points 2000        # Clear existing data first
        """
    )

    parser.add_argument(
        '--points',
        type=int,
        default=1000,
        help='Number of data points to generate (default: 1000)'
    )

    parser.add_argument(
        '--scenario',
        type=str,
        choices=['normal', 'voltage_sag', 'overcurrent', 'frequency_drift', 'mixed'],
        default='normal',
        help='Data generation scenario (default: normal)'
    )

    parser.add_argument(
        '--clear',
        action='store_true',
        help='Clear existing data before seeding'
    )

    parser.add_argument(
        '--sample',
        action='store_true',
        help='Display sample data after seeding'
    )

    args = parser.parse_args()

    print("=" * 80)
    print(" REMHART DIGITAL TWIN - DATABASE SEEDING")
    print("=" * 80)

    # Check database connection
    if not check_db_connection():
        print("\n✗ Failed to connect to database!")
        print("Please check your .env file and ensure MySQL is running.")
        return 1

    # Initialize database
    init_database()

    # Create session
    db = SessionLocal()

    try:
        # Clear data if requested
        if args.clear:
            clear_data(db)

        # Seed data
        seed_grid_data(db, num_points=args.points, scenario=args.scenario)

        # Verify data
        verify_data(db)

        # Show sample data if requested
        if args.sample:
            get_sample_data(db)

        print("\n" + "=" * 80)
        print(" ✓ DATABASE SEEDING COMPLETE!")
        print("=" * 80)
        print(f"\nSeeded {args.points} data points with '{args.scenario}' scenario.")
        print("Your REMHART Digital Twin is ready to use!")

        return 0

    except Exception as e:
        print(f"\n✗ Error during seeding: {e}")
        import traceback
        traceback.print_exc()
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    exit(main())
