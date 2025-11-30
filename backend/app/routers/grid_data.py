"""
REMHART Digital Twin - Grid Data Router
========================================
Handles all grid data operations including CRUD and querying.

Endpoints:
- POST /api/grid/data - Add new grid data point
- GET /api/grid/data - Get grid data (with filters)
- GET /api/grid/data/latest - Get latest data point
- GET /api/grid/status - Get current grid status
- POST /api/grid/generate - Generate test data

Author: REMHART Team
Date: 2025
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import math

from app.database import get_db
from app.models.db_models import (
    DateTimeTable, VoltageTable, CurrentTable, 
    FrequencyTable, ActivePowerTable, ReactivePowerTable
)
from app.schemas.grid_data import GridDataPoint, GridDataResponse, GridStatus
from app.utils.security import get_current_user, check_role
from app.services.data_generator import grid_generator
from app.routers.websocket_router import manager

router = APIRouter()




@router.post("/data", response_model=dict)
async def add_grid_data(
    data: GridDataPoint,
    db: Session = Depends(get_db),
    current_user: dict = Depends(check_role("operator"))
):
    """
    Add new grid data point to database and broadcast to WebSocket clients.
    """
    try:
        # Create datetime entry
        dt_entry = DateTimeTable(timestamp=data.timestamp)
        db.add(dt_entry)
        db.flush()
        
        # Create related entries
        db.add(VoltageTable(timestamp_id=dt_entry.id, **data.voltage.dict()))
        db.add(CurrentTable(timestamp_id=dt_entry.id, **data.current.dict()))
        db.add(FrequencyTable(timestamp_id=dt_entry.id, **data.frequency.dict()))
        db.add(ActivePowerTable(timestamp_id=dt_entry.id, **data.active_power.dict()))
        db.add(ReactivePowerTable(timestamp_id=dt_entry.id, **data.reactive_power.dict()))
        
        db.commit()
        
        # Prepare data for broadcast
        broadcast_data = {
            "id": dt_entry.id,
            "timestamp": data.timestamp.isoformat(),
            "voltage": data.voltage.dict(),
            "current": data.current.dict(),
            "frequency": {"value": data.frequency.frequency_value},
            "active_power": data.active_power.dict(),
            "reactive_power": data.reactive_power.dict()
        }
        
        # Broadcast to all WebSocket clients immediately
        await manager.broadcast({
            "type": "new_data",
            "data": broadcast_data
        })
        
        return {
            "success": True,
            "message": "Grid data added and broadcasted successfully",
            "data_id": dt_entry.id,
            "timestamp": dt_entry.timestamp
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding grid data: {str(e)}")

@router.get("/data", response_model=List[dict])
async def get_grid_data(
    limit: int = Query(default=100, ge=1, le=1000, description="Number of records to return"),
    offset: int = Query(default=0, ge=0, description="Number of records to skip"),
    start_time: Optional[datetime] = Query(default=None, description="Start time filter"),
    end_time: Optional[datetime] = Query(default=None, description="End time filter"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get grid data with optional filters.
    
    Returns time-series grid data for visualization and analysis.
    
    Args:
        limit: Maximum number of records (1-1000)
        offset: Records to skip for pagination
        start_time: Filter data after this time
        end_time: Filter data before this time
        db: Database session
        current_user: Authenticated user
        
    Returns:
        List of grid data points
        
    Example:
        GET /api/grid/data?limit=50&start_time=2025-01-15T00:00:00
    """
    # Build query
    query = db.query(DateTimeTable).order_by(desc(DateTimeTable.timestamp))
    
    # Apply time filters
    if start_time:
        query = query.filter(DateTimeTable.timestamp >= start_time)
    if end_time:
        query = query.filter(DateTimeTable.timestamp <= end_time)
    
    # Apply pagination
    query = query.offset(offset).limit(limit)
    
    # Execute query
    datetime_records = query.all()
    
    # Build response
    result = []
    for dt_record in datetime_records:
        result.append({
            "id": dt_record.id,
            "timestamp": dt_record.timestamp.isoformat(),
            "voltage": {
                "phaseA": dt_record.voltage[0].phaseA if dt_record.voltage else 0,
                "phaseB": dt_record.voltage[0].phaseB if dt_record.voltage else 0,
                "phaseC": dt_record.voltage[0].phaseC if dt_record.voltage else 0,
                "average": dt_record.voltage[0].average if dt_record.voltage else 0
            },
            "current": {
                "phaseA": dt_record.current[0].phaseA if dt_record.current else 0,
                "phaseB": dt_record.current[0].phaseB if dt_record.current else 0,
                "phaseC": dt_record.current[0].phaseC if dt_record.current else 0,
                "average": dt_record.current[0].average if dt_record.current else 0
            },
            "frequency": {
                "value": dt_record.frequency[0].frequency_value if dt_record.frequency else 50.0
            },
            "active_power": {
                "phaseA": dt_record.active_power[0].phaseA if dt_record.active_power else 0,
                "phaseB": dt_record.active_power[0].phaseB if dt_record.active_power else 0,
                "phaseC": dt_record.active_power[0].phaseC if dt_record.active_power else 0,
                "total": dt_record.active_power[0].total if dt_record.active_power else 0
            },
            "reactive_power": {
                "phaseA": dt_record.reactive_power[0].phaseA if dt_record.reactive_power else 0,
                "phaseB": dt_record.reactive_power[0].phaseB if dt_record.reactive_power else 0,
                "phaseC": dt_record.reactive_power[0].phaseC if dt_record.reactive_power else 0,
                "total": dt_record.reactive_power[0].total if dt_record.reactive_power else 0
            }
        })
    
    return result


@router.get("/data/latest", response_model=dict)
async def get_latest_data(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the most recent grid data point.
    
    Used for real-time dashboard display.
    
    Args:
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Latest grid data point
    """
    # Get latest timestamp
    latest_dt = db.query(DateTimeTable).order_by(desc(DateTimeTable.timestamp)).first()
    
    if not latest_dt:
        raise HTTPException(status_code=404, detail="No data available")
    
    return {
        "id": latest_dt.id,
        "timestamp": latest_dt.timestamp.isoformat(),
        "voltage": {
            "phaseA": latest_dt.voltage[0].phaseA if latest_dt.voltage else 0,
            "phaseB": latest_dt.voltage[0].phaseB if latest_dt.voltage else 0,
            "phaseC": latest_dt.voltage[0].phaseC if latest_dt.voltage else 0,
            "average": latest_dt.voltage[0].average if latest_dt.voltage else 0
        },
        "current": {
            "phaseA": latest_dt.current[0].phaseA if latest_dt.current else 0,
            "phaseB": latest_dt.current[0].phaseB if latest_dt.current else 0,
            "phaseC": latest_dt.current[0].phaseC if latest_dt.current else 0,
            "average": latest_dt.current[0].average if latest_dt.current else 0
        },
        "frequency": {
            "value": latest_dt.frequency[0].frequency_value if latest_dt.frequency else 50.0
        },
        "active_power": {
            "phaseA": latest_dt.active_power[0].phaseA if latest_dt.active_power else 0,
            "phaseB": latest_dt.active_power[0].phaseB if latest_dt.active_power else 0,
            "phaseC": latest_dt.active_power[0].phaseC if latest_dt.active_power else 0,
            "total": latest_dt.active_power[0].total if latest_dt.active_power else 0
        },
        "reactive_power": {
            "phaseA": latest_dt.reactive_power[0].phaseA if latest_dt.reactive_power else 0,
            "phaseB": latest_dt.reactive_power[0].phaseB if latest_dt.reactive_power else 0,
            "phaseC": latest_dt.reactive_power[0].phaseC if latest_dt.reactive_power else 0,
            "total": latest_dt.reactive_power[0].total if latest_dt.reactive_power else 0
        }
    }


@router.get("/status", response_model=GridStatus)
async def get_grid_status(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get current grid operational status.
    
    Analyzes latest data to determine grid health status.
    
    Args:
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Grid status summary
    """
    # Get latest data
    latest_dt = db.query(DateTimeTable).order_by(desc(DateTimeTable.timestamp)).first()
    
    if not latest_dt:
        raise HTTPException(status_code=404, detail="No data available")
    
    # Extract values
    voltage_avg = latest_dt.voltage[0].average if latest_dt.voltage else 230.0
    current_avg = latest_dt.current[0].average if latest_dt.current else 15.0
    frequency = latest_dt.frequency[0].frequency_value if latest_dt.frequency else 50.0
    active_power_total = latest_dt.active_power[0].total if latest_dt.active_power else 0
    reactive_power_total = latest_dt.reactive_power[0].total if latest_dt.reactive_power else 0
    
    # Calculate power factor
    apparent_power = math.sqrt(active_power_total**2 + reactive_power_total**2)
    power_factor = active_power_total / apparent_power if apparent_power > 0 else 0
    
    # Determine status
    status = "Normal"
    if voltage_avg < 220 or voltage_avg > 240:
        status = "Warning"
    if frequency < 49.85 or frequency > 50.15:
        status = "Critical"
    if current_avg > 25:
        status = "Critical"
    
    return {
        "voltage_avg": round(voltage_avg, 2),
        "current_avg": round(current_avg, 2),
        "frequency": round(frequency, 2),
        "active_power_total": round(active_power_total, 2),
        "reactive_power_total": round(reactive_power_total, 2),
        "power_factor": round(power_factor, 3),
        "status": status,
        "timestamp": latest_dt.timestamp
    }


@router.get("/generate", response_model=dict)
async def generate_test_data(
    num_points: int = Query(default=100, ge=1, le=1000),
    scenario: str = Query(default="normal", regex="^(normal|voltage_sag|overcurrent|frequency_drift|mixed)$"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(check_role("admin"))
):
    """
    Generate test data for development and testing.
    
    Only admins can generate test data.
    
    Args:
        num_points: Number of data points to generate
        scenario: Test scenario (normal, voltage_sag, overcurrent, frequency_drift, mixed)
        db: Database session
        current_user: Admin user
        
    Returns:
        Success message with count
    """
    try:
        # Generate data based on scenario
        if scenario == "normal":
            data_points = grid_generator.generate_time_series(num_points=num_points)
        else:
            data_points = grid_generator.generate_scenario_data(scenario=scenario)
        
        # Insert into database
        for data_point in data_points:
            # Create datetime entry
            dt_entry = DateTimeTable(timestamp=data_point["timestamp"])
            db.add(dt_entry)
            db.flush()
            
            # Create related entries
            db.add(VoltageTable(timestamp_id=dt_entry.id, **data_point["voltage"]))
            db.add(CurrentTable(timestamp_id=dt_entry.id, **data_point["current"]))
            db.add(FrequencyTable(timestamp_id=dt_entry.id, **data_point["frequency"]))
            db.add(ActivePowerTable(timestamp_id=dt_entry.id, **data_point["active_power"]))
            db.add(ReactivePowerTable(timestamp_id=dt_entry.id, **data_point["reactive_power"]))
        
        db.commit()
        
        return {
            "success": True,
            "message": f"Generated {len(data_points)} test data points",
            "scenario": scenario,
            "count": len(data_points)
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating data: {str(e)}")
    
@router.get("/generate-simple")
async def generate_test_data_simple(
    num_points: int = Query(default=100, ge=1, le=1000),
    scenario: str = Query(default="normal"),
    db: Session = Depends(get_db)
):
    """
    Generate test data by calling add_grid_data for each point.
    This ensures all data goes through the same pipeline.
    """
    try:
        from app.services.data_generator import grid_generator
        from app.schemas.grid_data import GridDataPoint
        import asyncio
        
        # Generate data points
        if scenario == "normal":
            data_points = grid_generator.generate_time_series(num_points=num_points)
        else:
            data_points = grid_generator.generate_scenario_data(scenario=scenario)
        
        count = 0
        
        # Create a mock user for authentication (since this is a test endpoint)
        mock_user = {"sub": "system", "user_id": 0, "role": "admin"}
        
        # Process each data point through the standard pipeline
        for dp in data_points:
            # Convert to GridDataPoint schema
            grid_data = GridDataPoint(
                timestamp=dp["timestamp"],
                voltage=dp["voltage"],
                current=dp["current"],
                frequency=dp["frequency"],
                active_power=dp["active_power"],
                reactive_power=dp["reactive_power"]
            )
            
            # Call the standard add_grid_data function
            await add_grid_data(grid_data, db, mock_user)
            
            count += 1
            
            # Wait 1 second between insertions for real-time streaming effect
            await asyncio.sleep(1)
        
        return {
            "success": True,
            "message": f"Generated {count} data points via standard pipeline",
            "count": count,
            "note": "All data was inserted and broadcasted through add_grid_data"
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}