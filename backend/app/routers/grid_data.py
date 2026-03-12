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

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from typing import List, Optional
from datetime import datetime, timedelta
import math
import asyncio

from app.database import get_db, SessionLocal
from app.models.db_models import (
    DateTimeTable, VoltageTable, CurrentTable,
    FrequencyTable, ActivePowerTable, ReactivePowerTable
)
from app.schemas.grid_data import GridDataPoint, GridDataResponse, GridStatus
from app.utils.security import get_current_user, check_role
from app.services.data_generator import grid_generator
from app.routers.websocket_router import manager, broadcast_new_data, broadcast_ml_predictions

router = APIRouter()

# Stream control flag — set False to stop a running generate-simple stream
_stream_active: bool = False




async def _run_ml_and_broadcast(data_id: int, is_simulation: bool):
    """
    Background task: load saved record, run 10-model inference, broadcast predictions.
    Uses its own DB session so it doesn't block the request.
    """
    from app.services.ml_inference_engine import ml_inference_engine
    db = SessionLocal()
    try:
        dp = db.query(DateTimeTable).options(
            joinedload(DateTimeTable.voltage),
            joinedload(DateTimeTable.current),
            joinedload(DateTimeTable.frequency),
            joinedload(DateTimeTable.active_power),
            joinedload(DateTimeTable.reactive_power),
        ).filter(DateTimeTable.id == data_id).first()

        if dp:
            preds = ml_inference_engine.process_data_point(dp)
            await broadcast_ml_predictions(preds, is_simulation=is_simulation)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"ML background broadcast failed: {e}")
    finally:
        db.close()


@router.post("/data", response_model=dict)
async def add_grid_data(
    data: GridDataPoint,
    db: Session = Depends(get_db),
    current_user: dict = Depends(check_role("operator"))
):
    """
    Add new grid data point to database.
    Immediately broadcasts raw data + triggers async ML inference broadcast.
    """
    try:
        # Persist
        dt_entry = DateTimeTable(timestamp=data.timestamp)
        db.add(dt_entry)
        db.flush()

        db.add(VoltageTable(timestamp_id=dt_entry.id, **data.voltage.dict()))
        db.add(CurrentTable(timestamp_id=dt_entry.id, **data.current.dict()))
        db.add(FrequencyTable(timestamp_id=dt_entry.id, **data.frequency.dict()))
        db.add(ActivePowerTable(timestamp_id=dt_entry.id, **data.active_power.dict()))
        db.add(ReactivePowerTable(timestamp_id=dt_entry.id, **data.reactive_power.dict()))

        db.commit()

        # Determine simulation flag (schema may carry it)
        is_sim = getattr(data, 'is_simulation', False) or False

        # 1) Broadcast raw grid data immediately
        await broadcast_new_data({
            "id":            dt_entry.id,
            "timestamp":     data.timestamp.isoformat(),
            "voltage":       data.voltage.dict(),
            "current":       data.current.dict(),
            "frequency":     {"value": data.frequency.frequency_value},
            "active_power":  data.active_power.dict(),
            "reactive_power": data.reactive_power.dict(),
            "is_simulation": is_sim,
        })

        # 2) Run ML inference + broadcast predictions in background (non-blocking)
        asyncio.create_task(_run_ml_and_broadcast(dt_entry.id, is_sim))

        return {
            "success":   True,
            "message":   "Grid data added and broadcasted",
            "data_id":   dt_entry.id,
            "timestamp": dt_entry.timestamp.isoformat()
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
    is_simulation: Optional[bool] = Query(default=None, description="Filter by simulation mode (true) or real-time (false)"),
    simulation_id: Optional[str] = Query(default=None, description="Filter by specific simulation ID"),
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
        is_simulation: Filter by simulation (true) or real-time (false) data
        simulation_id: Filter by specific simulation ID (requires is_simulation=true)
        db: Database session
        current_user: Authenticated user

    Returns:
        List of grid data points

    Examples:
        GET /api/grid/data?limit=50&start_time=2025-01-15T00:00:00
        GET /api/grid/data?limit=50&is_simulation=false
        GET /api/grid/data?limit=50&is_simulation=true&simulation_id=abc123
    """
    # Build query
    query = db.query(DateTimeTable).order_by(desc(DateTimeTable.timestamp))

    # Apply time filters
    if start_time:
        query = query.filter(DateTimeTable.timestamp >= start_time)
    if end_time:
        query = query.filter(DateTimeTable.timestamp <= end_time)

    # Apply simulation mode filters
    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    # Apply simulation ID filter (only if is_simulation is True)
    if simulation_id is not None:
        query = query.filter(DateTimeTable.simulation_id == simulation_id)

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
            },
            "is_simulation": dt_record.is_simulation if hasattr(dt_record, 'is_simulation') else False,
            "simulation_id": dt_record.simulation_id if hasattr(dt_record, 'simulation_id') else None,
            "simulation_name": dt_record.simulation_name if hasattr(dt_record, 'simulation_name') else None,
            "simulation_scenario": dt_record.simulation_scenario if hasattr(dt_record, 'simulation_scenario') else None
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
    
@router.get("/live-stream/stop")
async def stop_live_stream():
    """Stop a running generate-simple stream."""
    global _stream_active
    _stream_active = False
    return {"success": True, "message": "Stream stop signal sent"}


@router.get("/generate-simple")
async def generate_test_data_simple(
    request: Request,
    num_points: int = Query(default=3600, ge=1, le=3600),
    scenario: str = Query(default="normal"),
    db: Session = Depends(get_db)
):
    """
    Continuously stream real-time grid data at 1-second intervals.
    Broadcasts every point system-wide via WebSocket.
    Stops when: Stop endpoint is called, client disconnects, or num_points reached.
    """
    global _stream_active
    _stream_active = True
    count = 0
    mock_user = {"sub": "system", "user_id": 0, "role": "admin"}

    # Scenario anomaly windows (repeat every 60 points for ongoing streams)
    def should_force_anomaly(n: int) -> bool:
        pos = n % 60
        if scenario == "voltage_sag":      return 20 <= pos <= 30
        if scenario == "overcurrent":      return 15 <= pos <= 35
        if scenario == "frequency_drift":  return 10 <= pos <= 40
        if scenario == "mixed":            return pos % 5 == 0
        return False

    try:
        while _stream_active and count < num_points:
            if await request.is_disconnected():
                break

            dp = grid_generator.generate_single_datapoint(
                force_anomaly=should_force_anomaly(count)
            )

            await add_grid_data(
                GridDataPoint(
                    timestamp=dp["timestamp"],
                    voltage=dp["voltage"],
                    current=dp["current"],
                    frequency=dp["frequency"],
                    active_power=dp["active_power"],
                    reactive_power=dp["reactive_power"]
                ),
                db,
                mock_user
            )
            count += 1
            await asyncio.sleep(1)

        return {
            "success": True,
            "count": count,
            "message": f"Stream ended: {count} points inserted"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        _stream_active = False