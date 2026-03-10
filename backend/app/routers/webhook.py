"""
REMHART Digital Twin - Webhook Router
======================================
Provides webhook endpoints for external systems to push real-time grid data.
When data arrives, it is saved to the database, ML inference is run, and
results are broadcast to all connected WebSocket clients.

Endpoints:
- POST /api/webhook/data       - Push a single grid data point
- POST /api/webhook/data/batch - Push multiple data points at once
- GET  /api/webhook/status     - Check webhook health

Author: REMHART Team
Date: 2025
"""

from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.orm import Session, joinedload
from typing import List, Optional
from datetime import datetime
import logging

from ..database import get_db
from ..models.db_models import (
    DateTimeTable, VoltageTable, CurrentTable,
    FrequencyTable, ActivePowerTable, ReactivePowerTable
)
from ..schemas.grid_data import GridDataPoint
from ..services.ml_inference_engine import ml_inference_engine
from ..routers.websocket_router import manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/webhook", tags=["Webhook"])


def _save_data_point(db: Session, data: GridDataPoint) -> DateTimeTable:
    """Save a single grid data point to the database and return the entry."""
    dt_entry = DateTimeTable(
        timestamp=data.timestamp,
        is_simulation=False,
        simulation_id=None
    )
    db.add(dt_entry)
    db.flush()

    db.add(VoltageTable(timestamp_id=dt_entry.id, **data.voltage.dict()))
    db.add(CurrentTable(timestamp_id=dt_entry.id, **data.current.dict()))
    db.add(FrequencyTable(timestamp_id=dt_entry.id, **data.frequency.dict()))
    db.add(ActivePowerTable(timestamp_id=dt_entry.id, **data.active_power.dict()))
    db.add(ReactivePowerTable(timestamp_id=dt_entry.id, **data.reactive_power.dict()))

    db.commit()
    db.refresh(dt_entry)
    return dt_entry


@router.post("/data", response_model=dict)
async def webhook_data(
    data: GridDataPoint,
    db: Session = Depends(get_db)
):
    """
    Receive a real-time grid data point from an external source.

    - Saves the data point to the MySQL database
    - Broadcasts raw data to all connected WebSocket clients (type: new_data)
    - Runs ML inference on the new data
    - Broadcasts ML predictions to all WebSocket clients (type: ml_update)

    This endpoint does NOT require authentication so that IoT sensors
    and external systems can push data without OAuth complexity.
    Use a reverse proxy / VPN to restrict access in production.
    """
    try:
        dt_entry = _save_data_point(db, data)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save data: {str(e)}")

    # Broadcast raw grid data
    try:
        await manager.broadcast({
            "type": "new_data",
            "data": {
                "id": dt_entry.id,
                "timestamp": data.timestamp.isoformat(),
                "voltage": data.voltage.dict(),
                "current": data.current.dict(),
                "frequency": {"value": data.frequency.frequency_value},
                "active_power": data.active_power.dict(),
                "reactive_power": data.reactive_power.dict(),
                "source": "webhook"
            }
        })
    except Exception as e:
        logger.warning(f"Webhook: raw data broadcast failed: {e}")

    # Run ML inference and broadcast predictions
    ml_result = {}
    try:
        dt_for_ml = db.query(DateTimeTable).options(
            joinedload(DateTimeTable.voltage),
            joinedload(DateTimeTable.current),
            joinedload(DateTimeTable.frequency),
            joinedload(DateTimeTable.active_power),
            joinedload(DateTimeTable.reactive_power)
        ).filter(DateTimeTable.id == dt_entry.id).first()

        if dt_for_ml:
            ml_result = ml_inference_engine.process_data_point(dt_for_ml)
            await manager.broadcast({
                "type": "ml_update",
                "data": ml_result,
                "timestamp": data.timestamp.isoformat(),
                "source": "webhook"
            })
    except Exception as e:
        logger.warning(f"Webhook: ML inference/broadcast failed: {e}")

    return {
        "success": True,
        "data_id": dt_entry.id,
        "timestamp": data.timestamp.isoformat(),
        "ml_inference_ran": bool(ml_result and not ml_result.get("error")),
        "websocket_clients_notified": len(manager.active_connections)
    }


@router.post("/data/batch", response_model=dict)
async def webhook_data_batch(
    data_points: List[GridDataPoint],
    db: Session = Depends(get_db)
):
    """
    Receive multiple real-time grid data points at once.

    Saves all points, broadcasts raw data and ML predictions for the last point.
    Useful for bulk uploads or catching up after a connection gap.
    Maximum 100 points per request.
    """
    if len(data_points) > 100:
        raise HTTPException(
            status_code=422,
            detail="Maximum 100 data points per batch request"
        )

    saved_ids = []
    last_entry = None

    try:
        for data in data_points:
            entry = _save_data_point(db, data)
            saved_ids.append(entry.id)
            last_entry = (entry, data)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Batch save failed: {str(e)}")

    # Broadcast raw data summary
    try:
        await manager.broadcast({
            "type": "batch_data",
            "count": len(saved_ids),
            "first_id": saved_ids[0] if saved_ids else None,
            "last_id": saved_ids[-1] if saved_ids else None
        })
    except Exception as e:
        logger.warning(f"Webhook batch: raw broadcast failed: {e}")

    # ML inference on last data point only
    ml_ran = False
    if last_entry:
        dt_entry, data = last_entry
        try:
            dt_for_ml = db.query(DateTimeTable).options(
                joinedload(DateTimeTable.voltage),
                joinedload(DateTimeTable.current),
                joinedload(DateTimeTable.frequency),
                joinedload(DateTimeTable.active_power),
                joinedload(DateTimeTable.reactive_power)
            ).filter(DateTimeTable.id == dt_entry.id).first()

            if dt_for_ml:
                ml_result = ml_inference_engine.process_data_point(dt_for_ml)
                await manager.broadcast({
                    "type": "ml_update",
                    "data": ml_result,
                    "timestamp": data.timestamp.isoformat(),
                    "source": "webhook_batch"
                })
                ml_ran = not ml_result.get("error", False)
        except Exception as e:
            logger.warning(f"Webhook batch: ML inference failed: {e}")

    return {
        "success": True,
        "saved_count": len(saved_ids),
        "data_ids": saved_ids,
        "ml_inference_ran": ml_ran,
        "websocket_clients_notified": len(manager.active_connections)
    }


@router.get("/status", response_model=dict)
async def webhook_status():
    """
    Check webhook health and active WebSocket connections.
    """
    return {
        "status": "operational",
        "active_websocket_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "single_point": "POST /api/webhook/data",
            "batch_points": "POST /api/webhook/data/batch"
        }
    }
