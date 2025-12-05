"""
REMHART Digital Twin - WebSocket Router
========================================
Handles real-time WebSocket connections for live grid data streaming.

Features:
- Real-time data broadcasting
- Connection management
- Automatic reconnection support
- Heartbeat/ping-pong mechanism

Author: REMHART Team
Date: 2025
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from typing import List
import asyncio
import json
from datetime import datetime

from app.database import get_db, SessionLocal
from app.models.db_models import (
    DateTimeTable, VoltageTable, CurrentTable,
    FrequencyTable, ActivePowerTable, ReactivePowerTable
)
from app.services.ml_inference_engine import ml_inference_engine

router = APIRouter()


class ConnectionManager:
    """
    Manages WebSocket connections for real-time data streaming.
    
    Handles multiple simultaneous client connections and broadcasts
    grid data updates to all connected clients.
    """
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """
        Accept new WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
        """
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """
        Remove WebSocket connection.
        
        Args:
            websocket: WebSocket connection object
        """
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        print(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: dict, websocket: WebSocket):
        """
        Send message to specific client.
        
        Args:
            message: Data to send
            websocket: Target WebSocket connection
        """
        try:
            await websocket.send_json(message)
        except Exception as e:
            print(f"Error sending message: {e}")
    
    async def broadcast(self, message: dict):
        """
        Broadcast message to all connected clients.
        
        Args:
            message: Data to broadcast
        """
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error broadcasting to client: {e}")
                disconnected.append(connection)
        
        # Clean up disconnected clients
        for connection in disconnected:
            self.disconnect(connection)


# Global connection manager instance
manager = ConnectionManager()


@router.websocket("/grid-data")
# @app.websocket("/ws/grid-data")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time data streaming
    """
    await manager.connect(websocket)
    try:
        # Send initial data on connection
        db = SessionLocal()
        try:
            latest_data = db.query(DateTimeTable).order_by(
                DateTimeTable.timestamp.desc()
            ).limit(50).all()
            
            # Build response with simulation info
            data_list = []
            for dt in latest_data:
                # Get related data
                voltage = db.query(VoltageTable).filter(VoltageTable.timestamp_id == dt.id).first()
                current = db.query(CurrentTable).filter(CurrentTable.timestamp_id == dt.id).first()
                frequency = db.query(FrequencyTable).filter(FrequencyTable.timestamp_id == dt.id).first()
                active_power = db.query(ActivePowerTable).filter(ActivePowerTable.timestamp_id == dt.id).first()
                reactive_power = db.query(ReactivePowerTable).filter(ReactivePowerTable.timestamp_id == dt.id).first()
                
                if all([voltage, current, frequency, active_power, reactive_power]):
                    data_list.append({
                        "id": dt.id,
                        "timestamp": dt.timestamp.isoformat(),
                        "voltage": {
                            "phaseA": voltage.phaseA,
                            "phaseB": voltage.phaseB,
                            "phaseC": voltage.phaseC,
                            "average": voltage.average
                        },
                        "current": {
                            "phaseA": current.phaseA,
                            "phaseB": current.phaseB,
                            "phaseC": current.phaseC,
                            "average": current.average
                        },
                        "frequency": {
                            "value": frequency.frequency_value
                        },
                        "active_power": {
                            "phaseA": active_power.phaseA,
                            "phaseB": active_power.phaseB,
                            "phaseC": active_power.phaseC,
                            "total": active_power.total
                        },
                        "reactive_power": {
                            "phaseA": reactive_power.phaseA,
                            "phaseB": reactive_power.phaseB,
                            "phaseC": reactive_power.phaseC,
                            "total": reactive_power.total
                        },
                        "is_simulation": dt.is_simulation if hasattr(dt, 'is_simulation') else False,
                        "simulation_id": dt.simulation_id if hasattr(dt, 'simulation_id') else None
                    })
            
            await websocket.send_json({
                "type": "initial_data",
                "data": data_list
            })
        finally:
            db.close()
        
        # Keep connection alive
        while True:
            try:
                data = await websocket.receive_text()
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
            
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

async def broadcast_new_data(data: dict):
    """
    Broadcast new grid data to all connected WebSocket clients.
    
    Call this function whenever new data is added to the database.
    
    Args:
        data: Grid data point to broadcast
        
    Example:
        await broadcast_new_data({
            "timestamp": "2025-01-15T10:30:00",
            "voltage": {...},
            "current": {...},
            ...
        })
    """
    await manager.broadcast({
        "type": "new_data",
        "data": data,
        "timestamp": datetime.now().isoformat()
    })


@router.websocket("/ml-predictions")
async def ml_predictions_websocket(websocket: WebSocket, is_simulation: bool = False):
    """
    WebSocket endpoint for real-time ML predictions streaming.

    Streams predictions from all 16 ML models organized by category:
    - Real-time Monitoring (4 models)
    - Predictive Maintenance (4 models)
    - Energy Flow (4 models)
    - Decision Making (4 models)

    Args:
        websocket: WebSocket connection
        is_simulation: Filter for simulation data
    """
    await manager.connect(websocket)
    try:
        # Send initial ML predictions on connection
        db = SessionLocal()
        try:
            # Get latest data point
            query = db.query(DateTimeTable).options(
                joinedload(DateTimeTable.voltage),
                joinedload(DateTimeTable.current),
                joinedload(DateTimeTable.frequency),
                joinedload(DateTimeTable.active_power),
                joinedload(DateTimeTable.reactive_power)
            )

            if is_simulation:
                query = query.filter(DateTimeTable.is_simulation == True)
            else:
                query = query.filter(DateTimeTable.is_simulation == False)

            latest = query.order_by(DateTimeTable.timestamp.desc()).first()

            if latest:
                # Run ML inference
                predictions = ml_inference_engine.process_data_point(latest)
                # # --- ADD CHECK HERE ---
                # print(f"Predictions Type: {type(predictions)}")
                # print(f"Predictions Content Sample: {predictions}")
                # # -----------------------
                # # Send initial predictions
                await websocket.send_json({
                    "type": "initial_predictions",
                    "data": predictions,
                    "timestamp": latest.timestamp.isoformat()
                })
        finally:
            db.close()

        # Keep connection alive and listen for requests
        while True:
            try:
                data = await websocket.receive_text()

                if data == "ping":
                    await websocket.send_json({"type": "pong"})

                elif data == "refresh":
                    # Client requested fresh predictions
                    db = SessionLocal()
                    try:
                        query = db.query(DateTimeTable).options(
                            joinedload(DateTimeTable.voltage),
                            joinedload(DateTimeTable.current),
                            joinedload(DateTimeTable.frequency),
                            joinedload(DateTimeTable.active_power),
                            joinedload(DateTimeTable.reactive_power)
                        )

                        if is_simulation:
                            query = query.filter(DateTimeTable.is_simulation == True)
                        else:
                            query = query.filter(DateTimeTable.is_simulation == False)

                        latest = query.order_by(DateTimeTable.timestamp.desc()).first()

                        if latest:
                            predictions = ml_inference_engine.process_data_point(latest)

                            await websocket.send_json({
                                "type": "update",
                                "data": predictions,
                                "timestamp": latest.timestamp.isoformat()
                            })
                    finally:
                        db.close()

            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error processing WebSocket message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "message": str(e)
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)


async def broadcast_ml_predictions(predictions: dict):
    """
    Broadcast new ML predictions to all connected WebSocket clients.

    Call this function whenever new ML predictions are generated.

    Args:
        predictions: ML predictions dictionary from ml_inference_engine

    Example:
        await broadcast_ml_predictions({
            "real_time_monitoring": {...},
            "predictive_maintenance": {...},
            "energy_flow": {...},
            "decision_making": {...}
        })
    """
    await manager.broadcast({
        "type": "ml_update",
        "data": predictions,
        "timestamp": datetime.now().isoformat()
    })


@router.get("/connections")
async def get_connection_count():
    """
    Get number of active WebSocket connections.

    Useful for monitoring and debugging.

    Returns:
        Active connection count
    """
    return {
        "active_connections": len(manager.active_connections),
        "timestamp": datetime.now().isoformat()
    }