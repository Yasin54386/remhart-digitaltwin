"""
REMHART Digital Twin – WebSocket Router
========================================
Two independent WebSocket channels:

  /ws/grid-data       – raw grid measurements  (ConnectionManager → grid_manager)
  /ws/ml-predictions  – all 10 ML model outputs (MLConnectionManager → ml_manager)

On every POST /api/grid/data the grid_data router calls:
  broadcast_new_data(data_dict)       – pushes to /ws/grid-data clients
  broadcast_ml_predictions(preds)    – pushes to /ws/ml-predictions clients

Both managers handle dead connections gracefully.
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import joinedload
from sqlalchemy import desc
from typing import List, Optional
import asyncio
import json
from datetime import datetime

from app.database import SessionLocal
from app.models.db_models import (
    DateTimeTable, VoltageTable, CurrentTable,
    FrequencyTable, ActivePowerTable, ReactivePowerTable
)
from app.services.ml_inference_engine import ml_inference_engine

router = APIRouter()


# ════════════════════════════════════════════════════════════════════
# Connection managers
# ════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages /ws/grid-data connections."""

    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c is not ws]

    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


class MLConnectionManager:
    """
    Manages /ws/ml-predictions connections.
    Each connection records the is_simulation filter the client wants.
    """

    def __init__(self):
        self.active: List[dict] = []   # [{ws, is_simulation}]

    async def connect(self, ws: WebSocket, is_simulation: Optional[bool]):
        await ws.accept()
        self.active.append({'ws': ws, 'is_simulation': is_simulation})

    def disconnect(self, ws: WebSocket):
        self.active = [c for c in self.active if c['ws'] is not ws]

    async def broadcast(self, msg: dict, is_simulation: Optional[bool] = None):
        """
        Broadcast to all clients whose is_simulation filter matches.
        If is_simulation is None, broadcast to everyone.
        """
        dead = []
        for conn in self.active:
            # Filter: only push to clients that requested this data type
            if is_simulation is not None and conn['is_simulation'] is not None:
                if conn['is_simulation'] != is_simulation:
                    continue
            try:
                await conn['ws'].send_json(msg)
            except Exception:
                dead.append(conn['ws'])
        for ws in dead:
            self.disconnect(ws)

    @property
    def count(self):
        return len(self.active)


# Global singletons
grid_manager = ConnectionManager()
ml_manager   = MLConnectionManager()

# Keep backward-compat alias used by grid_data.py
manager = grid_manager


# ════════════════════════════════════════════════════════════════════
# /ws/grid-data
# ════════════════════════════════════════════════════════════════════

@router.websocket("/grid-data")
async def ws_grid_data(websocket: WebSocket):
    """Stream raw grid measurements to connected clients."""
    await grid_manager.connect(websocket)
    try:
        db = SessionLocal()
        try:
            rows = db.query(DateTimeTable).order_by(
                DateTimeTable.timestamp.desc()
            ).limit(50).all()

            payload = []
            for dt in rows:
                v  = db.query(VoltageTable).filter_by(timestamp_id=dt.id).first()
                i  = db.query(CurrentTable).filter_by(timestamp_id=dt.id).first()
                f  = db.query(FrequencyTable).filter_by(timestamp_id=dt.id).first()
                ap = db.query(ActivePowerTable).filter_by(timestamp_id=dt.id).first()
                rp = db.query(ReactivePowerTable).filter_by(timestamp_id=dt.id).first()
                if all([v, i, f, ap, rp]):
                    payload.append({
                        "id": dt.id,
                        "timestamp": dt.timestamp.isoformat(),
                        "voltage":  {"phaseA": v.phaseA, "phaseB": v.phaseB, "phaseC": v.phaseC, "average": v.average},
                        "current":  {"phaseA": i.phaseA, "phaseB": i.phaseB, "phaseC": i.phaseC, "average": i.average},
                        "frequency": {"value": f.frequency_value},
                        "active_power":   {"phaseA": ap.phaseA, "phaseB": ap.phaseB, "phaseC": ap.phaseC, "total": ap.total},
                        "reactive_power": {"phaseA": rp.phaseA, "phaseB": rp.phaseB, "phaseC": rp.phaseC, "total": rp.total},
                        "is_simulation": getattr(dt, 'is_simulation', False),
                    })

            await websocket.send_json({"type": "initial_data", "data": payload})
        finally:
            db.close()

        while True:
            try:
                msg = await websocket.receive_text()
                if msg == "ping":
                    await websocket.send_json({"type": "pong"})
            except WebSocketDisconnect:
                break
    except WebSocketDisconnect:
        pass
    finally:
        grid_manager.disconnect(websocket)


# ════════════════════════════════════════════════════════════════════
# /ws/ml-predictions
# ════════════════════════════════════════════════════════════════════

@router.websocket("/ml-predictions")
async def ws_ml_predictions(websocket: WebSocket, is_simulation: Optional[bool] = None):
    """
    Stream all 10 ML model predictions.
    On connect: sends latest predictions immediately.
    On new data (POST /api/grid/data): receives a server-push update.
    Supports client commands: "ping" (heartbeat), "refresh" (manual re-fetch).
    """
    await ml_manager.connect(websocket, is_simulation)
    try:
        # ── Initial predictions on connect ──────────────────────────
        preds = _fetch_latest_predictions(is_simulation)
        if preds:
            await websocket.send_json({
                "type": "initial_predictions",
                "data": preds,
                "timestamp": datetime.now().isoformat()
            })
        else:
            await websocket.send_json({"type": "no_data", "message": "No data available yet"})

        # ── Keep-alive / client-command loop ────────────────────────
        while True:
            try:
                cmd = await websocket.receive_text()
                if cmd == "ping":
                    await websocket.send_json({"type": "pong"})
                elif cmd == "refresh":
                    preds = _fetch_latest_predictions(is_simulation)
                    if preds:
                        await websocket.send_json({
                            "type": "update",
                            "data": preds,
                            "timestamp": datetime.now().isoformat()
                        })
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({"type": "error", "message": str(e)})

    except WebSocketDisconnect:
        pass
    finally:
        ml_manager.disconnect(websocket)


# ════════════════════════════════════════════════════════════════════
# Broadcast helpers (called from grid_data router on new data)
# ════════════════════════════════════════════════════════════════════

async def broadcast_new_data(data: dict):
    """Push raw grid data to all /ws/grid-data clients."""
    await grid_manager.broadcast({
        "type": "new_data",
        "data": data,
        "timestamp": datetime.now().isoformat()
    })


async def broadcast_ml_predictions(predictions: dict, is_simulation: Optional[bool] = None):
    """Push ML predictions to all matching /ws/ml-predictions clients."""
    await ml_manager.broadcast(
        {
            "type": "ml_update",
            "data": predictions,
            "timestamp": datetime.now().isoformat()
        },
        is_simulation=is_simulation
    )


# ════════════════════════════════════════════════════════════════════
# Internal helper
# ════════════════════════════════════════════════════════════════════

def _fetch_latest_predictions(is_simulation: Optional[bool]) -> Optional[dict]:
    db = SessionLocal()
    try:
        q = db.query(DateTimeTable).options(
            joinedload(DateTimeTable.voltage),
            joinedload(DateTimeTable.current),
            joinedload(DateTimeTable.frequency),
            joinedload(DateTimeTable.active_power),
            joinedload(DateTimeTable.reactive_power),
        )
        if is_simulation is not None:
            q = q.filter(DateTimeTable.is_simulation == is_simulation)
        dp = q.order_by(desc(DateTimeTable.timestamp)).first()
        if dp:
            return ml_inference_engine.process_data_point(dp)
        return None
    finally:
        db.close()


# ════════════════════════════════════════════════════════════════════
# Status endpoint
# ════════════════════════════════════════════════════════════════════

@router.get("/connections")
async def connection_count():
    return {
        "grid_connections": len(grid_manager.active),
        "ml_connections":   ml_manager.count,
        "timestamp":        datetime.now().isoformat()
    }
