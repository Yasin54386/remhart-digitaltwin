"""
Real-time Monitoring ML API
Models: voltage_anomaly (IsolationForest), energy_anomaly (XGBClassifier)
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import desc
from typing import Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models.db_models import DateTimeTable
from ..services.ml_inference_engine import ml_inference_engine
from ..utils.security import get_optional_user

router = APIRouter(prefix="/api/ml/monitoring", tags=["ML Monitoring"])

_OPTS = [
    joinedload(DateTimeTable.voltage),
    joinedload(DateTimeTable.current),
    joinedload(DateTimeTable.frequency),
    joinedload(DateTimeTable.active_power),
    joinedload(DateTimeTable.reactive_power),
]


def _history(db, is_simulation, limit=1010):
    q = db.query(DateTimeTable).options(*_OPTS)
    if is_simulation is not None:
        q = q.filter(DateTimeTable.is_simulation == is_simulation)
    rows = q.order_by(desc(DateTimeTable.timestamp)).limit(limit).all()
    rows.reverse()
    return rows


@router.get("/latest")
async def latest(
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _history(db, is_simulation)
    if not history:
        return {"error": "No data available"}
    record = history[-1]
    preds = ml_inference_engine.process(record, history)
    return {
        "timestamp": record.timestamp,
        "is_simulation": record.is_simulation,
        "insights": preds["realtime_monitoring"],
        "metadata": preds["metadata"],
    }


@router.get("/voltage-anomaly")
async def voltage_anomaly(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _history(db, is_simulation)
    if hours:
        since = datetime.now() - timedelta(hours=hours)
        history = [r for r in history if r.timestamp >= since]

    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        va = preds["realtime_monitoring"].get("voltage_anomaly", {})
        results.append({"timestamp": record.timestamp, **va})

    return {
        "model": "Isolation Forest (voltage_anomaly_iforest.joblib)",
        "features": "volt_A/B/C, VUF%, v_stability_delta, roll_mean/std, hour_sin/cos, is_workday",
        "data": results,
    }


@router.get("/energy-anomaly")
async def energy_anomaly(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _history(db, is_simulation)
    if hours:
        since = datetime.now() - timedelta(hours=hours)
        history = [r for r in history if r.timestamp >= since]

    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        ea = preds["realtime_monitoring"].get("energy_anomaly", {})
        results.append({"timestamp": record.timestamp, **ea})

    return {
        "model": "XGBoost Classifier (anomaly_detection_model.joblib)",
        "features": "volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T, Frec, hour",
        "data": results,
    }
