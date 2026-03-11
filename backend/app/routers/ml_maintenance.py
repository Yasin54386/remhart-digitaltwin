"""
Predictive Maintenance ML API
Models: equipment_failure, overload_risk, epqi_score, voltage_sag
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

router = APIRouter(prefix="/api/ml/maintenance", tags=["ML Maintenance"])

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


def _since_filter(history, hours):
    if not hours:
        return history
    since = datetime.now() - timedelta(hours=hours)
    return [r for r in history if r.timestamp >= since]


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
        "insights": preds["predictive_maintenance"],
        "metadata": preds["metadata"],
    }


@router.get("/equipment-failure")
async def equipment_failure(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        ef = preds["predictive_maintenance"].get("equipment_failure", {})
        results.append({"timestamp": record.timestamp, **ef})

    return {
        "model": "XGBoost Regressor (equipment_failure_model.joblib)",
        "features": "volt_A/B/C, I_A/B/C, P_A/B/C, Q_A/B/C",
        "data": results,
    }


@router.get("/overload-risk")
async def overload_risk(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        ov = preds["predictive_maintenance"].get("overload_risk", {})
        results.append({"timestamp": record.timestamp, **ov})

    return {
        "model": "XGBoost Classifier (overload_risk_model.joblib)",
        "features": "volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T",
        "classes": "Low (0) / Medium (1) / High (2)",
        "data": results,
    }


@router.get("/epqi-score")
async def epqi_score(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        pq = preds["predictive_maintenance"].get("epqi_score", {})
        results.append({"timestamp": record.timestamp, **pq})

    return {
        "model": "XGBoost Regressor (epqi_score_model.joblib)",
        "features": "volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T, Frec",
        "grades": "Excellent ≥90 | Good ≥70 | Fair ≥50 | Poor <50",
        "data": results,
    }


@router.get("/voltage-sag")
async def voltage_sag(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        vs = preds["predictive_maintenance"].get("voltage_sag", {})
        results.append({"timestamp": record.timestamp, **vs})

    return {
        "model": "XGBoost Regressor (voltage_sag_xgb_regressor.joblib)",
        "features": "volt_A/B/C, I_A/B/C, P_A/B/C/T, Q_A/B/C/T, FP_A/B/C/T, Frec",
        "output": "sag_probability: 0.0=Normal, 0.25=Precursor, 0.5=Load-induced, 0.75=Fault-induced, 1.0=Severe",
        "data": results,
    }
