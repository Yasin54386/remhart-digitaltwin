"""
Energy Flow ML API
Models: load_forecasting, health_index (energy imbalance & health)
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

router = APIRouter(prefix="/api/ml/energy", tags=["ML Energy"])

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
        "insights": preds["energy_flow"],
        "metadata": preds["metadata"],
    }


@router.get("/load-forecast")
async def load_forecast(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        lf = preds["energy_flow"].get("load_forecasting", {})
        results.append({"timestamp": record.timestamp, **lf})

    return {
        "model": "XGBoost Regressor (load_forecasting_model.joblib)",
        "features": "all raw + temporal (hour_sin/cos, dow_sin/cos, is_weekend, holiday) + lags (1h,24h,168h) + rolling stats (3h,6h,24h)",
        "data": results,
    }


@router.get("/health-index")
async def health_index(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        hi = preds["energy_flow"].get("health_index", {})
        results.append({"timestamp": record.timestamp, **hi})

    return {
        "model": "XGBoost Regressor (energy_imbalance_and_health_index_model.joblib)",
        "features": "volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T, Frec",
        "output": "health_score (0-100), energy_imbalance_kw, current_unbalance_pct, imbalance_loss_flag",
        "data": results,
    }
