"""
Decision Making ML API
Models: reactive_q_forecast + pf_forecast (XGBRegressors, combined endpoint)
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

router = APIRouter(prefix="/api/ml/decision", tags=["ML Decision"])

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
        "insights": preds["decision_making"],
        "metadata": preds["metadata"],
    }


@router.get("/reactive-pf-forecast")
async def reactive_pf_forecast(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    current_user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """
    Combined endpoint returning both reactive power (Q) and power factor (PF)
    forecasts for the next time step.

    Uses two models:
    - reactive_q_forecast_model.joblib  → predicts Q_T at t+1
    - pf_forecast_model.joblib          → predicts FP_T at t+1

    Both share the same 16 features:
        Q_lag_1h, Q_lag_24h, PF_lag_1h,
        hour_sin, hour_cos, is_workday,
        P_T, volt_A/B/C, I_A/B/C, Frec, FP_T, Q_T
    """
    history = _since_filter(_history(db, is_simulation), hours)
    results = []
    for i, record in enumerate(history):
        preds = ml_inference_engine.process(record, history[: i + 1])
        rp = preds["decision_making"].get("reactive_and_pf", {})
        results.append({"timestamp": record.timestamp, **rp})

    return {
        "models": [
            "XGBoost Regressor (reactive_q_forecast_model.joblib)",
            "XGBoost Regressor (pf_forecast_model.joblib)",
        ],
        "features": "Q_lag_1h/24h, PF_lag_1h, hour_sin/cos, is_workday, P_T, volt_A/B/C, I_A/B/C, Frec, FP_T, Q_T",
        "data": results,
    }
