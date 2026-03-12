"""
Decision Making ML API
Models: reactive_q_forecast_model, pf_forecast_model
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


def _load_points(db, is_simulation, hours, limit=500):
    q = db.query(DateTimeTable).options(
        joinedload(DateTimeTable.voltage),
        joinedload(DateTimeTable.current),
        joinedload(DateTimeTable.frequency),
        joinedload(DateTimeTable.active_power),
        joinedload(DateTimeTable.reactive_power),
    )
    if hours is not None:
        q = q.filter(DateTimeTable.timestamp >= datetime.now() - timedelta(hours=hours))
    if is_simulation is not None:
        q = q.filter(DateTimeTable.is_simulation == is_simulation)
    pts = q.order_by(desc(DateTimeTable.timestamp)).limit(limit).all()
    pts.reverse()
    return pts


def _diag(total, success, errors, skipped):
    return {
        "total": total, "success": success, "errors": errors, "skipped": skipped,
        "success_rate": f"{success/total*100:.1f}%" if total else "0%",
    }


@router.get("/latest")
async def get_latest(
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    pts = _load_points(db, is_simulation, None, 1)
    if not pts:
        return {"error": "No data available"}
    preds = ml_inference_engine.process_data_point(pts[0])
    return {
        "timestamp":     pts[0].timestamp,
        "is_simulation": pts[0].is_simulation,
        "insights":      preds["decision_making"],
        "metadata":      preds["metadata"],
    }


@router.get("/reactive-q-forecast")
async def get_reactive_q_forecast(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series reactive power (kVAR) forecast — XGBRegressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        rq = preds["decision_making"].get("reactive_q_forecast")
        if not rq:
            skipped += 1; continue
        results.append({
            "timestamp":           p.timestamp,
            "predicted_q_kvar":    rq["predicted_q_kvar"],
            "current_q_kvar":      rq["current_q_kvar"],
            "trend":               rq["trend"],
            "compensation_needed": rq["compensation_needed"],
        })
    return {
        "algorithm":   "XGBoost Regressor",
        "model_file":  "reactive_q_forecast_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }


@router.get("/pf-forecast")
async def get_pf_forecast(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series power factor forecast with compensation advice — XGBRegressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        pf = preds["decision_making"].get("pf_forecast")
        if not pf:
            skipped += 1; continue
        results.append({
            "timestamp":                   p.timestamp,
            "predicted_pf":                pf["predicted_pf"],
            "current_pf":                  pf["current_pf"],
            "target_pf":                   pf["target_pf"],
            "required_compensation_kvar":  pf["required_compensation_kvar"],
        })
    return {
        "algorithm":   "XGBoost Regressor",
        "model_file":  "pf_forecast_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }
