"""
Energy Flow ML API
Models: load_forecasting_model, energy_imbalance_and_health_index_model
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
        "insights":      preds["energy_flow"],
        "metadata":      preds["metadata"],
    }


@router.get("/load-forecast")
async def get_load_forecast(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series load forecast with 24-hour ahead predictions — XGBRegressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        lf = preds["energy_flow"].get("load_forecasting")
        if not lf:
            skipped += 1; continue
        results.append({
            "timestamp":       p.timestamp,
            "current_load_kw": lf["current_load_kw"],
            "hourly_forecast": lf["hourly_forecast"],
            "peak_load_time":  lf["peak_load_time"],
            "trend":           lf["trend"],
        })
    return {
        "algorithm":   "XGBoost Regressor (time-series lag features)",
        "model_file":  "load_forecasting_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }


@router.get("/energy-health-index")
async def get_energy_health_index(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series grid health score (IEC 61000-4-30 inspired) — XGBRegressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        hi = preds["energy_flow"].get("energy_health_index")
        if not hi:
            skipped += 1; continue
        results.append({
            "timestamp":         p.timestamp,
            "health_score":      hi["health_score"],
            "grade":             hi["grade"],
            "efficiency_status": hi["efficiency_status"],
            "risk_indicators":   hi["risk_indicators"],
        })
    return {
        "algorithm":   "XGBoost Regressor (health watchdog)",
        "model_file":  "energy_imbalance_and_health_index_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }
