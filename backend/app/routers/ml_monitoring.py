"""
Real-Time Monitoring ML API
Models: voltage_anomaly_iforest, anomaly_detection_model
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
        "insights":      preds["real_time_monitoring"],
        "metadata":      preds["metadata"],
    }


@router.get("/voltage-anomaly")
async def get_voltage_anomaly(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series voltage anomaly scores — Isolation Forest."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        va = preds["real_time_monitoring"].get("voltage_anomaly")
        if not va:
            skipped += 1; continue
        results.append({
            "timestamp":     p.timestamp,
            "is_anomaly":    va["is_anomaly"],
            "anomaly_score": va["anomaly_score"],
            "confidence":    va["confidence"],
            "severity":      va["severity"],
        })
    return {
        "algorithm":   "Isolation Forest",
        "model_file":  "voltage_anomaly_iforest.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }


@router.get("/anomaly-detection")
async def get_anomaly_detection(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series multivariate anomaly classification — XGBClassifier."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        ad = preds["real_time_monitoring"].get("anomaly_detection")
        if not ad:
            skipped += 1; continue
        results.append({
            "timestamp":            p.timestamp,
            "is_anomaly":           ad["is_anomaly"],
            "anomaly_probability":  ad["anomaly_probability"],
            "risk_level":           ad["risk_level"],
            "contributing_factors": ad["contributing_factors"],
        })
    return {
        "algorithm":   "XGBoost Classifier",
        "model_file":  "anomaly_detection_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }
