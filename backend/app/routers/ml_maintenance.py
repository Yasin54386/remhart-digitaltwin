"""
Predictive Maintenance ML API
Models: equipment_failure_model, overload_risk_model, epqi_score_model, voltage_sag_xgb_regressor
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
        "insights":      preds["predictive_maintenance"],
        "metadata":      preds["metadata"],
    }


@router.get("/equipment-failure")
async def get_equipment_failure(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series equipment failure probability — XGBRegressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        ef = preds["predictive_maintenance"].get("equipment_failure")
        if not ef:
            skipped += 1; continue
        results.append({
            "timestamp":                 p.timestamp,
            "failure_probability":       ef["failure_probability"],
            "risk_level":                ef["risk_level"],
            "estimated_days_to_failure": ef["estimated_days_to_failure"],
            "contributing_factors":      ef["contributing_factors"],
        })
    return {
        "algorithm":   "XGBoost Regressor",
        "model_file":  "equipment_failure_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }


@router.get("/overload-risk")
async def get_overload_risk(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series overload risk classification — XGBClassifier."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        ov = preds["predictive_maintenance"].get("overload_risk")
        if not ov:
            skipped += 1; continue
        results.append({
            "timestamp":         p.timestamp,
            "risk_level":        ov["risk_level"],
            "current_load_pct":  ov["current_load_pct"],
            "peak_phase":        ov["peak_phase"],
            "mitigation_needed": ov["mitigation_needed"],
        })
    return {
        "algorithm":   "XGBoost Classifier",
        "model_file":  "overload_risk_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }


@router.get("/power-quality-index")
async def get_power_quality_index(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series Enhanced Power Quality Index (EPQI) — XGBRegressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        pq = preds["predictive_maintenance"].get("power_quality_index")
        if not pq:
            skipped += 1; continue
        results.append({
            "timestamp":             p.timestamp,
            "pqi_score":             pq["pqi_score"],
            "grade":                 pq["grade"],
            "voltage_quality":       pq["voltage_quality"],
            "frequency_quality":     pq["frequency_quality"],
            "power_factor_quality":  pq["power_factor_quality"],
            "improvement_areas":     pq["improvement_areas"],
        })
    return {
        "algorithm":   "XGBoost Regressor (EPQI)",
        "model_file":  "epqi_score_model.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }


@router.get("/voltage-sag")
async def get_voltage_sag(
    hours: Optional[int] = Query(None),
    is_simulation: Optional[bool] = Query(None),
    _user=Depends(get_optional_user),
    db: Session = Depends(get_db),
):
    """Time-series voltage sag severity prediction — XGBoost Regressor."""
    pts = _load_points(db, is_simulation, hours)
    results, errors, skipped = [], 0, 0
    for p in pts:
        preds = ml_inference_engine.process_data_point(p)
        if preds.get("error"):
            errors += 1; continue
        vs = preds["predictive_maintenance"].get("voltage_sag")
        if not vs:
            skipped += 1; continue
        results.append({
            "timestamp":            p.timestamp,
            "sag_probability":      vs["sag_probability"],
            "risk_level":           vs["risk_level"],
            "expected_duration_ms": vs["expected_duration_ms"],
            "affected_phases":      vs["affected_phases"],
        })
    return {
        "algorithm":   "XGBoost Regressor",
        "model_file":  "voltage_sag_xgb_regressor.joblib",
        "data":        results[-200:],
        "diagnostics": _diag(len(pts), len(results), errors, skipped),
    }
