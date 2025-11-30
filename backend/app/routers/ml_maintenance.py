"""
Predictive Maintenance ML API
Provides AI-powered predictive maintenance insights
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


@router.get("/latest")
async def get_latest_maintenance_insights(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Get latest predictive maintenance ML insights"""
    query = db.query(DateTimeTable).options(
        joinedload(DateTimeTable.voltage),
        joinedload(DateTimeTable.current),
        joinedload(DateTimeTable.frequency),
        joinedload(DateTimeTable.active_power),
        joinedload(DateTimeTable.reactive_power)
    )

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    latest = query.order_by(desc(DateTimeTable.timestamp)).first()

    if not latest:
        return {"error": "No data available"}

    predictions = ml_inference_engine.process_data_point(latest)

    return {
        "timestamp": latest.timestamp,
        "is_simulation": latest.is_simulation,
        "insights": predictions['predictive_maintenance'],
        "metadata": predictions['metadata']
    }


@router.get("/equipment-failure")
async def get_equipment_failure_prediction(
    hours: int = Query(None, description="Hours of historical data (optional, defaults to all data)"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get equipment failure probability predictions

    Returns:
        Time-series of failure probability and contributing factors
    """
    query = db.query(DateTimeTable).options(
        joinedload(DateTimeTable.voltage),
        joinedload(DateTimeTable.current),
        joinedload(DateTimeTable.frequency),
        joinedload(DateTimeTable.active_power),
        joinedload(DateTimeTable.reactive_power)
    )

    # Only filter by time if hours parameter is provided
    if hours is not None:
        since = datetime.now() - timedelta(hours=hours)
        query = query.filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(desc(DateTimeTable.timestamp)).limit(500).all()
    data_points.reverse()  # Show oldest to newest for charts

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        failure = predictions['predictive_maintenance']['equipment_failure_prediction']

        results.append({
            'timestamp': point.timestamp,
            'failure_probability': failure['failure_probability'],
            'risk_level': failure['risk_level'],
            'time_to_failure': failure['estimated_days_to_failure'],
            'contributing_factors': failure['contributing_factors']
        })

    return {
        "algorithm": "XGBoost (Gradient Boosting)",
        "training_dataset": "Synthetic data based on equipment stress indicators: high current, voltage/current variance, poor power factor, imbalances",
        "benefits": "Prevents unexpected equipment failures, optimizes maintenance scheduling, reduces downtime costs",
        "predictions": results
    }


@router.get("/overload-risk")
async def get_overload_risk_classification(
    hours: int = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get overload risk classification

    Returns:
        Time-series of overload risk levels
    """
    query = db.query(DateTimeTable).options(
        joinedload(DateTimeTable.voltage),
        joinedload(DateTimeTable.current),
        joinedload(DateTimeTable.frequency),
        joinedload(DateTimeTable.active_power),
        joinedload(DateTimeTable.reactive_power)
    )

    if hours is not None:
        since = datetime.now() - timedelta(hours=hours)
        query = query.filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(desc(DateTimeTable.timestamp)).limit(500).all()
    data_points.reverse()

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        overload = predictions['predictive_maintenance']['overload_risk_classification']

        results.append({
            'timestamp': point.timestamp,
            'risk_level': overload['risk_level'],
            'load_percentage': overload['current_load_pct'],
            'peak_phase': overload['peak_phase'],
            'mitigation_needed': overload['mitigation_needed']
        })

    return {
        "algorithm": "SVM (Support Vector Machine)",
        "training_dataset": "500 samples with varying load levels from 50% to 200% of rated capacity",
        "benefits": "Prevents equipment overheating and damage, enables proactive load shedding decisions",
        "predictions": results
    }


@router.get("/power-quality-index")
async def get_power_quality_index(
    hours: int = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get Power Quality Index (PQI) scores

    Returns:
        Time-series of comprehensive power quality scores
    """
    query = db.query(DateTimeTable).options(
        joinedload(DateTimeTable.voltage),
        joinedload(DateTimeTable.current),
        joinedload(DateTimeTable.frequency),
        joinedload(DateTimeTable.active_power),
        joinedload(DateTimeTable.reactive_power)
    )

    if hours is not None:
        since = datetime.now() - timedelta(hours=hours)
        query = query.filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(desc(DateTimeTable.timestamp)).limit(500).all()
    data_points.reverse()

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        pqi = predictions['predictive_maintenance']['power_quality_index']

        results.append({
            'timestamp': point.timestamp,
            'pqi_score': pqi['pqi_score'],
            'quality_grade': pqi['grade'],
            'voltage_quality': pqi['voltage_quality'],
            'frequency_quality': pqi['frequency_quality'],
            'power_factor_quality': pqi['pf_quality_score'],
            'improvement_areas': pqi['improvement_areas']
        })

    return {
        "algorithm": "Neural Network (Multi-layer Perceptron)",
        "training_dataset": "Comprehensive dataset combining voltage quality (40%), frequency quality (30%), and power factor quality (30%) metrics",
        "benefits": "Provides single metric for overall power quality, helps prioritize improvement investments",
        "predictions": results
    }


@router.get("/voltage-sag")
async def get_voltage_sag_prediction(
    hours: int = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get voltage sag event predictions

    Returns:
        Time-series of voltage sag probabilities
    """
    query = db.query(DateTimeTable).options(
        joinedload(DateTimeTable.voltage),
        joinedload(DateTimeTable.current),
        joinedload(DateTimeTable.frequency),
        joinedload(DateTimeTable.active_power),
        joinedload(DateTimeTable.reactive_power)
    )

    if hours is not None:
        since = datetime.now() - timedelta(hours=hours)
        query = query.filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(desc(DateTimeTable.timestamp)).limit(500).all()
    data_points.reverse()

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        sag = predictions['predictive_maintenance']['voltage_sag_prediction']

        results.append({
            'timestamp': point.timestamp,
            'sag_probability': sag['sag_probability'],
            'risk_level': sag['risk_level'],
            'expected_duration_ms': sag['expected_duration_ms'],
            'affected_phases': sag['affected_phases']
        })

    return {
        "algorithm": "Random Forest Classifier",
        "training_dataset": "Data including normal conditions and voltage sag events (< 0.9 pu voltage)",
        "benefits": "Enables installation of protective devices before sags occur, protects sensitive equipment",
        "predictions": results
    }
