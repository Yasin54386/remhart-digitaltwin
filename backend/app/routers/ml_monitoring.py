"""
Real-time Monitoring ML API
Provides AI-powered real-time grid monitoring insights
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models.db_models import DateTimeTable
from ..services.ml_inference_engine import ml_inference_engine
from ..utils.security import get_current_user

router = APIRouter(prefix="/api/ml/monitoring", tags=["ML Monitoring"])


@router.get("/latest")
async def get_latest_monitoring_insights(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get latest real-time monitoring ML insights

    Returns:
        All 4 monitoring model predictions for the latest data point
    """
    # Get latest data point
    query = db.query(DateTimeTable).options(
        db.joinedload(DateTimeTable.voltage),
        db.joinedload(DateTimeTable.current),
        db.joinedload(DateTimeTable.frequency),
        db.joinedload(DateTimeTable.active_power),
        db.joinedload(DateTimeTable.reactive_power)
    )

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    latest = query.order_by(desc(DateTimeTable.timestamp)).first()

    if not latest:
        return {"error": "No data available"}

    # Run ML inference
    predictions = ml_inference_engine.process_data_point(latest)

    return {
        "timestamp": latest.timestamp,
        "is_simulation": latest.is_simulation,
        "insights": predictions['real_time_monitoring'],
        "metadata": predictions['metadata']
    }


@router.get("/voltage-anomaly")
async def get_voltage_anomaly_detection(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get voltage anomaly detection over time

    Returns:
        Time-series of voltage anomaly predictions
    """
    since = datetime.now() - timedelta(hours=hours)

    query = db.query(DateTimeTable).options(
        db.joinedload(DateTimeTable.voltage),
        db.joinedload(DateTimeTable.current),
        db.joinedload(DateTimeTable.frequency),
        db.joinedload(DateTimeTable.active_power),
        db.joinedload(DateTimeTable.reactive_power)
    ).filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(DateTimeTable.timestamp).limit(1000).all()

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        voltage_anomaly = predictions['real_time_monitoring']['voltage_anomaly_detection']

        results.append({
            'timestamp': point.timestamp,
            'is_anomaly': voltage_anomaly['is_anomaly'],
            'anomaly_score': voltage_anomaly['anomaly_score'],
            'severity': voltage_anomaly['severity'],
            'v_avg': point.voltage[0].average if point.voltage else 0
        })

    return {
        "algorithm": "Isolation Forest",
        "training_dataset": "2000 normal samples + 500 anomaly samples (voltage sags, swells, imbalances)",
        "benefits": "Early detection of voltage quality issues prevents equipment damage and improves grid reliability",
        "data": results
    }


@router.get("/harmonic-analysis")
async def get_harmonic_analysis(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get harmonic distortion analysis

    Returns:
        Time-series of THD estimates and harmonic components
    """
    since = datetime.now() - timedelta(hours=hours)

    query = db.query(DateTimeTable).options(
        db.joinedload(DateTimeTable.voltage),
        db.joinedload(DateTimeTable.current),
        db.joinedload(DateTimeTable.frequency),
        db.joinedload(DateTimeTable.active_power),
        db.joinedload(DateTimeTable.reactive_power)
    ).filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(DateTimeTable.timestamp).limit(1000).all()

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        harmonic = predictions['real_time_monitoring']['harmonic_analysis']

        results.append({
            'timestamp': point.timestamp,
            'thd_percentage': harmonic['thd_percentage'],
            'thd_category': harmonic['thd_category'],
            'harmonics': harmonic['harmonics'],
            'quality_impact': harmonic['quality_impact']
        })

    return {
        "algorithm": "Random Forest + FFT (Fast Fourier Transform)",
        "training_dataset": "500 samples with varying power quality conditions, labeled by THD levels",
        "benefits": "Identifies harmonic pollution sources, helps specify filter requirements, and improves power quality",
        "data": results
    }


@router.get("/frequency-stability")
async def get_frequency_stability(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get frequency stability prediction

    Returns:
        Current frequency and future predictions
    """
    # Get latest 100 data points for LSTM input
    query = db.query(DateTimeTable).options(
        db.joinedload(DateTimeTable.voltage),
        db.joinedload(DateTimeTable.current),
        db.joinedload(DateTimeTable.frequency),
        db.joinedload(DateTimeTable.active_power),
        db.joinedload(DateTimeTable.reactive_power)
    )

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    recent_points = query.order_by(desc(DateTimeTable.timestamp)).limit(100).all()

    if not recent_points:
        return {"error": "Insufficient data"}

    # Get prediction for latest point
    latest = recent_points[0]
    predictions = ml_inference_engine.process_data_point(latest)
    freq_stability = predictions['real_time_monitoring']['frequency_stability']

    # Also get historical frequency
    historical = []
    for point in reversed(recent_points[-50:]):  # Last 50 points
        freq = point.frequency[0].frequency_value if point.frequency else 50.0
        historical.append({
            'timestamp': point.timestamp,
            'frequency': freq
        })

    return {
        "algorithm": "LSTM (Long Short-Term Memory) Neural Network",
        "training_dataset": "30 days of time-series data with daily and weekly load patterns",
        "benefits": "Predicts frequency instability before it occurs, enables proactive generator dispatch adjustments",
        "current": {
            'frequency': freq_stability['current_frequency'],
            'stability_score': freq_stability['stability_score'],
            'trend': freq_stability['trend']
        },
        "predictions": freq_stability['predicted_frequencies'],
        "historical": historical
    }


@router.get("/phase-imbalance")
async def get_phase_imbalance(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get phase imbalance classification

    Returns:
        Time-series of phase imbalance severity
    """
    since = datetime.now() - timedelta(hours=hours)

    query = db.query(DateTimeTable).options(
        db.joinedload(DateTimeTable.voltage),
        db.joinedload(DateTimeTable.current),
        db.joinedload(DateTimeTable.frequency),
        db.joinedload(DateTimeTable.active_power),
        db.joinedload(DateTimeTable.reactive_power)
    ).filter(DateTimeTable.timestamp >= since)

    if is_simulation is not None:
        query = query.filter(DateTimeTable.is_simulation == is_simulation)

    data_points = query.order_by(DateTimeTable.timestamp).limit(1000).all()

    results = []
    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)
        imbalance = predictions['real_time_monitoring']['phase_imbalance_classification']

        results.append({
            'timestamp': point.timestamp,
            'severity': imbalance['severity'],
            'voltage_imbalance': imbalance['voltage_imbalance'],
            'current_imbalance': imbalance['current_imbalance'],
            'power_imbalance': imbalance['power_imbalance'],
            'balance_score': imbalance['balance_score'],
            'action_required': imbalance['action_required']
        })

    return {
        "algorithm": "Decision Tree Classifier",
        "training_dataset": "500 samples with varying degrees of phase imbalance (normal, warning, critical)",
        "benefits": "Identifies unbalanced loads early, guides load redistribution, extends equipment life",
        "data": results
    }
