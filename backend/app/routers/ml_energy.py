"""
Energy Flow ML API
Provides AI-powered energy flow optimization insights
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

router = APIRouter(prefix="/api/ml/energy", tags=["ML Energy"])


@router.get("/latest")
async def get_latest_energy_insights(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get latest energy flow ML insights"""
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

    predictions = ml_inference_engine.process_data_point(latest)

    return {
        "timestamp": latest.timestamp,
        "is_simulation": latest.is_simulation,
        "insights": predictions['energy_flow'],
        "metadata": predictions['metadata']
    }


@router.get("/load-forecast")
async def get_load_forecasting(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get load forecasting predictions

    Returns:
        Historical load and 24-hour forecast
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

    data_points = query.order_by(DateTimeTable.timestamp).limit(500).all()

    # Get historical loads
    historical = []
    for point in data_points:
        if point.active_power:
            load = sum([point.active_power[0].phaseA,
                       point.active_power[0].phaseB,
                       point.active_power[0].phaseC])
            historical.append({
                'timestamp': point.timestamp,
                'load_kw': load
            })

    # Get forecast from latest point
    if data_points:
        predictions = ml_inference_engine.process_data_point(data_points[-1])
        forecast_data = predictions['energy_flow']['load_forecasting']
    else:
        forecast_data = {'hourly_forecast': [], 'current_load_kw': 0, 'trend': 'stable'}

    return {
        "algorithm": "Prophet / LSTM Time-series Forecasting",
        "training_dataset": "30 days of load data with daily and weekly patterns (peak hours 9-17, 17-22)",
        "benefits": "Enables optimal generator scheduling, reduces fuel costs, improves grid reliability",
        "historical": historical,
        "forecast": forecast_data
    }


@router.get("/energy-loss")
async def get_energy_loss_estimation(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get energy loss estimations

    Returns:
        Time-series of energy losses and efficiency metrics
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
        loss = predictions['energy_flow']['energy_loss_estimation']

        results.append({
            'timestamp': point.timestamp,
            'loss_kw': loss['loss_kw'],
            'loss_percentage': loss['loss_percentage'],
            'efficiency': loss['efficiency'],
            'loss_breakdown': loss['loss_breakdown']
        })

    return {
        "algorithm": "Linear Regression (I²R loss model)",
        "training_dataset": "Physics-based model using current, power, imbalance, and power factor data",
        "benefits": "Identifies inefficiency sources, guides conductor upgrades, calculates cost savings from improvements",
        "data": results
    }


@router.get("/power-flow-optimization")
async def get_power_flow_optimization(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get power flow optimization recommendations

    Returns:
        Current distribution and optimal redistribution plan
    """
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

    predictions = ml_inference_engine.process_data_point(latest)
    optimization = predictions['energy_flow']['power_flow_optimization']

    return {
        "algorithm": "Linear Programming Optimization",
        "training_dataset": "Rule-based optimization using power flow equations and constraint satisfaction",
        "benefits": "Minimizes transmission losses, improves voltage profiles, reduces operational costs",
        "current_state": {
            'timestamp': latest.timestamp,
            'phase_distribution': {
                'phase_a': latest.active_power[0].phaseA if latest.active_power else 0,
                'phase_b': latest.active_power[0].phaseB if latest.active_power else 0,
                'phase_c': latest.active_power[0].phaseC if latest.active_power else 0
            }
        },
        "optimization": optimization
    }


@router.get("/demand-response")
async def get_demand_response_potential(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get demand response potential assessment

    Returns:
        Load clustering and DR recommendations
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
        dr = predictions['energy_flow']['demand_response_assessment']

        load = point.active_power[0].total if point.active_power else 0

        results.append({
            'timestamp': point.timestamp,
            'load_kw': load,
            'load_cluster': dr['load_cluster'],
            'dr_potential_kw': dr['dr_potential_kw'],
            'flexibility_score': dr['flexibility_score'],
            'recommended_actions': dr['recommended_actions']
        })

    return {
        "algorithm": "K-Means Clustering",
        "training_dataset": "Load profiles clustered into 3 categories: low-load, medium-load, high-load",
        "benefits": "Identifies opportunities for demand response programs, reduces peak demand charges",
        "data": results
    }
