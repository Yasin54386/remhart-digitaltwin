"""
Energy Flow ML API
Provides AI-powered energy flow optimization insights
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


@router.get("/latest")
async def get_latest_energy_insights(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Get latest energy flow ML insights"""
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
        "insights": predictions['energy_flow'],
        "metadata": predictions['metadata']
    }


@router.get("/load-forecast")
async def get_load_forecasting(
    hours: Optional[int] = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get load forecasting predictions

    Returns:
        Historical load and 24-hour forecast with time-series data
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

    # Get time-series load data with forecasts
    results = []
    total_points = len(data_points)
    skipped_count = 0
    error_count = 0

    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)

        # Track errors
        if 'error' in predictions:
            error_count += 1
            continue

        if not predictions.get('energy_flow'):
            skipped_count += 1
            continue

        forecast = predictions['energy_flow'].get('load_forecasting')
        if not forecast:
            skipped_count += 1
            continue

        results.append({
            'timestamp': point.timestamp,
            'current_load_kw': forecast.get('current_load_kw', 0),
            'trend': forecast.get('trend', 'stable'),
            'hourly_forecast': forecast.get('hourly_forecast', []),
            'peak_load_time': forecast.get('peak_load_time', 'N/A')
        })

    return {
        "algorithm": "Prophet / LSTM Time-series Forecasting",
        "training_dataset": "30 days of load data with daily and weekly patterns (peak hours 9-17, 17-22)",
        "benefits": "Enables optimal generator scheduling, reduces fuel costs, improves grid reliability",
        "data": results,
        "diagnostics": {
            "total_data_points": total_points,
            "successful_predictions": len(results),
            "errors": error_count,
            "skipped": skipped_count,
            "success_rate": f"{(len(results)/total_points*100):.1f}%" if total_points > 0 else "0%"
        }
    }


@router.get("/energy-loss")
async def get_energy_loss_estimation(
    hours: Optional[int] = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get energy loss estimations

    Returns:
        Time-series of energy losses and efficiency metrics
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
    total_points = len(data_points)
    skipped_count = 0
    error_count = 0

    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)

        # Track errors
        if 'error' in predictions:
            error_count += 1
            continue

        if not predictions.get('energy_flow'):
            skipped_count += 1
            continue

        loss = predictions['energy_flow'].get('energy_loss_estimation')
        if not loss:
            skipped_count += 1
            continue

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
        "data": results,
        "diagnostics": {
            "total_data_points": total_points,
            "successful_predictions": len(results),
            "errors": error_count,
            "skipped": skipped_count,
            "success_rate": f"{(len(results)/total_points*100):.1f}%" if total_points > 0 else "0%"
        }
    }


@router.get("/power-flow")
async def get_power_flow_optimization(
    hours: Optional[int] = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get power flow optimization recommendations

    Returns:
        Time-series of power distribution and optimization recommendations
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
    total_points = len(data_points)
    skipped_count = 0
    error_count = 0

    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)

        # Track errors
        if 'error' in predictions:
            error_count += 1
            continue

        if not predictions.get('energy_flow'):
            skipped_count += 1
            continue

        optimization = predictions['energy_flow'].get('power_flow_optimization')
        if not optimization:
            skipped_count += 1
            continue

        results.append({
            'timestamp': point.timestamp,
            'current_distribution': {
                'phase_a': point.active_power[0].phaseA if point.active_power else 0,
                'phase_b': point.active_power[0].phaseB if point.active_power else 0,
                'phase_c': point.active_power[0].phaseC if point.active_power else 0
            },
            'optimal_distribution': optimization.get('optimal_distribution', {}),
            'potential_savings_pct': optimization.get('potential_savings_pct', 0),
            'rebalancing_needed': optimization.get('rebalancing_needed', False),
            'suggested_adjustments': optimization.get('suggested_adjustments', {})
        })

    return {
        "algorithm": "Linear Programming Optimization",
        "training_dataset": "Rule-based optimization using power flow equations and constraint satisfaction",
        "benefits": "Minimizes transmission losses, improves voltage profiles, reduces operational costs",
        "data": results,
        "diagnostics": {
            "total_data_points": total_points,
            "successful_predictions": len(results),
            "errors": error_count,
            "skipped": skipped_count,
            "success_rate": f"{(len(results)/total_points*100):.1f}%" if total_points > 0 else "0%"
        }
    }


@router.get("/demand-response")
async def get_demand_response_potential(
    hours: Optional[int] = Query(None, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get demand response potential assessment

    Returns:
        Load clustering and DR recommendations
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
    total_points = len(data_points)
    skipped_count = 0
    error_count = 0

    for point in data_points:
        predictions = ml_inference_engine.process_data_point(point)

        # Track errors
        if 'error' in predictions:
            error_count += 1
            continue

        if not predictions.get('energy_flow'):
            skipped_count += 1
            continue

        dr = predictions['energy_flow'].get('demand_response_assessment')
        if not dr:
            skipped_count += 1
            continue

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
        "data": results,
        "diagnostics": {
            "total_data_points": total_points,
            "successful_predictions": len(results),
            "errors": error_count,
            "skipped": skipped_count,
            "success_rate": f"{(len(results)/total_points*100):.1f}%" if total_points > 0 else "0%"
        }
    }
