"""
Decision Making ML API
Provides AI-powered decision support for grid operations
"""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import Optional
from datetime import datetime, timedelta

from ..database import get_db
from ..models.db_models import DateTimeTable
from ..services.ml_inference_engine import ml_inference_engine
from ..utils.security import get_optional_user

router = APIRouter(prefix="/api/ml/decision", tags=["ML Decision"])


@router.get("/latest")
async def get_latest_decision_insights(
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """Get latest decision-making ML insights"""
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
        "insights": predictions['decision_making'],
        "metadata": predictions['metadata']
    }


@router.get("/reactive-compensation")
async def get_reactive_power_compensation(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get reactive power compensation recommendations

    Returns:
        Time-series of power factor and compensation requirements
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
        compensation = predictions['decision_making']['reactive_power_compensation']

        results.append({
            'timestamp': point.timestamp,
            'current_pf': compensation['current_pf'],
            'target_pf': compensation['target_pf'],
            'required_compensation_kvar': compensation['required_compensation_kvar'],
            'capacitor_size_kvar': compensation['capacitor_size_kvar'],
            'expected_savings': compensation['expected_savings']
        })

    return {
        "algorithm": "Neural Network Optimizer",
        "training_dataset": "Power factor correction scenarios targeting 0.95 PF, calculated using power triangle equations",
        "benefits": "Reduces reactive power charges, improves voltage regulation, increases system capacity",
        "data": results
    }


@router.get("/load-balancing")
async def get_load_balancing_optimization(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get load balancing optimization recommendations

    Returns:
        Time-series of load distribution and redistribution plans
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
        balancing = predictions['decision_making']['load_balancing_optimization']

        # Get current phase currents
        i_a = point.current[0].phaseA if point.current else 0
        i_b = point.current[0].phaseB if point.current else 0
        i_c = point.current[0].phaseC if point.current else 0

        results.append({
            'timestamp': point.timestamp,
            'current_distribution': {
                'phase_a': i_a,
                'phase_b': i_b,
                'phase_c': i_c
            },
            'current_imbalance': balancing['current_imbalance'],
            'redistribution_plan': balancing['redistribution_plan'],
            'expected_improvement_pct': balancing['expected_improvement_pct']
        })

    return {
        "algorithm": "Multi-Criteria Decision Analysis (MCDA)",
        "training_dataset": "Optimization scenarios balancing load distribution, losses, and voltage stability",
        "benefits": "Reduces neutral current, extends transformer life, improves efficiency",
        "data": results
    }


@router.get("/grid-stability")
async def get_grid_stability_scoring(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get grid stability scores

    Returns:
        Time-series of comprehensive stability scores
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
        stability = predictions['decision_making']['grid_stability_scoring']

        results.append({
            'timestamp': point.timestamp,
            'stability_score': stability['stability_score'],
            'status': stability['status'],
            'risk_factors': stability['risk_factors'],
            'recommendations': stability['recommendations']
        })

    return {
        "algorithm": "Ensemble Model (Random Forest + Rule-based)",
        "training_dataset": "Stability scores calculated from voltage, frequency, power factor, and balance metrics",
        "benefits": "Provides single metric for grid health, guides operator decisions, prevents blackouts",
        "data": results
    }


@router.get("/optimal-dispatch")
async def get_optimal_dispatch_advisory(
    hours: int = Query(24, description="Hours of historical data"),
    is_simulation: Optional[bool] = Query(None),
    current_user = Depends(get_optional_user),
    db: Session = Depends(get_db)
):
    """
    Get optimal generation dispatch recommendations

    Returns:
        Time-series of load and recommended generation
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
        dispatch = predictions['decision_making']['optimal_dispatch_advisory']

        results.append({
            'timestamp': point.timestamp,
            'current_load_kw': dispatch['current_load_kw'],
            'recommended_generation_kw': dispatch['recommended_generation_kw'],
            'reserve_margin_pct': dispatch['reserve_margin_pct'],
            'dispatch_plan': dispatch['dispatch_plan']
        })

    return {
        "algorithm": "SVR (Support Vector Regression)",
        "training_dataset": "Load patterns with optimal generation including 15% spinning reserve",
        "benefits": "Optimizes fuel consumption, maintains reliability reserves, reduces generation costs",
        "data": results
    }
