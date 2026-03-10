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
# Active model: demand_response_assessment (K-Means, trained)
# Removed: load_forecasting (stub), energy_loss (stub), power_flow (rule-based dict)


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
