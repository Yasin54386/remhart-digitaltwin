"""
REMHART Digital Twin - Simulator Router
========================================
Handles simulation creation, management, and data generation.

Endpoints:
- POST /api/simulator/run - Create and run simulation
- GET /api/simulator/list - List all simulations
- GET /api/simulator/active - Get active simulation
- GET /api/simulator/{id} - Get simulation details
- DELETE /api/simulator/{id} - Delete simulation
- POST /api/simulator/{id}/activate - Set as active simulation
- GET /api/simulator/templates - Get scenario templates

Author: REMHART Team
Date: 2025
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List
import uuid
from datetime import datetime
import asyncio

from app.database import get_db
from app.models.db_models import SimulationMetadata, DateTimeTable
from app.schemas.simulation import (
    SimulationRequest, SimulationResponse, 
    SimulationListItem, SimulationDetail, SimulationParameters
)
from app.services.simulation_generator import simulation_generator
from app.routers.grid_data import add_grid_data
from app.schemas.grid_data import GridDataPoint
from app.utils.security import get_current_user

router = APIRouter()


@router.post("/run", response_model=SimulationResponse)
async def run_simulation(
    request: SimulationRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Create and run a new simulation.
    
    This will:
    1. Deactivate any currently active simulation
    2. Create simulation metadata
    3. Generate data points based on parameters
    4. Insert data with simulation flags
    5. Broadcast to WebSocket clients
    
    Args:
        request: Simulation configuration
        db: Database session
        current_user: Authenticated user
        
    Returns:
        Simulation details and status
    """
    try:
        # Generate unique simulation ID
        sim_id = str(uuid.uuid4())
        
        # Deactivate any currently active simulation
        active_sims = db.query(SimulationMetadata).filter(
            SimulationMetadata.is_active == True
        ).all()
        
        for sim in active_sims:
            sim.is_active = False
        
        # Get parameters (use template or custom)
        if request.scenario_type != "custom":
            # Use template parameters
            params = simulation_generator.generate_scenario_from_template(
                request.scenario_type,
                request.parameters.num_points
            )
            # Override with any user-provided values
            params.update(request.parameters.dict(exclude_unset=True))
        else:
            # Use all user-provided parameters
            params = request.parameters.dict()
        
        # Create simulation metadata
        simulation = SimulationMetadata(
            simulation_id=sim_id,
            name=request.name,
            scenario_type=request.scenario_type,
            is_active=True,
            created_by=current_user.get("sub", "unknown"),
            created_at=datetime.now(),
            total_points=params['num_points'],
            status='running'
        )
        simulation.set_parameters(params)
        
        db.add(simulation)
        db.commit()
        db.refresh(simulation)
        
        # Generate simulation data
        data_points = simulation_generator.generate_simulation(params)
        
        # Mock user for data insertion
        mock_user = {"sub": current_user.get("sub"), "user_id": current_user.get("user_id"), "role": "operator"}
        
        # Insert data points with simulation flags
        count = 0
        for dp in data_points:
            # Convert to GridDataPoint schema
            grid_data = GridDataPoint(
                timestamp=dp["timestamp"],
                voltage=dp["voltage"],
                current=dp["current"],
                frequency=dp["frequency"],
                active_power=dp["active_power"],
                reactive_power=dp["reactive_power"]
            )
            
            # Add via standard pipeline with simulation metadata
            await add_grid_data_with_simulation(
                grid_data, 
                db, 
                mock_user,
                is_simulation=True,
                simulation_id=sim_id,
                simulation_name=request.name,
                simulation_scenario=request.scenario_type
            )
            
            count += 1
            
            # Small delay for real-time streaming (1 second)
            await asyncio.sleep(params.get('interval', 1))
        
        # Update simulation status
        simulation.status = 'completed'
        simulation.total_points = count
        db.commit()
        
        return SimulationResponse(
            success=True,
            simulation_id=sim_id,
            name=request.name,
            scenario_type=request.scenario_type,
            total_points=count,
            message=f"Simulation '{request.name}' completed successfully with {count} data points"
        )
        
    except Exception as e:
        # Mark simulation as failed
        if 'simulation' in locals():
            simulation.status = 'failed'
            db.commit()
        
        raise HTTPException(status_code=500, detail=f"Simulation failed: {str(e)}")


@router.get("/list", response_model=List[SimulationListItem])
async def list_simulations(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    List all simulations, most recent first.
    
    Returns:
        List of simulations with metadata
    """
    simulations = db.query(SimulationMetadata).order_by(
        desc(SimulationMetadata.created_at)
    ).all()
    
    return simulations


@router.get("/active", response_model=SimulationDetail)
async def get_active_simulation(
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get the currently active simulation.
    
    Returns:
        Active simulation details
        
    Raises:
        404: If no active simulation exists
    """
    active_sim = db.query(SimulationMetadata).filter(
        SimulationMetadata.is_active == True
    ).first()
    
    if not active_sim:
        raise HTTPException(status_code=404, detail="No active simulation")
    
    return SimulationDetail(
        id=active_sim.id,
        simulation_id=active_sim.simulation_id,
        name=active_sim.name,
        scenario_type=active_sim.scenario_type,
        parameters=active_sim.get_parameters(),
        is_active=active_sim.is_active,
        created_by=active_sim.created_by,
        created_at=active_sim.created_at,
        total_points=active_sim.total_points,
        status=active_sim.status
    )


@router.get("/{simulation_id}", response_model=SimulationDetail)
async def get_simulation(
    simulation_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Get details of a specific simulation.
    
    Args:
        simulation_id: Simulation UUID
        
    Returns:
        Simulation details
    """
    simulation = db.query(SimulationMetadata).filter(
        SimulationMetadata.simulation_id == simulation_id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    return SimulationDetail(
        id=simulation.id,
        simulation_id=simulation.simulation_id,
        name=simulation.name,
        scenario_type=simulation.scenario_type,
        parameters=simulation.get_parameters(),
        is_active=simulation.is_active,
        created_by=simulation.created_by,
        created_at=simulation.created_at,
        total_points=simulation.total_points,
        status=simulation.status
    )


@router.delete("/{simulation_id}")
async def delete_simulation(
    simulation_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Delete a simulation and all its data.
    
    This will:
    1. Delete simulation metadata
    2. Delete all data points associated with this simulation
    
    Args:
        simulation_id: Simulation UUID
        
    Returns:
        Success message
    """
    # Find simulation
    simulation = db.query(SimulationMetadata).filter(
        SimulationMetadata.simulation_id == simulation_id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Delete all data points for this simulation
    deleted_count = db.query(DateTimeTable).filter(
        DateTimeTable.simulation_id == simulation_id
    ).delete()
    
    # Delete simulation metadata
    db.delete(simulation)
    db.commit()
    
    return {
        "success": True,
        "message": f"Deleted simulation '{simulation.name}' and {deleted_count} data points",
        "deleted_points": deleted_count
    }


@router.post("/{simulation_id}/activate")
async def activate_simulation(
    simulation_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user)
):
    """
    Set a simulation as active.
    
    Only one simulation can be active at a time.
    This determines which simulation data is shown when "Simulator" mode is enabled.
    
    Args:
        simulation_id: Simulation UUID to activate
        
    Returns:
        Success message
    """
    # Find the simulation
    simulation = db.query(SimulationMetadata).filter(
        SimulationMetadata.simulation_id == simulation_id
    ).first()
    
    if not simulation:
        raise HTTPException(status_code=404, detail="Simulation not found")
    
    # Deactivate all other simulations
    db.query(SimulationMetadata).filter(
        SimulationMetadata.is_active == True
    ).update({"is_active": False})
    
    # Activate this simulation
    simulation.is_active = True
    db.commit()
    
    return {
        "success": True,
        "message": f"Activated simulation '{simulation.name}'",
        "simulation_id": simulation_id
    }


@router.get("/templates/list")
async def get_scenario_templates():
    """
    Get list of available scenario templates with descriptions.
    
    Returns:
        List of scenario templates
    """
    templates = [
        {
            "id": "normal",
            "name": "Normal Operation",
            "description": "Stable grid operation with nominal parameters",
            "icon": "check-circle"
        },
        {
            "id": "peak_load",
            "name": "Peak Load",
            "description": "High demand period with increased current and voltage sag",
            "icon": "trending-up"
        },
        {
            "id": "voltage_sag",
            "name": "Voltage Sag Event",
            "description": "Temporary voltage drop due to fault or overload",
            "icon": "arrow-down"
        },
        {
            "id": "overcurrent",
            "name": "Overcurrent/Overload",
            "description": "Excessive current flow causing potential equipment damage",
            "icon": "alert-triangle"
        },
        {
            "id": "frequency_drift",
            "name": "Frequency Instability",
            "description": "Grid frequency deviation from 50Hz standard",
            "icon": "activity"
        },
        {
            "id": "renewable_intermittency",
            "name": "Renewable Intermittency",
            "description": "Fluctuations due to solar/wind variability",
            "icon": "sun"
        },
        {
            "id": "fault",
            "name": "Grid Fault",
            "description": "Severe fault condition with multiple parameter anomalies",
            "icon": "x-circle"
        },
        {
            "id": "custom",
            "name": "Custom Scenario",
            "description": "Define your own parameters",
            "icon": "settings"
        }
    ]
    
    return {"templates": templates}


# Helper function for add_grid_data with simulation metadata
async def add_grid_data_with_simulation(
    data: GridDataPoint,
    db: Session,
    current_user: dict,
    is_simulation: bool = False,
    simulation_id: str = None,
    simulation_name: str = None,
    simulation_scenario: str = None
):
    """
    Modified version of add_grid_data that includes simulation metadata.
    """
    from app.models.db_models import (
        DateTimeTable, VoltageTable, CurrentTable,
        FrequencyTable, ActivePowerTable, ReactivePowerTable
    )
    from app.routers.websocket_router import manager
    
    try:
        # Create datetime entry with simulation metadata
        dt_entry = DateTimeTable(
            timestamp=data.timestamp,
            is_simulation=is_simulation,
            simulation_id=simulation_id,
            simulation_name=simulation_name,
            simulation_scenario=simulation_scenario
        )
        db.add(dt_entry)
        db.flush()
        
        # Create related entries
        db.add(VoltageTable(timestamp_id=dt_entry.id, **data.voltage.dict()))
        db.add(CurrentTable(timestamp_id=dt_entry.id, **data.current.dict()))
        db.add(FrequencyTable(timestamp_id=dt_entry.id, **data.frequency.dict()))
        db.add(ActivePowerTable(timestamp_id=dt_entry.id, **data.active_power.dict()))
        db.add(ReactivePowerTable(timestamp_id=dt_entry.id, **data.reactive_power.dict()))
        
        db.commit()
        
        # Prepare data for broadcast
        broadcast_data = {
            "id": dt_entry.id,
            "timestamp": data.timestamp.isoformat(),
            "voltage": data.voltage.dict(),
            "current": data.current.dict(),
            "frequency": {"value": data.frequency.frequency_value},
            "active_power": data.active_power.dict(),
            "reactive_power": data.reactive_power.dict(),
            "is_simulation": is_simulation,
            "simulation_id": simulation_id
        }
        
        # Broadcast to all WebSocket clients
        await manager.broadcast({
            "type": "new_data",
            "data": broadcast_data
        })
        
        return {
            "success": True,
            "data_id": dt_entry.id
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding grid data: {str(e)}")
    

print("Simulator router loaded successfully")
