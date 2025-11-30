"""
REMHART Digital Twin - Simulation Schemas
=========================================
Pydantic schemas for simulation requests and responses.

Author: REMHART Team
Date: 2025
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any
from datetime import datetime


class SimulationParameters(BaseModel):
    """
    Complete simulation parameters.
    Includes basic and advanced options.
    """
    # Basic parameters
    num_points: int = Field(ge=10, le=1000, description="Number of data points")
    interval: int = Field(ge=1, le=60, description="Seconds between points")
    
    # Voltage parameters
    voltage_min: float = Field(ge=150, le=240, description="Minimum voltage (V)")
    voltage_max: float = Field(ge=200, le=280, description="Maximum voltage (V)")
    voltage_pattern: str = Field(default="stable", pattern="^(stable|fluctuating|sag|swell)$")
    
    # Current parameters
    current_min: float = Field(ge=0, le=50, description="Minimum current (A)")
    current_max: float = Field(ge=5, le=100, description="Maximum current (A)")
    current_pattern: str = Field(default="stable", pattern="^(stable|increasing|decreasing|spike)$")
    
    # Frequency parameters
    frequency_min: float = Field(ge=48.0, le=50.0, description="Minimum frequency (Hz)")
    frequency_max: float = Field(ge=50.0, le=52.0, description="Maximum frequency (Hz)")
    frequency_pattern: str = Field(default="stable", pattern="^(stable|drift)$")
    
    # Advanced parameters
    power_factor: float = Field(default=0.95, ge=0.5, le=1.0, description="Power factor")
    renewable_percentage: int = Field(default=30, ge=0, le=100, description="Renewable energy %")
    
    # Fault injection
    inject_fault: bool = Field(default=False)
    fault_type: Optional[str] = Field(default=None, pattern="^(voltage_sag|overcurrent|frequency_drift)$")
    fault_start_point: Optional[int] = Field(default=None, ge=1)
    fault_duration: Optional[int] = Field(default=None, ge=1)
    
    # Load profile
    load_profile: str = Field(default="constant", pattern="^(constant|ramping_up|ramping_down|random)$")
    
    @validator('voltage_max')
    def voltage_max_greater_than_min(cls, v, values):
        if 'voltage_min' in values and v <= values['voltage_min']:
            raise ValueError('voltage_max must be greater than voltage_min')
        return v
    
    @validator('current_max')
    def current_max_greater_than_min(cls, v, values):
        if 'current_min' in values and v <= values['current_min']:
            raise ValueError('current_max must be greater than current_min')
        return v
    
    @validator('frequency_max')
    def frequency_max_greater_than_min(cls, v, values):
        if 'frequency_min' in values and v <= values['frequency_min']:
            raise ValueError('frequency_max must be greater than frequency_min')
        return v


class SimulationRequest(BaseModel):
    """Request to create a new simulation"""
    name: str = Field(min_length=3, max_length=100)
    scenario_type: str = Field(
        pattern="^(normal|peak_load|voltage_sag|overcurrent|frequency_drift|renewable_intermittency|fault|custom)$"
    )
    parameters: SimulationParameters


class SimulationResponse(BaseModel):
    """Response after creating simulation"""
    success: bool
    simulation_id: str
    name: str
    scenario_type: str
    total_points: int
    message: str
    
    class Config:
        from_attributes = True


class SimulationListItem(BaseModel):
    """Simulation item in list"""
    id: int
    simulation_id: str
    name: str
    scenario_type: str
    is_active: bool
    created_at: datetime
    total_points: int
    status: str
    
    class Config:
        from_attributes = True


class SimulationDetail(BaseModel):
    """Detailed simulation information"""
    id: int
    simulation_id: str
    name: str
    scenario_type: str
    parameters: Dict[str, Any]
    is_active: bool
    created_by: str
    created_at: datetime
    total_points: int
    status: str
    
    class Config:
        from_attributes = True