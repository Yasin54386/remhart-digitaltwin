"""
REMHART Digital Twin - Pydantic Schemas
========================================
Data validation and serialization schemas using Pydantic.
These define the structure of API requests and responses.

Author: REMHART Team
Date: 2025
"""

from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional, List


# ============================================
# Grid Data Schemas
# ============================================

class VoltageData(BaseModel):
    """Voltage measurement data (3-phase)"""
    phaseA: float = Field(..., description="Phase A voltage in Volts", ge=0, le=300)
    phaseB: float = Field(..., description="Phase B voltage in Volts", ge=0, le=300)
    phaseC: float = Field(..., description="Phase C voltage in Volts", ge=0, le=300)
    average: float = Field(..., description="Average voltage", ge=0, le=300)
    
    class Config:
        json_schema_extra = {
            "example": {
                "phaseA": 230.5,
                "phaseB": 231.2,
                "phaseC": 229.8,
                "average": 230.5
            }
        }


class CurrentData(BaseModel):
    """Current measurement data (3-phase)"""
    phaseA: float = Field(..., description="Phase A current in Amperes", ge=0)
    phaseB: float = Field(..., description="Phase B current in Amperes", ge=0)
    phaseC: float = Field(..., description="Phase C current in Amperes", ge=0)
    average: float = Field(..., description="Average current", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "phaseA": 15.3,
                "phaseB": 14.8,
                "phaseC": 15.1,
                "average": 15.1
            }
        }


class FrequencyData(BaseModel):
    """Grid frequency measurement"""
    frequency_value: float = Field(
        ..., 
        description="Grid frequency in Hz",
        ge=49.0,
        le=51.0
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "frequency_value": 50.02
            }
        }


class ActivePowerData(BaseModel):
    """Active power measurement (3-phase)"""
    phaseA: float = Field(..., description="Phase A active power in Watts")
    phaseB: float = Field(..., description="Phase B active power in Watts")
    phaseC: float = Field(..., description="Phase C active power in Watts")
    total: float = Field(..., description="Total active power in Watts")
    
    class Config:
        json_schema_extra = {
            "example": {
                "phaseA": 3521.5,
                "phaseB": 3410.2,
                "phaseC": 3465.8,
                "total": 10397.5
            }
        }


class ReactivePowerData(BaseModel):
    """Reactive power measurement (3-phase)"""
    phaseA: float = Field(..., description="Phase A reactive power in VAR")
    phaseB: float = Field(..., description="Phase B reactive power in VAR")
    phaseC: float = Field(..., description="Phase C reactive power in VAR")
    total: float = Field(..., description="Total reactive power in VAR")
    
    class Config:
        json_schema_extra = {
            "example": {
                "phaseA": 245.3,
                "phaseB": 238.7,
                "phaseC": 242.1,
                "total": 726.1
            }
        }


class GridDataPoint(BaseModel):
    """
    Complete grid data point - all measurements at one timestamp.
    This is what SCADA would send every few seconds.
    """
    timestamp: datetime
    voltage: VoltageData
    current: CurrentData
    frequency: FrequencyData
    active_power: ActivePowerData
    reactive_power: ReactivePowerData
    
    class Config:
        json_schema_extra = {
            "example": {
                "timestamp": "2025-01-15T10:30:45.123456",
                "voltage": {
                    "phaseA": 230.5,
                    "phaseB": 231.2,
                    "phaseC": 229.8,
                    "average": 230.5
                },
                "current": {
                    "phaseA": 15.3,
                    "phaseB": 14.8,
                    "phaseC": 15.1,
                    "average": 15.1
                },
                "frequency": {
                    "frequency_value": 50.02
                },
                "active_power": {
                    "phaseA": 3521.5,
                    "phaseB": 3410.2,
                    "phaseC": 3465.8,
                    "total": 10397.5
                },
                "reactive_power": {
                    "phaseA": 245.3,
                    "phaseB": 238.7,
                    "phaseC": 242.1,
                    "total": 726.1
                }
            }
        }


class GridDataResponse(BaseModel):
    """Response schema for grid data with ID"""
    id: int
    timestamp: datetime
    voltage: VoltageData
    current: CurrentData
    frequency: FrequencyData
    active_power: ActivePowerData
    reactive_power: ReactivePowerData
    
    class Config:
        from_attributes = True


# ============================================
# Authentication Schemas
# ============================================

class UserLogin(BaseModel):
    """User login credentials"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    
    class Config:
        json_schema_extra = {
            "example": {
                "username": "admin",
                "password": "securepassword123"
            }
        }


class Token(BaseModel):
    """JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    """User information response"""
    id: int
    username: str
    email: str
    full_name: Optional[str]
    role: str
    is_active: bool
    
    class Config:
        from_attributes = True


# ============================================
# Query Parameters
# ============================================

class GridDataQuery(BaseModel):
    """Query parameters for fetching grid data"""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    limit: int = Field(default=100, ge=1, le=10000)
    offset: int = Field(default=0, ge=0)


# ============================================
# Real-time Status
# ============================================

class GridStatus(BaseModel):
    """Current grid status summary"""
    voltage_avg: float
    current_avg: float
    frequency: float
    active_power_total: float
    reactive_power_total: float
    power_factor: float
    status: str  # "Normal", "Warning", "Critical"
    timestamp: datetime