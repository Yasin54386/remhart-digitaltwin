"""
REMHART Digital Twin - Database Models
======================================
SQLAlchemy ORM models matching your existing database schema.
These models represent the smart grid data structure.

Tables:
- datetime_table: Central timestamp reference
- voltage_table: 3-phase voltage readings
- current_table: 3-phase current readings
- frequency_table: Grid frequency measurements
- active_power_table: 3-phase active power
- reactive_power_table: 3-phase reactive power

Author: REMHART Team
Date: 2025
"""

from sqlalchemy import Column, Integer, Float, DateTime, ForeignKey, String, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects import mysql
from sqlalchemy import Boolean, Text
import json
from app.database import Base


# Add to imports at top
from sqlalchemy import Boolean, Text
import json

# Update DateTimeTable class
class DateTimeTable(Base):
    """
    Central timestamp table for all grid measurements.
    Now supports both real and simulated data.
    """
    __tablename__ = 'datetime_table'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(mysql.DATETIME(fsp=6), nullable=False, index=True)
    
    # Simulation tracking fields
    is_simulation = Column(Boolean, default=False, index=True, nullable=False)
    simulation_id = Column(String(50), nullable=True, index=True)
    simulation_name = Column(String(100), nullable=True)
    simulation_scenario = Column(String(50), nullable=True)
    
    # Relationships to other tables
    voltage = relationship("VoltageTable", back_populates="timestamp_ref", cascade="all, delete-orphan")
    current = relationship("CurrentTable", back_populates="timestamp_ref", cascade="all, delete-orphan")
    frequency = relationship("FrequencyTable", back_populates="timestamp_ref", cascade="all, delete-orphan")
    active_power = relationship("ActivePowerTable", back_populates="timestamp_ref", cascade="all, delete-orphan")
    reactive_power = relationship("ReactivePowerTable", back_populates="timestamp_ref", cascade="all, delete-orphan")

    def __repr__(self):
        sim_flag = " [SIM]" if self.is_simulation else ""
        return f"<DateTimeTable(id={self.id}, timestamp={self.timestamp}{sim_flag})>"


# Add new SimulationMetadata table at the end, before User class
class SimulationMetadata(Base):
    """
    Stores metadata about simulation runs.
    Only one simulation can be active at a time.
    """
    __tablename__ = 'simulation_metadata'
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(String(50), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    scenario_type = Column(String(50), nullable=False)
    
    # Store all input parameters as JSON
    parameters = Column(Text, nullable=False)
    
    # Only one simulation can be active
    is_active = Column(Boolean, default=False, index=True)
    
    # Tracking
    created_by = Column(String(50), nullable=False)
    created_at = Column(DateTime, nullable=False)
    total_points = Column(Integer, default=0)
    
    # Status
    status = Column(String(20), default='running')  # running, completed, cancelled
    
    def __repr__(self):
        active = " [ACTIVE]" if self.is_active else ""
        return f"<Simulation(name={self.name}, scenario={self.scenario_type}{active})>"
    
    def get_parameters(self):
        """Parse JSON parameters"""
        return json.loads(self.parameters)
    
    def set_parameters(self, params_dict):
        """Store parameters as JSON"""
        self.parameters = json.dumps(params_dict)
        
class VoltageTable(Base):
    """
    3-Phase voltage measurements.
    Typical range: 220-240V per phase for Australian grid
    """
    __tablename__ = 'voltage_table'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_id = Column(Integer, ForeignKey('datetime_table.id'), nullable=False, index=True)
    
    # Phase voltages in Volts (V)
    phaseA = Column(Float, nullable=False)
    phaseB = Column(Float, nullable=False)
    phaseC = Column(Float, nullable=False)
    average = Column(Float, nullable=False)  # Average of 3 phases
    
    timestamp_ref = relationship("DateTimeTable", back_populates="voltage")

    def __repr__(self):
        return f"<VoltageTable(id={self.id}, avg={self.average}V)>"


class CurrentTable(Base):
    """
    3-Phase current measurements.
    Measured in Amperes (A)
    """
    __tablename__ = 'current_table'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_id = Column(Integer, ForeignKey('datetime_table.id'), nullable=False, index=True)
    
    # Phase currents in Amperes (A)
    phaseA = Column(Float, nullable=False)
    phaseB = Column(Float, nullable=False)
    phaseC = Column(Float, nullable=False)
    average = Column(Float, nullable=False)  # Average of 3 phases
    
    timestamp_ref = relationship("DateTimeTable", back_populates="current")

    def __repr__(self):
        return f"<CurrentTable(id={self.id}, avg={self.average}A)>"


class FrequencyTable(Base):
    """
    Grid frequency measurements.
    Standard: 50Hz for Australian grid
    Acceptable range: 49.85 - 50.15 Hz
    """
    __tablename__ = 'frequency_table'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_id = Column(Integer, ForeignKey('datetime_table.id'), nullable=False, index=True)
    
    # Frequency in Hertz (Hz)
    frequency_value = Column(Float, nullable=False)
    
    timestamp_ref = relationship("DateTimeTable", back_populates="frequency")

    def __repr__(self):
        return f"<FrequencyTable(id={self.id}, freq={self.frequency_value}Hz)>"


class ActivePowerTable(Base):
    """
    3-Phase active (real) power measurements.
    Measured in Watts (W) or Kilowatts (kW)
    Active power does actual work in the circuit.
    """
    __tablename__ = 'active_power_table'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_id = Column(Integer, ForeignKey('datetime_table.id'), nullable=False, index=True)
    
    # Active power per phase in Watts (W)
    phaseA = Column(Float, nullable=False)
    phaseB = Column(Float, nullable=False)
    phaseC = Column(Float, nullable=False)
    total = Column(Float, nullable=False)  # Sum of all phases
    
    timestamp_ref = relationship("DateTimeTable", back_populates="active_power")

    def __repr__(self):
        return f"<ActivePowerTable(id={self.id}, total={self.total}W)>"


class ReactivePowerTable(Base):
    """
    3-Phase reactive power measurements.
    Measured in Volt-Amperes Reactive (VAR) or kVAR
    Reactive power maintains voltage levels in the grid.
    """
    __tablename__ = 'reactive_power_table'
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp_id = Column(Integer, ForeignKey('datetime_table.id'), nullable=False, index=True)
    
    # Reactive power per phase in VAR
    phaseA = Column(Float, nullable=False)
    phaseB = Column(Float, nullable=False)
    phaseC = Column(Float, nullable=False)
    total = Column(Float, nullable=False)  # Sum of all phases
    
    timestamp_ref = relationship("DateTimeTable", back_populates="reactive_power")

    def __repr__(self):
        return f"<ReactivePowerTable(id={self.id}, total={self.total}VAR)>"


class User(Base):
    """
    User authentication table.
    Supports role-based access control (RBAC)
    """
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False, index=True)
    email = Column(String(100), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100))
    
    # Role: 'admin', 'operator', 'analyst', 'viewer'
    role = Column(String(20), nullable=False, default='viewer')
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime)
    last_login = Column(DateTime)

    def __repr__(self):
        return f"<User(username={self.username}, role={self.role})>"