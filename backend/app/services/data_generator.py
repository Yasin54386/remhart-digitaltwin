"""
REMHART Digital Twin - Smart Grid Data Generator
=================================================
Generates realistic seeded data for testing and development.
Simulates SCADA data with normal and anomaly conditions.

Features:
- Realistic 3-phase power grid data
- Anomaly injection for testing ML models
- Configurable data patterns
- Time-series simulation

Author: REMHART Team
Date: 2025
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List
import math


class GridDataGenerator:
    """
    Smart Grid Data Generator for REMHART.
    
    Generates realistic power grid measurements including:
    - 3-phase voltage (220-240V nominal)
    - 3-phase current (10-20A nominal)
    - Grid frequency (50Hz nominal for Australia)
    - Active and reactive power
    
    Can inject anomalies for testing:
    - Voltage sags/swells
    - Frequency deviations
    - Overcurrent conditions
    - Power factor issues
    """
    
    def __init__(self):
        # Australian grid standards
        self.NOMINAL_VOLTAGE = 230.0  # Volts
        self.NOMINAL_FREQUENCY = 50.0  # Hz
        self.NOMINAL_CURRENT = 15.0  # Amperes
        
        # Acceptable ranges
        self.VOLTAGE_TOLERANCE = 0.05  # ±5%
        self.FREQUENCY_TOLERANCE = 0.003  # ±0.15 Hz (0.3%)
        
        # Anomaly probabilities
        self.ANOMALY_PROBABILITY = 0.05  # 5% chance of anomaly
        
    def generate_voltage(self, anomaly: bool = False) -> Dict[str, float]:
        """
        Generate 3-phase voltage readings.
        
        Normal range: 218.5 - 241.5V (±5%)
        Anomaly: Outside normal range
        
        Args:
            anomaly: If True, generate anomalous readings
            
        Returns:
            dict: {phaseA, phaseB, phaseC, average}
        """
        if anomaly:
            # Voltage sag (undervoltage) or swell (overvoltage)
            if random.random() < 0.5:
                # Sag: 200-220V
                base_voltage = random.uniform(200, 220)
            else:
                # Swell: 240-250V
                base_voltage = random.uniform(240, 250)
        else:
            # Normal operation
            base_voltage = self.NOMINAL_VOLTAGE
        
        # Generate 3-phase voltages with slight variations
        phaseA = base_voltage + random.uniform(-3, 3)
        phaseB = base_voltage + random.uniform(-3, 3)
        phaseC = base_voltage + random.uniform(-3, 3)
        
        return {
            "phaseA": round(phaseA, 2),
            "phaseB": round(phaseB, 2),
            "phaseC": round(phaseC, 2),
            "average": round((phaseA + phaseB + phaseC) / 3, 2)
        }
    
    def generate_current(self, anomaly: bool = False) -> Dict[str, float]:
        """
        Generate 3-phase current readings.
        
        Normal range: 10-20A
        Anomaly: >25A (overcurrent)
        
        Args:
            anomaly: If True, generate anomalous readings
            
        Returns:
            dict: {phaseA, phaseB, phaseC, average}
        """
        if anomaly:
            # Overcurrent condition
            base_current = random.uniform(25, 35)
        else:
            # Normal operation
            base_current = self.NOMINAL_CURRENT
        
        # Generate 3-phase currents with variations
        phaseA = base_current + random.uniform(-2, 2)
        phaseB = base_current + random.uniform(-2, 2)
        phaseC = base_current + random.uniform(-2, 2)
        
        return {
            "phaseA": round(phaseA, 2),
            "phaseB": round(phaseB, 2),
            "phaseC": round(phaseC, 2),
            "average": round((phaseA + phaseB + phaseC) / 3, 2)
        }
    
    def generate_frequency(self, anomaly: bool = False) -> float:
        """
        Generate grid frequency reading.
        
        Normal range: 49.85 - 50.15 Hz
        Anomaly: Outside normal range
        
        Args:
            anomaly: If True, generate anomalous reading
            
        Returns:
            float: Frequency in Hz
        """
        if anomaly:
            # Frequency deviation
            if random.random() < 0.5:
                # Under-frequency
                frequency = random.uniform(49.0, 49.8)
            else:
                # Over-frequency
                frequency = random.uniform(50.2, 51.0)
        else:
            # Normal operation
            frequency = self.NOMINAL_FREQUENCY + random.uniform(-0.1, 0.1)
        
        return round(frequency, 2)
    
    def generate_power(self, voltage: Dict[str, float], current: Dict[str, float], 
                      power_factor: float = 0.95) -> tuple:
        """
        Calculate active and reactive power from voltage and current.
        
        Active Power (P) = V × I × cos(φ)
        Reactive Power (Q) = V × I × sin(φ)
        
        Args:
            voltage: Voltage readings per phase
            current: Current readings per phase
            power_factor: Power factor (cos φ), default 0.95
            
        Returns:
            tuple: (active_power_dict, reactive_power_dict)
        """
        # Calculate power factor angle
        phi = math.acos(power_factor)
        
        # Active power per phase
        active_A = voltage["phaseA"] * current["phaseA"] * power_factor
        active_B = voltage["phaseB"] * current["phaseB"] * power_factor
        active_C = voltage["phaseC"] * current["phaseC"] * power_factor
        
        # Reactive power per phase
        reactive_A = voltage["phaseA"] * current["phaseA"] * math.sin(phi)
        reactive_B = voltage["phaseB"] * current["phaseB"] * math.sin(phi)
        reactive_C = voltage["phaseC"] * current["phaseC"] * math.sin(phi)
        
        active_power = {
            "phaseA": round(active_A, 2),
            "phaseB": round(active_B, 2),
            "phaseC": round(active_C, 2),
            "total": round(active_A + active_B + active_C, 2)
        }
        
        reactive_power = {
            "phaseA": round(reactive_A, 2),
            "phaseB": round(reactive_B, 2),
            "phaseC": round(reactive_C, 2),
            "total": round(reactive_A + reactive_B + reactive_C, 2)
        }
        
        return active_power, reactive_power
    
    def generate_single_datapoint(self, timestamp: datetime = None, 
                                 force_anomaly: bool = False) -> Dict:
        """
        Generate a single complete grid data point.
        
        Args:
            timestamp: Timestamp for the data point (default: now)
            force_anomaly: Force generation of anomaly
            
        Returns:
            dict: Complete grid data point
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Determine if this should be an anomaly
        is_anomaly = force_anomaly or (random.random() < self.ANOMALY_PROBABILITY)
        
        # Generate measurements
        voltage = self.generate_voltage(anomaly=is_anomaly)
        current = self.generate_current(anomaly=is_anomaly)
        frequency = self.generate_frequency(anomaly=is_anomaly)
        
        # Calculate power
        power_factor = random.uniform(0.90, 0.98) if not is_anomaly else random.uniform(0.70, 0.85)
        active_power, reactive_power = self.generate_power(voltage, current, power_factor)
        
        return {
            "timestamp": timestamp,
            "voltage": voltage,
            "current": current,
            "frequency": {"frequency_value": frequency},
            "active_power": active_power,
            "reactive_power": reactive_power
        }
    
    def generate_time_series(self, num_points: int = 100, 
                           interval_seconds: int = 3,
                           start_time: datetime = None) -> List[Dict]:
        """
        Generate time-series grid data.
        
        Args:
            num_points: Number of data points to generate
            interval_seconds: Time interval between points
            start_time: Starting timestamp (default: now)
            
        Returns:
            list: List of grid data points
        """
        if start_time is None:
            start_time = datetime.now()
        
        data_points = []
        current_time = start_time
        
        for i in range(num_points):
            data_point = self.generate_single_datapoint(timestamp=current_time)
            data_points.append(data_point)
            current_time += timedelta(seconds=interval_seconds)
        
        return data_points
    
    def generate_scenario_data(self, scenario: str = "normal") -> List[Dict]:
        """
        Generate data for specific test scenarios.
        
        Scenarios:
        - "normal": Stable grid operation
        - "voltage_sag": Voltage drop event
        - "overcurrent": Overload condition
        - "frequency_drift": Frequency instability
        - "mixed": Mixed anomalies
        
        Args:
            scenario: Scenario name
            
        Returns:
            list: List of grid data points for scenario
        """
        num_points = 50
        data_points = []
        current_time = datetime.now()
        
        for i in range(num_points):
            force_anomaly = False
            
            if scenario == "normal":
                force_anomaly = False
            elif scenario == "voltage_sag" and 20 <= i <= 30:
                force_anomaly = True
            elif scenario == "overcurrent" and 15 <= i <= 35:
                force_anomaly = True
            elif scenario == "frequency_drift" and 10 <= i <= 40:
                force_anomaly = True
            elif scenario == "mixed" and i % 5 == 0:
                force_anomaly = True
            
            data_point = self.generate_single_datapoint(
                timestamp=current_time,
                force_anomaly=force_anomaly
            )
            data_points.append(data_point)
            current_time += timedelta(seconds=3)
        
        return data_points


# Singleton instance
grid_generator = GridDataGenerator()