"""
REMHART Digital Twin - Advanced Simulation Generator
====================================================
Generates grid data based on user-defined parameters and scenarios.

Author: REMHART Team
Date: 2025
"""

import random
from datetime import datetime, timedelta
from typing import Dict, List
import math


class SimulationGenerator:
    """
    Advanced grid data generator for simulation scenarios.
    Supports custom parameters and realistic scenario patterns.
    """
    
    def __init__(self):
        self.NOMINAL_VOLTAGE = 230.0
        self.NOMINAL_FREQUENCY = 50.0
        self.NOMINAL_CURRENT = 15.0
    
    def generate_voltage_value(self, point_index: int, total_points: int, params: Dict) -> float:
        """
        Generate voltage based on pattern and parameters.
        
        Args:
            point_index: Current point number (0-based)
            total_points: Total number of points
            params: Simulation parameters
            
        Returns:
            Voltage value in Volts
        """
        v_min = params['voltage_min']
        v_max = params['voltage_max']
        pattern = params['voltage_pattern']
        
        if pattern == "stable":
            # Stable with small random variation
            base = (v_min + v_max) / 2
            return base + random.uniform(-2, 2)
        
        elif pattern == "fluctuating":
            # Sinusoidal fluctuation
            progress = point_index / total_points
            amplitude = (v_max - v_min) / 2
            center = (v_min + v_max) / 2
            return center + amplitude * math.sin(progress * 4 * math.pi)
        
        elif pattern == "sag":
            # Voltage drop in middle
            if 0.3 <= point_index / total_points <= 0.7:
                return v_min + random.uniform(-5, 5)
            else:
                return (v_min + v_max) / 2 + random.uniform(-2, 2)
        
        elif pattern == "swell":
            # Voltage increase in middle
            if 0.3 <= point_index / total_points <= 0.7:
                return v_max + random.uniform(-5, 5)
            else:
                return (v_min + v_max) / 2 + random.uniform(-2, 2)
        
        return (v_min + v_max) / 2
    
    def generate_current_value(self, point_index: int, total_points: int, params: Dict) -> float:
        """Generate current based on pattern and parameters"""
        i_min = params['current_min']
        i_max = params['current_max']
        pattern = params['current_pattern']
        load_profile = params.get('load_profile', 'constant')
        
        # Apply load profile first
        if load_profile == "ramping_up":
            progress = point_index / total_points
            base = i_min + (i_max - i_min) * progress
        elif load_profile == "ramping_down":
            progress = point_index / total_points
            base = i_max - (i_max - i_min) * progress
        elif load_profile == "random":
            base = random.uniform(i_min, i_max)
        else:  # constant
            base = (i_min + i_max) / 2
        
        # Apply pattern on top
        if pattern == "stable":
            return base + random.uniform(-1, 1)
        
        elif pattern == "increasing":
            progress = point_index / total_points
            return base + (i_max - base) * progress * 0.3
        
        elif pattern == "decreasing":
            progress = point_index / total_points
            return base - (base - i_min) * progress * 0.3
        
        elif pattern == "spike":
            # Random spikes
            if random.random() < 0.1:  # 10% chance of spike
                return i_max + random.uniform(5, 15)
            return base
        
        return base
    
    def generate_frequency_value(self, point_index: int, total_points: int, params: Dict) -> float:
        """Generate frequency based on pattern and parameters"""
        f_min = params['frequency_min']
        f_max = params['frequency_max']
        pattern = params['frequency_pattern']
        
        if pattern == "stable":
            return (f_min + f_max) / 2 + random.uniform(-0.02, 0.02)
        
        elif pattern == "drift":
            # Gradual drift from min to max
            progress = point_index / total_points
            return f_min + (f_max - f_min) * progress + random.uniform(-0.05, 0.05)
        
        return 50.0
    
    def apply_fault_injection(self, point_index: int, data: Dict, params: Dict) -> Dict:
        """
        Apply fault if configured and at right point.
        
        Args:
            point_index: Current point index
            data: Generated data point
            params: Simulation parameters
            
        Returns:
            Modified data point
        """
        if not params.get('inject_fault', False):
            return data
        
        fault_start = params.get('fault_start_point', 0)
        fault_duration = params.get('fault_duration', 10)
        fault_type = params.get('fault_type')
        
        # Check if we're in fault period
        if fault_start <= point_index < (fault_start + fault_duration):
            if fault_type == "voltage_sag":
                # Severe voltage drop
                data['voltage']['phaseA'] *= 0.7
                data['voltage']['phaseB'] *= 0.7
                data['voltage']['phaseC'] *= 0.7
                data['voltage']['average'] *= 0.7
            
            elif fault_type == "overcurrent":
                # Severe current spike
                data['current']['phaseA'] *= 2.5
                data['current']['phaseB'] *= 2.5
                data['current']['phaseC'] *= 2.5
                data['current']['average'] *= 2.5
            
            elif fault_type == "frequency_drift":
                # Frequency deviation
                data['frequency']['frequency_value'] = random.uniform(48.5, 49.0)
        
        return data
    
    def calculate_power(self, voltage: Dict, current: Dict, power_factor: float) -> tuple:
        """Calculate active and reactive power"""
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
    
    def generate_datapoint(self, point_index: int, total_points: int, 
                          params: Dict, start_time: datetime) -> Dict:
        """
        Generate single data point based on parameters.
        
        Args:
            point_index: Current point (0-based)
            total_points: Total points to generate
            params: Simulation parameters
            start_time: Starting timestamp
            
        Returns:
            Complete grid data point
        """
        # Calculate timestamp
        interval = params.get('interval', 1)
        timestamp = start_time + timedelta(seconds=point_index * interval)
        
        # Generate base values
        voltage_avg = self.generate_voltage_value(point_index, total_points, params)
        current_avg = self.generate_current_value(point_index, total_points, params)
        frequency = self.generate_frequency_value(point_index, total_points, params)
        
        # Generate 3-phase values with slight variations
        voltage = {
            "phaseA": round(voltage_avg + random.uniform(-2, 2), 2),
            "phaseB": round(voltage_avg + random.uniform(-2, 2), 2),
            "phaseC": round(voltage_avg + random.uniform(-2, 2), 2),
            "average": round(voltage_avg, 2)
        }
        
        current = {
            "phaseA": round(current_avg + random.uniform(-1, 1), 2),
            "phaseB": round(current_avg + random.uniform(-1, 1), 2),
            "phaseC": round(current_avg + random.uniform(-1, 1), 2),
            "average": round(current_avg, 2)
        }
        
        # Calculate power
        power_factor = params.get('power_factor', 0.95)
        active_power, reactive_power = self.calculate_power(voltage, current, power_factor)
        
        # Build data point
        data_point = {
            "timestamp": timestamp,
            "voltage": voltage,
            "current": current,
            "frequency": {"frequency_value": round(frequency, 2)},
            "active_power": active_power,
            "reactive_power": reactive_power
        }
        
        # Apply fault injection if configured
        data_point = self.apply_fault_injection(point_index, data_point, params)
        
        return data_point
    
    def generate_scenario_from_template(self, scenario_type: str, num_points: int = 100) -> Dict:
        """
        Generate parameters based on scenario template.
        
        Args:
            scenario_type: Template name
            num_points: Number of points to generate
            
        Returns:
            Parameters dictionary
        """
        templates = {
            "normal": {
                "voltage_min": 225,
                "voltage_max": 235,
                "voltage_pattern": "stable",
                "current_min": 12,
                "current_max": 18,
                "current_pattern": "stable",
                "frequency_min": 49.9,
                "frequency_max": 50.1,
                "frequency_pattern": "stable",
                "power_factor": 0.95,
                "renewable_percentage": 30,
                "inject_fault": False,
                "load_profile": "constant"
            },
            "peak_load": {
                "voltage_min": 215,
                "voltage_max": 225,
                "voltage_pattern": "sag",
                "current_min": 20,
                "current_max": 35,
                "current_pattern": "increasing",
                "frequency_min": 49.8,
                "frequency_max": 50.0,
                "frequency_pattern": "drift",
                "power_factor": 0.85,
                "renewable_percentage": 20,
                "inject_fault": False,
                "load_profile": "ramping_up"
            },
            "voltage_sag": {
                "voltage_min": 200,
                "voltage_max": 220,
                "voltage_pattern": "sag",
                "current_min": 15,
                "current_max": 20,
                "current_pattern": "stable",
                "frequency_min": 49.9,
                "frequency_max": 50.1,
                "frequency_pattern": "stable",
                "power_factor": 0.90,
                "renewable_percentage": 30,
                "inject_fault": True,
                "fault_type": "voltage_sag",
                "fault_start_point": int(num_points * 0.3),
                "fault_duration": int(num_points * 0.2),
                "load_profile": "constant"
            },
            "overcurrent": {
                "voltage_min": 225,
                "voltage_max": 235,
                "voltage_pattern": "stable",
                "current_min": 25,
                "current_max": 40,
                "current_pattern": "spike",
                "frequency_min": 49.9,
                "frequency_max": 50.1,
                "frequency_pattern": "stable",
                "power_factor": 0.80,
                "renewable_percentage": 25,
                "inject_fault": True,
                "fault_type": "overcurrent",
                "fault_start_point": int(num_points * 0.4),
                "fault_duration": int(num_points * 0.1),
                "load_profile": "constant"
            },
            "frequency_drift": {
                "voltage_min": 225,
                "voltage_max": 235,
                "voltage_pattern": "stable",
                "current_min": 15,
                "current_max": 20,
                "current_pattern": "stable",
                "frequency_min": 49.5,
                "frequency_max": 50.5,
                "frequency_pattern": "drift",
                "power_factor": 0.92,
                "renewable_percentage": 30,
                "inject_fault": True,
                "fault_type": "frequency_drift",
                "fault_start_point": int(num_points * 0.2),
                "fault_duration": int(num_points * 0.5),
                "load_profile": "constant"
            },
            "renewable_intermittency": {
                "voltage_min": 220,
                "voltage_max": 240,
                "voltage_pattern": "fluctuating",
                "current_min": 10,
                "current_max": 25,
                "current_pattern": "spike",
                "frequency_min": 49.8,
                "frequency_max": 50.2,
                "frequency_pattern": "drift",
                "power_factor": 0.88,
                "renewable_percentage": 60,
                "inject_fault": False,
                "load_profile": "random"
            },
            "fault": {
                "voltage_min": 200,
                "voltage_max": 240,
                "voltage_pattern": "sag",
                "current_min": 20,
                "current_max": 40,
                "current_pattern": "spike",
                "frequency_min": 49.0,
                "frequency_max": 51.0,
                "frequency_pattern": "drift",
                "power_factor": 0.75,
                "renewable_percentage": 30,
                "inject_fault": True,
                "fault_type": "voltage_sag",
                "fault_start_point": int(num_points * 0.5),
                "fault_duration": int(num_points * 0.15),
                "load_profile": "constant"
            }
        }
        
        params = templates.get(scenario_type, templates["normal"])
        params["num_points"] = num_points
        params["interval"] = 1
        
        return params
    
    def generate_simulation(self, params: Dict, start_time: datetime = None) -> List[Dict]:
        """
        Generate complete simulation dataset.
        
        Args:
            params: Simulation parameters
            start_time: Start timestamp (default: now)
            
        Returns:
            List of data points
        """
        if start_time is None:
            start_time = datetime.now()
        
        num_points = params['num_points']
        data_points = []
        
        for i in range(num_points):
            data_point = self.generate_datapoint(i, num_points, params, start_time)
            data_points.append(data_point)
        
        return data_points


# Global instance
simulation_generator = SimulationGenerator()