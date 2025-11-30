"""
Training Data Generator
Generates synthetic grid data for training all 16 ML models
Uses physics-based models to create realistic scenarios
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import math


class GridDataGenerator:
    """
    Generates synthetic electrical grid data based on physical models.
    Creates various scenarios for training ML models.
    """

    def __init__(self, seed=42):
        """Initialize with random seed for reproducibility"""
        np.random.seed(seed)
        self.nominal_voltage = 230.0  # V (Line-to-Neutral)
        self.nominal_frequency = 50.0  # Hz
        self.base_current = 50.0  # A
        self.base_power_factor = 0.92

    def generate_normal_operation(self, num_samples=1000) -> pd.DataFrame:
        """
        Generate normal grid operation data

        Returns:
            DataFrame with voltage, current, frequency, active and reactive power
        """
        data = []

        for i in range(num_samples):
            # Normal operation with small variations
            v_a = self.nominal_voltage + np.random.normal(0, 2)
            v_b = self.nominal_voltage + np.random.normal(0, 2)
            v_c = self.nominal_voltage + np.random.normal(0, 2)

            # Currents with slight imbalance
            i_base = self.base_current + np.random.normal(0, 2)
            i_a = i_base * (1 + np.random.normal(0, 0.02))
            i_b = i_base * (1 + np.random.normal(0, 0.02))
            i_c = i_base * (1 + np.random.normal(0, 0.02))

            # Frequency stability
            freq = self.nominal_frequency + np.random.normal(0, 0.05)

            # Power calculations
            pf = self.base_power_factor + np.random.normal(0, 0.02)
            pf = max(0.1, min(1.0, pf))  # Clamp to valid range [0.1, 1.0]
            p_a = v_a * i_a * pf / 1000  # kW
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            # Reactive power
            theta = math.acos(pf)
            q_a = v_a * i_a * math.sin(theta) / 1000  # kVAR
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': 'normal'
            })

        return pd.DataFrame(data)

    def generate_voltage_anomalies(self, num_samples=500) -> pd.DataFrame:
        """Generate voltage anomaly scenarios (sags, swells, imbalances)"""
        data = []

        for i in range(num_samples):
            anomaly_type = np.random.choice(['sag', 'swell', 'imbalance'])

            if anomaly_type == 'sag':
                # Voltage sag (0.7-0.9 pu)
                factor = np.random.uniform(0.7, 0.9)
                v_a = self.nominal_voltage * factor
                v_b = self.nominal_voltage * factor * (1 + np.random.normal(0, 0.05))
                v_c = self.nominal_voltage * factor * (1 + np.random.normal(0, 0.05))

            elif anomaly_type == 'swell':
                # Voltage swell (1.1-1.2 pu)
                factor = np.random.uniform(1.1, 1.2)
                v_a = self.nominal_voltage * factor
                v_b = self.nominal_voltage * factor * (1 + np.random.normal(0, 0.05))
                v_c = self.nominal_voltage * factor * (1 + np.random.normal(0, 0.05))

            else:  # imbalance
                # Severe phase imbalance
                v_a = self.nominal_voltage * (1 + np.random.uniform(-0.15, 0.15))
                v_b = self.nominal_voltage * (1 + np.random.uniform(-0.15, 0.15))
                v_c = self.nominal_voltage * (1 + np.random.uniform(-0.15, 0.15))

            # Current adjusts to maintain power (simplified)
            i_base = self.base_current
            i_a = i_base + np.random.normal(0, 5)
            i_b = i_base + np.random.normal(0, 5)
            i_c = i_base + np.random.normal(0, 5)

            freq = self.nominal_frequency + np.random.normal(0, 0.1)
            pf = self.base_power_factor + np.random.normal(0, 0.05)
            pf = max(0.1, min(1.0, pf))  # Clamp to valid range [0.1, 1.0]

            # Power calculations
            p_a = v_a * i_a * pf / 1000
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            theta = math.acos(pf)
            q_a = v_a * i_a * math.sin(theta) / 1000
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': f'voltage_{anomaly_type}'
            })

        return pd.DataFrame(data)

    def generate_frequency_deviations(self, num_samples=500) -> pd.DataFrame:
        """Generate frequency deviation scenarios"""
        data = []

        for i in range(num_samples):
            # Frequency deviations (under-frequency or over-frequency)
            deviation_type = np.random.choice(['under', 'over'])

            if deviation_type == 'under':
                freq = np.random.uniform(48.5, 49.8)
            else:
                freq = np.random.uniform(50.2, 51.5)

            # Voltages remain relatively stable
            v_a = self.nominal_voltage + np.random.normal(0, 3)
            v_b = self.nominal_voltage + np.random.normal(0, 3)
            v_c = self.nominal_voltage + np.random.normal(0, 3)

            # Currents
            i_base = self.base_current + np.random.normal(0, 3)
            i_a = i_base * (1 + np.random.normal(0, 0.03))
            i_b = i_base * (1 + np.random.normal(0, 0.03))
            i_c = i_base * (1 + np.random.normal(0, 0.03))

            pf = self.base_power_factor + np.random.normal(0, 0.03)
            pf = max(0.1, min(1.0, pf))  # Clamp to valid range [0.1, 1.0]

            p_a = v_a * i_a * pf / 1000
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            theta = math.acos(pf)
            q_a = v_a * i_a * math.sin(theta) / 1000
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': f'frequency_{deviation_type}'
            })

        return pd.DataFrame(data)

    def generate_overload_scenarios(self, num_samples=500) -> pd.DataFrame:
        """Generate equipment overload scenarios"""
        data = []

        for i in range(num_samples):
            severity = np.random.choice(['moderate', 'severe'])

            # Normal voltages
            v_a = self.nominal_voltage + np.random.normal(0, 2)
            v_b = self.nominal_voltage + np.random.normal(0, 2)
            v_c = self.nominal_voltage + np.random.normal(0, 2)

            # High currents (overload)
            if severity == 'moderate':
                i_base = self.base_current * np.random.uniform(1.2, 1.5)
            else:
                i_base = self.base_current * np.random.uniform(1.5, 2.0)

            # One or more phases may be heavily loaded
            i_a = i_base * (1 + np.random.uniform(0, 0.2))
            i_b = i_base * (1 + np.random.uniform(-0.1, 0.1))
            i_c = i_base * (1 + np.random.uniform(-0.1, 0.1))

            freq = self.nominal_frequency + np.random.normal(0, 0.1)
            pf = self.base_power_factor + np.random.normal(0, 0.03)
            pf = max(0.1, min(1.0, pf))  # Clamp to valid range [0.1, 1.0]

            p_a = v_a * i_a * pf / 1000
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            theta = math.acos(pf)
            q_a = v_a * i_a * math.sin(theta) / 1000
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': f'overload_{severity}'
            })

        return pd.DataFrame(data)

    def generate_poor_power_quality(self, num_samples=500) -> pd.DataFrame:
        """Generate poor power quality scenarios (low PF, harmonics)"""
        data = []

        for i in range(num_samples):
            # Poor power factor
            pf = np.random.uniform(0.65, 0.85)

            # Voltages with harmonics (simulated by higher variance)
            v_a = self.nominal_voltage + np.random.normal(0, 5)
            v_b = self.nominal_voltage + np.random.normal(0, 5)
            v_c = self.nominal_voltage + np.random.normal(0, 5)

            # Currents
            i_base = self.base_current + np.random.normal(0, 5)
            i_a = i_base * (1 + np.random.normal(0, 0.05))
            i_b = i_base * (1 + np.random.normal(0, 0.05))
            i_c = i_base * (1 + np.random.normal(0, 0.05))

            freq = self.nominal_frequency + np.random.normal(0, 0.15)

            p_a = v_a * i_a * pf / 1000
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            theta = math.acos(pf)
            # Higher reactive power due to poor PF
            q_a = v_a * i_a * math.sin(theta) / 1000
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': 'poor_power_quality'
            })

        return pd.DataFrame(data)

    def generate_phase_imbalance(self, num_samples=500) -> pd.DataFrame:
        """Generate severe phase imbalance scenarios"""
        data = []

        for i in range(num_samples):
            # Severely unbalanced loads
            v_a = self.nominal_voltage + np.random.normal(0, 2)
            v_b = self.nominal_voltage + np.random.normal(0, 2)
            v_c = self.nominal_voltage + np.random.normal(0, 2)

            # One phase heavily loaded
            heavy_phase = np.random.choice(['a', 'b', 'c'])
            i_base = self.base_current

            if heavy_phase == 'a':
                i_a = i_base * np.random.uniform(1.5, 2.0)
                i_b = i_base * np.random.uniform(0.5, 0.8)
                i_c = i_base * np.random.uniform(0.5, 0.8)
            elif heavy_phase == 'b':
                i_a = i_base * np.random.uniform(0.5, 0.8)
                i_b = i_base * np.random.uniform(1.5, 2.0)
                i_c = i_base * np.random.uniform(0.5, 0.8)
            else:
                i_a = i_base * np.random.uniform(0.5, 0.8)
                i_b = i_base * np.random.uniform(0.5, 0.8)
                i_c = i_base * np.random.uniform(1.5, 2.0)

            freq = self.nominal_frequency + np.random.normal(0, 0.05)
            pf = self.base_power_factor + np.random.normal(0, 0.03)
            pf = max(0.1, min(1.0, pf))  # Clamp to valid range [0.1, 1.0]

            p_a = v_a * i_a * pf / 1000
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            theta = math.acos(pf)
            q_a = v_a * i_a * math.sin(theta) / 1000
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': 'phase_imbalance'
            })

        return pd.DataFrame(data)

    def generate_load_variation_timeseries(self, days=7, samples_per_hour=60) -> pd.DataFrame:
        """
        Generate realistic load variation over time (for LSTM training)

        Args:
            days: Number of days to simulate
            samples_per_hour: Samples per hour

        Returns:
            DataFrame with time-series data
        """
        data = []
        total_samples = days * 24 * samples_per_hour
        start_time = datetime.now()

        for i in range(total_samples):
            timestamp = start_time + timedelta(minutes=i / samples_per_hour * 60)
            hour = timestamp.hour

            # Daily load pattern
            if 0 <= hour < 6:  # Night (low load)
                load_factor = 0.5
            elif 6 <= hour < 9:  # Morning ramp
                load_factor = 0.5 + (hour - 6) * 0.15
            elif 9 <= hour < 17:  # Day (high load)
                load_factor = 0.9 + np.random.normal(0, 0.05)
            elif 17 <= hour < 22:  # Evening peak
                load_factor = 1.0 + np.random.normal(0, 0.05)
            else:  # Evening decline
                load_factor = 0.8 - (hour - 22) * 0.15

            # Add day-of-week variation
            if timestamp.weekday() >= 5:  # Weekend
                load_factor *= 0.8

            # Current scales with load
            i_base = self.base_current * load_factor
            i_a = i_base * (1 + np.random.normal(0, 0.02))
            i_b = i_base * (1 + np.random.normal(0, 0.02))
            i_c = i_base * (1 + np.random.normal(0, 0.02))

            # Voltages remain stable
            v_a = self.nominal_voltage + np.random.normal(0, 2)
            v_b = self.nominal_voltage + np.random.normal(0, 2)
            v_c = self.nominal_voltage + np.random.normal(0, 2)

            # Frequency very stable
            freq = self.nominal_frequency + np.random.normal(0, 0.02)

            pf = self.base_power_factor + np.random.normal(0, 0.01)
            pf = max(0.1, min(1.0, pf))  # Clamp to valid range [0.1, 1.0]

            p_a = v_a * i_a * pf / 1000
            p_b = v_b * i_b * pf / 1000
            p_c = v_c * i_c * pf / 1000

            theta = math.acos(pf)
            q_a = v_a * i_a * math.sin(theta) / 1000
            q_b = v_b * i_b * math.sin(theta) / 1000
            q_c = v_c * i_c * math.sin(theta) / 1000

            data.append({
                'timestamp': timestamp,
                'v_a': v_a, 'v_b': v_b, 'v_c': v_c,
                'i_a': i_a, 'i_b': i_b, 'i_c': i_c,
                'freq': freq,
                'p_a': p_a, 'p_b': p_b, 'p_c': p_c,
                'q_a': q_a, 'q_b': q_b, 'q_c': q_c,
                'scenario': 'timeseries'
            })

        return pd.DataFrame(data)

    def generate_comprehensive_dataset(self) -> pd.DataFrame:
        """
        Generate comprehensive dataset with all scenarios

        Returns:
            Combined DataFrame with labeled scenarios
        """
        print("Generating normal operation data...")
        normal = self.generate_normal_operation(2000)

        print("Generating voltage anomalies...")
        voltage = self.generate_voltage_anomalies(500)

        print("Generating frequency deviations...")
        frequency = self.generate_frequency_deviations(500)

        print("Generating overload scenarios...")
        overload = self.generate_overload_scenarios(500)

        print("Generating poor power quality...")
        poor_pq = self.generate_poor_power_quality(500)

        print("Generating phase imbalance...")
        imbalance = self.generate_phase_imbalance(500)

        print("Generating time-series data...")
        timeseries = self.generate_load_variation_timeseries(days=30)

        # Combine all datasets
        combined = pd.concat([
            normal, voltage, frequency, overload, poor_pq, imbalance, timeseries
        ], ignore_index=True)

        print(f"\nTotal samples generated: {len(combined)}")
        print(f"Scenario distribution:\n{combined['scenario'].value_counts()}")

        return combined

    def save_dataset(self, dataset: pd.DataFrame, filepath: str):
        """Save dataset to CSV"""
        dataset.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")


if __name__ == "__main__":
    # Generate and save comprehensive training dataset
    generator = GridDataGenerator(seed=42)
    dataset = generator.generate_comprehensive_dataset()

    # Save to trained models directory
    import os
    from pathlib import Path

    output_dir = Path(__file__).parent.parent / "trained"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "training_data.csv"

    generator.save_dataset(dataset, str(output_path))
    print(f"\n✓ Training data generated successfully!")
