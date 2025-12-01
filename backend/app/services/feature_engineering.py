"""
Feature Engineering Module
Extracts ML features from raw grid data for all models
"""

import numpy as np
from typing import Dict, List, Any
from collections import deque
import math


class FeatureEngineer:
    """
    Extracts features from raw grid data for ML model inference.
    Maintains sliding window history for time-series features.
    """

    def __init__(self, window_size=100):
        """
        Initialize feature engineer with history buffer

        Args:
            window_size: Number of data points to keep in history
        """
        self.window_size = window_size

        # Sliding window history for time-series features
        self.history = {
            'voltage_avg': deque(maxlen=window_size),
            'current_avg': deque(maxlen=window_size),
            'frequency': deque(maxlen=window_size),
            'active_power': deque(maxlen=window_size),
            'reactive_power': deque(maxlen=window_size),
            'power_factor': deque(maxlen=window_size)
        }

    def extract_all_features(self, data_point) -> Dict[str, Any]:
        """
        Extract all features needed for all ML models

        Args:
            data_point: Database record (DateTimeTable with relationships)

        Returns:
            Dictionary of features organized by category
        """
        # Update history first
        self._update_history(data_point)

        # Extract features for different categories
        features = {
            'voltage': self._extract_voltage_features(data_point),
            'current': self._extract_current_features(data_point),
            'frequency': self._extract_frequency_features(data_point),
            'power': self._extract_power_features(data_point),
            'balance': self._extract_balance_features(data_point),
            'quality': self._extract_quality_features(data_point),
            'time_series': self._extract_time_series_features()
        }

        return features

    def _update_history(self, data_point):
        """Update sliding window history with new data point"""
        # Extract values safely
        v_avg = data_point.voltage[0].average if data_point.voltage else 230.0
        i_avg = data_point.current[0].average if data_point.current else 0.0
        f = data_point.frequency[0].frequency_value if data_point.frequency else 50.0
        p = data_point.active_power[0].total if data_point.active_power else 0.0
        q = data_point.reactive_power[0].total if data_point.reactive_power else 0.0

        # Calculate power factor
        s = math.sqrt(p**2 + q**2) if (p != 0 or q != 0) else 0.0001
        pf = p / s if s > 0 else 0.0

        # Append to history
        self.history['voltage_avg'].append(v_avg)
        self.history['current_avg'].append(i_avg)
        self.history['frequency'].append(f)
        self.history['active_power'].append(p)
        self.history['reactive_power'].append(q)
        self.history['power_factor'].append(pf)

    def _extract_voltage_features(self, data_point) -> Dict:
        """Extract voltage-related features"""
        voltage = data_point.voltage[0] if data_point.voltage else None

        if not voltage:
            return self._default_voltage_features()

        v_avg = voltage.average
        v_a = voltage.phaseA
        v_b = voltage.phaseB
        v_c = voltage.phaseC

        # Calculate variance from history
        v_variance = np.var(list(self.history['voltage_avg'])[-50:]) if len(self.history['voltage_avg']) >= 10 else 0

        # Calculate imbalance
        v_imbalance = self._calculate_imbalance(v_a, v_b, v_c)

        # Deviation from nominal
        v_deviation = abs(v_avg - 230) / 230

        # Rate of change
        v_roc = self._calculate_roc('voltage_avg')

        return {
            'v_avg': v_avg,
            'v_phase_a': v_a,
            'v_phase_b': v_b,
            'v_phase_c': v_c,
            'v_variance': v_variance,
            'v_imbalance_pct': v_imbalance,
            'v_deviation_pct': v_deviation * 100,
            'v_rate_of_change': v_roc,
            'v_max_phase': max(v_a, v_b, v_c),
            'v_min_phase': min(v_a, v_b, v_c),
            'v_range': max(v_a, v_b, v_c) - min(v_a, v_b, v_c)
        }

    def _extract_current_features(self, data_point) -> Dict:
        """Extract current-related features"""
        current = data_point.current[0] if data_point.current else None

        if not current:
            return self._default_current_features()

        i_avg = current.average
        i_a = current.phaseA
        i_b = current.phaseB
        i_c = current.phaseC

        # Calculate variance
        i_variance = np.var(list(self.history['current_avg'])[-50:]) if len(self.history['current_avg']) >= 10 else 0

        # Calculate imbalance
        i_imbalance = self._calculate_imbalance(i_a, i_b, i_c)

        # Rate of change
        i_roc = self._calculate_roc('current_avg')

        # Detect spikes (current > 2x average in history)
        avg_historical = np.mean(list(self.history['current_avg'])) if len(self.history['current_avg']) > 10 else i_avg
        i_spike = 1 if i_avg > 2 * avg_historical else 0

        return {
            'i_avg': i_avg,
            'i_phase_a': i_a,
            'i_phase_b': i_b,
            'i_phase_c': i_c,
            'i_variance': i_variance,
            'i_imbalance_pct': i_imbalance,
            'i_rate_of_change': i_roc,
            'i_spike_detected': i_spike,
            'i_max_phase': max(i_a, i_b, i_c),
            'i_min_phase': min(i_a, i_b, i_c),
            'i_range': max(i_a, i_b, i_c) - min(i_a, i_b, i_c)
        }

    def _extract_frequency_features(self, data_point) -> Dict:
        """Extract frequency-related features"""
        frequency = data_point.frequency[0] if data_point.frequency else None

        if not frequency:
            return {'f_value': 50.0, 'f_deviation': 0.0, 'f_roc': 0.0, 'f_history': [50.0] * 10}

        f = frequency.frequency_value
        f_deviation = abs(f - 50.0)
        f_roc = self._calculate_roc('frequency')

        # Get recent history for LSTM
        f_history = list(self.history['frequency'])[-100:] if len(self.history['frequency']) > 0 else [50.0]

        return {
            'f_value': f,
            'f_deviation': f_deviation,
            'f_rate_of_change': f_roc,
            'f_history': f_history,
            'f_above_nominal': 1 if f > 50.0 else 0,
            'f_below_nominal': 1 if f < 50.0 else 0
        }

    def _extract_power_features(self, data_point) -> Dict:
        """Extract power-related features"""
        active = data_point.active_power[0] if data_point.active_power else None
        reactive = data_point.reactive_power[0] if data_point.reactive_power else None

        if not active or not reactive:
            return self._default_power_features()

        p_total = active.total
        q_total = reactive.total

        # Calculate apparent power
        s_total = math.sqrt(p_total**2 + q_total**2) if (p_total != 0 or q_total != 0) else 0.0001

        # Power factor
        pf = p_total / s_total if s_total > 0 else 0.0

        # Power imbalance
        p_imbalance = self._calculate_imbalance(active.phaseA, active.phaseB, active.phaseC)
        q_imbalance = self._calculate_imbalance(reactive.phaseA, reactive.phaseB, reactive.phaseC)

        return {
            'p_total': p_total,
            'q_total': q_total,
            's_total': s_total,
            'power_factor': pf,
            'p_phase_a': active.phaseA,
            'p_phase_b': active.phaseB,
            'p_phase_c': active.phaseC,
            'q_phase_a': reactive.phaseA,
            'q_phase_b': reactive.phaseB,
            'q_phase_c': reactive.phaseC,
            'p_imbalance_pct': p_imbalance,
            'q_imbalance_pct': q_imbalance,
            'pf_lagging': 1 if q_total > 0 else 0,
            'pf_leading': 1 if q_total < 0 else 0
        }

    def _extract_balance_features(self, data_point) -> Dict:
        """Extract 3-phase balance features"""
        voltage = data_point.voltage[0] if data_point.voltage else None
        current = data_point.current[0] if data_point.current else None
        power = data_point.active_power[0] if data_point.active_power else None

        if not all([voltage, current, power]):
            return {'overall_balance_score': 100.0}

        v_imb = self._calculate_imbalance(voltage.phaseA, voltage.phaseB, voltage.phaseC)
        i_imb = self._calculate_imbalance(current.phaseA, current.phaseB, current.phaseC)
        p_imb = self._calculate_imbalance(power.phaseA, power.phaseB, power.phaseC)

        # Overall balance score (0-100, higher is better)
        balance_score = max(0, 100 - (v_imb + i_imb + p_imb))

        return {
            'v_imbalance': v_imb,
            'i_imbalance': i_imb,
            'p_imbalance': p_imb,
            'overall_balance_score': balance_score
        }

    def _extract_quality_features(self, data_point) -> Dict:
        """Extract power quality features"""
        v_features = self._extract_voltage_features(data_point)
        f_features = self._extract_frequency_features(data_point)
        p_features = self._extract_power_features(data_point)

        # Estimate THD from variance (simplified)
        v_thd_est = min(v_features.get('v_variance', 0) * 10, 20.0)  # Cap at 20%

        # Power Quality Index components
        v_quality = max(0, 100 - v_features.get('v_deviation_pct', 0) * 10)
        f_quality = max(0, 100 - f_features.get('f_deviation', 0) * 200)
        pf_quality = p_features.get('power_factor', 0.9) * 100

        # Overall PQI
        pqi = (v_quality * 0.4 + f_quality * 0.3 + pf_quality * 0.3)

        return {
            'v_thd_estimated': v_thd_est,
            'v_variance': v_features.get('v_variance', 0),  # Add for harmonic analyzer
            'power_factor': p_features.get('power_factor', 0.9),  # Add for harmonic analyzer
            'v_quality_score': v_quality,
            'f_quality_score': f_quality,
            'pf_quality_score': pf_quality,
            'power_quality_index': pqi
        }

    def _extract_time_series_features(self) -> Dict:
        """Extract time-series statistical features"""
        features = {}

        for param in ['voltage_avg', 'current_avg', 'frequency', 'active_power', 'reactive_power', 'power_factor']:
            if len(self.history[param]) >= 10:
                values = list(self.history[param])
                features[f'{param}_mean'] = np.mean(values)
                features[f'{param}_std'] = np.std(values)
                features[f'{param}_min'] = np.min(values)
                features[f'{param}_max'] = np.max(values)
                features[f'{param}_trend'] = self._calculate_trend(values)
            else:
                # Provide defaults when not enough history
                features[f'{param}_mean'] = 0
                features[f'{param}_std'] = 0
                features[f'{param}_min'] = 0
                features[f'{param}_max'] = 0
                features[f'{param}_trend'] = 0

        return features

    def _calculate_imbalance(self, a: float, b: float, c: float) -> float:
        """
        Calculate 3-phase imbalance percentage

        Args:
            a, b, c: Phase values

        Returns:
            Imbalance percentage
        """
        avg = (a + b + c) / 3
        if avg == 0:
            return 0.0

        max_dev = max(abs(a - avg), abs(b - avg), abs(c - avg))
        return (max_dev / avg) * 100

    def _calculate_roc(self, param: str) -> float:
        """
        Calculate rate of change for a parameter

        Args:
            param: Parameter name in history

        Returns:
            Rate of change
        """
        if len(self.history[param]) < 2:
            return 0.0

        values = list(self.history[param])
        return values[-1] - values[-2]

    def _calculate_trend(self, values: List[float]) -> float:
        """
        Calculate trend (linear regression slope)

        Args:
            values: List of values

        Returns:
            Trend slope
        """
        if len(values) < 2:
            return 0.0

        x = np.arange(len(values))
        y = np.array(values)

        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope

    # Default feature dictionaries for missing data
    def _default_voltage_features(self) -> Dict:
        return {
            'v_avg': 230.0, 'v_phase_a': 230.0, 'v_phase_b': 230.0, 'v_phase_c': 230.0,
            'v_variance': 0.0, 'v_imbalance_pct': 0.0, 'v_deviation_pct': 0.0,
            'v_rate_of_change': 0.0, 'v_max_phase': 230.0, 'v_min_phase': 230.0, 'v_range': 0.0
        }

    def _default_current_features(self) -> Dict:
        return {
            'i_avg': 0.0, 'i_phase_a': 0.0, 'i_phase_b': 0.0, 'i_phase_c': 0.0,
            'i_variance': 0.0, 'i_imbalance_pct': 0.0, 'i_rate_of_change': 0.0,
            'i_spike_detected': 0, 'i_max_phase': 0.0, 'i_min_phase': 0.0, 'i_range': 0.0
        }

    def _default_power_features(self) -> Dict:
        return {
            'p_total': 0.0, 'q_total': 0.0, 's_total': 0.0, 'power_factor': 0.9,
            'p_phase_a': 0.0, 'p_phase_b': 0.0, 'p_phase_c': 0.0,
            'q_phase_a': 0.0, 'q_phase_b': 0.0, 'q_phase_c': 0.0,
            'p_imbalance_pct': 0.0, 'q_imbalance_pct': 0.0,
            'pf_lagging': 0, 'pf_leading': 0
        }
