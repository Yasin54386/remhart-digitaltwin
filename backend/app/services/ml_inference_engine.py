"""
ML Inference Engine
Processes every new grid data point through all 16 ML models
"""

import logging
from typing import Dict, Any
from .feature_engineering import FeatureEngineer
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLInferenceEngine:
    """
    Main inference engine that coordinates feature extraction and model predictions.
    Runs all ML models on every new data point for real-time insights.
    """

    def __init__(self):
        """Initialize the inference engine with feature engineer and model manager"""
        self.feature_engineer = FeatureEngineer(window_size=100)
        self.model_manager = model_manager
        logger.info("ML Inference Engine initialized")

    def process_data_point(self, data_point) -> Dict[str, Any]:
        """
        Process a single data point through entire ML pipeline

        Args:
            data_point: Database record (DateTimeTable with relationships)

        Returns:
            Dictionary containing predictions from all 16 models organized by module
        """
        try:
            # Step 1: Extract features
            features = self.feature_engineer.extract_all_features(data_point)

            # Flatten features for easier model access
            flat_features = self._flatten_features(features)

            # Step 2: Run all model predictions
            predictions = {
                'real_time_monitoring': self._run_monitoring_models(features, flat_features),
                'predictive_maintenance': self._run_maintenance_models(features, flat_features),
                'energy_flow': self._run_energy_models(features, flat_features),
                'decision_making': self._run_decision_models(features, flat_features)
            }

            # Step 3: Add metadata
            predictions['metadata'] = {
                'timestamp': data_point.timestamp,
                'is_simulation': data_point.is_simulation,
                'data_quality': self._assess_data_quality(data_point)
            }

            return predictions

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Error processing data point: {e}\n{error_trace}")
            return self._get_error_response(str(e))

    def _flatten_features(self, features: Dict) -> Dict:
        """Flatten nested feature dictionary for easier access"""
        flat = {}
        for category, feature_dict in features.items():
            if isinstance(feature_dict, dict):
                for key, value in feature_dict.items():
                    flat[key] = value
        return flat

    def _run_monitoring_models(self, features: Dict, flat: Dict) -> Dict:
        """Run all Real-time Monitoring models"""
        voltage_features = features['voltage']
        frequency_features = features['frequency']
        quality_features = features['quality']
        balance_features = features['balance']

        return {
            'voltage_anomaly_detection': self.model_manager.predict_voltage_anomaly(voltage_features),
            'harmonic_analysis': self.model_manager.analyze_harmonics(quality_features),
            'frequency_stability': self.model_manager.predict_frequency_stability(frequency_features),
            'phase_imbalance_classification': self.model_manager.classify_phase_imbalance(balance_features)
        }

    def _run_maintenance_models(self, features: Dict, flat: Dict) -> Dict:
        """Run all Predictive Maintenance models"""
        return {
            'equipment_failure_prediction': self.model_manager.predict_equipment_failure(flat),
            'overload_risk_classification': self.model_manager.classify_overload_risk(flat),
            'power_quality_index': self.model_manager.calculate_power_quality_index(flat),
            'voltage_sag_prediction': self.model_manager.predict_voltage_sag(flat)
        }

    def _run_energy_models(self, features: Dict, flat: Dict) -> Dict:
        """Run all Energy Flow models"""
        return {
            'load_forecasting': self.model_manager.forecast_load(flat),
            'energy_loss_estimation': self.model_manager.estimate_energy_loss(flat),
            'power_flow_optimization': self.model_manager.optimize_power_flow(flat),
            'demand_response_assessment': self.model_manager.assess_demand_response(flat)
        }

    def _run_decision_models(self, features: Dict, flat: Dict) -> Dict:
        """Run all Decision Making models"""
        return {
            'reactive_power_compensation': self.model_manager.optimize_reactive_compensation(flat),
            'load_balancing_optimization': self.model_manager.optimize_load_balancing(flat),
            'grid_stability_scoring': self.model_manager.score_grid_stability(flat),
            'optimal_dispatch_advisory': self.model_manager.advise_optimal_dispatch(flat)
        }

    def _assess_data_quality(self, data_point) -> str:
        """Assess quality of incoming data"""
        # Check if essential data exists
        has_voltage = bool(data_point.voltage)
        has_current = bool(data_point.current)
        has_frequency = bool(data_point.frequency)
        has_power = bool(data_point.active_power)

        complete_count = sum([has_voltage, has_current, has_frequency, has_power])

        if complete_count == 4:
            return 'Excellent'
        elif complete_count == 3:
            return 'Good'
        elif complete_count == 2:
            return 'Fair'
        return 'Poor'

    def _get_error_response(self, error_msg: str) -> Dict:
        """Return error response structure"""
        return {
            'error': True,
            'message': error_msg,
            'real_time_monitoring': {},
            'predictive_maintenance': {},
            'energy_flow': {},
            'decision_making': {},
            'metadata': {
                'timestamp': None,
                'is_simulation': False,
                'data_quality': 'Error'
            }
        }

    def get_historical_predictions(self, data_points: list) -> list:
        """
        Process multiple historical data points

        Args:
            data_points: List of database records

        Returns:
            List of prediction dictionaries
        """
        predictions = []
        for data_point in data_points:
            pred = self.process_data_point(data_point)
            predictions.append(pred)

        return predictions


# Global singleton instance
ml_inference_engine = MLInferenceEngine()
