"""
ML Inference Engine
Processes every new grid data point through all 10 authorized ML models.

Output structure:
  real_time_monitoring:
    voltage_anomaly       <- voltage_anomaly_iforest
    anomaly_detection     <- anomaly_detection_model
  predictive_maintenance:
    equipment_failure     <- equipment_failure_model
    overload_risk         <- overload_risk_model
    power_quality_index   <- epqi_score_model
    voltage_sag           <- voltage_sag_xgb_regressor
  energy_flow:
    load_forecasting      <- load_forecasting_model
    energy_health_index   <- energy_imbalance_and_health_index_model
  decision_making:
    reactive_q_forecast   <- reactive_q_forecast_model
    pf_forecast           <- pf_forecast_model
  metadata:
    timestamp, is_simulation, data_quality
"""

import logging
from typing import Dict, Any
from .feature_engineering import FeatureEngineer
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLInferenceEngine:
    """
    Coordinates feature extraction and all 10 model predictions.
    Designed for real-time single-point inference.
    """

    def __init__(self):
        self.feature_engineer = FeatureEngineer(window_size=200)
        self.model_manager    = model_manager
        logger.info("ML Inference Engine initialized (10 models)")

    def process_data_point(self, data_point) -> Dict[str, Any]:
        """
        Run the full ML pipeline on a single DB record.

        Args:
            data_point: DateTimeTable ORM object with loaded relationships

        Returns:
            Nested dict with predictions for all 10 models + metadata
        """
        try:
            # Step 1 – extract category features (also updates sliding window history)
            features = self.feature_engineer.extract_all_features(data_point)

            # Step 2 – merge training-convention feature names (volt_A, I_A, P_T …)
            flat = self._flatten(features)
            flat.update(self.feature_engineer.extract_model_input_features(data_point))

            # Step 3 – run all 10 models
            return {
                'real_time_monitoring': self._monitoring(features, flat),
                'predictive_maintenance': self._maintenance(flat),
                'energy_flow': self._energy(flat),
                'decision_making': self._decision(flat),
                'metadata': {
                    'timestamp':     data_point.timestamp.isoformat() if data_point.timestamp else None,
                    'is_simulation': data_point.is_simulation,
                    'data_quality':  self._data_quality(data_point)
                }
            }

        except Exception as e:
            import traceback
            logger.error(f"Inference error: {e}\n{traceback.format_exc()}")
            return self._error_response(str(e))

    # ----------------------------------------------------------------
    # Private runners
    # ----------------------------------------------------------------

    def _monitoring(self, features: Dict, flat: Dict) -> Dict:
        return {
            'voltage_anomaly':    self.model_manager.predict_voltage_anomaly(flat),
            'anomaly_detection':  self.model_manager.predict_anomaly_detection(flat),
        }

    def _maintenance(self, flat: Dict) -> Dict:
        return {
            'equipment_failure':  self.model_manager.predict_equipment_failure(flat),
            'overload_risk':      self.model_manager.classify_overload_risk(flat),
            'power_quality_index': self.model_manager.calculate_power_quality_index(flat),
            'voltage_sag':        self.model_manager.predict_voltage_sag(flat),
        }

    def _energy(self, flat: Dict) -> Dict:
        return {
            'load_forecasting':   self.model_manager.forecast_load(flat),
            'energy_health_index': self.model_manager.estimate_health_index(flat),
        }

    def _decision(self, flat: Dict) -> Dict:
        return {
            'reactive_q_forecast': self.model_manager.forecast_reactive_q(flat),
            'pf_forecast':         self.model_manager.forecast_power_factor(flat),
        }

    # ----------------------------------------------------------------
    # Utilities
    # ----------------------------------------------------------------

    def _flatten(self, features: Dict) -> Dict:
        flat = {}
        for cat in features.values():
            if isinstance(cat, dict):
                flat.update(cat)
        return flat

    def _data_quality(self, dp) -> str:
        count = sum([bool(dp.voltage), bool(dp.current), bool(dp.frequency), bool(dp.active_power)])
        return ['Poor', 'Fair', 'Good', 'Good', 'Excellent'][count]

    def _error_response(self, msg: str) -> Dict:
        return {
            'error': True,
            'message': msg,
            'real_time_monitoring': {},
            'predictive_maintenance': {},
            'energy_flow': {},
            'decision_making': {},
            'metadata': {'timestamp': None, 'is_simulation': False, 'data_quality': 'Error'}
        }


# Global singleton
ml_inference_engine = MLInferenceEngine()
