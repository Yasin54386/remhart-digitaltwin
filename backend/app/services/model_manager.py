"""
Model Manager Module
Singleton that loads and manages all 10 ML models for grid monitoring.

Authorized models (must match ml_models/trained/):
  voltage_anomaly_iforest.joblib          - Isolation Forest (real-time monitoring)
  anomaly_detection_model.joblib          - XGBClassifier   (real-time monitoring)
  equipment_failure_model.joblib          - XGBRegressor    (predictive maintenance)
  overload_risk_model.joblib              - XGBClassifier   (predictive maintenance)
  epqi_score_model.joblib                 - XGBRegressor    (predictive maintenance)
  voltage_sag_xgb_regressor.joblib        - XGBRegressor    (predictive maintenance)
  load_forecasting_model.joblib           - XGBRegressor    (energy flow)
  energy_imbalance_and_health_index_model.joblib - XGBRegressor (energy flow)
  reactive_q_forecast_model.joblib        - XGBRegressor    (decision making)
  pf_forecast_model.joblib                - XGBRegressor    (decision making)
"""

import os
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any
import logging
import math

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class that manages all 10 ML models.
    Loads models at startup and provides prediction interfaces.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not ModelManager._initialized:
            self.models_dir = Path(__file__).parent.parent / "ml_models" / "trained"
            self.models_dir.mkdir(parents=True, exist_ok=True)
            self.models = {}
            self._load_models()
            ModelManager._initialized = True

    def _load_models(self):
        """Load all 10 authorized trained models from disk."""
        logger.info("Loading ML models...")

        model_files = {
            # Real-time Monitoring
            'voltage_anomaly': 'voltage_anomaly_iforest.joblib',            # dict {model, scaler, features}
            'anomaly_detection': 'anomaly_detection_model.joblib',          # raw XGBClassifier

            # Predictive Maintenance
            'equipment_failure': 'equipment_failure_model.joblib',          # raw XGBRegressor
            'overload_risk':     'overload_risk_model.joblib',              # raw XGBClassifier (LabelEncoded)
            'epqi_score':        'epqi_score_model.joblib',                 # raw XGBRegressor (0-100)
            'voltage_sag':       'voltage_sag_xgb_regressor.joblib',        # dict {model, features}

            # Energy Flow
            'load_forecast':     'load_forecasting_model.joblib',           # raw XGBRegressor
            'health_index':      'energy_imbalance_and_health_index_model.joblib',  # XGBRegressor (0-100)

            # Decision Making
            'reactive_q':        'reactive_q_forecast_model.joblib',        # raw XGBRegressor
            'pf_forecast':       'pf_forecast_model.joblib',                # raw XGBRegressor
        }

        for key, filename in model_files.items():
            path = self.models_dir / filename
            if path.exists():
                try:
                    self.models[key] = joblib.load(path)
                    logger.info(f"  [OK] {key} <- {filename}")
                except Exception as e:
                    logger.warning(f"  [FAIL] {key}: {e}")
                    self.models[key] = None
            else:
                logger.warning(f"  [MISSING] {filename}")
                self.models[key] = None

        loaded = sum(1 for m in self.models.values() if m is not None)
        logger.info(f"Models loaded: {loaded}/{len(model_files)}")

    # ================================================================
    # REAL-TIME MONITORING
    # ================================================================

    def predict_voltage_anomaly(self, features: Dict) -> Dict[str, Any]:
        """
        Isolation Forest on voltage features.
        Artifact: dict {model, scaler, features (13 names), chosen_contamination}
        Returns: {is_anomaly, anomaly_score, confidence, severity}
        """
        artifact = self.models.get('voltage_anomaly')
        if not isinstance(artifact, dict):
            return self._mock_voltage_anomaly(features)

        try:
            iforest    = artifact['model']
            scaler     = artifact['scaler']
            feat_names = artifact['features']

            X        = np.array([[features.get(f, 0.0) for f in feat_names]])
            X_scaled = scaler.transform(X)

            score      = float(iforest.decision_function(X_scaled)[0])
            prediction = int(iforest.predict(X_scaled)[0])   # -1=anomaly, 1=normal

            return {
                'is_anomaly':    bool(prediction == -1),
                'anomaly_score': score,
                'confidence':    abs(score),
                'severity':      self._score_to_severity(score)
            }
        except Exception as e:
            logger.warning(f"predict_voltage_anomaly failed: {e}")
            return self._mock_voltage_anomaly(features)

    def predict_anomaly_detection(self, features: Dict) -> Dict[str, Any]:
        """
        XGBClassifier (binary) on multivariate grid features.
        Training features: volt_A, volt_B, volt_C, I_A, I_B, I_C, P_T, Q_T, FP_T, Frec, hour
        Returns: {is_anomaly, anomaly_probability, risk_level, contributing_factors}
        """
        model = self.models.get('anomaly_detection')
        if model is None:
            return self._mock_anomaly_detection(features)

        try:
            feat_names = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C',
                          'P_T', 'Q_T', 'FP_T', 'Frec', 'hour']
            X     = np.array([[features.get(f, 0.0) for f in feat_names]])
            label = int(model.predict(X)[0])
            prob  = float(model.predict_proba(X)[0][1])   # probability of class=1 (anomaly)

            return {
                'is_anomaly':          bool(label == 1),
                'anomaly_probability': prob,
                'risk_level':          self._prob_to_risk(prob),
                'contributing_factors': self._anomaly_factors(features)
            }
        except Exception as e:
            logger.warning(f"predict_anomaly_detection failed: {e}")
            return self._mock_anomaly_detection(features)

    # ================================================================
    # PREDICTIVE MAINTENANCE
    # ================================================================

    def predict_equipment_failure(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → failure probability [0, 1].
        Training features: volt_A, volt_B, volt_C, I_A, I_B, I_C, P_A, P_B, P_C, Q_A, Q_B, Q_C
        Returns: {failure_probability, risk_level, estimated_days_to_failure, contributing_factors}
        """
        model = self.models.get('equipment_failure')
        if model is None:
            return self._mock_equipment_failure(features)

        try:
            feat_names = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C',
                          'P_A', 'P_B', 'P_C', 'Q_A', 'Q_B', 'Q_C']
            X    = np.array([[features.get(f, 0.0) for f in feat_names]])
            prob = float(np.clip(model.predict(X)[0], 0.0, 1.0))

            return {
                'failure_probability':       prob,
                'risk_level':                self._prob_to_risk(prob),
                'estimated_days_to_failure': self._ttf(prob),
                'contributing_factors':      self._failure_factors(features)
            }
        except Exception as e:
            logger.warning(f"predict_equipment_failure failed: {e}")
            return self._mock_equipment_failure(features)

    def classify_overload_risk(self, features: Dict) -> Dict[str, Any]:
        """
        XGBClassifier (LabelEncoded: High=0, Low=1, Medium=2).
        Training features: volt_A, volt_B, volt_C, I_A, I_B, I_C, P_T, Q_T, FP_T
        Returns: {risk_level, current_load_pct, peak_phase, mitigation_needed}
        """
        model = self.models.get('overload_risk')
        if model is None:
            return self._mock_overload_risk(features)

        try:
            feat_names = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C', 'P_T', 'Q_T', 'FP_T']
            X         = np.array([[features.get(f, 0.0) for f in feat_names]])
            code      = int(model.predict(X)[0])
            label_map = {0: 'High', 1: 'Low', 2: 'Medium'}
            risk      = label_map.get(code, 'Low')

            return {
                'risk_level':       risk,
                'current_load_pct': self._load_pct(features),
                'peak_phase':       self._peak_phase(features),
                'mitigation_needed': bool(risk in ['Medium', 'High'])
            }
        except Exception as e:
            logger.warning(f"classify_overload_risk failed: {e}")
            return self._mock_overload_risk(features)

    def calculate_power_quality_index(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → EPQI score [0, 100].
        Training features: volt_A, volt_B, volt_C, I_A, I_B, I_C, P_T, Q_T, FP_T, Frec
        Returns: {pqi_score, grade, voltage_quality, frequency_quality, power_factor_quality, improvement_areas}
        """
        model = self.models.get('epqi_score')
        if model is None:
            return self._mock_pqi(features)

        try:
            feat_names = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C',
                          'P_T', 'Q_T', 'FP_T', 'Frec']
            X   = np.array([[features.get(f, 0.0) for f in feat_names]])
            pqi = float(np.clip(model.predict(X)[0], 0.0, 100.0))

            return {
                'pqi_score':           pqi,
                'grade':               self._pqi_grade(pqi),
                'voltage_quality':     features.get('v_quality_score', pqi),
                'frequency_quality':   features.get('f_quality_score', pqi),
                'power_factor_quality':features.get('pf_quality_score', pqi),
                'improvement_areas':   self._pqi_improvements(features)
            }
        except Exception as e:
            logger.warning(f"calculate_power_quality_index failed: {e}")
            return self._mock_pqi(features)

    def predict_voltage_sag(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → sag severity [0, 1].
        Artifact: dict {model, features (list), target, time_aware_split}
        Returns: {sag_probability, risk_level, expected_duration_ms, affected_phases}
        """
        artifact = self.models.get('voltage_sag')
        if not isinstance(artifact, dict):
            return self._mock_voltage_sag(features)

        model      = artifact.get('model')
        feat_names = artifact.get('features', [])
        if model is None or not feat_names:
            return self._mock_voltage_sag(features)

        try:
            X        = np.array([[features.get(f, 0.0) for f in feat_names]])
            severity = float(np.clip(model.predict(X)[0], 0.0, 1.0))

            return {
                'sag_probability':    severity,
                'risk_level':         self._prob_to_risk(severity),
                'expected_duration_ms': int(severity * 500),
                'affected_phases':    self._sag_phases(features)
            }
        except Exception as e:
            logger.warning(f"predict_voltage_sag failed: {e}")
            return self._mock_voltage_sag(features)

    # ================================================================
    # ENERGY FLOW
    # ================================================================

    def forecast_load(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → next-period load (kW).
        Features: load lag/rolling stats + time cyclical encoding.
        Returns: {current_load_kw, hourly_forecast[24], peak_load_time, trend}
        """
        model = self.models.get('load_forecast')
        if model is None:
            return self._mock_load_forecast(features)

        try:
            feat_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [
                'load_lag_1h', 'load_lag_24h', 'load_lag_168h',
                'load_roll_mean_3h', 'load_roll_mean_6h', 'load_roll_mean_24h',
                'load_roll_std_3h', 'load_roll_std_6h', 'load_roll_std_24h',
                'load_roll_min_24h', 'load_roll_max_24h',
                'hour_sin', 'hour_cos', 'is_workday'
            ]
            X          = np.array([[features.get(f, 0.0) for f in feat_names]])
            next_load  = float(model.predict(X)[0])
            current    = features.get('P_T', 0.0)
            trend      = (next_load - current) / max(abs(current), 1e-6)
            forecasts  = [next_load] + [
                next_load * (1 + trend * i * 0.005) for i in range(1, 24)
            ]

            return {
                'current_load_kw': current,
                'hourly_forecast': forecasts,
                'peak_load_time':  int(np.argmax(forecasts)),
                'trend':           'increasing' if next_load > current else 'decreasing'
            }
        except Exception as e:
            logger.warning(f"forecast_load failed: {e}")
            return self._mock_load_forecast(features)

    def estimate_health_index(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → grid health score [0, 100] (IEC 61000-4-30 inspired).
        Training features: volt_A, volt_B, volt_C, I_A, I_B, I_C, P_T, Q_T, FP_T, Frec
        Returns: {health_score, grade, efficiency_status, risk_indicators}
        """
        model = self.models.get('health_index')
        if model is None:
            return self._mock_health_index(features)

        try:
            feat_names = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C',
                          'P_T', 'Q_T', 'FP_T', 'Frec']
            X     = np.array([[features.get(f, 0.0) for f in feat_names]])
            score = float(np.clip(model.predict(X)[0], 0.0, 100.0))

            return {
                'health_score':      score,
                'grade':             self._pqi_grade(score),
                'efficiency_status': self._health_status(score),
                'risk_indicators':   self._health_risks(features)
            }
        except Exception as e:
            logger.warning(f"estimate_health_index failed: {e}")
            return self._mock_health_index(features)

    # ================================================================
    # DECISION MAKING
    # ================================================================

    def forecast_reactive_q(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → next-period reactive power (kVAR).
        Returns: {predicted_q_kvar, current_q_kvar, trend, compensation_needed}
        """
        model = self.models.get('reactive_q')
        if model is None:
            return self._mock_reactive_q(features)

        try:
            feat_names = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [
                'Q_lag_1h', 'Q_lag_24h', 'PF_lag_1h', 'hour_sin', 'hour_cos', 'is_workday'
            ]
            X          = np.array([[features.get(f, 0.0) for f in feat_names]])
            predicted  = float(model.predict(X)[0])
            current_q  = features.get('Q_T', 0.0)

            return {
                'predicted_q_kvar':    predicted,
                'current_q_kvar':      current_q,
                'trend':               'increasing' if predicted > current_q else 'decreasing',
                'compensation_needed': bool(abs(predicted) > abs(current_q) * 1.1)
            }
        except Exception as e:
            logger.warning(f"forecast_reactive_q failed: {e}")
            return self._mock_reactive_q(features)

    def forecast_power_factor(self, features: Dict) -> Dict[str, Any]:
        """
        XGBRegressor → next-period power factor [0, 1].
        Returns: {predicted_pf, current_pf, target_pf, required_compensation_kvar}
        """
        model = self.models.get('pf_forecast')
        if model is None:
            return self._mock_pf_forecast(features)

        try:
            feat_names  = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else [
                'Q_lag_1h', 'Q_lag_24h', 'PF_lag_1h', 'hour_sin', 'hour_cos', 'is_workday'
            ]
            X           = np.array([[features.get(f, 0.0) for f in feat_names]])
            predicted   = float(np.clip(model.predict(X)[0], 0.0, 1.0))
            current_pf  = features.get('FP_T', features.get('power_factor', 0.9))
            current_p   = features.get('P_T', 0.0)
            target_pf   = 0.95

            safe_pf     = max(0.01, predicted)
            if safe_pf < target_pf:
                req_q = current_p * (math.tan(math.acos(safe_pf)) - math.tan(math.acos(target_pf)))
            else:
                req_q = 0.0

            return {
                'predicted_pf':              predicted,
                'current_pf':                current_pf,
                'target_pf':                 target_pf,
                'required_compensation_kvar': float(req_q)
            }
        except Exception as e:
            logger.warning(f"forecast_power_factor failed: {e}")
            return self._mock_pf_forecast(features)

    # ================================================================
    # HELPERS
    # ================================================================

    def _score_to_severity(self, score: float) -> str:
        if score < -0.5:   return 'High'
        if score < -0.2:   return 'Medium'
        return 'Low'

    def _prob_to_risk(self, prob: float) -> str:
        if prob > 0.7:  return 'High'
        if prob > 0.4:  return 'Medium'
        return 'Low'

    def _pqi_grade(self, score: float) -> str:
        if score >= 90: return 'Excellent'
        if score >= 75: return 'Good'
        if score >= 60: return 'Fair'
        return 'Poor'

    def _health_status(self, score: float) -> str:
        if score >= 85: return 'Optimal'
        if score >= 70: return 'Normal'
        if score >= 55: return 'Degraded'
        return 'Critical'

    def _load_pct(self, features: Dict) -> float:
        return (features.get('i_avg', features.get('I_A', 0)) / 100.0) * 100

    def _peak_phase(self, features: Dict) -> str:
        phases = {'A': features.get('I_A', 0), 'B': features.get('I_B', 0), 'C': features.get('I_C', 0)}
        return max(phases, key=phases.get)

    def _ttf(self, prob: float) -> int:
        if prob > 0.8:  return 7
        if prob > 0.6:  return 30
        if prob > 0.4:  return 90
        return 365

    def _failure_factors(self, features: Dict) -> list:
        factors = []
        if features.get('i_spike_detected', 0): factors.append('Current spikes detected')
        if features.get('v_imbalance_pct', 0) > 5: factors.append('Voltage imbalance')
        if features.get('FP_T', features.get('power_factor', 1.0)) < 0.85: factors.append('Low power factor')
        return factors

    def _anomaly_factors(self, features: Dict) -> list:
        factors = []
        volt_A = features.get('volt_A', 230)
        if abs(volt_A - 230) > 11.5: factors.append('Voltage deviation')
        if features.get('FP_T', 1.0) < 0.85: factors.append('Low power factor')
        if abs(features.get('Frec', 50) - 50) > 0.5: factors.append('Frequency deviation')
        return factors

    def _pqi_improvements(self, features: Dict) -> list:
        areas = []
        if features.get('v_quality_score', 100) < 80: areas.append('Voltage regulation')
        if features.get('f_quality_score', 100) < 80:  areas.append('Frequency control')
        if features.get('pf_quality_score', 100) < 80: areas.append('Power factor correction')
        return areas

    def _health_risks(self, features: Dict) -> list:
        risks = []
        v_avg = features.get('v_avg', (features.get('volt_A', 230) + features.get('volt_B', 230) + features.get('volt_C', 230)) / 3)
        if v_avg < 218 or v_avg > 242: risks.append('Voltage out of range')
        if features.get('FP_T', 1.0) < 0.85: risks.append('Low power factor')
        if abs(features.get('Frec', 50) - 50) > 0.5: risks.append('Frequency instability')
        return risks

    def _sag_phases(self, features: Dict) -> list:
        v_avg   = features.get('v_avg', 230)
        phases  = []
        if features.get('volt_A', v_avg) < v_avg * 0.9: phases.append('A')
        if features.get('volt_B', v_avg) < v_avg * 0.9: phases.append('B')
        if features.get('volt_C', v_avg) < v_avg * 0.9: phases.append('C')
        return phases

    # ================================================================
    # MOCK FALLBACKS (used when model file is missing)
    # ================================================================

    def _mock_voltage_anomaly(self, features: Dict) -> Dict:
        v_dev = abs(features.get('v_avg', features.get('volt_A', 230)) - 230) / 230
        is_a  = v_dev > 0.05
        return {'is_anomaly': is_a, 'anomaly_score': -0.3 if is_a else 0.1,
                'confidence': 0.75, 'severity': 'Medium' if is_a else 'Low'}

    def _mock_anomaly_detection(self, features: Dict) -> Dict:
        pf  = features.get('FP_T', features.get('power_factor', 0.95))
        prob = max(0.0, 0.3 * (1 - pf))
        return {'is_anomaly': prob > 0.3, 'anomaly_probability': prob,
                'risk_level': self._prob_to_risk(prob), 'contributing_factors': []}

    def _mock_equipment_failure(self, features: Dict) -> Dict:
        prob = min(0.9, features.get('i_variance', 0) * 0.1 + (1 - features.get('FP_T', 0.9)) * 0.4)
        return {'failure_probability': prob, 'risk_level': self._prob_to_risk(prob),
                'estimated_days_to_failure': self._ttf(prob), 'contributing_factors': self._failure_factors(features)}

    def _mock_overload_risk(self, features: Dict) -> Dict:
        pct  = self._load_pct(features)
        risk = 'High' if pct > 85 else 'Medium' if pct > 70 else 'Low'
        return {'risk_level': risk, 'current_load_pct': pct,
                'peak_phase': self._peak_phase(features), 'mitigation_needed': risk != 'Low'}

    def _mock_pqi(self, features: Dict) -> Dict:
        pqi = features.get('power_quality_index', 80.0)
        return {'pqi_score': pqi, 'grade': self._pqi_grade(pqi),
                'voltage_quality': features.get('v_quality_score', pqi),
                'frequency_quality': features.get('f_quality_score', pqi),
                'power_factor_quality': features.get('pf_quality_score', pqi),
                'improvement_areas': self._pqi_improvements(features)}

    def _mock_voltage_sag(self, features: Dict) -> Dict:
        prob = min(0.8, features.get('v_deviation_pct', 0) / 100)
        return {'sag_probability': prob, 'risk_level': self._prob_to_risk(prob),
                'expected_duration_ms': int(prob * 500), 'affected_phases': self._sag_phases(features)}

    def _mock_load_forecast(self, features: Dict) -> Dict:
        current   = features.get('P_T', 0.0)
        trend_val = features.get('active_power_trend', 0)
        forecasts = [current + trend_val * i for i in range(1, 25)]
        return {'current_load_kw': current, 'hourly_forecast': forecasts,
                'peak_load_time': int(np.argmax(forecasts)), 'trend': 'stable'}

    def _mock_health_index(self, features: Dict) -> Dict:
        score = features.get('power_quality_index', 75.0)
        return {'health_score': score, 'grade': self._pqi_grade(score),
                'efficiency_status': self._health_status(score),
                'risk_indicators': self._health_risks(features)}

    def _mock_reactive_q(self, features: Dict) -> Dict:
        q = features.get('Q_T', 0.0)
        return {'predicted_q_kvar': q, 'current_q_kvar': q,
                'trend': 'stable', 'compensation_needed': False}

    def _mock_pf_forecast(self, features: Dict) -> Dict:
        pf = features.get('FP_T', features.get('power_factor', 0.9))
        return {'predicted_pf': pf, 'current_pf': pf, 'target_pf': 0.95,
                'required_compensation_kvar': 0.0}


# Global singleton
model_manager = ModelManager()
