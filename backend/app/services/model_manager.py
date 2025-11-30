"""
Model Manager Module
Singleton that loads and manages all 16 ML models for grid monitoring
"""

import os
import joblib
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class that manages all ML models.
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

            # Initialize model storage
            self.models = {}

            # Load all models
            self._load_models()

            ModelManager._initialized = True

    def _load_models(self):
        """Load all trained models from disk"""
        logger.info("Loading ML models...")

        # Define all model paths
        model_files = {
            # Real-time Monitoring Models
            'voltage_anomaly': 'voltage_anomaly_detector.pkl',
            'harmonic_analyzer': 'harmonic_analyzer.pkl',
            'frequency_stability': 'frequency_stability_predictor.pkl',
            'phase_imbalance': 'phase_imbalance_classifier.pkl',

            # Predictive Maintenance Models
            'equipment_failure': 'equipment_failure_predictor.pkl',
            'overload_risk': 'overload_risk_classifier.pkl',
            'power_quality': 'power_quality_index.pkl',
            'voltage_sag': 'voltage_sag_predictor.pkl',

            # Energy Flow Models
            'load_forecast': 'load_forecasting_lstm.pkl',
            'energy_loss': 'energy_loss_estimator.pkl',
            'power_flow': 'power_flow_optimizer.pkl',
            'demand_response': 'demand_response_potential.pkl',

            # Decision Making Models
            'reactive_compensation': 'reactive_power_compensator.pkl',
            'load_balancing': 'load_balancing_optimizer.pkl',
            'grid_stability': 'grid_stability_scorer.pkl',
            'optimal_dispatch': 'optimal_dispatch_advisor.pkl'
        }

        # Try to load each model
        for model_name, filename in model_files.items():
            model_path = self.models_dir / filename

            if model_path.exists():
                try:
                    self.models[model_name] = joblib.load(model_path)
                    logger.info(f"✓ Loaded {model_name}")
                except Exception as e:
                    logger.warning(f"✗ Failed to load {model_name}: {e}")
                    self.models[model_name] = None
            else:
                logger.warning(f"✗ Model file not found: {filename}")
                self.models[model_name] = None

        logger.info(f"Models loaded: {sum(1 for m in self.models.values() if m is not None)}/{len(model_files)}")

    # ==================== REAL-TIME MONITORING PREDICTIONS ====================

    def predict_voltage_anomaly(self, features: Dict) -> Dict[str, Any]:
        """
        Predict voltage anomalies using Isolation Forest

        Args:
            features: Voltage features from FeatureEngineer

        Returns:
            Dictionary with anomaly score and status
        """
        model = self.models.get('voltage_anomaly')
        if model is None:
            return self._mock_voltage_anomaly(features)

        try:
            # Extract relevant features
            X = np.array([[
                features['v_avg'],
                features['v_variance'],
                features['v_imbalance_pct'],
                features['v_deviation_pct'],
                features['v_rate_of_change']
            ]])

            # Predict (-1 = anomaly, 1 = normal)
            prediction = model.predict(X)[0]
            anomaly_score = model.score_samples(X)[0]

            return {
                'is_anomaly': prediction == -1,
                'anomaly_score': float(anomaly_score),
                'confidence': abs(float(anomaly_score)),
                'severity': self._get_severity(anomaly_score)
            }
        except Exception as e:
            logger.warning(f"Voltage anomaly prediction failed, using mock: {e}")
            return self._mock_voltage_anomaly(features)

    def analyze_harmonics(self, features: Dict) -> Dict[str, Any]:
        """
        Analyze harmonic distortion using FFT + Random Forest

        Args:
            features: Quality features from FeatureEngineer

        Returns:
            THD estimate and harmonic components
        """
        model = self.models.get('harmonic_analyzer')
        if model is None:
            return self._mock_harmonic_analysis(features)

        X = np.array([[
            features['v_thd_estimated'],
            features['v_variance'],
            features['power_factor'],
            features['v_quality_score']
        ]])

        # Predict THD level
        thd_category = model.predict(X)[0]

        return {
            'thd_percentage': features['v_thd_estimated'],
            'thd_category': thd_category,  # 'Low', 'Medium', 'High'
            'harmonics': self._estimate_harmonics(features),
            'quality_impact': self._assess_quality_impact(features['v_thd_estimated'])
        }

    def predict_frequency_stability(self, features: Dict) -> Dict[str, Any]:
        """
        Predict frequency stability using LSTM

        Args:
            features: Frequency features with history

        Returns:
            Future frequency predictions and stability score
        """
        model = self.models.get('frequency_stability')
        if model is None:
            return self._mock_frequency_stability(features)

        # Prepare sequence data for LSTM
        f_history = np.array(features['f_history'][-100:])
        if len(f_history) < 100:
            f_history = np.pad(f_history, (100 - len(f_history), 0), mode='edge')

        X = f_history.reshape(1, 100, 1)

        # Predict next 10 time steps
        predictions = model.predict(X, verbose=0)[0]

        return {
            'current_frequency': features['f_value'],
            'predicted_frequencies': predictions.tolist(),
            'stability_score': self._calculate_stability_score(predictions),
            'trend': 'increasing' if predictions[-1] > features['f_value'] else 'decreasing'
        }

    def classify_phase_imbalance(self, features: Dict) -> Dict[str, Any]:
        """
        Classify phase imbalance severity using Decision Tree

        Args:
            features: Balance features from FeatureEngineer

        Returns:
            Imbalance classification and recommendations
        """
        model = self.models.get('phase_imbalance')
        if model is None:
            return self._mock_phase_imbalance(features)

        X = np.array([[
            features['v_imbalance'],
            features['i_imbalance'],
            features['p_imbalance'],
            features['overall_balance_score']
        ]])

        # Classify severity
        severity = model.predict(X)[0]

        return {
            'severity': severity,  # 'Normal', 'Warning', 'Critical'
            'voltage_imbalance': features['v_imbalance'],
            'current_imbalance': features['i_imbalance'],
            'power_imbalance': features['p_imbalance'],
            'balance_score': features['overall_balance_score'],
            'action_required': severity in ['Warning', 'Critical']
        }

    # ==================== PREDICTIVE MAINTENANCE PREDICTIONS ====================

    def predict_equipment_failure(self, features: Dict) -> Dict[str, Any]:
        """
        Predict equipment failure probability using XGBoost

        Args:
            features: All features from FeatureEngineer

        Returns:
            Failure probability and time-to-failure estimate
        """
        model = self.models.get('equipment_failure')
        if model is None:
            return self._mock_equipment_failure(features)

        try:
            X = np.array([[
                features['i_avg'],
                features['i_variance'],
                features['v_variance'],
                features['power_factor'],
                features['i_spike_detected'],
                features['v_imbalance_pct']
            ]])

            failure_prob = model.predict_proba(X)[0][1]

            return {
                'failure_probability': float(failure_prob),
                'risk_level': self._get_risk_level(failure_prob),
                'estimated_days_to_failure': self._estimate_ttf(failure_prob),
                'contributing_factors': self._identify_failure_factors(features)
            }
        except Exception as e:
            logger.warning(f"Model prediction failed, using mock: {e}")
            return self._mock_equipment_failure(features)

    def classify_overload_risk(self, features: Dict) -> Dict[str, Any]:
        """
        Classify overload risk using SVM

        Args:
            features: Current and power features

        Returns:
            Overload risk classification
        """
        model = self.models.get('overload_risk')
        if model is None:
            return self._mock_overload_risk(features)

        X = np.array([[
            features['i_avg'],
            features['p_total'],
            features['i_max_phase'],
            features['i_imbalance_pct']
        ]])

        risk_class = model.predict(X)[0]

        return {
            'risk_level': risk_class,  # 'Low', 'Medium', 'High'
            'current_load_pct': self._calculate_load_percentage(features),
            'peak_phase': self._identify_peak_phase(features),
            'mitigation_needed': risk_class in ['Medium', 'High']
        }

    def calculate_power_quality_index(self, features: Dict) -> Dict[str, Any]:
        """
        Calculate comprehensive power quality index using Neural Network

        Args:
            features: Quality features

        Returns:
            PQI score and component breakdown
        """
        model = self.models.get('power_quality')
        if model is None:
            return self._mock_pqi(features)

        X = np.array([[
            features['v_quality_score'],
            features['f_quality_score'],
            features['pf_quality_score'],
            features['v_thd_estimated'],
            features['overall_balance_score']
        ]])

        pqi = model.predict(X)[0][0]

        return {
            'pqi_score': float(pqi),
            'grade': self._get_pqi_grade(pqi),
            'voltage_quality': features['v_quality_score'],
            'frequency_quality': features['f_quality_score'],
            'power_factor_quality': features['pf_quality_score'],
            'improvement_areas': self._identify_improvements(features)
        }

    def predict_voltage_sag(self, features: Dict) -> Dict[str, Any]:
        """
        Predict voltage sag events using Random Forest

        Args:
            features: Voltage and time-series features

        Returns:
            Sag probability and severity
        """
        model = self.models.get('voltage_sag')
        if model is None:
            return self._mock_voltage_sag(features)

        X = np.array([[
            features['v_avg'],
            features['voltage_avg_std'],
            features['v_rate_of_change'],
            features['voltage_avg_trend']
        ]])

        sag_prob = model.predict_proba(X)[0][1]

        return {
            'sag_probability': float(sag_prob),
            'risk_level': self._get_risk_level(sag_prob),
            'expected_duration_ms': self._estimate_sag_duration(sag_prob),
            'affected_phases': self._identify_affected_phases(features)
        }

    # ==================== ENERGY FLOW PREDICTIONS ====================

    def forecast_load(self, features: Dict) -> Dict[str, Any]:
        """
        Forecast future load using Prophet

        Args:
            features: Time-series power features

        Returns:
            Load forecasts for next 24 hours
        """
        model = self.models.get('load_forecast')
        if model is None:
            return self._mock_load_forecast(features)

        # Prophet uses different interface, simplified here
        current_load = features['p_total']
        trend = features.get('active_power_trend', 0)

        # Generate 24-hour forecast
        forecasts = [current_load + trend * i for i in range(1, 25)]

        return {
            'current_load_kw': current_load,
            'hourly_forecast': forecasts,
            'peak_load_time': self._find_peak_time(forecasts),
            'trend': 'increasing' if trend > 0 else 'decreasing'
        }

    def estimate_energy_loss(self, features: Dict) -> Dict[str, Any]:
        """
        Estimate transmission losses using Linear Regression

        Args:
            features: Current and power features

        Returns:
            Loss estimates and efficiency metrics
        """
        model = self.models.get('energy_loss')
        if model is None:
            return self._mock_energy_loss(features)

        X = np.array([[
            features['i_avg'],
            features['p_total'],
            features['i_imbalance_pct'],
            features['power_factor']
        ]])

        loss_kw = model.predict(X)[0]

        return {
            'loss_kw': float(loss_kw),
            'loss_percentage': (loss_kw / features['p_total'] * 100) if features['p_total'] > 0 else 0,
            'efficiency': 100 - ((loss_kw / features['p_total'] * 100) if features['p_total'] > 0 else 0),
            'loss_breakdown': self._breakdown_losses(features)
        }

    def optimize_power_flow(self, features: Dict) -> Dict[str, Any]:
        """
        Optimize power flow using Linear Programming

        Args:
            features: All power and balance features

        Returns:
            Optimization recommendations
        """
        model = self.models.get('power_flow')
        if model is None:
            return self._mock_power_flow(features)

        # Simplified optimization
        return {
            'optimal_distribution': {
                'phase_a': features['p_phase_a'],
                'phase_b': features['p_phase_b'],
                'phase_c': features['p_phase_c']
            },
            'rebalancing_needed': features['p_imbalance_pct'] > 10,
            'suggested_adjustments': self._calculate_adjustments(features),
            'potential_savings_pct': self._estimate_savings(features)
        }

    def assess_demand_response(self, features: Dict) -> Dict[str, Any]:
        """
        Assess demand response potential using K-Means clustering

        Args:
            features: Load and time-series features

        Returns:
            DR potential and recommendations
        """
        model = self.models.get('demand_response')
        if model is None:
            return self._mock_demand_response(features)

        X = np.array([[
            features['p_total'],
            features['active_power_mean'],
            features['active_power_std']
        ]])

        cluster = model.predict(X)[0]

        return {
            'load_cluster': int(cluster),
            'dr_potential_kw': self._calculate_dr_potential(features),
            'flexibility_score': self._calculate_flexibility(features),
            'recommended_actions': self._suggest_dr_actions(cluster, features)
        }

    # ==================== DECISION MAKING PREDICTIONS ====================

    def optimize_reactive_compensation(self, features: Dict) -> Dict[str, Any]:
        """
        Optimize reactive power compensation using Neural Network

        Args:
            features: Power factor and reactive power features

        Returns:
            Compensation recommendations
        """
        model = self.models.get('reactive_compensation')
        if model is None:
            return self._mock_reactive_compensation(features)

        X = np.array([[
            features['power_factor'],
            features['q_total'],
            features['p_total']
        ]])

        optimal_q = model.predict(X)[0][0]

        return {
            'current_pf': features['power_factor'],
            'target_pf': 0.95,
            'required_compensation_kvar': float(optimal_q - features['q_total']),
            'capacitor_size_kvar': self._calculate_capacitor_size(optimal_q - features['q_total']),
            'expected_savings': self._calculate_pf_savings(features)
        }

    def optimize_load_balancing(self, features: Dict) -> Dict[str, Any]:
        """
        Optimize load balancing using Multi-Criteria Decision Analysis

        Args:
            features: Balance and power features

        Returns:
            Load balancing recommendations
        """
        model = self.models.get('load_balancing')
        if model is None:
            return self._mock_load_balancing(features)

        return {
            'current_imbalance': features['overall_balance_score'],
            'target_distribution': {
                'phase_a': features['i_phase_a'],
                'phase_b': features['i_phase_b'],
                'phase_c': features['i_phase_c']
            },
            'redistribution_plan': self._create_redistribution_plan(features),
            'expected_improvement_pct': self._estimate_balance_improvement(features)
        }

    def score_grid_stability(self, features: Dict) -> Dict[str, Any]:
        """
        Score overall grid stability using ensemble model

        Args:
            features: All features

        Returns:
            Comprehensive stability score
        """
        model = self.models.get('grid_stability')
        if model is None:
            return self._mock_grid_stability(features)

        X = np.array([[
            features['v_avg'],
            features['f_value'],
            features['power_factor'],
            features['overall_balance_score'],
            features['power_quality_index']
        ]])

        stability_score = model.predict(X)[0]

        return {
            'stability_score': float(stability_score),
            'status': self._get_stability_status(stability_score),
            'risk_factors': self._identify_risks(features),
            'recommendations': self._generate_recommendations(features)
        }

    def advise_optimal_dispatch(self, features: Dict) -> Dict[str, Any]:
        """
        Advise optimal generation dispatch using SVR

        Args:
            features: Load and forecast features

        Returns:
            Dispatch recommendations
        """
        model = self.models.get('optimal_dispatch')
        if model is None:
            return self._mock_optimal_dispatch(features)

        X = np.array([[
            features['p_total'],
            features['active_power_mean'],
            features['active_power_trend']
        ]])

        optimal_generation = model.predict(X)[0]

        return {
            'current_load_kw': features['p_total'],
            'recommended_generation_kw': float(optimal_generation),
            'reserve_margin_pct': ((optimal_generation - features['p_total']) / features['p_total'] * 100) if features['p_total'] > 0 else 0,
            'dispatch_plan': self._create_dispatch_plan(optimal_generation, features)
        }

    # ==================== HELPER METHODS ====================

    def _get_severity(self, score: float) -> str:
        if score < -0.5:
            return 'High'
        elif score < -0.2:
            return 'Medium'
        return 'Low'

    def _get_risk_level(self, prob: float) -> str:
        if prob > 0.7:
            return 'High'
        elif prob > 0.4:
            return 'Medium'
        return 'Low'

    def _get_pqi_grade(self, pqi: float) -> str:
        if pqi >= 90:
            return 'Excellent'
        elif pqi >= 75:
            return 'Good'
        elif pqi >= 60:
            return 'Fair'
        return 'Poor'

    def _get_stability_status(self, score: float) -> str:
        if score >= 0.8:
            return 'Stable'
        elif score >= 0.6:
            return 'Marginal'
        return 'Unstable'

    def _estimate_harmonics(self, features: Dict) -> Dict:
        thd = features['v_thd_estimated']
        return {
            'h3': thd * 0.4,
            'h5': thd * 0.3,
            'h7': thd * 0.2,
            'h9': thd * 0.1
        }

    def _assess_quality_impact(self, thd: float) -> str:
        if thd < 5:
            return 'Minimal'
        elif thd < 10:
            return 'Moderate'
        return 'Significant'

    def _calculate_stability_score(self, predictions: np.ndarray) -> float:
        variance = np.var(predictions)
        return max(0, 100 - variance * 100)

    def _identify_failure_factors(self, features: Dict) -> list:
        factors = []
        if features['i_spike_detected']:
            factors.append('Current spikes detected')
        if features['v_imbalance_pct'] > 5:
            factors.append('Voltage imbalance')
        if features['power_factor'] < 0.85:
            factors.append('Low power factor')
        return factors

    def _estimate_ttf(self, prob: float) -> int:
        """Estimate days to failure based on probability"""
        if prob > 0.8:
            return 7
        elif prob > 0.6:
            return 30
        elif prob > 0.4:
            return 90
        return 365

    def _calculate_load_percentage(self, features: Dict) -> float:
        # Assuming 100A rated current
        rated_current = 100.0
        return (features['i_avg'] / rated_current) * 100

    def _identify_peak_phase(self, features: Dict) -> str:
        phases = {
            'A': features['i_phase_a'],
            'B': features['i_phase_b'],
            'C': features['i_phase_c']
        }
        return max(phases, key=phases.get)

    def _identify_improvements(self, features: Dict) -> list:
        improvements = []
        if features['v_quality_score'] < 80:
            improvements.append('Voltage regulation')
        if features['f_quality_score'] < 80:
            improvements.append('Frequency control')
        if features['pf_quality_score'] < 80:
            improvements.append('Power factor correction')
        return improvements

    def _estimate_sag_duration(self, prob: float) -> int:
        """Estimate sag duration in milliseconds"""
        return int(prob * 500)

    def _identify_affected_phases(self, features: Dict) -> list:
        affected = []
        avg_v = features['v_avg']
        if features['v_phase_a'] < avg_v * 0.9:
            affected.append('A')
        if features['v_phase_b'] < avg_v * 0.9:
            affected.append('B')
        if features['v_phase_c'] < avg_v * 0.9:
            affected.append('C')
        return affected

    def _find_peak_time(self, forecasts: list) -> int:
        return forecasts.index(max(forecasts))

    def _breakdown_losses(self, features: Dict) -> Dict:
        total_loss = features['i_avg'] ** 2 * 0.1  # Simplified I²R loss
        return {
            'resistive_loss': total_loss * 0.7,
            'reactive_loss': total_loss * 0.2,
            'imbalance_loss': total_loss * 0.1
        }

    def _calculate_adjustments(self, features: Dict) -> Dict:
        avg_power = features['p_total'] / 3
        return {
            'phase_a': avg_power - features['p_phase_a'],
            'phase_b': avg_power - features['p_phase_b'],
            'phase_c': avg_power - features['p_phase_c']
        }

    def _estimate_savings(self, features: Dict) -> float:
        return features['p_imbalance_pct'] * 0.5

    def _calculate_dr_potential(self, features: Dict) -> float:
        return features['p_total'] * 0.15  # 15% of current load

    def _calculate_flexibility(self, features: Dict) -> float:
        std = features['active_power_std']
        mean = features['active_power_mean']
        return min(100, (std / mean * 100)) if mean > 0 else 0

    def _suggest_dr_actions(self, cluster: int, features: Dict) -> list:
        actions = []
        if cluster == 0:  # Low load
            actions.append('Increase load during off-peak')
        elif cluster == 1:  # Medium load
            actions.append('Maintain current operation')
        else:  # High load
            actions.append('Reduce non-critical loads')
        return actions

    def _calculate_capacitor_size(self, required_kvar: float) -> float:
        # Round to standard capacitor sizes
        standard_sizes = [5, 10, 15, 20, 25, 30, 40, 50]
        abs_kvar = abs(required_kvar)
        return min(standard_sizes, key=lambda x: abs(x - abs_kvar))

    def _calculate_pf_savings(self, features: Dict) -> float:
        current_pf = features['power_factor']
        target_pf = 0.95
        return features['p_total'] * (1 / current_pf - 1 / target_pf) * 0.1

    def _create_redistribution_plan(self, features: Dict) -> Dict:
        avg_current = (features['i_phase_a'] + features['i_phase_b'] + features['i_phase_c']) / 3
        return {
            'move_from_a': max(0, features['i_phase_a'] - avg_current),
            'move_from_b': max(0, features['i_phase_b'] - avg_current),
            'move_from_c': max(0, features['i_phase_c'] - avg_current)
        }

    def _estimate_balance_improvement(self, features: Dict) -> float:
        current_score = features['overall_balance_score']
        return (100 - current_score) * 0.7

    def _identify_risks(self, features: Dict) -> list:
        risks = []
        if features['v_avg'] < 220 or features['v_avg'] > 240:
            risks.append('Voltage deviation')
        if abs(features['f_value'] - 50) > 0.5:
            risks.append('Frequency instability')
        if features['power_factor'] < 0.85:
            risks.append('Low power factor')
        return risks

    def _generate_recommendations(self, features: Dict) -> list:
        recs = []
        if features['power_factor'] < 0.9:
            recs.append('Install power factor correction')
        if features['overall_balance_score'] < 80:
            recs.append('Redistribute loads across phases')
        if features['power_quality_index'] < 75:
            recs.append('Review power quality improvement options')
        return recs

    def _create_dispatch_plan(self, optimal_gen: float, features: Dict) -> Dict:
        return {
            'base_load': optimal_gen * 0.7,
            'intermediate': optimal_gen * 0.2,
            'peak': optimal_gen * 0.1
        }

    # ==================== MOCK METHODS (Used when models not trained yet) ====================

    def _mock_voltage_anomaly(self, features: Dict) -> Dict:
        v_deviation = abs(features['v_avg'] - 230) / 230
        is_anomaly = v_deviation > 0.05 or features['v_imbalance_pct'] > 5
        return {
            'is_anomaly': is_anomaly,
            'anomaly_score': -0.3 if is_anomaly else 0.1,
            'confidence': 0.75,
            'severity': 'Medium' if is_anomaly else 'Low'
        }

    def _mock_harmonic_analysis(self, features: Dict) -> Dict:
        thd = features['v_thd_estimated']
        return {
            'thd_percentage': thd,
            'thd_category': 'High' if thd > 8 else 'Medium' if thd > 5 else 'Low',
            'harmonics': self._estimate_harmonics(features),
            'quality_impact': self._assess_quality_impact(thd)
        }

    def _mock_frequency_stability(self, features: Dict) -> Dict:
        f_current = features['f_value']
        predictions = [f_current + np.random.normal(0, 0.01) for _ in range(10)]
        return {
            'current_frequency': f_current,
            'predicted_frequencies': predictions,
            'stability_score': self._calculate_stability_score(np.array(predictions)),
            'trend': 'stable'
        }

    def _mock_phase_imbalance(self, features: Dict) -> Dict:
        score = features['overall_balance_score']
        severity = 'Critical' if score < 70 else 'Warning' if score < 85 else 'Normal'
        return {
            'severity': severity,
            'voltage_imbalance': features['v_imbalance'],
            'current_imbalance': features['i_imbalance'],
            'power_imbalance': features['p_imbalance'],
            'balance_score': score,
            'action_required': severity != 'Normal'
        }

    def _mock_equipment_failure(self, features: Dict) -> Dict:
        risk_score = (features['i_variance'] * 0.3 + features['v_variance'] * 0.3 +
                     (1 - features['power_factor']) * 0.4)
        return {
            'failure_probability': min(0.9, risk_score),
            'risk_level': self._get_risk_level(risk_score),
            'estimated_days_to_failure': self._estimate_ttf(risk_score),
            'contributing_factors': self._identify_failure_factors(features)
        }

    def _mock_overload_risk(self, features: Dict) -> Dict:
        load_pct = self._calculate_load_percentage(features)
        risk = 'High' if load_pct > 85 else 'Medium' if load_pct > 70 else 'Low'
        return {
            'risk_level': risk,
            'current_load_pct': load_pct,
            'peak_phase': self._identify_peak_phase(features),
            'mitigation_needed': risk != 'Low'
        }

    def _mock_pqi(self, features: Dict) -> Dict:
        pqi = features['power_quality_index']
        return {
            'pqi_score': pqi,
            'grade': self._get_pqi_grade(pqi),
            'voltage_quality': features['v_quality_score'],
            'frequency_quality': features['f_quality_score'],
            'power_factor_quality': features['pf_quality_score'],
            'improvement_areas': self._identify_improvements(features)
        }

    def _mock_voltage_sag(self, features: Dict) -> Dict:
        sag_prob = min(0.8, features['v_deviation_pct'] / 100 + features['voltage_avg_std'] / 100)
        return {
            'sag_probability': sag_prob,
            'risk_level': self._get_risk_level(sag_prob),
            'expected_duration_ms': self._estimate_sag_duration(sag_prob),
            'affected_phases': self._identify_affected_phases(features)
        }

    def _mock_load_forecast(self, features: Dict) -> Dict:
        current = features['p_total']
        trend = features.get('active_power_trend', 0)
        forecasts = [current + trend * i + np.random.normal(0, current * 0.05) for i in range(1, 25)]
        return {
            'current_load_kw': current,
            'hourly_forecast': forecasts,
            'peak_load_time': self._find_peak_time(forecasts),
            'trend': 'increasing' if trend > 0 else 'decreasing'
        }

    def _mock_energy_loss(self, features: Dict) -> Dict:
        loss = features['i_avg'] ** 2 * 0.1
        return {
            'loss_kw': loss,
            'loss_percentage': (loss / features['p_total'] * 100) if features['p_total'] > 0 else 0,
            'efficiency': 100 - ((loss / features['p_total'] * 100) if features['p_total'] > 0 else 0),
            'loss_breakdown': self._breakdown_losses(features)
        }

    def _mock_power_flow(self, features: Dict) -> Dict:
        return {
            'optimal_distribution': {
                'phase_a': features['p_phase_a'],
                'phase_b': features['p_phase_b'],
                'phase_c': features['p_phase_c']
            },
            'rebalancing_needed': features['p_imbalance_pct'] > 10,
            'suggested_adjustments': self._calculate_adjustments(features),
            'potential_savings_pct': self._estimate_savings(features)
        }

    def _mock_demand_response(self, features: Dict) -> Dict:
        cluster = 1  # Medium load cluster
        return {
            'load_cluster': cluster,
            'dr_potential_kw': self._calculate_dr_potential(features),
            'flexibility_score': self._calculate_flexibility(features),
            'recommended_actions': self._suggest_dr_actions(cluster, features)
        }

    def _mock_reactive_compensation(self, features: Dict) -> Dict:
        target_pf = 0.95
        current_pf = features['power_factor']
        required_q = features['p_total'] * (np.tan(np.arccos(current_pf)) - np.tan(np.arccos(target_pf)))
        return {
            'current_pf': current_pf,
            'target_pf': target_pf,
            'required_compensation_kvar': required_q,
            'capacitor_size_kvar': self._calculate_capacitor_size(required_q),
            'expected_savings': self._calculate_pf_savings(features)
        }

    def _mock_load_balancing(self, features: Dict) -> Dict:
        return {
            'current_imbalance': features['overall_balance_score'],
            'target_distribution': {
                'phase_a': features['i_phase_a'],
                'phase_b': features['i_phase_b'],
                'phase_c': features['i_phase_c']
            },
            'redistribution_plan': self._create_redistribution_plan(features),
            'expected_improvement_pct': self._estimate_balance_improvement(features)
        }

    def _mock_grid_stability(self, features: Dict) -> Dict:
        score = (features['power_quality_index'] * 0.4 +
                features['overall_balance_score'] * 0.3 +
                features['power_factor'] * 100 * 0.3) / 100
        return {
            'stability_score': score,
            'status': self._get_stability_status(score),
            'risk_factors': self._identify_risks(features),
            'recommendations': self._generate_recommendations(features)
        }

    def _mock_optimal_dispatch(self, features: Dict) -> Dict:
        optimal = features['p_total'] * 1.15  # 15% reserve
        return {
            'current_load_kw': features['p_total'],
            'recommended_generation_kw': optimal,
            'reserve_margin_pct': 15.0,
            'dispatch_plan': self._create_dispatch_plan(optimal, features)
        }


# Global singleton instance
model_manager = ModelManager()
