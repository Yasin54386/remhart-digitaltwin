"""
ML Inference Engine
===================
Runs all 9 trained models on a new data point and returns structured
predictions organised by the 4 frontend dashboard pages.

Page → Models
─────────────────────────────────────────────────────────────────────
realtime_monitoring   : voltage_anomaly, energy_anomaly
predictive_maintenance: equipment_failure, overload_risk,
                        epqi_score, voltage_sag
energy_flow           : load_forecasting, health_index
decision_making       : reactive_q + pf_forecast (combined)
"""

import logging
from typing import Dict, Any, List, Optional

from .feature_engineering import feature_extractor
from .model_manager import model_manager

logger = logging.getLogger(__name__)


class MLInferenceEngine:

    def process(
        self,
        record,
        history: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Run all 9 models on *record*.

        Parameters
        ----------
        record  : DateTimeTable ORM object
        history : recent records (oldest→newest) for lag features.
                  Should include *record* as the last element.
                  If None, lag features fall back to current values.

        Returns
        -------
        {
            "realtime_monitoring":    {...},
            "predictive_maintenance": {...},
            "energy_flow":            {...},
            "decision_making":        {...},
            "metadata":               {...},
        }
        """
        try:
            f = feature_extractor.extract(record, history)

            return {
                "realtime_monitoring":    self._monitoring(f),
                "predictive_maintenance": self._maintenance(f),
                "energy_flow":            self._energy(f),
                "decision_making":        self._decision(f),
                "metadata": {
                    "timestamp":    record.timestamp,
                    "is_simulation": record.is_simulation,
                    "data_quality": self._quality(record),
                },
            }

        except Exception as e:
            logger.error(f"Inference error: {e}", exc_info=True)
            return self._error_response(str(e))

    # ------------------------------------------------------------------
    # Page 1 — Real-time Monitoring
    # ------------------------------------------------------------------

    def _monitoring(self, f: Dict) -> Dict:
        return {
            "voltage_anomaly": model_manager.predict_voltage_anomaly(f),
            "energy_anomaly":  model_manager.predict_energy_anomaly(f),
        }

    # ------------------------------------------------------------------
    # Page 2 — Predictive Maintenance
    # ------------------------------------------------------------------

    def _maintenance(self, f: Dict) -> Dict:
        return {
            "equipment_failure": model_manager.predict_equipment_failure(f),
            "overload_risk":     model_manager.predict_overload_risk(f),
            "epqi_score":        model_manager.predict_epqi_score(f),
            "voltage_sag":       model_manager.predict_voltage_sag(f),
        }

    # ------------------------------------------------------------------
    # Page 3 — Energy Flow
    # ------------------------------------------------------------------

    def _energy(self, f: Dict) -> Dict:
        return {
            "load_forecasting": model_manager.predict_load_forecast(f),
            "health_index":     model_manager.predict_health_index(f),
        }

    # ------------------------------------------------------------------
    # Page 4 — Decision Making
    # ------------------------------------------------------------------

    def _decision(self, f: Dict) -> Dict:
        return {
            "reactive_and_pf": model_manager.predict_reactive_and_pf(f),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _quality(self, record) -> str:
        checks = [
            bool(record.voltage),
            bool(record.current),
            bool(record.frequency),
            bool(record.active_power),
            bool(record.reactive_power),
        ]
        n = sum(checks)
        return "Excellent" if n == 5 else "Good" if n == 4 else "Fair" if n >= 2 else "Poor"

    def _error_response(self, msg: str) -> Dict:
        return {
            "error": True,
            "message": msg,
            "realtime_monitoring":    {},
            "predictive_maintenance": {},
            "energy_flow":            {},
            "decision_making":        {},
            "metadata": {
                "timestamp": None,
                "is_simulation": False,
                "data_quality": "Error",
            },
        }


# Singleton
ml_inference_engine = MLInferenceEngine()
