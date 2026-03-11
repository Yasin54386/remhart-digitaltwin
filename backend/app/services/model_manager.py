"""
Model Manager
Loads and manages the 9 trained ML models for REMHART Digital Twin.
All models were trained on real data from master_data.xlsx.
"""

import logging
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent.parent / "ml_models" / "trained"

# Overload risk class index → label
OVERLOAD_CLASSES = {0: "Low", 1: "Medium", 2: "High"}


def _pqi_grade(score: float) -> str:
    if score >= 90: return "Excellent"
    if score >= 70: return "Good"
    if score >= 50: return "Fair"
    return "Poor"


class ModelManager:
    """
    Singleton that loads all 9 trained models and provides a clean
    predict() interface for the ML inference engine.
    """

    _instance = None
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ModelManager._initialized:
            return
        self.models = {}
        self._load_all()
        ModelManager._initialized = True

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, key: str, filename: str):
        path = MODELS_DIR / filename
        if path.exists():
            try:
                self.models[key] = joblib.load(path)
                logger.info(f"Loaded {key} from {filename}")
            except Exception as e:
                logger.warning(f"Failed to load {key}: {e}")
                self.models[key] = None
        else:
            logger.warning(f"Model file not found: {filename}")
            self.models[key] = None

    def _load_all(self):
        logger.info("Loading REMHART ML models...")
        self._load("voltage_anomaly",      "voltage_anomaly_iforest.joblib")
        self._load("energy_anomaly",       "anomaly_detection_model.joblib")
        self._load("equipment_failure",    "equipment_failure_model.joblib")
        self._load("overload_risk",        "overload_risk_model.joblib")
        self._load("epqi_score",           "epqi_score_model.joblib")
        self._load("voltage_sag",          "voltage_sag_xgb_regressor.joblib")
        self._load("load_forecasting",     "load_forecasting_model.joblib")
        self._load("health_index",         "energy_imbalance_and_health_index_model.joblib")
        self._load("reactive_q_forecast",  "reactive_q_forecast_model.joblib")
        self._load("pf_forecast",          "pf_forecast_model.joblib")
        loaded = sum(1 for m in self.models.values() if m is not None)
        logger.info(f"Loaded {loaded}/{len(self.models)} models")

    # ------------------------------------------------------------------
    # 1. Real-time Monitoring: Voltage Anomaly (IsolationForest + Scaler)
    #    13 features: volt_A/B/C, vuf_pct, v_stability_delta,
    #                 v_ab/bc/ac_dev, v_avg_roll_mean/std,
    #                 hour_sin, hour_cos, is_workday
    # ------------------------------------------------------------------

    def predict_voltage_anomaly(self, f: Dict) -> Dict:
        bundle = self.models.get("voltage_anomaly")
        v_a, v_b, v_c = f["volt_A"], f["volt_B"], f["volt_C"]
        v_avg = (v_a + v_b + v_c) / 3.0

        vuf_pct = (max(abs(v_a - v_avg), abs(v_b - v_avg), abs(v_c - v_avg)) / max(v_avg, 1)) * 100
        v_stability_delta = f.get("v_stability_delta", 0.0)
        v_ab_dev = abs(v_a - v_b)
        v_bc_dev = abs(v_b - v_c)
        v_ac_dev = abs(v_a - v_c)
        v_avg_roll_mean = f.get("v_avg_roll_mean", v_avg)
        v_avg_roll_std  = f.get("v_avg_roll_std", 0.0)
        hour_sin  = f.get("hour_sin", 0.0)
        hour_cos  = f.get("hour_cos", 1.0)
        is_workday = float(f.get("is_workday", 1))

        X = np.array([[v_a, v_b, v_c, vuf_pct, v_stability_delta,
                        v_ab_dev, v_bc_dev, v_ac_dev,
                        v_avg_roll_mean, v_avg_roll_std,
                        hour_sin, hour_cos, is_workday]])

        if bundle is not None:
            try:
                X_scaled = bundle["scaler"].transform(X)
                pred  = bundle["model"].predict(X_scaled)[0]
                score = bundle["model"].score_samples(X_scaled)[0]
                is_anomaly = bool(pred == -1)
                severity = "Critical" if score < -0.3 else "Warning" if score < -0.1 else "Normal"
                return {
                    "is_anomaly": is_anomaly,
                    "anomaly_score": round(float(score), 4),
                    "vuf_pct": round(vuf_pct, 3),
                    "v_avg": round(v_avg, 2),
                    "severity": severity,
                }
            except Exception as e:
                logger.warning(f"voltage_anomaly predict failed: {e}")

        is_anomaly = vuf_pct > 3.0
        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(-vuf_pct / 10, 4),
            "vuf_pct": round(vuf_pct, 3),
            "v_avg": round(v_avg, 2),
            "severity": "Warning" if is_anomaly else "Normal",
        }

    # ------------------------------------------------------------------
    # 2. Real-time Monitoring: Energy / Multivariate Anomaly (XGBClassifier)
    #    11 features: volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T, Frec, hour
    # ------------------------------------------------------------------

    def predict_energy_anomaly(self, f: Dict) -> Dict:
        model = self.models.get("energy_anomaly")
        X = np.array([[
            f["volt_A"], f["volt_B"], f["volt_C"],
            f["I_A"],    f["I_B"],    f["I_C"],
            f["P_T"],    f["Q_T"],    f["FP_T"],
            f["Frec"],   f.get("hour", 0)
        ]])

        if model is not None:
            try:
                pred  = int(model.predict(X)[0])
                proba = float(model.predict_proba(X)[0][1])
                severity = "Critical" if proba > 0.7 else "Warning" if proba > 0.4 else "Normal"
                return {
                    "is_anomaly": bool(pred == 1),
                    "anomaly_probability": round(proba, 4),
                    "severity": severity,
                }
            except Exception as e:
                logger.warning(f"energy_anomaly predict failed: {e}")

        anomaly = f["FP_T"] < 0.7 or abs(f["volt_A"] - f["volt_B"]) > 10
        return {
            "is_anomaly": anomaly,
            "anomaly_probability": 0.6 if anomaly else 0.1,
            "severity": "Warning" if anomaly else "Normal",
        }

    # ------------------------------------------------------------------
    # 3. Predictive Maintenance: Equipment Failure (XGBRegressor)
    #    12 features: volt_A/B/C, I_A/B/C, P_A/B/C, Q_A/B/C
    #    Output: failure_probability (0–1)
    # ------------------------------------------------------------------

    def predict_equipment_failure(self, f: Dict) -> Dict:
        model = self.models.get("equipment_failure")
        X = np.array([[
            f["volt_A"], f["volt_B"], f["volt_C"],
            f["I_A"],    f["I_B"],    f["I_C"],
            f["P_A"],    f["P_B"],    f["P_C"],
            f["Q_A"],    f["Q_B"],    f["Q_C"],
        ]])

        prob = 0.0
        if model is not None:
            try:
                prob = float(np.clip(model.predict(X)[0], 0.0, 1.0))
            except Exception as e:
                logger.warning(f"equipment_failure predict failed: {e}")

        risk = "High" if prob > 0.7 else "Medium" if prob > 0.4 else "Low"
        return {
            "failure_probability": round(prob, 4),
            "risk_level": risk,
            "estimated_days_to_failure": max(1, int((1 - prob) * 90)),
        }

    # ------------------------------------------------------------------
    # 4. Predictive Maintenance: Overload Risk (XGBClassifier, 3 classes)
    #    9 features: volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T
    #    Output: 0=Low, 1=Medium, 2=High
    # ------------------------------------------------------------------

    def predict_overload_risk(self, f: Dict) -> Dict:
        model = self.models.get("overload_risk")
        X = np.array([[
            f["volt_A"], f["volt_B"], f["volt_C"],
            f["I_A"],    f["I_B"],    f["I_C"],
            f["P_T"],    f["Q_T"],    f["FP_T"],
        ]])

        i_avg = (f["I_A"] + f["I_B"] + f["I_C"]) / 3.0
        i_max = max(f["I_A"], f["I_B"], f["I_C"])
        load_pct = round((i_max / max(i_avg * 1.5, 0.01)) * 100, 2)
        peak_phase = "A" if f["I_A"] >= f["I_B"] and f["I_A"] >= f["I_C"] else \
                     "B" if f["I_B"] >= f["I_C"] else "C"

        if model is not None:
            try:
                cls   = int(model.predict(X)[0])
                proba = model.predict_proba(X)[0]
                risk  = OVERLOAD_CLASSES.get(cls, "Low")
                return {
                    "risk_level": risk,
                    "risk_index": cls,
                    "probabilities": {k: round(float(v), 3) for k, v in zip(["Low","Medium","High"], proba)},
                    "load_percentage": load_pct,
                    "peak_phase": peak_phase,
                }
            except Exception as e:
                logger.warning(f"overload_risk predict failed: {e}")

        risk = "High" if load_pct > 90 else "Medium" if load_pct > 80 else "Low"
        return {
            "risk_level": risk,
            "risk_index": {"Low":0,"Medium":1,"High":2}[risk],
            "probabilities": {"Low":0.8,"Medium":0.1,"High":0.1},
            "load_percentage": load_pct,
            "peak_phase": peak_phase,
        }

    # ------------------------------------------------------------------
    # 5. Predictive Maintenance: Equipment PQI Score (XGBRegressor)
    #    10 features: volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T, Frec
    #    Output: PQI_Score (0–100)
    # ------------------------------------------------------------------

    def predict_epqi_score(self, f: Dict) -> Dict:
        model = self.models.get("epqi_score")
        X = np.array([[
            f["volt_A"], f["volt_B"], f["volt_C"],
            f["I_A"],    f["I_B"],    f["I_C"],
            f["P_T"],    f["Q_T"],    f["FP_T"],
            f["Frec"],
        ]])

        score = 75.0
        if model is not None:
            try:
                score = float(np.clip(model.predict(X)[0], 0.0, 100.0))
            except Exception as e:
                logger.warning(f"epqi_score predict failed: {e}")

        return {
            "pqi_score": round(score, 1),
            "grade": _pqi_grade(score),
        }

    # ------------------------------------------------------------------
    # 6. Predictive Maintenance: Voltage Sag (XGBRegressor dict bundle)
    #    19 features: volt_A/B/C, I_A/B/C, P_A/B/C/T, Q_A/B/C/T,
    #                 FP_A/B/C/T, Frec
    #    Output: voltage_sag_probability (0.0–1.0)
    # ------------------------------------------------------------------

    def predict_voltage_sag(self, f: Dict) -> Dict:
        bundle = self.models.get("voltage_sag")
        X = np.array([[
            f["volt_A"], f["volt_B"], f["volt_C"],
            f["I_A"],    f["I_B"],    f["I_C"],
            f["P_A"],    f["P_B"],    f["P_C"],    f["P_T"],
            f["Q_A"],    f["Q_B"],    f["Q_C"],    f["Q_T"],
            f["FP_A"],   f["FP_B"],   f["FP_C"],   f["FP_T"],
            f["Frec"],
        ]])

        prob = 0.0
        if bundle is not None:
            try:
                prob = float(np.clip(bundle["model"].predict(X)[0], 0.0, 1.0))
            except Exception as e:
                logger.warning(f"voltage_sag predict failed: {e}")

        risk = "High" if prob >= 0.75 else "Medium" if prob >= 0.5 else "Precursor" if prob >= 0.25 else "Low"
        return {
            "sag_probability": round(prob, 4),
            "risk_level": risk,
        }

    # ------------------------------------------------------------------
    # 7. Energy Flow: Load Forecasting (XGBRegressor, 40 features)
    # ------------------------------------------------------------------

    def predict_load_forecast(self, f: Dict) -> Dict:
        model = self.models.get("load_forecasting")
        feature_order = [
            "volt_A","volt_B","volt_C","I_A","I_B","I_C",
            "P_A","P_B","P_C","P_T","Q_A","Q_B","Q_C","Q_T",
            "FP_A","FP_B","FP_C","FP_T","Frec",
            "hour_of_day","day_of_week","day_of_month","month",
            "hour_sin","hour_cos","dayofweek_sin","dayofweek_cos",
            "is_weekend","is_public_holiday",
            "load_lag_1h","load_lag_24h","load_lag_168h",
            "load_roll_mean_3h","load_roll_mean_6h","load_roll_mean_24h",
            "load_roll_std_3h","load_roll_std_6h","load_roll_std_24h",
            "load_roll_min_24h","load_roll_max_24h",
        ]
        X = np.array([[f.get(k, 0.0) for k in feature_order]])

        forecast_w = f.get("P_T", 0.0)
        if model is not None:
            try:
                forecast_w = float(model.predict(X)[0])
            except Exception as e:
                logger.warning(f"load_forecasting predict failed: {e}")

        current_w = f.get("P_T", 0.0)
        return {
            "current_load_kw": round(current_w / 1000, 3),
            "forecast_next_kw": round(forecast_w / 1000, 3),
            "trend": "rising"  if forecast_w > current_w * 1.02 else
                     "falling" if forecast_w < current_w * 0.98 else "stable",
        }

    # ------------------------------------------------------------------
    # 8. Energy Flow: Energy Imbalance & Health Index (XGBRegressor)
    #    10 features: volt_A/B/C, I_A/B/C, P_T, Q_T, FP_T, Frec
    #    Output: health_score (0–100)
    # ------------------------------------------------------------------

    def predict_health_index(self, f: Dict) -> Dict:
        model = self.models.get("health_index")
        X = np.array([[
            f["volt_A"], f["volt_B"], f["volt_C"],
            f["I_A"],    f["I_B"],    f["I_C"],
            f["P_T"],    f["Q_T"],    f["FP_T"],
            f["Frec"],
        ]])

        health = 80.0
        if model is not None:
            try:
                health = float(np.clip(model.predict(X)[0], 0.0, 100.0))
            except Exception as e:
                logger.warning(f"health_index predict failed: {e}")

        v_avg = (f["volt_A"] + f["volt_B"] + f["volt_C"]) / 3.0
        i_avg = (f["I_A"] + f["I_B"] + f["I_C"]) / 3.0
        calc_p = (np.sqrt(3) * v_avg * i_avg * f["FP_T"]) / 1000.0
        imbalance_err = abs(f["P_T"] / 1000.0 - calc_p)
        i_unbalance_pct = ((max(f["I_A"], f["I_B"], f["I_C"]) - i_avg) / max(i_avg, 1)) * 100

        status = "Excellent" if health >= 90 else "Good" if health >= 70 else "Fair" if health >= 50 else "Poor"
        return {
            "health_score": round(health, 1),
            "status": status,
            "energy_imbalance_kw": round(imbalance_err, 3),
            "current_unbalance_pct": round(i_unbalance_pct, 2),
            "imbalance_loss_flag": bool(i_unbalance_pct > 15),
        }

    # ------------------------------------------------------------------
    # 9. Decision Making: Reactive Q + PF Forecast (2 XGBRegressors)
    #    16 features: Q_lag_1h, Q_lag_24h, PF_lag_1h, hour_sin, hour_cos,
    #                 is_workday, P_T, volt_A/B/C, I_A/B/C, Frec, FP_T, Q_T
    # ------------------------------------------------------------------

    def predict_reactive_and_pf(self, f: Dict) -> Dict:
        model_q  = self.models.get("reactive_q_forecast")
        model_pf = self.models.get("pf_forecast")

        feature_order = [
            "Q_lag_1h", "Q_lag_24h", "PF_lag_1h",
            "hour_sin", "hour_cos", "is_workday",
            "P_T", "volt_A", "volt_B", "volt_C",
            "I_A", "I_B", "I_C",
            "Frec", "FP_T", "Q_T",
        ]
        X = np.array([[f.get(k, 0.0) for k in feature_order]])

        q_next  = f.get("Q_T", 0.0)
        pf_next = f.get("FP_T", 0.95)

        if model_q is not None:
            try:
                q_next = float(model_q.predict(X)[0])
            except Exception as e:
                logger.warning(f"reactive_q_forecast predict failed: {e}")

        if model_pf is not None:
            try:
                pf_next = float(np.clip(model_pf.predict(X)[0], 0.0, 1.0))
            except Exception as e:
                logger.warning(f"pf_forecast predict failed: {e}")

        comp_kvar = max(0.0, f.get("Q_T", 0.0) - q_next)
        return {
            "current_q_kvar": round(f.get("Q_T", 0) / 1000, 3),
            "forecast_q_next_kvar": round(q_next / 1000, 3),
            "current_pf": round(f.get("FP_T", 0), 4),
            "forecast_pf_next": round(pf_next, 4),
            "compensation_needed_kvar": round(comp_kvar / 1000, 3),
            "pf_status": "Good" if pf_next >= 0.95 else "Warning" if pf_next >= 0.85 else "Poor",
        }


# Singleton
model_manager = ModelManager()
