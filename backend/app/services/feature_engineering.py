"""
Feature Engineering
===================
Converts raw DB records into the exact feature dictionaries that each of
the 9 trained models expects.

All feature names match the training column names from master_data.xlsx:
    volt_A, volt_B, volt_C
    I_A, I_B, I_C
    P_A, P_B, P_C, P_T
    Q_A, Q_B, Q_C, Q_T
    FP_A, FP_B, FP_C, FP_T   ← computed from P/Q (not stored in DB)
    Frec
"""

import math
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(val, default=0.0) -> float:
    """Return float or default if None / NaN."""
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def _compute_pf(p: float, q: float) -> float:
    """Power factor from active and reactive power."""
    apparent = math.sqrt(p ** 2 + q ** 2)
    return p / apparent if apparent > 1e-9 else 1.0


def _hour_sin_cos(hour: int):
    angle = 2 * math.pi * hour / 24
    return math.sin(angle), math.cos(angle)


def _dow_sin_cos(dow: int):
    angle = 2 * math.pi * dow / 7
    return math.sin(angle), math.cos(angle)


# ---------------------------------------------------------------------------
# Feature Extractor
# ---------------------------------------------------------------------------

class FeatureExtractor:
    """
    Extracts a flat feature dictionary from a DateTimeTable ORM record.

    The extractor also accepts an optional list of recent records so that
    lag / rolling-window features can be computed for forecasting models.
    """

    def extract(
        self,
        record,
        history: Optional[List] = None,
    ) -> Dict[str, Any]:
        """
        Parameters
        ----------
        record  : DateTimeTable ORM object (with .voltage, .current, etc.)
        history : ordered list of recent DateTimeTable records
                  (oldest first, includes *record* as last element).
                  Required for lag features; if None, lags default to
                  the current P_T / Q_T value.

        Returns
        -------
        Flat dict of all feature values needed by every model.
        """

        # ── Raw measurements ────────────────────────────────────────────
        v = record.voltage[0]   if record.voltage    else None
        c = record.current[0]   if record.current    else None
        f = record.frequency[0] if record.frequency  else None
        p = record.active_power[0]   if record.active_power   else None
        q = record.reactive_power[0] if record.reactive_power else None

        volt_A = _safe(v.phaseA   if v else 0)
        volt_B = _safe(v.phaseB   if v else 0)
        volt_C = _safe(v.phaseC   if v else 0)

        I_A = _safe(c.phaseA if c else 0)
        I_B = _safe(c.phaseB if c else 0)
        I_C = _safe(c.phaseC if c else 0)

        Frec = _safe(f.frequency_value if f else 50.0, 50.0)

        P_A = _safe(p.phaseA if p else 0)
        P_B = _safe(p.phaseB if p else 0)
        P_C = _safe(p.phaseC if p else 0)
        P_T = _safe(p.total  if p else 0)

        Q_A = _safe(q.phaseA if q else 0)
        Q_B = _safe(q.phaseB if q else 0)
        Q_C = _safe(q.phaseC if q else 0)
        Q_T = _safe(q.total  if q else 0)

        # ── Derived: Power Factor (not in DB — computed from P, Q) ─────
        FP_A = _compute_pf(P_A, Q_A)
        FP_B = _compute_pf(P_B, Q_B)
        FP_C = _compute_pf(P_C, Q_C)
        FP_T = _compute_pf(P_T, Q_T)

        # ── Temporal ────────────────────────────────────────────────────
        ts: datetime = record.timestamp
        hour_of_day = ts.hour
        day_of_week = ts.weekday()          # 0=Mon … 6=Sun
        day_of_month = ts.day
        month = ts.month
        is_weekend = 1 if day_of_week >= 5 else 0
        is_workday  = 1 - is_weekend
        is_public_holiday = 0               # simplified; override if needed

        hour_sin, hour_cos = _hour_sin_cos(hour_of_day)
        dow_sin,  dow_cos  = _dow_sin_cos(day_of_week)

        # ── Voltage-anomaly extra features ──────────────────────────────
        v_avg = (volt_A + volt_B + volt_C) / 3.0
        v_stability_delta = 0.0
        v_avg_roll_mean   = v_avg
        v_avg_roll_std    = 0.0

        if history and len(history) >= 2:
            prev = history[-2]
            pv = prev.voltage[0] if prev.voltage else None
            if pv:
                prev_v_avg = (pv.phaseA + pv.phaseB + pv.phaseC) / 3.0
                v_stability_delta = abs(v_avg - prev_v_avg)

            # rolling mean/std over up to 18 points (~3 h at 10-min intervals)
            window = history[-18:]
            v_avgs = []
            for r in window:
                rv = r.voltage[0] if r.voltage else None
                if rv:
                    v_avgs.append((rv.phaseA + rv.phaseB + rv.phaseC) / 3.0)
            if v_avgs:
                v_avg_roll_mean = float(np.mean(v_avgs))
                v_avg_roll_std  = float(np.std(v_avgs))

        # ── Lag features for forecasting models ─────────────────────────
        # Data sampled every 10 min → 1h = 6 steps, 24h = 144, 168h = 1008

        def _lag_P_T(steps: int) -> float:
            if history and len(history) > steps:
                r = history[-(steps + 1)]
                pp = r.active_power[0] if r.active_power else None
                return _safe(pp.total if pp else P_T, P_T)
            return P_T

        def _lag_Q_T(steps: int) -> float:
            if history and len(history) > steps:
                r = history[-(steps + 1)]
                rq = r.reactive_power[0] if r.reactive_power else None
                return _safe(rq.total if rq else Q_T, Q_T)
            return Q_T

        def _lag_FP_T(steps: int) -> float:
            if history and len(history) > steps:
                r = history[-(steps + 1)]
                rp = r.active_power[0] if r.active_power else None
                rq = r.reactive_power[0] if r.reactive_power else None
                lp = _safe(rp.total if rp else P_T, P_T)
                lq = _safe(rq.total if rq else Q_T, Q_T)
                return _compute_pf(lp, lq)
            return FP_T

        load_lag_1h   = _lag_P_T(6)
        load_lag_24h  = _lag_P_T(144)
        load_lag_168h = _lag_P_T(1008)
        Q_lag_1h  = _lag_Q_T(6)
        Q_lag_24h = _lag_Q_T(144)
        PF_lag_1h = _lag_FP_T(6)

        # Rolling stats for load forecasting
        def _roll_P_T(window_steps: int):
            vals = []
            if history:
                for r in history[-window_steps:]:
                    rp = r.active_power[0] if r.active_power else None
                    if rp:
                        vals.append(_safe(rp.total))
            return vals if vals else [P_T]

        w3h  = _roll_P_T(18)    # 3h window
        w6h  = _roll_P_T(36)    # 6h window
        w24h = _roll_P_T(144)   # 24h window

        return {
            # ── Raw ─────────────────────────────────────────────────────
            "volt_A": volt_A, "volt_B": volt_B, "volt_C": volt_C,
            "I_A": I_A, "I_B": I_B, "I_C": I_C,
            "P_A": P_A, "P_B": P_B, "P_C": P_C, "P_T": P_T,
            "Q_A": Q_A, "Q_B": Q_B, "Q_C": Q_C, "Q_T": Q_T,
            "FP_A": FP_A, "FP_B": FP_B, "FP_C": FP_C, "FP_T": FP_T,
            "Frec": Frec,

            # ── Temporal ────────────────────────────────────────────────
            "hour": hour_of_day,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "month": month,
            "is_weekend": is_weekend,
            "is_workday": is_workday,
            "is_public_holiday": is_public_holiday,
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "dayofweek_sin": dow_sin, "dayofweek_cos": dow_cos,

            # ── Voltage-anomaly extras ───────────────────────────────────
            "v_stability_delta": v_stability_delta,
            "v_avg_roll_mean":   v_avg_roll_mean,
            "v_avg_roll_std":    v_avg_roll_std,

            # ── Load-forecasting lags / rolling ─────────────────────────
            "load_lag_1h":   load_lag_1h,
            "load_lag_24h":  load_lag_24h,
            "load_lag_168h": load_lag_168h,
            "load_roll_mean_3h":  float(np.mean(w3h)),
            "load_roll_mean_6h":  float(np.mean(w6h)),
            "load_roll_mean_24h": float(np.mean(w24h)),
            "load_roll_std_3h":   float(np.std(w3h)),
            "load_roll_std_6h":   float(np.std(w6h)),
            "load_roll_std_24h":  float(np.std(w24h)),
            "load_roll_min_24h":  float(np.min(w24h)),
            "load_roll_max_24h":  float(np.max(w24h)),

            # ── Reactive-forecast lags ───────────────────────────────────
            "Q_lag_1h":  Q_lag_1h,
            "Q_lag_24h": Q_lag_24h,
            "PF_lag_1h": PF_lag_1h,
        }


# Singleton
feature_extractor = FeatureExtractor()
