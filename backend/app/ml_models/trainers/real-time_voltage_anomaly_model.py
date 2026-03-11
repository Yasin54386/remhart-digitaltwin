import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


# ============================================================
# REAL-TIME VOLTAGE ANOMALY DETECTION (VAD) — TRAINING PIPELINE
# Unsupervised / Context-aware multivariate anomaly detection
#
# INPUT:
#   training_set/real_time_voltage_anomaly_data.xlsx
#
# OUTPUTS:
#   trained/voltage_anomaly_iforest.joblib
#   model_results/real_time_voltage_anomaly/run_<timestamp>/*
#       01_voltage_timeseries_with_anomalies.png
#       02_anomaly_score_distribution.png
#       03_event_density_by_hour.png
#       04_phase_deviation_during_anomalies.png
#       metrics.json
#
# NOTE:
#   - No ground-truth labels are required.
#   - Target is derived operationally:
#       anomaly_score  ->  is_anomaly (binary) using thresholding.
# ============================================================


def _safe_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _time_aware_split(df: pd.DataFrame, test_size: float = 0.20):
    """
    Chronological hold-out split (deployment-realistic).
    Train on early window, evaluate on later window.
    """
    n = len(df)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    return train_df, test_df


def _add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Explicit feature engineering for voltage stability:
      - Phase symmetry: VUF% proxy via max deviation from mean
      - Stability deltas: absolute change in average voltage
      - Phase-to-phase deviations: |A-B|, |B-C|, |A-C|
      - Rolling stability: rolling mean/std of average voltage
      - Socio-temporal: hour + cyclical encoding + is_workday
    """
    out = df.copy()

    # --- Core raw inputs ---
    for c in ["volt_A", "volt_B", "volt_C"]:
        if c not in out.columns:
            raise KeyError(f"Missing required column: {c}")

    # Average voltage (system baseline)
    out["v_avg"] = out[["volt_A", "volt_B", "volt_C"]].mean(axis=1)

    # Voltage unbalance (max deviation from average) and VUF% proxy
    out["v_unbalance_max_dev"] = out[["volt_A", "volt_B", "volt_C"]].sub(out["v_avg"], axis=0).abs().max(axis=1)
    out["vuf_pct"] = (out["v_unbalance_max_dev"] / out["v_avg"].replace(0, np.nan)) * 100.0

    # Hourly stability delta (captures sudden shifts)
    out["v_stability_delta"] = out["v_avg"].diff().abs()

    # Phase-to-phase absolute deviations
    out["v_ab_dev"] = (out["volt_A"] - out["volt_B"]).abs()
    out["v_bc_dev"] = (out["volt_B"] - out["volt_C"]).abs()
    out["v_ac_dev"] = (out["volt_A"] - out["volt_C"]).abs()

    # Rolling stability indicators (short horizon to preserve real events)
    # Using a conservative window (6) assuming ~10-min resolution (≈1 hour).
    window = 6
    out["v_avg_roll_mean"] = out["v_avg"].rolling(window=window, min_periods=1).mean()
    out["v_avg_roll_std"] = out["v_avg"].rolling(window=window, min_periods=1).std()

    # Socio-temporal features (hour + cyclical encoding + workday flag)
    if "Timestamp" in out.columns:
        out["Timestamp"] = _safe_datetime(out["Timestamp"])
        out["hour"] = out["Timestamp"].dt.hour
        out["is_workday"] = (out["Timestamp"].dt.dayofweek < 5).astype(int)
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
    else:
        # Fallback if Timestamp is absent
        out["hour"] = 0
        out["is_workday"] = 1
        out["hour_sin"] = 0.0
        out["hour_cos"] = 1.0

    return out


def train_real_time_voltage_anomaly_detector():
    # --------------------------------------------------------
    # 1) DIRECTORY SETUP (CONSISTENT WITH YOUR OTHER TRAINERS)
    # --------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "training_set", "real_time_voltage_anomaly_data.xlsx")
    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "real_time_voltage_anomaly")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # 2) LOAD & CLEAN
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_excel(input_path).replace([np.inf, -np.inf], np.nan).dropna().copy()

    if "Timestamp" in df.columns:
        df["Timestamp"] = _safe_datetime(df["Timestamp"])
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # 3) FEATURE ENGINEERING (PHASE SYMMETRY + STABILITY + CONTEXT)
    # --------------------------------------------------------
    df = _add_engineered_features(df)

    # Final feature set for Isolation Forest (explicit + stable)
    features = [
        "volt_A", "volt_B", "volt_C",
        "vuf_pct", "v_stability_delta",
        "v_ab_dev", "v_bc_dev", "v_ac_dev",
        "v_avg_roll_mean", "v_avg_roll_std",
        "hour_sin", "hour_cos", "is_workday"
    ]

    # Ensure feature availability
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing engineered features: {missing}")

    # Clean feature matrix
    X_all = df[features].copy()
    X_all = X_all.replace([np.inf, -np.inf], np.nan).fillna(0)

    # --------------------------------------------------------
    # 4) TIME-AWARE SPLIT (DEPLOYMENT-REALISTIC EVALUATION)
    # --------------------------------------------------------
    train_df, test_df = _time_aware_split(df, test_size=0.20)

    X_train = train_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    X_test = test_df[features].replace([np.inf, -np.inf], np.nan).fillna(0)

    # --------------------------------------------------------
    # 5) SCALE (ROBUST) + TRAIN ISOLATION FOREST
    # --------------------------------------------------------
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Contamination tuning (unsupervised, operationally bounded)
    # Goal: keep anomaly rate realistic (avoid alert flooding), while retaining sensitivity.
    candidate_contaminations = [0.005, 0.01, 0.02, 0.03]

    best = None
    for cont in candidate_contaminations:
        model = IsolationForest(
            n_estimators=300,
            contamination=cont,
            max_samples="auto",
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train_scaled)

        # decision_function: higher = more normal; lower = more anomalous
        train_norm_score = model.decision_function(X_train_scaled)
        test_norm_score = model.decision_function(X_test_scaled)

        # Use model’s contamination boundary implicitly via predict
        test_pred = model.predict(X_test_scaled)  # -1 anomaly, +1 normal
        test_anom_rate = float(np.mean(test_pred == -1))

        # Separation proxy: difference between normal mean score and anomaly mean score (test)
        if np.any(test_pred == -1) and np.any(test_pred == 1):
            sep = float(np.mean(test_norm_score[test_pred == 1]) - np.mean(test_norm_score[test_pred == -1]))
        else:
            sep = 0.0

        # Heuristic selection:
        #   - anomaly rate between 0.5% and 5% preferred
        #   - maximise score separation within acceptable rate
        acceptable = (test_anom_rate >= 0.005) and (test_anom_rate <= 0.05)
        score = sep if acceptable else (sep * 0.2)

        record = {
            "contamination": cont,
            "test_anomaly_rate": test_anom_rate,
            "separation_proxy": sep,
            "score": score,
            "model": model
        }

        if (best is None) or (record["score"] > best["score"]):
            best = record

    if best is None:
        raise RuntimeError("Isolation Forest tuning failed unexpectedly.")

    iforest = best["model"]
    chosen_contamination = best["contamination"]

    # --------------------------------------------------------
    # 6) TARGET DERIVATION (OPERATIONAL LABELS)
    # --------------------------------------------------------
    # anomaly_score: use "normality score" (decision_function)
    # lower = more anomalous
    df["anomaly_score"] = iforest.decision_function(scaler.transform(X_all))

    # derive binary anomaly flag using model boundary (predict)
    df["is_anomaly"] = (iforest.predict(scaler.transform(X_all)) == -1).astype(int)

    # --------------------------------------------------------
    # 7) METRICS (UNSUPERVISED, OPERATIONAL)
    # --------------------------------------------------------
    total_rows = int(len(df))
    anomaly_count = int(df["is_anomaly"].sum())
    anomaly_rate = float(anomaly_count / max(total_rows, 1))

    # Hourly distribution (sanity check for socio-temporal correlation)
    if "hour" in df.columns:
        hourly_counts = df.loc[df["is_anomaly"] == 1, "hour"].value_counts().sort_index().to_dict()
    else:
        hourly_counts = {}

    metrics = {
        "Model": "IsolationForest",
        "Features_used": features,
        "Train_rows": int(len(train_df)),
        "Test_rows": int(len(test_df)),
        "Time_aware_split": True,
        "Chosen_contamination": float(chosen_contamination),
        "Total_rows_scored": total_rows,
        "Anomaly_count": anomaly_count,
        "Anomaly_rate": anomaly_rate,
        "Anomaly_score_definition": "IsolationForest decision_function (higher=more normal, lower=more anomalous)",
        "Anomaly_flag_definition": "is_anomaly = 1 if IsolationForest predicts -1",
        "Hourly_anomaly_counts": hourly_counts
    }

    # --------------------------------------------------------
    # 8) VISUALISATION OUTPUTS (PROFESSIONAL TITLES)
    # --------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 130

    # Prepare axis
    if "Timestamp" in df.columns:
        x = df["Timestamp"]
        x_label = "Time"
    else:
        x = np.arange(len(df))
        x_label = "Sample Index"

    # Visual 01 — Voltage time-series with anomaly flags
    plt.figure(figsize=(13, 6))
    plt.plot(x, df["volt_A"], linewidth=1.2, label="Voltage Phase A")
    plt.plot(x, df["volt_B"], linewidth=1.2, label="Voltage Phase B")
    plt.plot(x, df["volt_C"], linewidth=1.2, label="Voltage Phase C")

    anom_idx = df["is_anomaly"] == 1
    if np.any(anom_idx):
        plt.scatter(
            x[anom_idx],
            df.loc[anom_idx, "v_avg"],
            s=18,
            marker="x",
            label="Detected Anomaly (Marker)"
        )

    plt.title("Real-Time Voltage Behaviour with Detected Anomaly Events")
    plt.xlabel(x_label)
    plt.ylabel("Voltage (V)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_voltage_timeseries_with_anomalies.png")
    plt.close()

    # Visual 02 — Anomaly score distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df["anomaly_score"], kde=True, bins=60)
    plt.title("Distribution of Voltage Anomaly Scores (Normality Index)")
    plt.xlabel("Anomaly Score (Higher = More Normal)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_anomaly_score_distribution.png")
    plt.close()

    # Visual 03 — Event density by hour
    if "hour" in df.columns:
        hourly = (
            df.loc[df["is_anomaly"] == 1, "hour"]
            .value_counts()
            .reindex(range(24), fill_value=0)
        )
        plt.figure(figsize=(10, 5))
        plt.bar(hourly.index, hourly.values)
        plt.title("Detected Voltage Anomaly Event Density by Hour of Day")
        plt.xlabel("Hour of Day")
        plt.ylabel("Number of Anomaly Events")
        plt.xticks(range(0, 24, 1))
        plt.tight_layout()
        plt.savefig(f"{run_dir}/03_event_density_by_hour.png")
        plt.close()

    # Visual 04 — Phase deviation during anomalies (distribution)
    plt.figure(figsize=(10, 5))
    plot_df = df.copy()
    plot_df["State"] = np.where(plot_df["is_anomaly"] == 1, "Anomalous", "Normal")
    sns.kdeplot(data=plot_df, x="vuf_pct", hue="State", common_norm=False)
    plt.title("Voltage Unbalance Factor (VUF%) Behaviour: Normal vs Anomalous States")
    plt.xlabel("VUF (%)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/04_phase_deviation_during_anomalies.png")
    plt.close()

    # --------------------------------------------------------
    # 9) SAVE MODEL + METADATA
    # --------------------------------------------------------
    artifact = {
        "model": iforest,
        "scaler": scaler,
        "features": features,
        "chosen_contamination": chosen_contamination,
    }

    model_path = os.path.join(model_dir, "voltage_anomaly_iforest.joblib")
    joblib.dump(artifact, model_path)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Also export scored dataset snapshot for auditability
    df.to_excel(os.path.join(run_dir, "scored_voltage_anomalies.xlsx"), index=False)

    print("\nReal-Time Voltage Anomaly Detection Training Complete")
    print(f"Model artifact saved: {model_path}")
    print(f"Run results saved:   {run_dir}")
    print(json.dumps(metrics, indent=4))

    return metrics


if __name__ == "__main__":
    train_real_time_voltage_anomaly_detector()
