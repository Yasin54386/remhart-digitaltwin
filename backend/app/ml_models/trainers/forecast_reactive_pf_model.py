import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# REACTIVE POWER (Q) + POWER FACTOR (PF) FORECAST TRAINER
# Visuals (Selected):
#   01) Time-series: Actual vs Predicted (Q and PF)
#   02) Parity scatter: Actual vs Predicted (Q and PF)
#   05) Error distribution: Residual histogram (Q and PF)
#   08) Metrics summary bar chart (Q and PF)
#
# Outputs:
#   trained/reactive_q_forecast_model.joblib
#   trained/pf_forecast_model.joblib
#   model_results/reactive_pf_forecast/run_<timestamp>/*
#       01_timeseries_actual_vs_pred.png
#       02_parity_plots.png
#       05_error_distribution.png
#       08_metrics_summary_bar.png
#       metrics.json
# ============================================================


def _rmse(y_true, y_pred) -> float:
    return float(mean_squared_error(y_true, y_pred, squared=False))


def _mape_safe(y_true, y_pred, eps=1e-6) -> float:
    """
    Safe MAPE for signals that can be close to zero.
    Returns percent.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _time_aware_split(df: pd.DataFrame, test_size: float = 0.20):
    """
    Chronological hold-out split: last `test_size` portion as test.
    """
    n = len(df)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    return train_df, test_df


def train_reactive_and_pf_forecaster():
    # --------------------------------------------------------
    # 1) DIRECTORY SETUP (MATCHING YOUR STANDARD)
    # --------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(
        BASE_DIR,
        "training_set",
        "reactive_and_pf_forecast_data.xlsx"
    )

    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "reactive_pf_forecast")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # 2) LOAD & CLEAN (TIME-SERIES INTEGRITY)
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_excel(input_path).replace([np.inf, -np.inf], np.nan)

    # Timestamp is important for time-aware split + time-series plot
    if "Timestamp" in df.columns:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")

    df = df.dropna().copy()

    if "Timestamp" in df.columns:
        df = df.sort_values("Timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # 3) FEATURE SET & TARGETS (BASED ON YOUR PREPROCESSING)
    # --------------------------------------------------------
    # Expected engineered features and targets from your dataset generator:
    # - hour_sin, hour_cos, is_workday
    # - Q_lag_1h, Q_lag_24h, PF_lag_1h
    # - targets: target_Q_next, target_PF_next
    required_cols = [
        "Q_lag_1h", "Q_lag_24h", "PF_lag_1h",
        "hour_sin", "hour_cos", "is_workday",
        "target_Q_next", "target_PF_next"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns in dataset: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Optional multivariate context if available (won't break if missing)
    optional_cols = ["P_T", "volt_A", "volt_B", "volt_C", "I_A", "I_B", "I_C", "Frec", "FP_T", "Q_T"]
    features = [
        "Q_lag_1h", "Q_lag_24h", "PF_lag_1h",
        "hour_sin", "hour_cos", "is_workday"
    ]
    for c in optional_cols:
        if c in df.columns and c not in features:
            features.append(c)

    X_all = df[features].copy()
    y_q_all = df["target_Q_next"].astype(float).copy()
    y_pf_all = df["target_PF_next"].astype(float).copy()

    # --------------------------------------------------------
    # 4) TIME-AWARE TRAIN/TEST SPLIT (NO LEAKAGE)
    # --------------------------------------------------------
    train_df, test_df = _time_aware_split(df, test_size=0.20)

    X_train = train_df[features]
    X_test = test_df[features]

    y_q_train = train_df["target_Q_next"].astype(float)
    y_q_test = test_df["target_Q_next"].astype(float)

    y_pf_train = train_df["target_PF_next"].astype(float)
    y_pf_test = test_df["target_PF_next"].astype(float)

    # --------------------------------------------------------
    # 5) MODEL DEFINITION + INDUSTRY-SAFE OPTIMISATION
    # --------------------------------------------------------
    # TimeSeriesSplit keeps chronology in CV.
    tscv = TimeSeriesSplit(n_splits=5)

    base_reg = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        # Conservative defaults; tuning will refine
        n_estimators=800,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.0,
        reg_lambda=1.0
    )

    # Small but effective search space (stable + avoids under/overfitting)
    param_distributions = {
        "n_estimators": [600, 800, 1200],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.01, 0.03, 0.05],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.75, 0.85, 0.95],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.1, 0.2],
        "reg_alpha": [0.0, 0.1],
        "reg_lambda": [1.0, 2.0]
    }

    # Train Q model
    q_search = RandomizedSearchCV(
        estimator=base_reg,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    q_search.fit(X_train, y_q_train)
    q_model = q_search.best_estimator_

    # Train PF model
    pf_search = RandomizedSearchCV(
        estimator=base_reg,
        param_distributions=param_distributions,
        n_iter=25,
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    pf_search.fit(X_train, y_pf_train)
    pf_model = pf_search.best_estimator_

    # --------------------------------------------------------
    # 6) PREDICTION & METRICS
    # --------------------------------------------------------
    q_pred = q_model.predict(X_test)
    pf_pred = pf_model.predict(X_test)

    q_res = q_pred - y_q_test.values
    pf_res = pf_pred - y_pf_test.values

    metrics = {
        "ReactivePower_Q": {
            "MAE": float(mean_absolute_error(y_q_test, q_pred)),
            "RMSE": _rmse(y_q_test, q_pred),
            "R2_Score": float(r2_score(y_q_test, q_pred)),
            "MAPE_percent": _mape_safe(y_q_test, q_pred),
            "Best_params": q_search.best_params_
        },
        "PowerFactor_PF": {
            "MAE": float(mean_absolute_error(y_pf_test, pf_pred)),
            "RMSE": _rmse(y_pf_test, pf_pred),
            "R2_Score": float(r2_score(y_pf_test, pf_pred)),
            "MAPE_percent": _mape_safe(y_pf_test, pf_pred),
            "Best_params": pf_search.best_params_
        },
        "Data": {
            "Features_used": features,
            "Train_rows": int(len(train_df)),
            "Test_rows": int(len(test_df)),
            "Time_aware_split": True
        }
    }

    # --------------------------------------------------------
    # 7) VISUAL STYLE (PROFESSIONAL)
    # --------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 140

    # --------------------------------------------------------
    # VISUAL 01 — TIME-SERIES ACTUAL vs PREDICTED (Q and PF)
    # --------------------------------------------------------
    # Use timestamp if available; otherwise use sample index
    if "Timestamp" in test_df.columns:
        x_axis = test_df["Timestamp"]
        x_label = "Time"
    else:
        x_axis = np.arange(len(test_df))
        x_label = "Sample Index"

    plt.figure(figsize=(12, 7))
    plt.plot(x_axis, y_q_test.values, linewidth=2, label="Measured Q (Target)")
    plt.plot(x_axis, q_pred, linewidth=2, linestyle="--", label="Predicted Q")
    plt.title("Reactive Power Forecast Validation (Measured vs Predicted)")
    plt.xlabel(x_label)
    plt.ylabel("Reactive Power (Q)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_timeseries_actual_vs_pred_Q.png")
    plt.close()

    plt.figure(figsize=(12, 7))
    plt.plot(x_axis, y_pf_test.values, linewidth=2, label="Measured PF (Target)")
    plt.plot(x_axis, pf_pred, linewidth=2, linestyle="--", label="Predicted PF")
    plt.title("Power Factor Forecast Validation (Measured vs Predicted)")
    plt.xlabel(x_label)
    plt.ylabel("Power Factor (PF)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_timeseries_actual_vs_pred_PF.png")
    plt.close()

    # --------------------------------------------------------
    # VISUAL 02 — PARITY (SCATTER) PLOTS (Q and PF)
    # --------------------------------------------------------
    plt.figure(figsize=(7.2, 7.2))
    plt.scatter(y_q_test.values, q_pred, alpha=0.65)
    min_v = min(y_q_test.min(), q_pred.min())
    max_v = max(y_q_test.max(), q_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=2)
    plt.title("Parity Plot — Reactive Power Forecast (Q)")
    plt.xlabel("Measured Q")
    plt.ylabel("Predicted Q")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_parity_Q.png")
    plt.close()

    plt.figure(figsize=(7.2, 7.2))
    plt.scatter(y_pf_test.values, pf_pred, alpha=0.65)
    min_v = min(y_pf_test.min(), pf_pred.min())
    max_v = max(y_pf_test.max(), pf_pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=2)
    plt.title("Parity Plot — Power Factor Forecast (PF)")
    plt.xlabel("Measured PF")
    plt.ylabel("Predicted PF")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_parity_PF.png")
    plt.close()

    # --------------------------------------------------------
    # VISUAL 05 — ERROR DISTRIBUTION (Q and PF)
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(q_res, kde=True)
    plt.title("Forecast Error Distribution — Reactive Power (Q)")
    plt.xlabel("Residual Error (Predicted − Measured)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/05_error_distribution_Q.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.histplot(pf_res, kde=True)
    plt.title("Forecast Error Distribution — Power Factor (PF)")
    plt.xlabel("Residual Error (Predicted − Measured)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/05_error_distribution_PF.png")
    plt.close()

    # --------------------------------------------------------
    # VISUAL 08 — METRICS SUMMARY BAR CHART (Q and PF)
    # --------------------------------------------------------
    metric_labels = [
        "Q_MAE", "Q_RMSE", "Q_R2",
        "PF_MAE", "PF_RMSE", "PF_R2"
    ]
    metric_values = [
        metrics["ReactivePower_Q"]["MAE"],
        metrics["ReactivePower_Q"]["RMSE"],
        metrics["ReactivePower_Q"]["R2_Score"],
        metrics["PowerFactor_PF"]["MAE"],
        metrics["PowerFactor_PF"]["RMSE"],
        metrics["PowerFactor_PF"]["R2_Score"],
    ]

    plt.figure(figsize=(10.5, 5.5))
    bars = plt.bar(metric_labels, metric_values)
    plt.title("Forecasting Performance Summary — Reactive Power (Q) and Power Factor (PF)")
    plt.xlabel("Metric")
    plt.ylabel("Value")

    for b, v in zip(bars, metric_values):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/08_metrics_summary_bar.png")
    plt.close()

    # --------------------------------------------------------
    # 8) SAVE MODELS + METRICS
    # --------------------------------------------------------
    q_model_path = os.path.join(model_dir, "reactive_q_forecast_model.joblib")
    pf_model_path = os.path.join(model_dir, "pf_forecast_model.joblib")

    joblib.dump(q_model, q_model_path)
    joblib.dump(pf_model, pf_model_path)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nReactive Power and Power Factor Forecast Training Complete")
    print(f"Q model saved:  {q_model_path}")
    print(f"PF model saved: {pf_model_path}")
    print(f"Run results:    {run_dir}")
    print(json.dumps(metrics, indent=4))

    return metrics


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train_reactive_and_pf_forecaster()
