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
# VOLTAGE SAG SEVERITY PREDICTION — XGBOOST REGRESSOR PIPELINE
#
# Target:
#   voltage_sag_probability  (continuous severity/risk index: 0.0–1.0)
#
# Outputs (ONLY Visualisations 1–5):
#   01_timeseries_actual_vs_pred.png
#   02_parity_pred_vs_actual.png
#   03_residuals_over_time.png
#   04_residual_distribution.png
#   05_metrics_summary_bar.png
#
# Artifacts:
#   trained/voltage_sag_xgb_regressor.joblib
#   model_results/voltage_sag/run_<timestamp>/metrics.json
# ============================================================


def _safe_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce")


def _time_aware_train_test_split(df: pd.DataFrame, test_size: float = 0.20):
    """
    Chronological hold-out split:
      - Train on earlier data
      - Test on later data
    """
    n = len(df)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    train_df = df.iloc[:n_train].copy()
    test_df = df.iloc[n_train:].copy()
    return train_df, test_df


def train_voltage_sag_regressor():
    # --------------------------------------------------------
    # 1) DIRECTORY SETUP (MATCHES YOUR REPO STANDARD)
    # --------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "training_set", "voltage_sag_processed_data.xlsx")
    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "voltage_sag")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # 2) LOAD & HARDEN DATA
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = (
        pd.read_excel(input_path)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    # If Timestamp exists, enforce chronological ordering (time-aware learning)
    if "Timestamp" in df.columns:
        df["Timestamp"] = _safe_datetime(df["Timestamp"])
        df = df.dropna(subset=["Timestamp"]).sort_values("Timestamp").reset_index(drop=True)

    # --------------------------------------------------------
    # 3) FEATURE MATRIX (X) + TARGET (y)
    # --------------------------------------------------------
    target_col = "voltage_sag_probability"
    if target_col not in df.columns:
        raise KeyError(
            f"Target column '{target_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    # Exclude non-feature columns
    drop_cols = {target_col}
    if "Timestamp" in df.columns:
        drop_cols.add("Timestamp")

    # Keep only numeric predictors
    X = df.drop(columns=list(drop_cols), errors="ignore")
    X = X.select_dtypes(include=[np.number]).copy()

    if X.shape[1] == 0:
        raise ValueError("No numeric feature columns found after preprocessing.")

    y = df[target_col].astype(float).copy()

    # --------------------------------------------------------
    # 4) TIME-AWARE SPLIT (DEPLOYMENT-REALISTIC)
    # --------------------------------------------------------
    train_df, test_df = _time_aware_train_test_split(df, test_size=0.20)

    X_train = train_df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    y_train = train_df[target_col].astype(float)

    X_test = test_df.drop(columns=list(drop_cols), errors="ignore").select_dtypes(include=[np.number]).fillna(0)
    y_test = test_df[target_col].astype(float)

    # Ensure train/test have identical columns
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

    feature_names = list(X_train.columns)

    # --------------------------------------------------------
    # 5) MODEL + HYPERPARAMETER OPTIMISATION (TIME SERIES CV)
    # --------------------------------------------------------
    base_model = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    # Conservative search space to avoid overfitting while still learning non-linear dynamics
    param_distributions = {
        "n_estimators": [600, 1000, 1500, 2000],
        "max_depth": [2, 3, 4, 5],
        "learning_rate": [0.005, 0.01, 0.03, 0.05],
        "subsample": [0.75, 0.85, 0.95],
        "colsample_bytree": [0.70, 0.80, 0.90, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "reg_alpha": [0.0, 0.05, 0.1, 0.2],
        "reg_lambda": [1.0, 1.5, 2.0, 3.0],
        "gamma": [0.0, 0.05, 0.1]
    }

    # Time-series cross validation
    tscv = TimeSeriesSplit(n_splits=5)

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=25,                      # fast but strong enough for industrial-grade tuning
        scoring="neg_mean_absolute_error",
        cv=tscv,
        random_state=42,
        verbose=0,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    best_model = search.best_estimator_

    # Final fit on full training set
    best_model.fit(X_train, y_train)

    # --------------------------------------------------------
    # 6) PREDICTION & METRICS
    # --------------------------------------------------------
    y_pred = best_model.predict(X_test)
    residuals = y_pred - y_test.values

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))
    r2 = float(r2_score(y_test, y_pred))

    y_range = float(max(y_test.max() - y_test.min(), 1e-9))
    nrmse = float(rmse / y_range)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2_Score": r2,
        "Best_params": {k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in search.best_params_.items()},
        "Train_rows": int(len(X_train)),
        "Test_rows": int(len(X_test)),
        "Time_aware_split": True,
        "Features_used": feature_names
    }

    # --------------------------------------------------------
    # 7) VISUALISATION STYLE (PROFESSIONAL)
    # --------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 140

    # X-axis for time-aware plots
    if "Timestamp" in test_df.columns:
        x_axis = test_df["Timestamp"]
        x_label = "Time"
    else:
        x_axis = np.arange(len(y_test))
        x_label = "Test Sample Index"

    # --------------------------------------------------------
    # 8) VISUAL 01 — TIME-SERIES: MEASURED VS PREDICTED SAG SEVERITY
    # --------------------------------------------------------
    plt.figure(figsize=(13, 5))
    plt.plot(x_axis, y_test.values, linewidth=1.6, label="Measured Sag Severity")
    plt.plot(x_axis, y_pred, linewidth=1.6, label="Predicted Sag Severity")
    plt.title("Voltage Sag Severity Prediction: Time-Series Validation")
    plt.xlabel(x_label)
    plt.ylabel("Sag Severity Score (0–1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_timeseries_actual_vs_pred.png")
    plt.close()

    # --------------------------------------------------------
    # 9) VISUAL 02 — PARITY PLOT: PREDICTED VS MEASURED
    # --------------------------------------------------------
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.65)
    min_v = float(min(y_test.min(), np.min(y_pred)))
    max_v = float(max(y_test.max(), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=2)
    plt.title("Voltage Sag Severity Prediction: Parity Analysis")
    plt.xlabel("Measured Sag Severity Score")
    plt.ylabel("Predicted Sag Severity Score")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_parity_pred_vs_actual.png")
    plt.close()

    # --------------------------------------------------------
    # 10) VISUAL 03 — RESIDUALS OVER TIME
    # --------------------------------------------------------
    plt.figure(figsize=(13, 4.8))
    plt.plot(x_axis, residuals, linewidth=1.2)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title("Voltage Sag Severity Prediction: Residual Error Over Time")
    plt.xlabel(x_label)
    plt.ylabel("Prediction Error (Predicted − Measured)")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/03_residuals_over_time.png")
    plt.close()

    # --------------------------------------------------------
    # 11) VISUAL 04 — RESIDUAL DISTRIBUTION
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=60)
    plt.title("Distribution of Voltage Sag Severity Prediction Errors")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/04_residual_distribution.png")
    plt.close()

    # --------------------------------------------------------
    # 12) VISUAL 05 — METRICS SUMMARY BAR CHART
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    keys = ["MAE", "RMSE", "NRMSE", "R2_Score"]
    vals = [metrics[k] for k in keys]
    plt.bar(keys, vals)

    plt.title("Voltage Sag Severity Prediction: Quantitative Performance Summary")
    plt.ylim(0, max(1.05, max(vals) * 1.15))
    for i, v in enumerate(vals):
        plt.text(i, v + (0.02 if v < 1 else 0.05), f"{v:.4f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/05_metrics_summary_bar.png")
    plt.close()

    # --------------------------------------------------------
    # 13) SAVE ARTIFACTS
    # --------------------------------------------------------
    model_artifact = {
        "model": best_model,
        "features": feature_names,
        "target": target_col,
        "time_aware_split": True
    }

    model_path = os.path.join(model_dir, "voltage_sag_xgb_regressor.joblib")
    joblib.dump(model_artifact, model_path)

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Optional: Save test predictions for auditability
    out_pred = test_df.copy()
    out_pred["predicted_voltage_sag_probability"] = y_pred
    out_pred["prediction_error"] = residuals
    out_pred.to_excel(os.path.join(run_dir, "test_predictions.xlsx"), index=False)

    print("\nVoltage Sag Severity Regression Training Complete")
    print(f"Model saved to: {model_path}")
    print(f"Run outputs in: {run_dir}")
    print(json.dumps(metrics, indent=4))

    return metrics


if __name__ == "__main__":
    train_voltage_sag_regressor()
