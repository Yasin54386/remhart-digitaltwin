import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# LOAD FORECASTING TRAINER (ABSOLUTE TARGET, INDUSTRY STANDARD)
# - Forecast-safe chronological split (train/val/test)
# - TimeSeriesSplit hyperparameter search (no leakage)
# - Early stopping (overfit/underfit guardrail)
# - Rolling-context features (trend + volatility)
# - Robust objective (pseudohuber) for spikes/noise
# - Baseline comparison (persistence using load_lag_1h)
# - Full visualization suite + metrics summary bar chart
# - Saves: .joblib model, metrics.json, predictions.csv, PNG figures
# ============================================================

def train_load_forecasting_regressor():
    # --------------------------------------------------------
    # 1) DIRECTORY SETUP (MATCHES YOUR REPO STYLE)
    # --------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(BASE_DIR, "training_set", "load_forecasting_data.xlsx")
    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "load_forecasting")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # 2) LOAD & BASIC CLEANING (CHRONOLOGICAL INTEGRITY)
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = pd.read_excel(input_path).replace([np.inf, -np.inf], np.nan)

    required_cols = ["Timestamp", "load_lag_1h", "target_load_next"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found. Check preprocessing pipeline.")

    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Drop holiday_name (you confirmed it is always 'None' => zero-variance, no signal)
    if "holiday_name" in df.columns:
        df = df.drop(columns=["holiday_name"])

    # Drop rows with missing values (including created rolling NaNs later)
    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # 3) FEATURE ENRICHMENT: ROLLING CONTEXT FEATURES
    # --------------------------------------------------------
    # Assumption: data frequency aligns with the lag naming (1h, 24h, 168h).
    # Rolling windows operate on load_lag_1h to summarise recent behaviour.
    # These features often improve R² without changing the target definition.

    # Trend (mean)
    df["load_roll_mean_3h"] = df["load_lag_1h"].rolling(window=3, min_periods=3).mean()
    df["load_roll_mean_6h"] = df["load_lag_1h"].rolling(window=6, min_periods=6).mean()
    df["load_roll_mean_24h"] = df["load_lag_1h"].rolling(window=24, min_periods=24).mean()

    # Volatility (std)
    df["load_roll_std_3h"] = df["load_lag_1h"].rolling(window=3, min_periods=3).std()
    df["load_roll_std_6h"] = df["load_lag_1h"].rolling(window=6, min_periods=6).std()
    df["load_roll_std_24h"] = df["load_lag_1h"].rolling(window=24, min_periods=24).std()

    # Local envelope (min/max)
    df["load_roll_min_24h"] = df["load_lag_1h"].rolling(window=24, min_periods=24).min()
    df["load_roll_max_24h"] = df["load_lag_1h"].rolling(window=24, min_periods=24).max()

    # Drop NaNs introduced by rolling windows
    df = df.dropna().reset_index(drop=True)

    # --------------------------------------------------------
    # 4) BUILD FEATURE MATRIX X AND TARGET y (ABSOLUTE TARGET)
    # --------------------------------------------------------
    TARGET = "target_load_next"

    y = df[TARGET].astype(float)

    # Drop non-feature columns from X
    X = df.drop(columns=["Timestamp", TARGET], errors="ignore")

    # Keep numeric features only
    non_numeric = [c for c in X.columns if not np.issubdtype(X[c].dtype, np.number)]
    if non_numeric:
        X = X.drop(columns=non_numeric)

    # --------------------------------------------------------
    # 5) FORECAST-SAFE SPLIT: TRAIN / VAL / TEST (CHRONOLOGICAL)
    # --------------------------------------------------------
    n = len(df)
    test_size = int(np.ceil(0.20 * n))
    trainval_size = n - test_size

    X_trainval, X_test = X.iloc[:trainval_size], X.iloc[trainval_size:]
    y_trainval, y_test = y.iloc[:trainval_size], y.iloc[trainval_size:]
    ts_test = df["Timestamp"].iloc[trainval_size:]

    # Validation window = last 20% of trainval (for early stopping)
    val_size = int(np.ceil(0.20 * len(X_trainval)))
    train_size = len(X_trainval) - val_size

    X_train, X_val = X_trainval.iloc[:train_size], X_trainval.iloc[train_size:]
    y_train, y_val = y_trainval.iloc[:train_size], y_trainval.iloc[train_size:]

    # --------------------------------------------------------
    # 6) BASELINE (PERSISTENCE): y_hat = load_lag_1h
    # --------------------------------------------------------
    # Baseline predictions are taken from the already-aligned dataframe rows.
    baseline_pred_test = df["load_lag_1h"].iloc[trainval_size:].values

    baseline_mae = float(mean_absolute_error(y_test, baseline_pred_test))
    baseline_rmse = float(mean_squared_error(y_test, baseline_pred_test, squared=False))
    baseline_r2 = float(r2_score(y_test, baseline_pred_test))

    # MASE scaling denominator (naive error on trainval)
    baseline_pred_trainval = df["load_lag_1h"].iloc[:trainval_size].values
    mase_denom = float(mean_absolute_error(y_trainval, baseline_pred_trainval))
    if mase_denom == 0:
        mase_denom = 1e-9

    # --------------------------------------------------------
    # 7) MODEL: TIME SERIES CV + REGULARISATION + EARLY STOPPING
    # --------------------------------------------------------
    # Robust objective reduces sensitivity to spikes/outliers.
    base_model = XGBRegressor(
        random_state=42,
        objective="reg:pseudohubererror",
        tree_method="hist"  # fast + stable for tabular
    )

    tscv = TimeSeriesSplit(n_splits=5)

    # Strong but safe search space (prevents under/overfitting extremes)
    param_dist = {
        "n_estimators": [1200, 2000, 3500],
        "learning_rate": [0.005, 0.01, 0.03, 0.05],
        "max_depth": [2, 3, 4, 5, 6],
        "min_child_weight": [1, 3, 5, 7, 10],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
        "reg_alpha": [0.0, 0.1, 0.5, 1.0],
        "reg_lambda": [1.0, 2.0, 5.0, 10.0],
        "gamma": [0.0, 0.05, 0.1, 0.2],
        "max_delta_step": [0, 1, 2]
    }

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=90,
        scoring="neg_root_mean_squared_error",
        cv=tscv,
        verbose=0,
        n_jobs=-1,
        random_state=42
    )

    print("\n[1/3] Hyperparameter search (TimeSeriesSplit, leakage-safe)...")
    search.fit(X_trainval, y_trainval)

    best_params = search.best_params_
    print("Best params:", best_params)

    model = XGBRegressor(
        random_state=42,
        objective="reg:pseudohubererror",
        tree_method="hist",
        **best_params
    )

    print("\n[2/3] Final training with early stopping...")
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
        early_stopping_rounds=150
    )

    # --------------------------------------------------------
    # 8) TEST PREDICTION + METRICS (ROBUST + OPERATIONAL)
    # --------------------------------------------------------
    y_pred = model.predict(X_test)
    residuals = y_pred - y_test.values

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))
    r2 = float(r2_score(y_test, y_pred))

    # Normalised RMSE helps interpret RMSE relative to test dynamic range
    y_range = float(np.max(y_test.values) - np.min(y_test.values))
    nrmse = float(rmse / (y_range if y_range > 0 else 1e-9))

    # Robust percentage metrics (avoid exploding % when values are small)
    eps = 1e-6
    denom_floor = 0.02  # safe floor for scaled targets; adjust if your scale differs
    denom = np.maximum(np.abs(y_test.values), denom_floor)
    mape = float(np.mean(np.abs((y_test.values - y_pred) / denom)) * 100.0)
    smape = float(np.mean(2.0 * np.abs(y_pred - y_test.values) /
                          np.maximum(np.abs(y_test.values) + np.abs(y_pred), eps)) * 100.0)

    # MASE (forecasting-standard)
    mase = float(mae / mase_denom)

    # Absolute tolerance compliance (meaningful for scaled targets)
    abs_tol_001 = float(np.mean(np.abs(residuals) <= 0.01) * 100.0)
    abs_tol_002 = float(np.mean(np.abs(residuals) <= 0.02) * 100.0)
    abs_tol_005 = float(np.mean(np.abs(residuals) <= 0.05) * 100.0)

    # Peak-window performance (top 10% measured load in test)
    peak_threshold = float(np.quantile(y_test.values, 0.90))
    peak_mask = y_test.values >= peak_threshold
    peak_mae = float(mean_absolute_error(y_test.values[peak_mask], y_pred[peak_mask])) if np.any(peak_mask) else None
    peak_rmse = float(mean_squared_error(y_test.values[peak_mask], y_pred[peak_mask], squared=False)) if np.any(peak_mask) else None

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "R2_Score": r2,
        "MAPE_percent_floor02": mape,
        "sMAPE_percent": smape,
        "MASE": mase,

        "Baseline_MAE": baseline_mae,
        "Baseline_RMSE": baseline_rmse,
        "Baseline_R2": baseline_r2,

        "AbsTol_0.01_percent": abs_tol_001,
        "AbsTol_0.02_percent": abs_tol_002,
        "AbsTol_0.05_percent": abs_tol_005,

        "Peak_threshold_90pct": peak_threshold,
        "Peak_MAE": peak_mae,
        "Peak_RMSE": peak_rmse,

        "Best_params": best_params,
        "Best_iteration": int(getattr(model, "best_iteration", best_params.get("n_estimators", 0)))
    }

    # --------------------------------------------------------
    # 9) SAVE PREDICTIONS FOR AUDITABILITY
    # --------------------------------------------------------
    pred_df = pd.DataFrame({
        "Timestamp": ts_test.values,
        "Actual_Load": y_test.values,
        "Predicted_Load": y_pred,
        "Residual": residuals,
        "Baseline_Pred": baseline_pred_test
    })
    pred_df.to_csv(os.path.join(run_dir, "predictions.csv"), index=False)

    # --------------------------------------------------------
    # 10) VISUALISATIONS (PROFESSIONAL TITLES + LABELS)
    # --------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120

    # 01: Time-series overlay (Actual vs Predicted vs Baseline)
    plt.figure(figsize=(12, 5))
    plt.plot(ts_test, y_test.values, label="Measured Load", linewidth=1.3)
    plt.plot(ts_test, y_pred, label="Predicted Load (XGBoost)", linewidth=1.3)
    plt.plot(ts_test, baseline_pred_test, label="Baseline (Persistence: load_lag_1h)", linewidth=1.0, alpha=0.85)
    plt.title("Load Forecasting Performance on Hold-Out Test Window")
    plt.xlabel("Time")
    plt.ylabel("Load")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_timeseries_actual_vs_pred.png")
    plt.close()

    # 02: Parity plot (Predicted vs Actual)
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test.values, y_pred, alpha=0.6)
    min_v = float(min(np.min(y_test.values), np.min(y_pred)))
    max_v = float(max(np.max(y_test.values), np.max(y_pred)))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", linewidth=2)
    plt.title("Parity Plot: Predicted vs Measured Load")
    plt.xlabel("Measured Load")
    plt.ylabel("Predicted Load")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_scatter_parity.png")
    plt.close()

    # 03: Residuals over time
    plt.figure(figsize=(12, 4))
    plt.plot(ts_test, residuals, linewidth=1.1)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title("Residual Diagnostics Over Time (Predicted − Measured)")
    plt.xlabel("Time")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/03_residuals_over_time.png")
    plt.close()

    # 04: Residuals vs measured load
    plt.figure(figsize=(7, 5))
    plt.scatter(y_test.values, residuals, alpha=0.6)
    plt.axhline(0, linestyle="--", linewidth=2)
    plt.title("Residuals vs Measured Load (Error Structure Check)")
    plt.xlabel("Measured Load")
    plt.ylabel("Residual (Predicted − Measured)")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/04_residuals_vs_actual.png")
    plt.close()

    # 05: Error distribution
    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)
    plt.title("Distribution of Forecast Errors (Residuals)")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/05_error_distribution.png")
    plt.close()

    # 06: Operational tolerance compliance (absolute error)
    plt.figure(figsize=(7, 5))
    labels = ["±0.01", "±0.02", "±0.05"]
    vals = [abs_tol_001, abs_tol_002, abs_tol_005]
    bars = plt.bar(labels, vals)
    plt.ylim(0, 100)
    plt.title("Operational Accuracy: Predictions Within Absolute Error Tolerance")
    plt.xlabel("Absolute Error Tolerance")
    plt.ylabel("Samples Within Tolerance (%)")
    for b in bars:
        h = b.get_height()
        plt.text(b.get_x() + b.get_width() / 2, h + 1, f"{h:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/06_absolute_tolerance_compliance.png")
    plt.close()

    # 07: Peak-load absolute error
    plt.figure(figsize=(10, 4))
    if np.any(peak_mask):
        peak_idx = np.where(peak_mask)[0]
        plt.scatter(peak_idx, np.abs(residuals[peak_mask]), alpha=0.7)
    plt.title("Peak-Load Error Magnitude (Top 10% Measured Load)")
    plt.xlabel("Peak Samples (Index Within Test Window)")
    plt.ylabel("Absolute Error")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/07_peak_period_analysis.png")
    plt.close()

    # 08: Feature importance (top 20)
    fi = pd.Series(model.feature_importances_, index=X.columns).sort_values()
    plt.figure(figsize=(9, 6))
    fi.tail(20).plot(kind="barh")
    plt.title("Top Feature Importances for Load Forecasting (XGBoost)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/08_feature_importance.png")
    plt.close()

    # 09: Metrics summary bar chart (Model vs Baseline)
    plt.figure(figsize=(9, 5))
    summary_labels = ["MAE", "RMSE", "R2", "MASE", "NRMSE"]
    model_vals = [mae, rmse, r2, mase, nrmse]
    base_vals = [baseline_mae, baseline_rmse, baseline_r2, np.nan, np.nan]

    x = np.arange(len(summary_labels))
    width = 0.35
    plt.bar(x - width / 2, model_vals, width, label="XGBoost")
    plt.bar(x + width / 2, base_vals, width, label="Baseline (Persistence: load_lag_1h)")
    plt.title("Forecasting Performance Summary (Model vs Baseline)")
    plt.xticks(x, summary_labels)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{run_dir}/09_metrics_summary_bar.png")
    plt.close()

    # --------------------------------------------------------
    # 11) SAVE MODEL + METADATA
    # --------------------------------------------------------
    joblib.dump(model, os.path.join(model_dir, "load_forecasting_model.joblib"))

    with open(os.path.join(run_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n[3/3] Training complete. Metrics:")
    print(json.dumps(metrics, indent=4))
    print(f"\nArtifacts saved to: {run_dir}")

    return metrics


if __name__ == "__main__":
    train_load_forecasting_regressor()
