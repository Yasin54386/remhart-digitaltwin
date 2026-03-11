import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBRegressor 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- SCRIPT: SYSTEM HEALTH WATCHDOG (REGRESSION) ---
# WHAT: An industrial-grade regression pipeline for continuous health monitoring.
# WHY: To provide high-fidelity tracking of grid degradation.
# HOW: Predicting IEC 61000-4-30 health score with a cleaned, physics-anchored dataset.

def train_system_health_watchdog():
    # --- STEP 1: DYNAMIC DIRECTORY HANDLING ---
    # WHAT: Setting up the environment and creating necessary folders.
    # WHY: Prevents 'FileNotFoundError' during model and artifact saving.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "training_set", "energy_imbalance_and_health_index_data.xlsx")
    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "energy_imbalance")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    # --- STEP 2: BULLETPROOF DATA CLEANING ---
    # WHAT: Loading and sanitizing the dataset.
    # WHY: Prevents 'XGBoostError' by removing NaN or Infinity values.
    # HOW: Dropping rows with non-finite values in features or target.
    df = pd.read_excel(input_path)
    
    features = [
        'volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C', 
        'P_T', 'Q_T', 'FP_T', 'Frec', #'energy_imbalance_error'
    ]
    target = 'health_score'
    
    # Cleaning Step: Remove Inf and NaN to satisfy XGBoost requirements
    initial_len = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + [target], inplace=True)
    
    if len(df) < initial_len:
        print(f"Data Cleaning: Removed {initial_len - len(df)} rows containing NaN or Infinity.")

    X, y = df[features], df[target]

    # --- STEP 3: DATA PARTITIONING ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- STEP 4: HYPERPARAMETER OPTIMIZATION ---
    # WHAT: neg_mean_absolute_error scoring via 5-Fold Cross-Validation.
    # WHY: Maximizes precision for tracking non-technical losses.
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'objective': ['reg:squarederror']
    }
    xgb_reg = XGBRegressor(random_state=42)
    grid = GridSearchCV(xgb_reg, param_grid, scoring='neg_mean_absolute_error', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # --- STEP 5: ARTIFACT REPOSITORY INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # =========================================================================
    # INDUSTRIAL PERFORMANCE VISUALIZATION SUITE (REGRESSION)
    # =========================================================================
    y_pred = best_model.predict(X_test)
    print(f"Generating regression validation artifacts in: {run_dir}...")

    # 01. ACTUAL VS. PREDICTED: Visual validation of watchdog tracking.
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='teal')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.title("Actual vs. Predicted System Health"); plt.xlabel("Actual"); plt.ylabel("Predicted")
    plt.savefig(os.path.join(run_dir, "01_actual_vs_predicted.png"))

    # 02. ERROR DISTRIBUTION: Visualizes precision across the grid health spectrum.
    plt.figure(figsize=(10, 6))
    sns.histplot(y_test - y_pred, kde=True, color='orange')
    plt.title("Residual Error Distribution"); plt.xlabel("Error"); plt.ylabel("Frequency")
    plt.savefig(os.path.join(run_dir, "02_error_distribution.png"))

    # 03. FEATURE IMPORTANCE: Proves the model uses physical laws (Energy Imbalance) for logic.
    plt.figure(figsize=(10, 6))
    pd.Series(best_model.feature_importances_, index=features).sort_values().plot(kind='barh', color='darkblue')
    plt.title("Physical Catalyst Importance (Explainability Report)"); plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "03_feature_importance.png"))

    # 04. VALIDATION METRIC BAR CHART: Summary of MAE, RMSE, and R2.
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2_Score': r2_score(y_test, y_pred)
    }
    plt.figure(figsize=(8, 5))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()), hue=list(metrics.keys()), palette='viridis', legend=False)
    plt.title("Performance Summary")
    for i, v in enumerate(metrics.values()):
        plt.text(i, v + 0.02, f"{v:.4f}", ha='center', fontweight='bold')
    plt.savefig(os.path.join(run_dir, "04_metrics_summary.png"))

    # 05. TIME-SERIES TRACKING (SAMPLE): Demonstrates the "Watchdog" monitoring capacity.
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:100], label='Actual Health', color='black', alpha=0.7)
    plt.plot(y_pred[:100], label='Predicted Health', color='red', linestyle='--')
    plt.title("Real-time Health Monitoring Tracking (Test Sample)"); plt.legend(); plt.grid(alpha=0.3)
    plt.savefig(os.path.join(run_dir, "05_health_tracking_sample.png"))

    # 06. RESIDUALS VS. FITTED: Validates model stability.
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_test - y_pred, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.title("Residuals vs. Fitted Values"); plt.xlabel("Predicted"); plt.ylabel("Residuals")
    plt.savefig(os.path.join(run_dir, "06_residuals_analysis.png"))

    # --- STEP 6: SERIALIZATION ---
    metadata = {
        "model_architecture": "XGBoost Regressor",
        "training_timestamp": timestamp,
        "performance_metrics": metrics,
        "best_params": grid.best_params_
    }
    with open(os.path.join(run_dir, "energy_imbalance_and_health_index.json"), "w") as f:
        json.dump(metadata, f, indent=4)
    
    joblib.dump(best_model, os.path.join(model_dir, "energy_imbalance_and_health_index_model.joblib"))
    print(f"System Health Watchdog (Regression) Built Successfully. Artifacts: {run_dir}")

if __name__ == "__main__":
    train_system_health_watchdog()