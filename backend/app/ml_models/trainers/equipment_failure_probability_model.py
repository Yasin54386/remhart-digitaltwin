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

# --- PROGNOSTIC TRAINING PIPELINE: EQUIPMENT FAILURE ---
def train_equipment_failure_model():
    # --- STEP 1: DYNAMIC DIRECTORY HANDLING ---
    # WHAT: Setting up the environment and creating necessary folders.
    # WHY: Prevents 'FileNotFoundError' during model and artifact saving.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "training_set", "equipment_failure_probability_data.xlsx")
    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "equipment_failure")
    
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # 2. DATA INGESTION
    df = pd.read_excel(input_path)
    
    # 3. MULTIVARIATE FEATURE MAPPING (The Raw Catalysts)
    # Excluding intermediate stressors to ensure AI learns the raw physics
    features = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C', 'P_A','P_B','P_C', 'Q_A','Q_B','Q_C']
    target = 'failure_probability'
    
    # Cleaning Step: Remove Inf and NaN to satisfy XGBoost requirements
    initial_len = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=features + [target], inplace=True)
    
    if len(df) < initial_len:
        print(f"Data Cleaning: Removed {initial_len - len(df)} rows containing NaN or Infinity.")

    X = df[features]
    y = df[target]

    # 4. DATA PARTITIONING (80/20 Stratified-style Split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. HYPERPARAMETER OPTIMIZATION (GridSearchCV)
    print("Optimizing XGBoost Regressor for prognostic precision...")
    param_grid = {
        'n_estimators': [500],
        'max_depth': [5, 7, 9],
        'learning_rate': [0.05, 0.1],
        'objective': ['reg:squarederror']
    }
    xgb = XGBRegressor(random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    # 6. METRIC CALCULATION
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2_Score": r2_score(y_test, y_pred),
        "MBE": np.mean(y_pred - y_test) # Mean Bias Error
    }

    # =========================================================================
    # VISUALIZATION GENERATION SUITE
    # =========================================================================
    print(f"Generating performance artifacts in: {run_dir}...")

    # 01. ACTUAL VS. PREDICTED PROBABILITY
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='darkred')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.title("Actual vs. Predicted Failure Risk")
    plt.xlabel("Ground Truth (Physics-Derived)"); plt.ylabel("AI Prediction")
    plt.grid(alpha=0.3); plt.savefig(f"{run_dir}/01_actual_vs_predicted.png")

    # 02. RISK CORRELATION HEATMAP (Feature Interactions)
    plt.figure(figsize=(10, 8))
    corr = df[features + [target]].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title("Risk Driver Correlation Matrix (Heatmap)")
    plt.savefig(f"{run_dir}/02_risk_correlation_heatmap.png")

    # 03. FEATURE IMPORTANCE (Explainability)
    plt.figure(figsize=(10, 6))
    feat_imp = pd.Series(best_model.feature_importances_, index=features).sort_values()
    feat_imp.plot(kind='barh', color='darkorange')
    plt.title("Feature Contribution to Equipment Stress (SHAP/Gain)")
    plt.xlabel("Relative Importance Score"); plt.savefig(f"{run_dir}/03_feature_importance.png")

    # 04. RESIDUAL ERROR DISTRIBUTION
    plt.figure(figsize=(8, 6))
    sns.histplot(y_pred - y_test, kde=True, color='purple')
    plt.title("Residual Analysis: Prediction Error Distribution")
    plt.xlabel("Error Magnitude"); plt.savefig(f"{run_dir}/04_residual_distribution.png")

    # 05. FAILURE RISK TREND (Sample Validation)
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:100], label='Actual Risk', color='blue', alpha=0.6)
    plt.plot(y_pred[:100], label='Predicted Risk', color='red', linestyle='--')
    plt.title("Time-Series Risk Tracking (Prognostic Sampling)")
    plt.legend(); plt.grid(alpha=0.2); plt.savefig(f"{run_dir}/05_risk_trend.png")

    # 06. PERFORMANCE METRICS SUMMARY (Bar Chart)
    plt.figure(figsize=(8, 5))
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    bars = plt.bar(metric_names, metric_values, color=['teal', 'navy', 'green', 'maroon'])
    plt.ylim(0, 1.1)
    plt.title("Model Validation Performance Summary")
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, round(yval, 4), ha='center', fontweight='bold')
    plt.savefig(f"{run_dir}/06_metrics_summary.png")

    # 7. SERIALIZATION
    joblib.dump(best_model, f"{model_dir}/equipment_failure_model.joblib")
    with open(f"{run_dir}/failure_model_metadata.json", "w") as f:
        json.dump(metrics, f, indent=4)
    
    print("Training Complete. All visuals and model saved.")

if __name__ == "__main__":
    train_equipment_failure_model()