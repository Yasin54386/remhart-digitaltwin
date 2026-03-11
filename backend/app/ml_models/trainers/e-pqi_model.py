import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# E-PQI SCORE REGRESSION TRAINING PIPELINE
# ============================================================

def train_epqi_score_regressor():

    # --------------------------------------------------------
    # 1. DIRECTORY SETUP (CONSISTENT WITH PREVIOUS SCRIPT)
    # --------------------------------------------------------

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(
        BASE_DIR,
        "training_set",
        "equipment_pqi_score_data.xlsx"
    )

    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "equipment_pqi")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # 2. LOAD AND CLEAN DATA
    # --------------------------------------------------------

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = (
        pd.read_excel(input_path)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    # --------------------------------------------------------
    # 3. FEATURE SELECTION & TARGET
    # --------------------------------------------------------

    features = [
        "volt_A", "volt_B", "volt_C",
        "I_A", "I_B", "I_C",
        "P_T", "Q_T",
        "FP_T", "Frec"
    ]

    X = df[features]
    y = df["PQI_Score"]   # CONTINUOUS TARGET

    # --------------------------------------------------------
    # 4. TRAIN / TEST SPLIT (NO STRATIFICATION NEEDED)
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    # --------------------------------------------------------
    # 5. MODEL DEFINITION (REGRESSION-OPTIMISED)
    # --------------------------------------------------------

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="reg:squarederror",
        random_state=42
    )

    model.fit(X_train, y_train)

    # --------------------------------------------------------
    # 6. PREDICTION & METRICS
    # --------------------------------------------------------

    y_pred = model.predict(X_test)
    residuals = y_pred - y_test

    metrics = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(mean_squared_error(y_test, y_pred, squared=False)),
        "R2_Score": float(r2_score(y_test, y_pred))
    }

    # --------------------------------------------------------
    # 7. VISUALISATION STYLE
    # --------------------------------------------------------

    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 120

    # --------------------------------------------------------
    # 8. VISUAL 01 — ACTUAL VS PREDICTED PQI SCORE
    # --------------------------------------------------------

    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.65)
    plt.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        linestyle="--",
        linewidth=2
    )

    plt.title("Actual vs Predicted Power Quality Index (PQI) Score")
    plt.xlabel("Measured PQI Score")
    plt.ylabel("Predicted PQI Score")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_actual_vs_predicted.png")
    plt.close()

    # --------------------------------------------------------
    # 9. VISUAL 02 — RESIDUAL ERROR DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))
    plt.scatter(y_test, residuals, alpha=0.65)
    plt.axhline(0, linestyle="--", linewidth=2)

    plt.title("Residual Error Distribution Across PQI Values")
    plt.xlabel("Measured PQI Score")
    plt.ylabel("Prediction Error (Predicted − Measured)")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_residuals.png")
    plt.close()

    # --------------------------------------------------------
    # 10. VISUAL 03 — ERROR HISTOGRAM
    # --------------------------------------------------------

    plt.figure(figsize=(7, 5))
    sns.histplot(residuals, kde=True)

    plt.title("Distribution of PQI Prediction Errors")
    plt.xlabel("Prediction Error")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/03_error_distribution.png")
    plt.close()

    # --------------------------------------------------------
    # 11. VISUAL 04 — ±5% OPERATIONAL TOLERANCE BAND
    # --------------------------------------------------------

    tolerance = 0.05 * y_test

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.6, label="Predictions")
    plt.plot(y_test, y_test, linestyle="--", linewidth=2, label="Ideal Prediction")

    plt.fill_between(
        y_test,
        y_test - tolerance,
        y_test + tolerance,
        alpha=0.25,
        label="±5% Operational Tolerance"
    )

    plt.title("PQI Prediction Accuracy Within Operational Tolerance")
    plt.xlabel("Measured PQI Score")
    plt.ylabel("Predicted PQI Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/04_tolerance_band.png")
    plt.close()

    # --------------------------------------------------------
    # 12. VISUAL 05 — FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.Series(
        model.feature_importances_,
        index=features
    ).sort_values()

    plt.figure(figsize=(9, 6))
    importance_df.plot(kind="barh")

    plt.title("Relative Importance of Electrical Parameters in PQI Prediction")
    plt.xlabel("Importance Score")
    plt.ylabel("Electrical Feature")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/05_feature_importance.png")
    plt.close()

    # --------------------------------------------------------
    # 13. SAVE MODEL & METADATA
    # --------------------------------------------------------

    joblib.dump(model, f"{model_dir}/epqi_score_model.joblib")

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nE-PQI Score Regression Training Complete")
    print(json.dumps(metrics, indent=4))

    return metrics


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train_epqi_score_regressor()
