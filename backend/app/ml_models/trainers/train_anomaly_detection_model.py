import pandas as pd
import numpy as np
import os
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    f1_score, precision_score, recall_score, balanced_accuracy_score, 
    roc_auc_score, roc_curve, auc
)

def train_anomaly_detection():
    # --- STEP 1: DYNAMIC PATH RESOLUTION ---
    # Automatically determines the directory structure relative to the script location.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(BASE_DIR, "training_set", "anomaly_detection_data.xlsx")
    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "anomaly_detection")
    os.makedirs(model_dir, exist_ok=True)
    
    if not os.path.exists(input_path):
        print(f"File Error: {input_path} not found.")
        return
        
    # --- STEP 2: MULTIVARIATE DATA INGESTION ---
    df = pd.read_excel(input_path)
    features = ['volt_A', 'volt_B', 'volt_C', 'I_A', 'I_B', 'I_C', 'P_T', 'Q_T', 'FP_T', 'Frec', 'hour']
    X, y = df[features], df['is_anomaly']

    # --- STEP 3: STRATIFIED SAMPLING (80/20 SPLIT) ---
    # Stratification ensures the minority class is proportionally represented in the test set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- STEP 4: HYPERPARAMETER OPTIMIZATION ---
    # GridSearchCV identifies the optimal XGBoost parameters to maximize the F1-Score.
    param_grid = {'n_estimators': [100, 200], 'max_depth': [3, 5, 7], 'scale_pos_weight': [1, 5, 10]}
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    grid = GridSearchCV(xgb, param_grid, scoring='f1', cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # --- STEP 5: PERFORMANCE INFERENCE ---
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    # --- STEP 6: ARTIFACT REPOSITORY INITIALIZATION ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = f"{results_base_dir}/run_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    # =========================================================================
    # UNIVERSAL PERFORMANCE VISUALIZATION SUITE
    # =========================================================================
    
    # 01. CONFUSION MATRIX
    # 
    plt.figure(figsize=(7, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title("Confusion Matrix for Anomaly Classification", fontsize=12)
    plt.xlabel("Predicted Class Label"); plt.ylabel("True Class Label")
    plt.savefig(f"{run_dir}/01_confusion_matrix.png")

    # 02. RECEIVER OPERATING CHARACTERISTIC (ROC) CURVE
    # 
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {auc(fpr, tpr):.4f}')
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=12)
    plt.xlabel("False Positive Rate (FPR)"); plt.ylabel("True Positive Rate (TPR)")
    plt.legend(loc="lower right"); plt.grid(alpha=0.2); plt.savefig(f"{run_dir}/02_roc_curve.png")

    # 03. PRECISION-RECALL (PR) CURVE
    # 
    prec, rec, thresh = precision_recall_curve(y_test, y_prob)
    plt.figure(figsize=(7, 6))
    plt.plot(rec, prec, color='darkorange', lw=2, label=f'PR-AUC = {auc(rec, prec):.4f}')
    plt.title("Precision-Recall Analysis", fontsize=12)
    plt.xlabel("Recall (Sensitivity)"); plt.ylabel("Precision (Positive Predictive Value)")
    plt.legend(); plt.grid(alpha=0.2); plt.savefig(f"{run_dir}/03_pr_curve.png")

    # 04. F1-SCORE VS. CLASSIFICATION THRESHOLD
    # 
    f1_scores = 2 * (prec * rec) / (prec + rec + 1e-8)
    best_threshold = thresh[np.argmax(f1_scores)]
    plt.figure(figsize=(7, 6))
    plt.plot(thresh, f1_scores[:-1], color='forestgreen', lw=2)
    plt.axvline(x=best_threshold, color='red', linestyle=':', label=f'Optimal Threshold: {best_threshold:.2f}')
    plt.title("F1-Score Optimization over Confidence Thresholds", fontsize=12)
    plt.xlabel("Classification Threshold"); plt.ylabel("F1-Score Value")
    plt.legend(); plt.grid(alpha=0.2); plt.savefig(f"{run_dir}/04_f1_threshold_optimization.png")

    # 05. CUMULATIVE GAIN ANALYSIS
    # 
    sorted_idx = np.argsort(y_prob)[::-1]
    y_test_sorted = y_test.values[sorted_idx]
    cum_gain = np.cumsum(y_test_sorted) / np.sum(y_test_sorted)
    plt.figure(figsize=(7, 6))
    plt.plot(np.linspace(0, 1, len(cum_gain)), cum_gain, lw=2, color='navy', label='Model Performance')
    plt.plot([0, 1], [0, 1], 'k--', label='Baseline (Random Selection)')
    plt.title("Cumulative Gain Chart", fontsize=12)
    plt.xlabel("Fraction of Total Population"); plt.ylabel("Fraction of Total Anomalies Captured")
    plt.legend(); plt.grid(alpha=0.2); plt.savefig(f"{run_dir}/05_cumulative_gain.png")

    # --- STEP 7: METADATA SERIALIZATION (JSON) ---
    # Encapsulates model performance metrics for systematic tracking.
    metadata = {
        "training_timestamp": timestamp,
        "model_architecture": "XGBoost Classifier",
        "performance_metrics": {
            "f1_score": float(f1_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_prob))
        },
        "optimal_operating_point": {
            "threshold": round(float(best_threshold), 4)
        }
    }
    with open(f"{run_dir}/run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    
    # --- STEP 8: MODEL EXPORT ---
    # Serializes the trained model object for production inference.
    joblib.dump(best_model, f"{model_dir}/anomaly_detection_model.joblib")
    plt.close('all')
    print(f"\nTraining Complete. Artifacts stored in: {run_dir}")

if __name__ == "__main__":
    train_anomaly_detection()