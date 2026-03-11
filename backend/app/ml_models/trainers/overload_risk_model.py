import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    accuracy_score,
    roc_curve,
    auc
)

# ============================================================
# OVERLOAD RISK CLASSIFICATION — XGBOOST TRAINING PIPELINE
# Outputs:
#   - trained/overload_risk_model.joblib
#   - model_results/overload_risk/run_<timestamp>/*
#       01_confusion_matrix.png
#       02_precision_recall_by_class.png
#       03_roc_high_vs_rest.png
#       04_metrics_summary.png
#       metrics.json
# ============================================================

def _safe_int(x, default=2):
    try:
        x = int(x)
        return x if x >= 2 else default
    except Exception:
        return default


def train_overload_risk_classifier():
    # --------------------------------------------------------
    # 1. DIRECTORY SETUP (CONSISTENT WITH YOUR PIPELINE)
    # --------------------------------------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    input_path = os.path.join(
        BASE_DIR,
        "training_set",
        "overload_risk_classification_data.xlsx"
    )

    model_dir = os.path.join(BASE_DIR, "trained")
    results_base_dir = os.path.join(BASE_DIR, "model_results", "overload_risk")

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(results_base_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_dir = os.path.join(results_base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # --------------------------------------------------------
    # 2. LOAD & CLEAN DATA
    # --------------------------------------------------------
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Dataset not found: {input_path}")

    df = (
        pd.read_excel(input_path)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    # Target column created by preprocessing
    if "overload_risk" not in df.columns:
        raise KeyError("Missing required target column: 'overload_risk'")

    # Remove unknowns if any
    df = df[df["overload_risk"].astype(str).str.lower() != "unknown"].copy()

    # --------------------------------------------------------
    # 3. FEATURE MATRIX (X) & TARGET (y)
    # --------------------------------------------------------
    # Use physically grounded predictors: phase currents + computed load %
    candidate_features = ["volt_A", "volt_B", "volt_C","I_A", "I_B", "I_C","P_T","Q_T","FP_T"]

    missing = [c for c in candidate_features if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")

    X = df[candidate_features].copy()
    y = df["overload_risk"].astype(str).copy()

    # --------------------------------------------------------
    # 4. ENCODE TARGET CLASSES
    # --------------------------------------------------------
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)

    # Identify the encoded index for "High" (case-safe)
    # Expect labels like: Low / Medium / High
    high_label_candidates = [i for i, c in enumerate(class_names) if c.lower() == "high"]
    if not high_label_candidates:
        raise ValueError(f"'High' class not found in overload_risk labels: {class_names}")
    high_class_idx = high_label_candidates[0]

    # --------------------------------------------------------
    # 5. HANDLE RARE-CLASS STRATIFICATION SAFELY
    # --------------------------------------------------------
    counts = pd.Series(y_enc).value_counts().sort_index()
    min_count = int(counts.min())

    # Decide stratification and CV folds safely
    use_stratify = min_count >= 2
    n_splits = min(5, min_count) if min_count >= 2 else 2
    n_splits = _safe_int(n_splits, default=2)

    # Train/test split
    if use_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=0.20,
            random_state=42,
            stratify=y_enc
        )
    else:
        # Fallback when a class has only 1 sample
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=0.20,
            random_state=42
        )

    # --------------------------------------------------------
    # 6. MODEL + GRID SEARCH (INDUSTRY-SAFE REGULARISED SETUP)
    # --------------------------------------------------------
    # Class imbalance handling via scale_pos_weight is binary in XGBoost.
    # For multi-class, we focus on robust metrics (macro F1, balanced accuracy)
    # and controlled model capacity to prevent overfitting.
    base_model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        num_class=len(class_names),
        random_state=42,
        n_jobs=-1,
        # Conservative defaults (grid will tune important ones)
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0
    )

    param_grid = {
        "n_estimators": [300, 600],
        "max_depth": [2, 3, 4],
        "learning_rate": [0.03, 0.05, 0.08],
        "min_child_weight": [1, 3, 5],
        "gamma": [0.0, 0.2]
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if use_stratify else 2

    grid = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="f1_macro",          # robust under imbalance across classes
        cv=cv,
        n_jobs=-1,
        verbose=0
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # --------------------------------------------------------
    # 7. PREDICTION & METRICS
    # --------------------------------------------------------
    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)

    # Core metrics
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    acc = float(accuracy_score(y_test, y_pred))

    # High-risk focused metrics
    high_precision = float(precision_score(y_test, y_pred, labels=[high_class_idx], average="macro", zero_division=0))
    high_recall = float(recall_score(y_test, y_pred, labels=[high_class_idx], average="macro", zero_division=0))
    high_f1 = float(f1_score(y_test, y_pred, labels=[high_class_idx], average="macro", zero_division=0))

    metrics = {
        "Macro_F1": macro_f1,
        "Weighted_F1": weighted_f1,
        "Balanced_Accuracy": bal_acc,
        "Overall_Accuracy": acc,
        "HighRisk_Precision": high_precision,
        "HighRisk_Recall": high_recall,
        "HighRisk_F1": high_f1,
        "Best_params": grid.best_params_,
        "CV_Folds_Used": int(n_splits),
        "Class_Distribution": {class_names[i]: int(counts.iloc[i]) for i in range(len(class_names))}
    }

    # --------------------------------------------------------
    # 8. VISUALISATION STYLE (PROFESSIONAL)
    # --------------------------------------------------------
    sns.set_style("whitegrid")
    plt.rcParams["figure.dpi"] = 140

    # --------------------------------------------------------
    # 01 — CONFUSION MATRIX (Counts + Normalised)
    # --------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(9, 4.5))

    # Left: raw counts
    plt.subplot(1, 2, 1)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix (Counts)")
    plt.xlabel("Predicted Risk Class")
    plt.ylabel("Actual Risk Class")

    # Right: normalised
    plt.subplot(1, 2, 2)
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names
    )
    plt.title("Confusion Matrix (Normalised)")
    plt.xlabel("Predicted Risk Class")
    plt.ylabel("Actual Risk Class")

    plt.tight_layout()
    plt.savefig(f"{run_dir}/01_confusion_matrix.png")
    plt.close()

    # --------------------------------------------------------
    # 02 — PER-CLASS PRECISION & RECALL BAR CHART
    # --------------------------------------------------------
    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    precisions = [report[c]["precision"] for c in class_names]
    recalls = [report[c]["recall"] for c in class_names]

    x = np.arange(len(class_names))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width/2, precisions, width, label="Precision")
    plt.bar(x + width/2, recalls, width, label="Recall")

    plt.xticks(x, class_names)
    plt.ylim(0, 1.05)
    plt.title("Per-Class Detection Reliability (Precision & Recall)")
    plt.xlabel("Overload Risk Class")
    plt.ylabel("Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{run_dir}/02_precision_recall_by_class.png")
    plt.close()

    # --------------------------------------------------------
    # 03 — ROC CURVE (HIGH RISK vs REST)
    # --------------------------------------------------------
    # Convert test labels to binary: High vs Rest
    y_test_high = (y_test == high_class_idx).astype(int)
    y_prob_high = y_prob[:, high_class_idx]

    # If test set has only one class (rare edge case), skip ROC gracefully
    if len(np.unique(y_test_high)) == 2:
        fpr, tpr, _ = roc_curve(y_test_high, y_prob_high)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, lw=2, label=f"High Risk vs Rest (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title("ROC Curve — High Risk Detection Capability")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{run_dir}/03_roc_high_vs_rest.png")
        plt.close()

        metrics["HighRisk_ROC_AUC"] = float(roc_auc)
    else:
        metrics["HighRisk_ROC_AUC"] = None

    # --------------------------------------------------------
    # 04 — METRICS SUMMARY BAR CHART
    # --------------------------------------------------------
    metric_keys = [
        "Balanced_Accuracy",
        "Macro_F1",
        "HighRisk_Recall",
        "HighRisk_Precision",
        "Overall_Accuracy"
    ]
    metric_vals = [metrics[k] if metrics[k] is not None else 0 for k in metric_keys]

    plt.figure(figsize=(9, 5))
    bars = plt.bar(metric_keys, metric_vals)
    plt.ylim(0, 1.05)
    plt.title("Overload Risk Classification — Performance Summary")
    plt.xlabel("Evaluation Metric")
    plt.ylabel("Score")

    for b, v in zip(bars, metric_vals):
        plt.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", va="bottom", fontweight="bold")

    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(f"{run_dir}/04_metrics_summary.png")
    plt.close()

    # --------------------------------------------------------
    # 9. SAVE MODEL & METADATA
    # --------------------------------------------------------
    model_path = os.path.join(model_dir, "overload_risk_model.joblib")
    joblib.dump(best_model, model_path)

    with open(f"{run_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nOverload Risk Classification Training Complete")
    print(f"Model saved: {model_path}")
    print(f"Run results saved: {run_dir}")
    print(json.dumps(metrics, indent=4))

    return metrics


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================
if __name__ == "__main__":
    train_overload_risk_classifier()
