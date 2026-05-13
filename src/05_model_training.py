"""
=============================================================================
Telecom Customer Sentiment Analysis — Model Training & Evaluation
=============================================================================
Script  : 05_model_training.py
Author  : Data Science Team
Purpose : Trains a Random Forest Classifier on the preprocessed feature
          matrix to predict customer churn. Includes:
            - Baseline model training
            - Hyperparameter tuning via RandomizedSearchCV
            - Full evaluation: accuracy, ROC-AUC, precision, recall, F1
            - Feature importance ranking (Gini impurity based)
            - SHAP value computation for explainability
            - Model persistence (.pkl)

Run     : python src/05_model_training.py
          (requires: pip install scikit-learn shap joblib matplotlib)
Output  : models/random_forest_churn_model.pkl
          reports/figures/09_feature_importance.png
          reports/figures/10_roc_curve.png
          reports/figures/11_confusion_matrix.png
          reports/model_evaluation_report.txt
=============================================================================
"""

import os
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
)
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_DIR    = Path(__file__).parent.parent / "data"
MODEL_DIR   = Path(__file__).parent.parent / "models"
FIG_DIR     = Path(__file__).parent.parent / "reports" / "figures"
REPORT_DIR  = Path(__file__).parent.parent / "reports"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE    = 42
CV_FOLDS        = 5
N_ITER_SEARCH   = 30       # number of hyperparameter combinations to try

# ─────────────────────────────────────────────
# HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────
PARAM_DISTRIBUTIONS = {
    "n_estimators":       [100, 200, 300, 400, 500],
    "max_depth":          [None, 10, 20, 30, 40],
    "min_samples_split":  [2, 5, 10],
    "min_samples_leaf":   [1, 2, 4],
    "max_features":       ["sqrt", "log2", 0.3, 0.5],
    "class_weight":       [None, "balanced"],       # "balanced" handles churn class imbalance
    "bootstrap":          [True, False],
}


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_splits(data_dir: Path) -> tuple:
    X_train = pd.read_csv(data_dir / "X_train.csv")
    X_test  = pd.read_csv(data_dir / "X_test.csv")
    y_train = pd.read_csv(data_dir / "y_train.csv").squeeze()
    y_test  = pd.read_csv(data_dir / "y_test.csv").squeeze()
    with open(data_dir / "feature_names.json") as f:
        feature_names = json.load(f)
    print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
    print(f"Train churn rate: {y_train.mean()*100:.2f}%  |  Test churn rate: {y_test.mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test, feature_names


# ─────────────────────────────────────────────
# BASELINE MODEL
# ─────────────────────────────────────────────

def train_baseline(X_train, y_train) -> RandomForestClassifier:
    """
    Train a baseline Random Forest with sensible default hyperparameters.
    Used to establish a performance floor before tuning.
    """
    print("\n── Training baseline Random Forest ──")
    baseline = RandomForestClassifier(
        n_estimators=100,
        max_depth=None,
        class_weight="balanced",    # compensate for class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,                  # use all CPU cores
    )
    baseline.fit(X_train, y_train)
    print("  Baseline training complete.")
    return baseline


# ─────────────────────────────────────────────
# HYPERPARAMETER TUNING
# ─────────────────────────────────────────────

def tune_model(X_train, y_train) -> RandomForestClassifier:
    """
    Randomized hyperparameter search with stratified k-fold cross-validation.

    Scoring metric: roc_auc — appropriate for imbalanced binary classification.
    RandomizedSearchCV is preferred over GridSearchCV because:
        - Faster: samples N random combinations vs exhaustive grid
        - Equally effective: shown to match or exceed grid search performance
          (Bergstra & Bengio, 2012)
    """
    print(f"\n── Hyperparameter tuning (n_iter={N_ITER_SEARCH}, cv={CV_FOLDS}) ──")
    print("  This may take a few minutes ...")

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_distributions=PARAM_DISTRIBUTIONS,
        n_iter=N_ITER_SEARCH,
        scoring="roc_auc",
        cv=cv,
        refit=True,          # refit best model on full training set
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    print(f"\n  Best ROC-AUC (CV): {search.best_score_:.4f}")
    print(f"  Best parameters:")
    for k, v in search.best_params_.items():
        print(f"    {k}: {v}")

    return search.best_estimator_


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_model(model: RandomForestClassifier, X_train, X_test,
                   y_train, y_test) -> dict:
    """
    Full evaluation suite:
        - Train / test accuracy
        - ROC-AUC
        - Precision, Recall, F1 per class
        - Cross-validated ROC-AUC on training set
    """
    print("\n── Model Evaluation ──")

    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Core metrics
    test_acc    = accuracy_score(y_test, y_pred)
    roc_auc     = roc_auc_score(y_test, y_prob)
    avg_prec    = average_precision_score(y_test, y_prob)
    train_acc   = accuracy_score(y_train, model.predict(X_train))

    # Cross-validated AUC on training set (generalisation check)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv,
                             scoring="roc_auc", n_jobs=-1)

    print(f"\n  Train Accuracy : {train_acc * 100:.2f}%")
    print(f"  Test  Accuracy : {test_acc * 100:.2f}%")
    print(f"  Test  ROC-AUC  : {roc_auc:.4f}")
    print(f"  Avg Precision  : {avg_prec:.4f}")
    print(f"  CV ROC-AUC     : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Retained", "Churned"]))

    results = {
        "train_accuracy": train_acc,
        "test_accuracy":  test_acc,
        "roc_auc":        roc_auc,
        "avg_precision":  avg_prec,
        "cv_auc_mean":    cv_auc.mean(),
        "cv_auc_std":     cv_auc.std(),
        "y_pred":         y_pred,
        "y_prob":         y_prob,
    }
    return results


# ─────────────────────────────────────────────
# VISUALISATIONS
# ─────────────────────────────────────────────

def plot_feature_importance(model: RandomForestClassifier, feature_names: list,
                             top_n: int = 25) -> None:
    """Bar chart of top N features by Gini importance."""
    importances = pd.Series(model.feature_importances_, index=feature_names)
    top = importances.nlargest(top_n).sort_values()

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#ef4444" if i >= len(top) - 5 else "#1B3A6B" for i in range(len(top))]
    top.plot(kind="barh", ax=ax, color=colors)
    ax.set_title(f"Top {top_n} Feature Importances (Gini Impurity Reduction)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Decrease in Gini Impurity", fontsize=11)
    ax.axvline(top.mean(), color="orange", linestyle="--", linewidth=1.5, label="Mean importance")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "09_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 09_feature_importance.png")


def plot_roc_curve(y_test, y_prob) -> None:
    """ROC curve with AUC annotation."""
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="#1B3A6B", linewidth=2.5, label=f"Random Forest (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random baseline (AUC = 0.50)")
    ax.fill_between(fpr, tpr, alpha=0.08, color="#1B3A6B")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curve — Churn Prediction Model", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    plt.savefig(FIG_DIR / "10_roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 10_roc_curve.png")


def plot_confusion_matrix(y_test, y_pred) -> None:
    """Normalised + count confusion matrix side by side."""
    cm      = confusion_matrix(y_test, y_pred)
    cm_norm = confusion_matrix(y_test, y_pred, normalize="true")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    labels = ["Retained", "Churned"]

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, linewidths=0.5, ax=axes[0])
    axes[0].set_title("Confusion Matrix (Counts)", fontweight="bold")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=labels,
                yticklabels=labels, linewidths=0.5, ax=axes[1])
    axes[1].set_title("Confusion Matrix (Normalised)", fontweight="bold")
    axes[1].set_ylabel("Actual")
    axes[1].set_xlabel("Predicted")

    plt.suptitle("Random Forest — Churn Prediction", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "11_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 11_confusion_matrix.png")


def plot_churn_score_distribution(y_test, y_prob) -> None:
    """Predicted probability distribution separated by actual churn label."""
    fig, ax = plt.subplots(figsize=(9, 5))
    retained = y_prob[y_test == 0]
    churned  = y_prob[y_test == 1]

    ax.hist(retained, bins=50, alpha=0.6, color="#3b82f6", label="Retained (actual)")
    ax.hist(churned,  bins=50, alpha=0.6, color="#ef4444", label="Churned (actual)")
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.5, label="Default threshold (0.5)")
    ax.set_xlabel("Predicted Churn Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Predicted Churn Probability Distribution by Actual Label",
                 fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "12_churn_probability_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 12_churn_probability_distribution.png")


# ─────────────────────────────────────────────
# SHAP EXPLAINABILITY (optional — requires shap)
# ─────────────────────────────────────────────

def compute_shap_values(model: RandomForestClassifier, X_test: pd.DataFrame,
                        feature_names: list, n_samples: int = 500) -> None:
    """
    Compute SHAP TreeExplainer values for a sample of test rows.
    Produces a summary plot showing feature impact on model output.

    SHAP (SHapley Additive exPlanations) quantifies how much each feature
    contributes to a specific prediction, enabling individual-level
    explainability in addition to global feature importance.
    """
    try:
        import shap
    except ImportError:
        print("  shap not installed — skipping SHAP analysis. Run: pip install shap")
        return

    print(f"\n── Computing SHAP values (sample n={n_samples}) ──")
    sample = X_test.sample(n=min(n_samples, len(X_test)), random_state=RANDOM_STATE)

    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # For binary classification, shap_values is a list [class_0, class_1]
    shap_class1 = shap_values[1] if isinstance(shap_values, list) else shap_values

    # Summary bar plot (global feature importance via mean |SHAP|)
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_class1, sample, feature_names=feature_names,
        plot_type="bar", show=False, max_display=20
    )
    plt.title("SHAP Feature Importance (Mean |SHAP Value|) — Churn Class",
              fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "13_shap_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 13_shap_feature_importance.png")


# ─────────────────────────────────────────────
# SAVE MODEL & REPORT
# ─────────────────────────────────────────────

def save_model(model: RandomForestClassifier, model_dir: Path) -> None:
    path = model_dir / "random_forest_churn_model.pkl"
    joblib.dump(model, path)
    print(f"\n✅ Model saved to: {path}")


def write_evaluation_report(results: dict, model: RandomForestClassifier,
                             feature_names: list, report_dir: Path) -> None:
    path = report_dir / "model_evaluation_report.txt"
    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("RANDOM FOREST CHURN PREDICTION — EVALUATION REPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Train Accuracy : {results['train_accuracy']*100:.2f}%\n")
        f.write(f"Test  Accuracy : {results['test_accuracy']*100:.2f}%\n")
        f.write(f"Test  ROC-AUC  : {results['roc_auc']:.4f}\n")
        f.write(f"Avg  Precision : {results['avg_precision']:.4f}\n")
        f.write(f"CV   ROC-AUC   : {results['cv_auc_mean']:.4f} ± {results['cv_auc_std']:.4f}\n\n")

        f.write("── Best Parameters ──\n")
        for k, v in model.get_params().items():
            f.write(f"  {k}: {v}\n")

        f.write("\n── Top 20 Features by Gini Importance ──\n")
        importances = sorted(zip(feature_names, model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        for name, imp in importances[:20]:
            f.write(f"  {imp:.4f}  {name}\n")

    print(f"Evaluation report saved to: {path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    X_train, X_test, y_train, y_test, feature_names = load_splits(DATA_DIR)

    # Train baseline first
    baseline = train_baseline(X_train, y_train)

    # Tune
    best_model = tune_model(X_train, y_train)

    # Evaluate tuned model
    results = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    # Plots
    plot_feature_importance(best_model, feature_names, top_n=25)
    plot_roc_curve(y_test, results["y_prob"])
    plot_confusion_matrix(y_test, results["y_pred"])
    plot_churn_score_distribution(y_test, results["y_prob"])
    compute_shap_values(best_model, X_test, feature_names)

    # Persist
    save_model(best_model, MODEL_DIR)
    write_evaluation_report(results, best_model, feature_names, REPORT_DIR)

    print("\n🎯 Model training and evaluation complete.")


if __name__ == "__main__":
    main()
