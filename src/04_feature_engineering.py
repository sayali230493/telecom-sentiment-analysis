"""
=============================================================================
Telecom Customer Sentiment Analysis — Feature Engineering
=============================================================================
Script  : 04_feature_engineering.py
Author  : Data Science Team
Purpose : Prepares the enriched dataset for model training by:
            1. Encoding categorical variables (one-hot + ordinal)
            2. Scaling numeric features (StandardScaler)
            3. Verifying engineered feature integrity
            4. Producing a model-ready feature matrix (X) and target (y)
            5. Splitting into train / validation / test sets
            6. Saving processed artifacts for use in model training

Run     : python src/04_feature_engineering.py
          (requires: pip install scikit-learn pandas numpy)
Output  : data/X_train.csv, data/X_test.csv, data/y_train.csv, data/y_test.csv
          reports/feature_summary.txt
=============================================================================
"""

import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH       = Path(__file__).parent.parent / "data" / "telecom_sentiment_with_nlp.csv"
OUT_DIR         = Path(__file__).parent.parent / "data"
REPORT_DIR      = Path(__file__).parent.parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL      = "churned"
TEST_SIZE       = 0.20      # 80/20 train-test split
RANDOM_STATE    = 42

# Columns to drop before modelling (IDs, raw text, redundant, or data-leakage columns)
DROP_COLS = [
    "customer_id",          # identifier only
    "csat_verbatim_text",   # raw text — processed by NLP separately
    "interaction_date",     # date — could be used for time-series, dropped for baseline
    "sentiment_score",      # generated from same source as sentiment_label (leakage risk)
    "sentiment_label",      # encoded separately below
    "vader_sentiment_label",# redundant with vader_compound_score
    "churned",              # target variable
]

# Categorical columns → one-hot encoded
OHE_COLS = [
    "region",
    "plan_type",
    "contract_type",
    "payment_method",
    "device_type",
    "support_channel",
    "issue_type",
]

# Ordinal columns → mapped to integers
ORDINAL_MAPS = {
    "loyalty_tier":    {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3},
    "sentiment_label": {"Negative": 0, "Neutral": 1, "Positive": 2},
}

# Numeric columns → standardised (mean=0, std=1)
NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges_usd",
    "total_charges_usd",
    "avg_monthly_data_gb",
    "num_support_contacts",
    "nps_score",
    "csat_score",
    "churn_risk_score",
    "revenue_per_tenure_month",
    "support_intensity_index",
    "weighted_sentiment_index",
    "vader_compound_score",
    "vader_confidence",
    "vader_pos",
    "vader_neu",
    "vader_neg",
]

# Boolean columns → int
BOOL_COLS = ["first_call_resolution"]


# ─────────────────────────────────────────────
# LOADING
# ─────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# FEATURE ENGINEERING VERIFICATION
# ─────────────────────────────────────────────

def verify_engineered_features(df: pd.DataFrame) -> None:
    """
    Re-validate the three engineered features to confirm formula integrity.
    Logs a warning if any row deviates by more than floating-point tolerance.
    """
    print("\n── Verifying engineered feature formulas ──")
    tol = 0.01  # tolerance for float rounding

    # 1. Weighted Sentiment Index
    norm_csat   = (df["csat_score"] - 1) / 4
    norm_nps    = (df["nps_score"] + 100) / 200
    fcr_penalty = df["first_call_resolution"].map({True: 0.0, False: 0.2})
    expected_wsi = (0.40 * norm_csat + 0.35 * norm_nps + 0.25 * (1 - fcr_penalty)).round(4)
    wsi_diff = (df["weighted_sentiment_index"] - expected_wsi).abs()
    print(f"  WSI  — max deviation: {wsi_diff.max():.6f}  {'✅ OK' if wsi_diff.max() < tol else '⚠ WARN'}")

    # 2. Support Intensity Index
    expected_sii = (df["num_support_contacts"] / (df["tenure_months"] + 1)).round(4)
    sii_diff = (df["support_intensity_index"] - expected_sii).abs()
    print(f"  SII  — max deviation: {sii_diff.max():.6f}  {'✅ OK' if sii_diff.max() < tol else '⚠ WARN'}")

    # 3. Revenue per Tenure Month
    expected_rpm = (df["total_charges_usd"] / df["tenure_months"]).round(2)
    rpm_diff = (df["revenue_per_tenure_month"] - expected_rpm).abs()
    print(f"  RPM  — max deviation: {rpm_diff.max():.6f}  {'✅ OK' if rpm_diff.max() < tol else '⚠ WARN'}")


# ─────────────────────────────────────────────
# PREPROCESSING PIPELINE
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, StandardScaler, list]:
    """
    Full preprocessing pipeline.

    Steps:
        1. Extract target variable y
        2. Apply ordinal encoding to loyalty_tier and sentiment_label
        3. Convert booleans to int
        4. One-hot encode categorical columns
        5. Standardise numeric columns
        6. Drop irrelevant columns
        7. Return (X, y, fitted_scaler, feature_names)
    """
    y = df[TARGET_COL].copy()

    # Work on a copy
    X = df.drop(columns=DROP_COLS, errors="ignore").copy()

    # Step 1 — Ordinal encode
    for col, mapping in ORDINAL_MAPS.items():
        if col in X.columns:
            X[col] = X[col].map(mapping)
            print(f"  Ordinal encoded: {col}  {mapping}")

    # Step 2 — Boolean → int
    for col in BOOL_COLS:
        if col in X.columns:
            X[col] = X[col].astype(int)

    # Step 3 — One-hot encode
    X = pd.get_dummies(X, columns=[c for c in OHE_COLS if c in X.columns], drop_first=False)
    print(f"  After OHE: {X.shape[1]} columns")

    # Step 4 — Scale numerics
    numeric_present = [c for c in NUMERIC_COLS if c in X.columns]
    scaler = StandardScaler()
    X[numeric_present] = scaler.fit_transform(X[numeric_present])
    print(f"  Standardised {len(numeric_present)} numeric columns")

    feature_names = list(X.columns)
    print(f"  Total features in model matrix: {len(feature_names)}")

    return X, y, scaler, feature_names


# ─────────────────────────────────────────────
# TRAIN / TEST SPLIT
# ─────────────────────────────────────────────

def split_data(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Stratified 80/20 train-test split.
    Stratification ensures both splits maintain the same churn ratio.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n── Data split ──")
    print(f"  Train: {len(X_train):,} rows  |  Churn rate: {y_train.mean()*100:.2f}%")
    print(f"  Test:  {len(X_test):,} rows  |  Churn rate: {y_test.mean()*100:.2f}%")
    return X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────
# SAVE ARTIFACTS
# ─────────────────────────────────────────────

def save_splits(X_train, X_test, y_train, y_test, feature_names: list, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(out_dir / "X_train.csv", index=False)
    X_test.to_csv(out_dir / "X_test.csv", index=False)
    y_train.to_csv(out_dir / "y_train.csv", index=False)
    y_test.to_csv(out_dir / "y_test.csv", index=False)

    # Save feature names for model loading
    with open(out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"\n✅ Saved splits + feature names to: {out_dir}")


def write_feature_report(feature_names: list, report_dir: Path) -> None:
    report_path = report_dir / "feature_summary.txt"
    with open(report_path, "w") as f:
        f.write("FEATURE SUMMARY — TELECOM SENTIMENT MODEL\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total features: {len(feature_names)}\n\n")
        for i, name in enumerate(feature_names, 1):
            f.write(f"  {i:03d}. {name}\n")
    print(f"Feature summary written to: {report_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    df = load_data(DATA_PATH)
    verify_engineered_features(df)

    print("\n── Preprocessing ──")
    X, y, scaler, feature_names = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    save_splits(X_train, X_test, y_train, y_test, feature_names, OUT_DIR)
    write_feature_report(feature_names, REPORT_DIR)


if __name__ == "__main__":
    main()
