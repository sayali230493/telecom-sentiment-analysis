"""
=============================================================================
Telecom Customer Sentiment Analysis — Inference / Scoring Pipeline
=============================================================================
Script  : 06_score_new_customers.py
Author  : Data Science Team
Purpose : Loads the trained Random Forest model and applies it to new or
          unseen customer records to:
            - Predict churn probability (0–100 score)
            - Assign a risk tier (Low / Medium / High / Critical)
            - Produce a scored output CSV for CRM ingestion or
              retention team triage

          This script is designed to be the operationalisation layer —
          it can be wrapped in a FastAPI endpoint or called from an
          ETL scheduler (Airflow, dbt, etc.).

Run     : python src/06_score_new_customers.py
          (or pass a custom CSV path via --input flag)
Output  : data/scored_customers.csv
=============================================================================
"""

import argparse
import json
import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
MODEL_PATH      = BASE_DIR / "models" / "random_forest_churn_model.pkl"
FEATURE_PATH    = BASE_DIR / "data" / "feature_names.json"
DEFAULT_INPUT   = BASE_DIR / "data" / "telecom_sentiment_with_nlp.csv"
OUTPUT_PATH     = BASE_DIR / "data" / "scored_customers.csv"

# Risk tier thresholds (churn probability × 100)
RISK_TIERS = {
    "Critical": (80, 100),
    "High":     (60,  80),
    "Medium":   (40,  60),
    "Low":      (0,   40),
}


# ─────────────────────────────────────────────
# LOAD ARTEFACTS
# ─────────────────────────────────────────────

def load_model(model_path: Path) -> object:
    model = joblib.load(model_path)
    print(f"Model loaded: {model_path.name}")
    return model


def load_feature_names(feature_path: Path) -> list:
    with open(feature_path) as f:
        return json.load(f)


# ─────────────────────────────────────────────
# PREPROCESSING (mirrors 04_feature_engineering.py)
# ─────────────────────────────────────────────

DROP_COLS = [
    "customer_id", "csat_verbatim_text", "interaction_date",
    "sentiment_score", "sentiment_label", "vader_sentiment_label",
    "churned",
]

OHE_COLS = [
    "region", "plan_type", "contract_type", "payment_method",
    "device_type", "support_channel", "issue_type",
]

ORDINAL_MAPS = {
    "loyalty_tier":    {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3},
    "sentiment_label": {"Negative": 0, "Neutral": 1, "Positive": 2},
}

NUMERIC_COLS = [
    "tenure_months", "monthly_charges_usd", "total_charges_usd",
    "avg_monthly_data_gb", "num_support_contacts",
    "nps_score", "csat_score", "churn_risk_score",
    "revenue_per_tenure_month", "support_intensity_index",
    "weighted_sentiment_index", "vader_compound_score",
    "vader_confidence", "vader_pos", "vader_neu", "vader_neg",
]


def preprocess_for_inference(df_raw: pd.DataFrame,
                              feature_names: list) -> pd.DataFrame:
    """
    Apply the same transformations used during training so that the
    feature matrix aligns with what the model expects.

    Key invariant: output column set must exactly match feature_names.
    Missing OHE columns (e.g., unseen categories) are zero-filled.
    """
    df = df_raw.copy()

    # Ordinal encoding
    for col, mapping in ORDINAL_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Boolean → int
    if "first_call_resolution" in df.columns:
        df["first_call_resolution"] = df["first_call_resolution"].astype(int)

    # One-hot encode
    df = pd.get_dummies(df, columns=[c for c in OHE_COLS if c in df.columns], drop_first=False)

    # Drop irrelevant columns
    df.drop(columns=[c for c in DROP_COLS if c in df.columns], inplace=True, errors="ignore")

    # Align columns to training feature set
    # Add missing columns as 0 (handles unseen OHE categories)
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    # Remove any extra columns not seen during training
    df = df[feature_names]

    return df


# ─────────────────────────────────────────────
# SCORING
# ─────────────────────────────────────────────

def assign_risk_tier(prob: float) -> str:
    """Map a churn probability (0–1) to a labelled risk tier."""
    score = prob * 100
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= score <= hi:
            return tier
    return "Low"


def score_customers(model, df_features: pd.DataFrame) -> pd.DataFrame:
    """
    Generate churn probability scores and risk tiers for each customer.

    Returns a DataFrame with:
        churn_probability   : model's predicted probability of churn [0.0–1.0]
        churn_score_100     : churn_probability × 100 for readability
        risk_tier           : Low / Medium / High / Critical
        predicted_churn     : binary prediction at 0.5 threshold
    """
    probs  = model.predict_proba(df_features)[:, 1]
    preds  = (probs >= 0.5).astype(int)

    result = pd.DataFrame({
        "churn_probability": probs.round(4),
        "churn_score_100":   (probs * 100).round(2),
        "risk_tier":         [assign_risk_tier(p) for p in probs],
        "predicted_churn":   preds,
    })
    return result


# ─────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────

def print_scoring_summary(scored: pd.DataFrame) -> None:
    print("\n── Scoring Summary ──")
    tier_counts = scored["risk_tier"].value_counts()
    total = len(scored)
    for tier in ["Critical", "High", "Medium", "Low"]:
        count = tier_counts.get(tier, 0)
        pct   = count / total * 100
        print(f"  {tier:10s}: {count:6,}  ({pct:.1f}%)")
    print(f"\n  Avg churn probability : {scored['churn_probability'].mean():.4f}")
    print(f"  % flagged for churn   : {scored['predicted_churn'].mean()*100:.2f}%")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(input_path: Path = DEFAULT_INPUT):
    print(f"\nScoring customers from: {input_path}")

    # Load artefacts
    model         = load_model(MODEL_PATH)
    feature_names = load_feature_names(FEATURE_PATH)

    # Load raw customer data
    df_raw = pd.read_csv(input_path)
    print(f"Input records: {len(df_raw):,}")

    # Keep identifiers separate
    id_cols = df_raw[["customer_id"]].copy() if "customer_id" in df_raw.columns else pd.DataFrame()

    # Preprocess
    df_features = preprocess_for_inference(df_raw, feature_names)

    # Score
    scores = score_customers(model, df_features)

    # Combine with identifiers and key fields for CRM
    output_cols = [
        "customer_id", "region", "plan_type", "contract_type",
        "loyalty_tier", "sentiment_label", "issue_type",
        "weighted_sentiment_index", "num_support_contacts",
        "interaction_date",
    ]
    available_cols = [c for c in output_cols if c in df_raw.columns]
    output = pd.concat([df_raw[available_cols].reset_index(drop=True), scores], axis=1)

    # Sort by risk (Critical first) for triage
    tier_order = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
    output["_sort"] = output["risk_tier"].map(tier_order)
    output.sort_values("_sort", inplace=True)
    output.drop(columns=["_sort"], inplace=True)

    print_scoring_summary(scores)

    # Save
    output.to_csv(OUTPUT_PATH, index=False)
    print(f"\n✅ Scored output saved to: {OUTPUT_PATH}")
    print(f"   {len(output[output['risk_tier']=='Critical']):,} Critical-risk customers flagged for immediate outreach.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score customer churn risk")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                        help="Path to input CSV with customer records")
    args = parser.parse_args()
    main(input_path=args.input)
