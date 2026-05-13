"""
=============================================================================
Telecom Customer Sentiment Analysis — NLP Sentiment Scoring (VADER)
=============================================================================
Script  : 03_nlp_sentiment.py
Author  : Data Science Team
Purpose : Applies VADER (Valence Aware Dictionary and sEntiment Reasoner)
          to the csat_verbatim_text column to produce:
            - vader_compound_score  : raw compound score [-1.0, 1.0]
            - vader_sentiment_label : Positive / Neutral / Negative
            - vader_confidence      : |compound| score as a proxy for certainty

          Compares VADER labels against the ground-truth sentiment_label
          and outputs a classification report + confusion matrix.

Run     : python src/03_nlp_sentiment.py
          (requires: pip install nltk)
Output  : data/telecom_sentiment_with_nlp.csv
          reports/figures/07_vader_confusion_matrix.png
=============================================================================
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── VADER setup ──────────────────────────────────────────────────────────────
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
except ImportError:
    sys.exit("Please install nltk: pip install nltk")

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = Path(__file__).parent.parent / "data" / "telecom_sentiment_dataset.csv"
OUT_PATH    = Path(__file__).parent.parent / "data" / "telecom_sentiment_with_nlp.csv"
FIG_DIR     = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# VADER thresholds (industry standard):
#   compound >= 0.05  → Positive
#   compound <= -0.05 → Negative
#   else              → Neutral
VADER_POS_THRESHOLD =  0.05
VADER_NEG_THRESHOLD = -0.05


# ─────────────────────────────────────────────
# VADER SCORING FUNCTIONS
# ─────────────────────────────────────────────

def get_vader_scores(text: str, sia: SentimentIntensityAnalyzer) -> dict:
    """
    Score a single text string with VADER.

    Returns a dict with:
        compound : overall sentiment polarity score [-1.0, 1.0]
        pos      : proportion of positive tokens
        neu      : proportion of neutral tokens
        neg      : proportion of negative tokens
    """
    return sia.polarity_scores(str(text))


def compound_to_label(compound: float) -> str:
    """Convert VADER compound score to a sentiment class label."""
    if compound >= VADER_POS_THRESHOLD:
        return "Positive"
    elif compound <= VADER_NEG_THRESHOLD:
        return "Negative"
    else:
        return "Neutral"


def apply_vader(df: pd.DataFrame, text_col: str = "csat_verbatim_text") -> pd.DataFrame:
    """
    Apply VADER to every row in the DataFrame and append scoring columns.

    New columns added:
        vader_compound_score  : float, compound polarity [-1.0, 1.0]
        vader_pos             : float, positive token ratio
        vader_neu             : float, neutral token ratio
        vader_neg             : float, negative token ratio
        vader_sentiment_label : str,   Positive / Neutral / Negative
        vader_confidence      : float, abs(compound) — proxy for certainty
    """
    sia = SentimentIntensityAnalyzer()

    scores = df[text_col].apply(lambda t: get_vader_scores(t, sia))
    scores_df = pd.DataFrame(scores.tolist(), index=df.index)

    df = df.copy()
    df["vader_compound_score"]  = scores_df["compound"].round(4)
    df["vader_pos"]             = scores_df["pos"].round(4)
    df["vader_neu"]             = scores_df["neu"].round(4)
    df["vader_neg"]             = scores_df["neg"].round(4)
    df["vader_sentiment_label"] = df["vader_compound_score"].apply(compound_to_label)
    df["vader_confidence"]      = df["vader_compound_score"].abs().round(4)

    return df


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_vader(df: pd.DataFrame) -> None:
    """
    Compare VADER-predicted labels against ground-truth sentiment_label.
    Prints a classification report and saves a confusion matrix plot.

    Note: Because csat_verbatim_text in this dataset uses templated phrases
    (not raw free-form text), VADER accuracy will be high by design.
    In production, accuracy on raw verbatim text typically ranges 70–82%.
    """
    y_true = df["sentiment_label"]
    y_pred = df["vader_sentiment_label"]

    acc = accuracy_score(y_true, y_pred)
    print(f"\n── VADER Accuracy vs Ground Truth: {acc * 100:.2f}% ──")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Negative", "Neutral", "Positive"]))

    # Confusion matrix plot
    cm = confusion_matrix(y_true, y_pred, labels=["Positive", "Neutral", "Negative"])
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Positive", "Neutral", "Negative"],
        yticklabels=["Positive", "Neutral", "Negative"],
        linewidths=0.5, ax=ax
    )
    ax.set_title("VADER Sentiment — Confusion Matrix\n(Predicted vs Ground Truth)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Ground Truth", fontsize=11)
    ax.set_xlabel("VADER Prediction", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "07_vader_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 07_vader_confusion_matrix.png")


# ─────────────────────────────────────────────
# COMPOUND SCORE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_compound_distribution(df: pd.DataFrame) -> None:
    """Plot the distribution of VADER compound scores by true sentiment tier."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    palette = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}

    for ax, label in zip(axes, ["Positive", "Neutral", "Negative"]):
        subset = df[df["sentiment_label"] == label]["vader_compound_score"]
        ax.hist(subset, bins=30, color=palette[label], edgecolor="white", alpha=0.85)
        ax.set_title(f"{label} Sentiment\n(n={len(subset):,})", fontweight="bold")
        ax.set_xlabel("VADER Compound Score")
        ax.axvline(0.05, color="green", linestyle="--", linewidth=1, label="Pos threshold")
        ax.axvline(-0.05, color="red", linestyle="--", linewidth=1, label="Neg threshold")
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Count")
    plt.suptitle("VADER Compound Score Distribution by Sentiment Tier",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "08_vader_compound_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 08_vader_compound_distribution.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("Loading data ...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df):,} rows")

    print("\nApplying VADER sentiment scoring ...")
    df = apply_vader(df, text_col="csat_verbatim_text")

    evaluate_vader(df)
    plot_compound_distribution(df)

    df.to_csv(OUT_PATH, index=False)
    print(f"\n✅ NLP-enriched dataset saved to: {OUT_PATH}")

    # Summary comparison
    print("\n── Sentiment label comparison (ground truth vs VADER) ──")
    comparison = pd.crosstab(
        df["sentiment_label"], df["vader_sentiment_label"],
        rownames=["Ground Truth"], colnames=["VADER Prediction"],
        normalize="index"
    ).round(3).mul(100)
    print(comparison.to_string())


if __name__ == "__main__":
    main()
