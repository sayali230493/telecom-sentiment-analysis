"""
=============================================================================
Telecom Customer Sentiment Analysis — Exploratory Data Analysis (EDA)
=============================================================================
Script  : 02_eda.py
Author  : Data Science Team
Purpose : Performs comprehensive EDA on the telecom sentiment dataset.
          Produces summary statistics, class balance checks, correlation
          analysis, and key visualizations saved to reports/figures/.

Run     : python src/02_eda.py
Output  : Console summary + reports/figures/*.png
=============================================================================
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = Path(__file__).parent.parent / "data" / "telecom_sentiment_dataset.csv"
FIG_DIR     = Path(__file__).parent.parent / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE     = {"Positive": "#22c55e", "Neutral": "#f59e0b", "Negative": "#ef4444"}
BLUE        = "#1B3A6B"
sns.set_theme(style="whitegrid", palette="muted")


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["interaction_date"])
    df["first_call_resolution"] = df["first_call_resolution"].astype(bool)
    print(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────
# 1. BASIC SUMMARY
# ─────────────────────────────────────────────

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(df.describe(include="all").T.to_string())

    print("\n── Missing values ──")
    print(df.isnull().sum().to_string())

    print("\n── Target class balance (churned) ──")
    vc = df["churned"].value_counts()
    print(vc.to_string())
    print(f"Churn rate: {vc[1] / len(df) * 100:.2f}%")

    print("\n── Sentiment distribution ──")
    print(df["sentiment_label"].value_counts(normalize=True).mul(100).round(2).to_string())


# ─────────────────────────────────────────────
# 2. UNIVARIATE — NUMERIC
# ─────────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    numeric_cols = [
        "tenure_months", "monthly_charges_usd", "total_charges_usd",
        "avg_monthly_data_gb", "num_support_contacts",
        "nps_score", "csat_score", "churn_risk_score",
        "sentiment_score", "support_intensity_index",
        "revenue_per_tenure_month", "weighted_sentiment_index",
    ]
    fig, axes = plt.subplots(4, 3, figsize=(16, 14))
    axes = axes.flatten()

    for ax, col in zip(axes, numeric_cols):
        ax.hist(df[col], bins=40, color=BLUE, alpha=0.75, edgecolor="white")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count", fontsize=8)
        ax.tick_params(labelsize=8)

    plt.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_numeric_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 01_numeric_distributions.png")


# ─────────────────────────────────────────────
# 3. SENTIMENT DISTRIBUTION
# ─────────────────────────────────────────────

def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Overall pie
    counts = df["sentiment_label"].value_counts()
    axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=[PALETTE[k] for k in counts.index],
        startangle=140, wedgeprops={"edgecolor": "white", "linewidth": 2}
    )
    axes[0].set_title("Overall Sentiment Split", fontweight="bold")

    # By region — stacked bar
    region_sent = (
        df.groupby(["region", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda r: r / r.sum() * 100, axis=1)
    )
    region_sent[["Positive", "Neutral", "Negative"]].plot(
        kind="barh", stacked=True, ax=axes[1],
        color=[PALETTE["Positive"], PALETTE["Neutral"], PALETTE["Negative"]]
    )
    axes[1].set_title("Sentiment by Region (%)", fontweight="bold")
    axes[1].set_xlabel("Percentage")
    axes[1].xaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].legend(loc="lower right", fontsize=8)

    # By plan type — grouped bar
    plan_sent = (
        df.groupby(["plan_type", "sentiment_label"])
        .size()
        .unstack(fill_value=0)
        .apply(lambda r: r / r.sum() * 100, axis=1)
    )
    plan_sent[["Positive", "Neutral", "Negative"]].plot(
        kind="bar", ax=axes[2], rot=0,
        color=[PALETTE["Positive"], PALETTE["Neutral"], PALETTE["Negative"]]
    )
    axes[2].set_title("Sentiment by Plan Type (%)", fontweight="bold")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Percentage")
    axes[2].legend(fontsize=8)

    plt.suptitle("Sentiment Distribution Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_sentiment_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 02_sentiment_distribution.png")


# ─────────────────────────────────────────────
# 4. CHURN ANALYSIS
# ─────────────────────────────────────────────

def plot_churn_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Churn rate by sentiment
    churn_by_sent = df.groupby("sentiment_label")["churned"].mean() * 100
    churn_by_sent = churn_by_sent.reindex(["Positive", "Neutral", "Negative"])
    axes[0, 0].bar(
        churn_by_sent.index, churn_by_sent.values,
        color=[PALETTE[k] for k in churn_by_sent.index]
    )
    axes[0, 0].set_title("Churn Rate by Sentiment (%)", fontweight="bold")
    axes[0, 0].set_ylabel("Churn Rate (%)")
    for i, v in enumerate(churn_by_sent.values):
        axes[0, 0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    # Churn rate by contract type
    churn_by_contract = df.groupby("contract_type")["churned"].mean() * 100
    axes[0, 1].bar(churn_by_contract.index, churn_by_contract.values, color=BLUE, alpha=0.8)
    axes[0, 1].set_title("Churn Rate by Contract Type (%)", fontweight="bold")
    axes[0, 1].set_ylabel("Churn Rate (%)")
    for i, v in enumerate(churn_by_contract.values):
        axes[0, 1].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")

    # Churn risk score distribution by churned
    df[df["churned"] == 0]["churn_risk_score"].plot(
        kind="hist", bins=40, alpha=0.6, label="Retained", ax=axes[1, 0], color="#3b82f6"
    )
    df[df["churned"] == 1]["churn_risk_score"].plot(
        kind="hist", bins=40, alpha=0.6, label="Churned", ax=axes[1, 0], color="#ef4444"
    )
    axes[1, 0].set_title("Churn Risk Score: Retained vs Churned", fontweight="bold")
    axes[1, 0].set_xlabel("Churn Risk Score")
    axes[1, 0].legend()

    # WSI by churned
    df.boxplot(column="weighted_sentiment_index", by="churned", ax=axes[1, 1],
               boxprops=dict(color=BLUE), medianprops=dict(color="red", linewidth=2))
    axes[1, 1].set_title("Weighted Sentiment Index by Churn Status", fontweight="bold")
    axes[1, 1].set_xlabel("Churned (0=Retained, 1=Churned)")
    axes[1, 1].set_ylabel("WSI Score")
    plt.suptitle("")  # suppress auto-title from boxplot

    plt.suptitle("Churn Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_churn_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 03_churn_analysis.png")


# ─────────────────────────────────────────────
# 5. FEATURE CORRELATION HEATMAP
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        center=0, linewidths=0.5, ax=ax,
        annot_kws={"size": 8}, vmin=-1, vmax=1
    )
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 04_correlation_heatmap.png")


# ─────────────────────────────────────────────
# 6. ENGINEERED FEATURE ANALYSIS
# ─────────────────────────────────────────────

def plot_engineered_features(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # WSI by loyalty tier
    wsi_tier = df.groupby("loyalty_tier")["weighted_sentiment_index"].mean().reindex(
        ["Bronze", "Silver", "Gold", "Platinum"]
    )
    colors = ["#ef4444", "#f59e0b", "#3b82f6", "#22c55e"]
    axes[0].bar(wsi_tier.index, wsi_tier.values, color=colors)
    axes[0].set_title("Avg WSI by Loyalty Tier", fontweight="bold")
    axes[0].set_ylabel("Avg Weighted Sentiment Index")
    axes[0].set_ylim(0, 1)
    for i, v in enumerate(wsi_tier.values):
        axes[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")

    # Support Intensity Index by issue type
    sii_issue = df.groupby("issue_type")["support_intensity_index"].mean().sort_values(ascending=False)
    sii_issue.plot(kind="barh", ax=axes[1], color=BLUE, alpha=0.8)
    axes[1].set_title("Avg Support Intensity Index\nby Issue Type", fontweight="bold")
    axes[1].set_xlabel("Avg SII")

    # Revenue per tenure vs churn risk — scatter
    sample = df.sample(500, random_state=42)
    scatter_colors = [PALETTE[s] for s in sample["sentiment_label"]]
    axes[2].scatter(
        sample["revenue_per_tenure_month"], sample["churn_risk_score"],
        c=scatter_colors, alpha=0.5, edgecolors="none", s=30
    )
    axes[2].set_title("Revenue/Tenure vs Churn Risk\n(colored by sentiment)", fontweight="bold")
    axes[2].set_xlabel("Revenue per Tenure Month ($)")
    axes[2].set_ylabel("Churn Risk Score")
    # Manual legend
    from matplotlib.patches import Patch
    legend_els = [Patch(color=v, label=k) for k, v in PALETTE.items()]
    axes[2].legend(handles=legend_els, fontsize=8)

    plt.suptitle("Engineered Feature Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_engineered_features.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 05_engineered_features.png")


# ─────────────────────────────────────────────
# 7. ISSUE TYPE & CHANNEL ANALYSIS
# ─────────────────────────────────────────────

def plot_issue_channel_analysis(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Avg churn risk by issue type
    risk_issue = df.groupby("issue_type")["churn_risk_score"].mean().sort_values(ascending=True)
    colors = ["#22c55e" if v < 40 else ("#f59e0b" if v < 60 else "#ef4444")
              for v in risk_issue.values]
    risk_issue.plot(kind="barh", ax=axes[0], color=colors)
    axes[0].set_title("Avg Churn Risk Score by Issue Type", fontweight="bold")
    axes[0].set_xlabel("Avg Churn Risk Score")
    axes[0].axvline(50, color="gray", linestyle="--", linewidth=1, label="Risk=50")
    axes[0].legend()

    # FCR rate by channel
    fcr_channel = df.groupby("support_channel")["first_call_resolution"].mean() * 100
    fcr_channel.sort_values().plot(kind="barh", ax=axes[1], color=BLUE, alpha=0.8)
    axes[1].set_title("First Call Resolution Rate by Channel (%)", fontweight="bold")
    axes[1].set_xlabel("FCR Rate (%)")
    axes[1].axvline(75, color="orange", linestyle="--", linewidth=1.5, label="Avg FCR=75%")
    axes[1].legend()

    plt.suptitle("Issue Type & Channel Diagnostics", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "06_issue_channel_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: 06_issue_channel_analysis.png")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    df = load_data(DATA_PATH)
    print_summary(df)
    plot_numeric_distributions(df)
    plot_sentiment_distribution(df)
    plot_churn_analysis(df)
    plot_correlation_heatmap(df)
    plot_engineered_features(df)
    plot_issue_channel_analysis(df)
    print(f"\n✅ All EDA figures saved to: {FIG_DIR}")


if __name__ == "__main__":
    main()
