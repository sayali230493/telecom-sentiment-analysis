"""
=============================================================================
Telecom Customer Sentiment Analysis — Dataset Generation
=============================================================================
Script  : 01_generate_dataset.py
Author  : Data Science Team
Purpose : Generates a synthetic but statistically realistic dataset of 5,000
          telecom customers with behavioral, transactional, and sentiment
          attributes suitable for churn prediction and NLP analysis.

Output  : data/telecom_sentiment_dataset.csv
=============================================================================
"""

import csv
import random
import math
from datetime import datetime, timedelta
import os

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
RANDOM_SEED     = 42
NUM_CUSTOMERS   = 5000
START_DATE      = datetime(2022, 1, 1)
END_DATE        = datetime(2024, 12, 31)
OUTPUT_PATH     = os.path.join(os.path.dirname(__file__), "..", "data", "telecom_sentiment_dataset.csv")

random.seed(RANDOM_SEED)

# ─────────────────────────────────────────────
# DOMAIN REFERENCE DATA
# ─────────────────────────────────────────────
PLANS           = ["Basic", "Standard", "Premium", "Enterprise"]
REGIONS         = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
CHANNELS        = ["Chat", "Phone", "Email", "In-Store", "App"]
DEVICE_TYPES    = ["Smartphone", "Tablet", "Hotspot", "Home Internet", "Business Line"]
ISSUE_TYPES     = ["Billing Dispute", "Network Outage", "Speed Issue",
                   "Device Problem", "Plan Change", "Contract Issue", "No Issue"]
CONTRACT_TYPES  = ["Month-to-Month", "1-Year", "2-Year"]
PAYMENT_METHODS = ["Auto-Pay", "Manual-Online", "In-Store", "Mail"]

# Verbatim feedback phrases mapped to sentiment tier
# In production, these would come from real CRM transcripts
CSAT_VERBATIMS = {
    "Positive": [
        "Very happy with service",
        "Excellent support experience",
        "Agent was very helpful",
        "Problem resolved quickly",
        "Outstanding customer care",
        "Highly satisfied with resolution",
        "Agent went above and beyond",
        "Seamless experience overall",
        "Great network coverage",
        "Very professional support team",
    ],
    "Neutral": [
        "Service was okay",
        "Issue eventually resolved",
        "Average experience",
        "Expected better response time",
        "Decent support",
        "Nothing exceptional",
        "Got what I needed but took a while",
        "Could be improved",
        "Acceptable service",
        "Waited longer than expected",
    ],
    "Negative": [
        "Very disappointed with service",
        "Issue still not resolved",
        "Poor network reliability",
        "Agent was unhelpful",
        "Billing overcharge not fixed",
        "Terrible wait times",
        "Worst customer service experience",
        "Problem kept recurring",
        "No accountability at all",
        "Will consider switching providers",
    ],
}


# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def random_date(start: datetime, end: datetime) -> str:
    """Return a random date string between start and end."""
    delta = end - start
    random_days = random.randint(0, delta.days)
    return (start + timedelta(days=random_days)).strftime("%Y-%m-%d")


def get_sentiment_tier(csat_score: int) -> str:
    """Map a 1–5 CSAT score to a sentiment tier."""
    if csat_score >= 4:
        return "Positive"
    elif csat_score == 3:
        return "Neutral"
    else:
        return "Negative"


def get_loyalty_tier(tenure_months: int) -> str:
    """
    Assign loyalty tier based on customer tenure.

    Bronze   : < 12 months
    Silver   : 12–35 months
    Gold     : 36–59 months
    Platinum : 60+ months
    """
    if tenure_months < 12:
        return "Bronze"
    elif tenure_months < 36:
        return "Silver"
    elif tenure_months < 60:
        return "Gold"
    else:
        return "Platinum"


def compute_churn_risk(csat_score: int, nps_score: int, num_support_contacts: int) -> float:
    """
    Compute a composite churn risk score (0–100).

    Formula:
        raw_risk = (0.30 × CSAT_inverse_norm)
                 + (0.40 × NPS_inverse_norm)
                 + (0.30 × support_contact_norm)

    Where:
        CSAT_inverse_norm     = (5 - csat_score) / 4
        NPS_inverse_norm      = (100 - nps_score) / 200
        support_contact_norm  = num_support_contacts / 15  [max = 15]

    Higher score = higher churn risk.
    """
    csat_inv   = (5 - csat_score) / 4
    nps_inv    = (100 - nps_score) / 200
    support_n  = num_support_contacts / 15

    raw_risk   = 0.30 * csat_inv + 0.40 * nps_inv + 0.30 * support_n
    return round(min(max(raw_risk * 100, 0.0), 100.0), 2)


def compute_weighted_sentiment_index(csat_score: int, nps_score: int,
                                     first_call_resolution: bool) -> float:
    """
    Compute the Weighted Sentiment Index (WSI) — a unified customer health score.

    Formula:
        WSI = (0.40 × norm_CSAT)
            + (0.35 × norm_NPS)
            + (0.25 × (1 - FCR_penalty))

    Where:
        norm_CSAT   = (csat_score - 1) / 4        [0.0 – 1.0]
        norm_NPS    = (nps_score + 100) / 200      [0.0 – 1.0]
        FCR_penalty = 0.20 if NOT resolved on first call, else 0.00

    A WSI closer to 1.0 indicates a highly satisfied, low-risk customer.
    A WSI closer to 0.0 indicates a frustrated, high-risk customer.
    """
    norm_csat    = (csat_score - 1) / 4
    norm_nps     = (nps_score + 100) / 200
    fcr_penalty  = 0.0 if first_call_resolution else 0.20

    wsi = 0.40 * norm_csat + 0.35 * norm_nps + 0.25 * (1 - fcr_penalty)
    return round(wsi, 4)


def determine_churned(churn_risk_score: float) -> int:
    """
    Derive binary churn label from the churn risk score.

    Rules:
        risk_score > 80  → always churned (deterministic)
        risk_score > 65  → churned with 70% probability (stochastic)
        otherwise        → retained
    """
    if churn_risk_score > 80:
        return 1
    elif churn_risk_score > 65 and random.random() > 0.30:
        return 1
    return 0


# ─────────────────────────────────────────────
# RECORD GENERATION
# ─────────────────────────────────────────────

def generate_customer_record(customer_id: int) -> dict:
    """Generate a single synthetic customer record."""

    # ── Categorical attributes ──────────────────────────────────────────────
    region          = random.choice(REGIONS)
    plan            = random.choice(PLANS)
    contract        = random.choice(CONTRACT_TYPES)
    payment         = random.choice(PAYMENT_METHODS)
    device          = random.choice(DEVICE_TYPES)
    channel         = random.choice(CHANNELS)
    issue           = random.choice(ISSUE_TYPES)

    # ── Behavioral / transactional attributes ───────────────────────────────
    tenure_months           = random.randint(1, 84)
    monthly_charges         = round(random.uniform(29.99, 249.99), 2)
    # total_charges formula: monthly × tenure × noise_factor
    # noise_factor ~ Uniform(0.85, 1.05) simulates promo credits, adjustments
    noise_factor            = random.uniform(0.85, 1.05)
    total_charges           = round(monthly_charges * tenure_months * noise_factor, 2)
    avg_monthly_data_gb     = round(random.uniform(1.0, 120.0), 2)
    num_support_contacts    = random.randint(0, 15)

    # FCR set to True 75% of the time (industry benchmark)
    first_call_resolution   = random.random() < 0.75

    # ── Satisfaction scores ─────────────────────────────────────────────────
    nps_score   = random.randint(-100, 100)
    csat_score  = random.randint(1, 5)

    # ── Sentiment ───────────────────────────────────────────────────────────
    sentiment_label = get_sentiment_tier(csat_score)
    csat_verbatim   = random.choice(CSAT_VERBATIMS[sentiment_label])

    # sentiment_score: VADER compound mapped to [0,1] range per tier
    score_ranges = {"Positive": (0.60, 1.00), "Neutral": (0.30, 0.60), "Negative": (0.00, 0.30)}
    lo, hi = score_ranges[sentiment_label]
    sentiment_score = round(random.uniform(lo, hi), 4)

    # ── Derived / engineered features ───────────────────────────────────────
    churn_risk_score = compute_churn_risk(csat_score, nps_score, num_support_contacts)
    churned          = determine_churned(churn_risk_score)

    # revenue_per_tenure_month = total_charges / tenure_months
    revenue_per_tenure_month = round(total_charges / tenure_months, 2) if tenure_months > 0 else 0.0

    # support_intensity_index = contacts / (tenure + 1)
    # +1 prevents division-by-zero for brand-new customers
    support_intensity_index = round(num_support_contacts / (tenure_months + 1), 4)

    loyalty_tier            = get_loyalty_tier(tenure_months)
    weighted_sentiment_index = compute_weighted_sentiment_index(csat_score, nps_score, first_call_resolution)
    interaction_date        = random_date(START_DATE, END_DATE)

    return {
        "customer_id":               f"CUST{customer_id:05d}",
        "region":                    region,
        "plan_type":                 plan,
        "contract_type":             contract,
        "payment_method":            payment,
        "device_type":               device,
        "support_channel":           channel,
        "issue_type":                issue,
        "tenure_months":             tenure_months,
        "monthly_charges_usd":       monthly_charges,
        "total_charges_usd":         total_charges,
        "avg_monthly_data_gb":       avg_monthly_data_gb,
        "num_support_contacts":      num_support_contacts,
        "first_call_resolution":     first_call_resolution,
        "nps_score":                 nps_score,
        "csat_score":                csat_score,
        "csat_verbatim_text":        csat_verbatim,
        "sentiment_label":           sentiment_label,
        "sentiment_score":           sentiment_score,
        "churn_risk_score":          churn_risk_score,
        "churned":                   churned,
        "revenue_per_tenure_month":  revenue_per_tenure_month,
        "support_intensity_index":   support_intensity_index,
        "loyalty_tier":              loyalty_tier,
        "weighted_sentiment_index":  weighted_sentiment_index,
        "interaction_date":          interaction_date,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print(f"Generating {NUM_CUSTOMERS} customer records ...")

    records = [generate_customer_record(i) for i in range(1, NUM_CUSTOMERS + 1)]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    # Summary statistics
    churned_count   = sum(r["churned"] for r in records)
    neg_count       = sum(1 for r in records if r["sentiment_label"] == "Negative")
    avg_risk        = sum(r["churn_risk_score"] for r in records) / len(records)

    print(f"\n✅ Dataset saved to: {OUTPUT_PATH}")
    print(f"   Total records      : {len(records):,}")
    print(f"   Churned customers  : {churned_count:,}  ({churned_count/len(records)*100:.1f}%)")
    print(f"   Negative sentiment : {neg_count:,}  ({neg_count/len(records)*100:.1f}%)")
    print(f"   Avg churn risk     : {avg_risk:.2f}")


if __name__ == "__main__":
    main()
