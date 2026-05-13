# 📡 Telecom Customer Sentiment Analysis — Churn Prediction

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=flat&logo=scikit-learn)
![NLTK](https://img.shields.io/badge/NLTK-VADER-green?style=flat)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat)

> **End-to-end data science project** combining NLP sentiment scoring and Random Forest machine learning to predict customer churn for a telecommunication company.

---

## 🎯 Project Overview

Customer churn costs telecom carriers $300–$400 per lost subscriber. 
This project builds a **sentiment-aware churn prediction system** that identifies at-risk customers *before* they cancel — by combining structured CRM data with NLP-derived emotional signals.

**Business Problem:** How do we identify customers who are emotionally disengaged before they cancel their service?

**Solution:** A machine learning pipeline that scores every customer with a churn risk probability, a Weighted Sentiment Index (WSI), and a risk tier — enabling proactive, data-driven retention outreach.

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Records | 5,000 customers |
| Features | 26 columns |
| Date Range | Jan 2022 – Dec 2024 |
| Target | `churned` (binary: 0 = Retained, 1 = Churned) |
| Sentiment Target | `sentiment_label` (Positive / Neutral / Negative) |

**3 Engineered Features:**

| Feature | Formula |
|---|---|
| Weighted Sentiment Index (WSI) | `(0.40 × norm_CSAT) + (0.35 × norm_NPS) + (0.25 × (1 - FCR_penalty))` |
| Support Intensity Index (SII) | `num_support_contacts / (tenure_months + 1)` |
| Revenue Per Tenure Month | `total_charges_usd / tenure_months` |

---

## 🤖 Algorithm

**Primary Model:** Random Forest Classifier  
**NLP Engine:** VADER (Valence Aware Dictionary and sEntiment Reasoner)  
**Explainability:** SHAP (SHapley Additive exPlanations)

Random Forest was chosen over XGBoost (interpretability), Logistic Regression (handles non-linear sentiment interactions), and Neural Networks (insufficient data, explainability requirements).

---

## 📁 Project Structure

```
telecom_sentiment_project/
│
├── data/                        # Raw and processed datasets
│   ├── telecom_sentiment_dataset.csv
│   ├── telecom_sentiment_with_nlp.csv
│   ├── X_train.csv / X_test.csv
│   ├── y_train.csv / y_test.csv
│   ├── feature_names.json
│   └── scored_customers.csv
│
├── src/                         # Python scripts (run in order)
│   ├── 01_generate_dataset.py   # Synthetic dataset generation
│   ├── 02_eda.py                # Exploratory data analysis + visualisations
│   ├── 03_nlp_sentiment.py      # VADER sentiment scoring + evaluation
│   ├── 04_feature_engineering.py # Encoding, scaling, train/test split
│   ├── 05_model_training.py     # Random Forest training + evaluation
│   └── 06_score_new_customers.py # Inference pipeline for new records
│
├── models/
│   └── random_forest_churn_model.pkl
│
├── reports/
│   ├── figures/                 # All generated charts (EDA, model, SHAP)
│   ├── feature_summary.txt
│   └── model_evaluation_report.txt
│
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sayali230493/telecom-sentiment-analysis.git
cd telecom-sentiment-analysis
```

### 2. Set up environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 3. Run the full pipeline (in order)
```bash
python src/01_generate_dataset.py       # Generate dataset
python src/02_eda.py                    # Exploratory analysis
python src/03_nlp_sentiment.py          # NLP scoring
python src/04_feature_engineering.py   # Feature prep + split
python src/05_model_training.py         # Train & evaluate model
python src/06_score_new_customers.py    # Score all customers
```

---

## 📈 Key Findings

- **32%** of the customer base carries Negative sentiment
- Negative-sentiment Month-to-Month customers churn at **68%** vs **4%** for Positive 2-year customers
- **Billing Disputes** and **Network Outages** are the top two churn accelerators
- First Call Resolution failure increases WSI on the negative side by ~0.18 points
- Top model features: `weighted_sentiment_index`, `churn_risk_score`, `support_intensity_index`

---

## 🏭 Who Can Use This

| Industry | Department | Use Case |
|---|---|---|
| Telecommunications | Customer Retention | Proactive high-risk outreach |
| Cable & Internet | Marketing | Personalised retention offers |
| Utilities | CX / UX | Identify top friction drivers |
| Streaming / SaaS | Customer Success | Monitor at-risk accounts |

---

## 🔮 Future Scope

- [ ] Real-time REST API scoring endpoint (FastAPI)
- [ ] BERT/RoBERTa upgrade for NLP layer
- [ ] Multi-modal sentiment from call audio
- [ ] Survival analysis (time-to-churn prediction)
- [ ] LLM-generated personalised retention scripts
- [ ] Airflow/dbt pipeline for automated daily scoring

---

## 🛠️ Tools & Technologies

`Python 3.10+` · `pandas` · `NumPy` · `scikit-learn` · `VADER (NLTK)` · `SHAP` · `matplotlib` · `seaborn` · `joblib`

---

## 👤 Author

**Sayali Khopade**  
[GitHub](https://github.com/sayali230493) · [LinkedIn](https://www.linkedin.com/in/sayalikhopade/)  
2026

---

## 📬 Contact

Feel free to reach out if you have questions about this project or want to collaborate:

- 💼 LinkedIn: [linkedin.com/in/YOUR_LINKEDIN](https://www.linkedin.com/in/sayalikhopade/)
- 📧 Email: sayali23khopade23@gmail.com
- 🐙 GitHub: [github.com/sayali230493](https://github.com/sayali230493)

---

## 📄 License

This project is for portfolio and educational purposes.
© 2026 Sayali Khopade. All rights reserved.
