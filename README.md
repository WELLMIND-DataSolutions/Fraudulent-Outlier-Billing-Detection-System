# Medicare Fraud & Outlier Billing Detection System

> **Automatically flag providers billing like outliers compared to their specialty peers — the same way OIG auditors do, but running daily instead of annually.**

---

## 📌 Project Overview

This system detects fraudulent and outlier billing patterns in Medicare Part B claims using public CMS data. It mirrors published OIG audit methodology and delivers explainable, per-provider risk scores through an interactive B2B compliance dashboard.

**Built for:** RCM companies, hospital compliance teams, billing auditors  
**Data:** 100% public, free to download — no approval needed

---

## 🎯 Solution Provided

| # | Problem | Our Solution |
|---|---------|--------------|
| 42 | Fraudulent billing | Isolation Forest anomaly detection |
| 43 | Compliance monitoring | Daily LEIE cross-reference pipeline |
| 45 | Duplicate claims | Synthetic duplicate classifier |
| 46 | Outlier billing detection | Specialty peer Z-score benchmarking |

---

## 📂 Data Sources

| Dataset | Source | Update Frequency |
|---------|--------|-----------------|
| CMS Medicare Part B Physician & Supplier PUF | [data.cms.gov](https://data.cms.gov) | Annual |
| OIG LEIE (Excluded Individuals/Entities) | [oig.hhs.gov](https://oig.hhs.gov/exclusions/exclusions_list.asp) | Monthly |

> ⚠️ Open Payments data excluded from v1.0 scope — planned for v2.0

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         DATA SOURCES                                │
│  ┌─────────────────────┐          ┌─────────────────────────────┐   │
│  │   CMS Part B PUF    │          │       OIG LEIE              │   │
│  │   (Annual · ~3GB)   │          │  (Exclusion List · Monthly) │   │
│  └──────────┬──────────┘          └──────────────┬──────────────┘   │
└─────────────┼────────────────────────────────────┼─────────────────┘
              │                                     │
              └────────────────┬────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                                │
│   Load  ──▶  Clean Columns  ──▶  Remove Nulls  ──▶  100k Sample    │
│                               ──▶  Feature Engineering              │
│          Output: cleaned_cms_100k.csv                               │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DETECTION MODULES                              │
│  ┌───────────────────┐  ┌──────────────────┐  ┌─────────────────┐  │
│  │ Z-Score Peer      │  │ Isolation Forest │  │ Upcoding        │  │
│  │ Benchmarking      │  │ (Unsupervised ML)│  │ Detector        │  │
│  └───────────────────┘  └──────────────────┘  └─────────────────┘  │
│  ┌───────────────────┐  ┌──────────────────┐                        │
│  │ Duplicate Claim   │  │ LEIE             │                        │
│  │ Classifier (RF)   │  │ Cross-Reference  │                        │
│  └───────────────────┘  └──────────────────┘                        │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    UNIFIED RISK SCORE  (0 – 100)                    │
│         + Explainable Reason per Provider                           │
│   0–30 Low Risk │ 31–60 Review Needed │ 61–100 Flag       │
└─────────────────────────────────┬───────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STREAMLIT DASHBOARD  (B2B Demo)                   │
│         Live via ngrok  ·  Multi-tab  ·  Filter by Specialty        │
└─────────────────────────────────────────────────────────────────────┘
```

---


## 🔬 Detection Methods — How Each One Works

### 1. Z-Score Peer Benchmarking

```
All Providers in Dataset
         │
         ▼
  Group by Specialty
  (e.g. Cardiology, Oncology, GP...)
         │
         ▼
  For each specialty group:
  ┌─────────────────────────────┐
  │  Mean (μ) = avg payment     │
  │  StdDev (σ) = spread        │
  │  Z = (provider - μ) / σ     │
  └──────────────┬──────────────┘
                 │
        ┌────────┴────────┐
        ▼                 ▼
    |Z| ≤ 3.0         |Z| > 3.0
    Normal          OUTLIER FLAGGED
```

---

### 2. Isolation Forest (Unsupervised ML)

```
Feature Matrix (per provider):
  [avg_payment, total_services, unique_procedures, payment_ratio]
                      │
                      ▼
           ┌──────────────────────┐
           │   Isolation Forest   │
           │   contamination=0.02 │  ← top 2% flagged
           │   n_estimators=100   │
           └──────────┬───────────┘
                      │
           ┌──────────┴───────────┐
           ▼                      ▼
    anomaly_score > 0        anomaly_score ≤ 0
    Normal billing        Anomalous pattern
```

---

### 3. 💉 Upcoding Detection

```
Provider's CPT Code History
           │
           ▼
  Count CPT 99215 (highest complexity E&M)
           │
           ▼
  upcoding_ratio = CPT_99215_count / total_visits
           │
           ▼
  Compare vs specialty average ratio
           │
      ┌────┴────┐
      ▼         ▼
   Within     ratio >> specialty norm
   normal     UPCODING SUSPECTED
   range
```

---

### 4. 🔁 Duplicate Claim Classifier

```
Real CMS Data
      │
      ├──▶  Keep original rows  ──▶  label = 0 (genuine)
      │
      └──▶  Generate synthetic      label = 1 (duplicate)
            duplicates (same NPI,
            slight date/amount
            variation)
                      │
                      ▼
             Combined Dataset
                      │
                      ▼
            Random Forest Classifier
            train_test_split (80/20)
                      │
                      ▼
             Binary prediction:
             0 = genuine claim 
             1 = duplicate 
```

---

### 5. 🚫 LEIE Cross-Reference

```
CMS Provider List
  [NPI_1, NPI_2, NPI_3 ... NPI_n]
           │
           ▼
   OIG LEIE Database
   (Excluded providers)
           │
           ▼
   Exact NPI match lookup
           │
      ┌────┴────┐
      ▼         ▼
   No match   NPI found in LEIE
   Clear    🚨 EXCLUDED PROVIDER
               Immediate flag
               Report to compliance
```

---

## Risk Score

```
Signal               Points    Condition
─────────────────────────────────────────────────────────────
Z-score flag          +20 pts   |Z| > 3.0 on payment or volume
Isolation Forest      +25 pts   anomaly_score > threshold
Upcoding flag         +20 pts   ratio >> specialty average
Duplicate flag        +15 pts   classifier predicts duplicate
LEIE match            +20 pts   NPI found in exclusion list
─────────────────────────────────────────────────────────────
TOTAL POSSIBLE        100 pts
─────────────────────────────────────────────────────────────

Risk Bands:
  0  – 30  │██░░░░░░░░│  ✅ Low Risk    → No action needed
 31  – 60  │█████░░░░░│  ⚠️  Review     → Manual review recommended
 61  – 100 │██████████│  🚨 High Risk  → Escalate to compliance
```

---

## 📁 Project Structure

```
medicare-fraud-detection/
│
├── README.md
├── .gitignore
│
├── notebooks/
│   └── Medicare_Fraud_Detection.ipynb    ← Main Colab notebook (21 cells)
│
├── data/
│   ├── raw/                              ← Original downloaded files (gitignored)
│   └── processed/
│       └── cleaned_cms_100k.csv         ← 100k row clean sample
│
├── models/
│   ├── isolation_forest.pkl             ← Trained anomaly detector
│   └── duplicate_classifier.pkl         ← Duplicate detection model
│
├── src/
│   ├── features.py                      ← Feature engineering functions
│   ├── models.py                        ← Model training functions
│   └── explainer.py                     ← Risk score + reason generator
│
└── dashboard/
    └── app.py                           ← Streamlit B2B dashboard
```

---



## Key Features

- **Daily-ready pipeline** — automated, not just annual audits
- **Explainable flags** — every risk score has a human-readable reason
- **Peer-based benchmarking** — specialty-specific, not one-size-fits-all
- **Multi-signal risk score** — combines 5 detection methods into one 0-100 score
- **B2B demo dashboard** — client-ready Streamlit UI
- **100% public data** — no data access approval required
- **Open methodology** — mirrors published OIG audit approach


---

*Last updated: May 2026*
