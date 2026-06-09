# 🏥 Healthcare Fraud Detection System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.3+-orange?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Data-CMS%20Medicare-red?style=for-the-badge" />
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" />
</p>

> **An AI-powered pipeline that analyzes Medicare billing data to detect suspicious healthcare providers using machine learning, statistical analysis, and federal exclusion list cross-referencing.**

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results](#-key-results)
- [Features](#-features)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Methodology](#-methodology)
- [Models Used](#-models-used)
- [Output Files](#-output-files)
- [Technologies](#-technologies)
- [Important Note](#%EF%B8%8F-important-note)
- [Author](#-author)

---

## 🔍 Overview

Medicare fraud costs the U.S. healthcare system an estimated **$60–100 billion annually**. Manual detection is slow, inconsistent, and unable to scale across millions of billing records.

This project builds an **end-to-end fraud detection pipeline** using:
- **CMS Medicare** physician & practitioner billing data
- **OIG LEIE** federal exclusion list
- **Statistical outlier detection** (Z-scores) + **ML anomaly detection** (Isolation Forest)
- **Supervised classification** (Random Forest) for duplicate claim patterns
- **Upcoding detection** for E&M billing abuse

Every provider receives a **Fraud Risk Score** and is classified into **High / Medium / Low Risk** with an **Investigation Priority** label for actionable triaging.

---

## 📊 Key Results

| Metric | Value |
|--------|-------|
| Total Providers Analyzed | 44,528 |
| High Risk Providers | 3,842 |
| Medium Risk Providers | 11,276 |
| Low Risk Providers | 29,410 |
| Critical Priority Providers | 1,247 |
| Upcoding Flags | 2,914 |
| LEIE Matches Found | — (varies by run date) |
| Peer Groups Created | 1,200+ |
| Duplicate Classifier Accuracy | ~96% |
| Provider Types Analyzed | 90+ |

---

## ✨ Features

- ✅ **Peer-Group Benchmarking** — Groups providers by specialty × state and computes Z-scores within each group
- ✅ **Z-Score Outlier Detection** — Flags providers exceeding 3 standard deviations from their peer mean on any key metric
- ✅ **Isolation Forest Anomaly Detection** — Unsupervised ML model trained on provider-level billing features
- ✅ **E&M Upcoding Detection** — Identifies providers over-billing high-value consultation codes (HCPCS 99211–99215)
- ✅ **Duplicate Claim Classifier** — Random Forest classifier trained to detect duplicate-like billing patterns
- ✅ **LEIE Cross-Reference** — Matches active excluded providers against CMS billing data by NPI
- ✅ **Fraud Risk Score** — Composite score combining all signals into a single risk label per provider
- ✅ **Investigation Priority** — Ranks providers as Critical / High / Medium / Standard for investigator triaging

---

## 📂 Dataset

### 1. CMS Medicare — Physician & Other Practitioners by Provider and Service
| Detail | Info |
|--------|------|
| Source | [CMS.gov — Medicare Physician & Other Practitioners](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners) |
| File | `MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv` |
| Size | ~2–3 GB |
| Key Columns | `NPI`, `Provider Type`, `State`, `HCPCS_Cd`, `Tot_Benes`, `Tot_Srvcs`, `Avg_Sbmtd_Chrg`, `Avg_Mdcr_Pymt_Amt` |

### 2. OIG LEIE — List of Excluded Individuals/Entities
| Detail | Info |
|--------|------|
| Source | [OIG HHS — Exclusions List](https://oig.hhs.gov/exclusions/exclusions_list.asp) |
| File | `UPDATED.csv` (auto-downloaded) |
| Key Columns | `NPI`, `EXCLTYPE`, `EXCLDATE`, `REINDATE`, `STATE` |

> ⚠️ **Note:** Raw data files are not included in this repo due to size. Download links above. See [Usage](#-usage) for setup instructions.

---

## 📁 Project Structure

```
fraud_detection/
│
├── data/
│   ├── raw/
│   │   ├── MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv   # CMS data (download separately)
│   │   └── leie_exclusions.csv                       # OIG LEIE (auto-downloaded)
│   │
│   └── processed/
│       ├── provider_fraud_features_final.csv
│       ├── outlier_flags.csv
│       ├── upcoding_scores.csv
│       ├── critical_priority_providers.csv
│       └── active_leie_matches_YYYYMMDD.csv
│
├── models/
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── duplicate_classifier.pkl
│   └── duplicate_classifier_threshold_info.csv
│
├── reports/
│   ├── final_project_summary.csv
│   ├── final_top100_suspicious_providers.csv
│   ├── final_high_priority_providers.csv
│   ├── duplicate_classifier_final_report.csv
│   ├── graph5_zscore_distribution.png
│   ├── graph6_top_outlier_provider_types.png
│   ├── graph8_em_code_distribution.html
│   ├── graph10_confusion_matrix.png
│   ├── graph11_feature_importance.png
│   ├── graph12_leie_exclusions_per_year.html
│   └── graph13_exclusion_reasons.png
│
├── Fraud_detection.ipynb    # Main notebook (run day by day)
└── README.md
```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

### 2. Install Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib plotly joblib requests
```

### 3. Download the Data

**CMS Data** — Download manually from CMS.gov:
```
https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners
```
Save to: `data/raw/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv`

**LEIE Data** — Auto-downloaded by the notebook in Step 1.2, or manually from:
```
https://oig.hhs.gov/exclusions/exclusions_list.asp
```
Save to: `data/raw/leie_exclusions.csv`

### 4. (Google Colab Users)
Mount your Drive and update the `BASE` path at the top of the notebook:
```python
BASE = '/content/drive/MyDrive/fraud_detection'
cms_path = '/content/drive/MyDrive/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv'
```

---

## 🚀 Usage

The notebook is organized into **6 sequential days** — run each section in order:

| Day | Steps | What Happens |
|-----|-------|--------------|
| **Day 1** | 0.1 → 1.3 | Setup, folder structure, data download |
| **Day 2** | 2.1 → 2.19 | Data cleaning, feature engineering, peer groups, Z-scores, risk scoring |
| **Day 3** | 3.1 → 3.3 | Isolation Forest training, outlier flagging, graphs |
| **Day 4** | 4.1 → 4.2E | Upcoding detection, Random Forest duplicate classifier |
| **Day 5** | 5.1 → 5.4 | LEIE cross-reference, active exclusion matching |
| **Day 6** | 6.1 → 6.4 | Final summary report, zip deliverables |

Run the full notebook from top to bottom. Each cell prints a `✅` confirmation on success.

---

## 🧠 Methodology

### Step 1 — Feature Engineering
Provider-level features computed from raw CMS data:

| Feature | Description |
|---------|-------------|
| `Payment_Per_Bene` | Total Medicare payment ÷ total beneficiaries |
| `Services_Per_Bene` | Total services ÷ total beneficiaries |
| `Payment_Per_Service` | Total payment ÷ total services |
| `Charge_To_Payment_Ratio` | Avg submitted charge ÷ avg Medicare payment |
| `Unique_HCPCS_Count` | Number of distinct billing codes used |

### Step 2 — Peer Group Benchmarking
Providers grouped by `Provider Type × State` (1,200+ groups). Z-scores computed within each group for all features. A provider is flagged if **any Z-score exceeds ±3**.

### Step 3 — Fraud Risk Score
```
Fraud_Risk_Score = count of features with Z-score > 2 (positive deviation only)

High Risk   → Score ≥ 3
Medium Risk → Score 1–2
Low Risk    → Score = 0
```

### Step 4 — Anomaly Detection
`IsolationForest(contamination=0.05, n_estimators=200)` trained on scaled provider features. Providers predicted as `-1` are flagged as anomalies.

Combined flag:
```
combined_model_risk = outlier_zscore OR iso_flag
```

### Step 5 — Upcoding Detection
For HCPCS codes 99211–99215 (E&M consultation codes), a weighted upcoding score is computed. Providers significantly above their peer mean are flagged.

### Step 6 — LEIE Cross-Reference
Active excluded providers (no reinstatement date) with valid 10-digit NPIs are matched against the CMS provider dataset. Any match is a critical flag.

### Step 7 — Investigation Priority
```
Critical Priority → High Risk + LEIE match
High Priority     → High Risk OR (Medium Risk + LEIE match)
Medium Priority   → Medium Risk
Standard          → Low Risk
```

---

## 🤖 Models Used

| Model | Type | Purpose | Key Params |
|-------|------|---------|------------|
| **Isolation Forest** | Unsupervised | Anomaly detection | `contamination=0.05`, `n_estimators=200` |
| **Random Forest** | Supervised | Duplicate claim detection | `n_estimators=100`, `random_state=42` |
| **Z-Score Analysis** | Statistical | Peer-group outlier detection | Threshold `> 3` std devs |
| **E&M Upcoding Score** | Rule-based | Detect overbilling of high-value codes | HCPCS 99211–99215 |

---

## 📤 Output Files

| File | Description |
|------|-------------|
| `outlier_flags.csv` | All providers with Z-score, Isolation Forest, and combined risk flags |
| `upcoding_scores.csv` | E&M upcoding scores per provider |
| `active_leie_matches_YYYYMMDD.csv` | Providers matched against active LEIE exclusions |
| `critical_priority_providers.csv` | High risk + LEIE matched providers |
| `final_top100_suspicious_providers.csv` | Top 100 by Fraud Risk Score |
| `final_high_priority_providers.csv` | All High + Critical priority providers |
| `final_project_summary.csv` | Aggregate summary numbers |
| `duplicate_classifier_final_report.csv` | Classification report for Random Forest model |

---

## 🛠️ Technologies

| Tool | Version | Use |
|------|---------|-----|
| Python | 3.10+ | Core language |
| Pandas | 2.x | Data wrangling |
| NumPy | 1.x | Numerical ops |
| Scikit-learn | 1.3+ | ML models |
| Matplotlib | 3.x | Static charts |
| Plotly | 5.x | Interactive charts |
| Joblib | — | Model serialization |
| Google Colab | — | GPU-free cloud environment |

---

## ⚠️ Important Note

> This system **does not prove fraud**. It identifies statistically unusual billing patterns and cross-references federal exclusion lists to **prioritize providers for further human investigation**.
>
> All results should be treated as **investigative leads**, not definitive conclusions.

---

## 👩‍💻 Author

**Abeera Ahmad**  
BS Computer Science — Riphah International University, Faisalabad  
Specialization: AI & Data Science  
AI Intern @ Well Mind

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/abeera-ahmad-a26983363/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/abeeraahmad666-ship-it)


<p align="center">Made with ❤️ for healthcare integrity — detecting fraud, protecting Medicare.</p>
