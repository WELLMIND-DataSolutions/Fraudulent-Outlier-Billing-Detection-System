# Healthcare Fraud Detection System

<p align="center">
  <img src=".\Assests\image.png"/>
  

<p align="center">
  An end-to-end ML pipeline that analyzes Medicare billing data to surface suspicious providers — combining statistical analysis, unsupervised anomaly detection, and federal exclusion list cross-referencing into a single actionable fraud risk score.
</p>

---

## What This Project Does

Medicare fraud costs the U.S. healthcare system an estimated **$60–100 billion per year**. Manual review cannot scale across millions of billing records.

This system processes raw CMS Medicare billing data and outputs a prioritized list of providers for human investigators — ranked by fraud risk and flagged with specific anomalies that triggered each alert.

**Key capabilities:**

- Benchmarks every provider against their specialty-state peer group using Z-score analysis
- Detects statistical outliers with Isolation Forest (unsupervised ML)
- Identifies E&M upcoding abuse across 90+ provider types
- Cross-references active federal exclusion lists (OIG LEIE) by NPI
- Produces a single composite **Fraud Risk Score** per provider with investigation priority labels

**Results on the 2023 CMS dataset:**

| Metric | Value |
|---|---|
| Total providers analyzed | 44,528 |
| High-risk providers flagged | 3,842 |
| Critical priority providers | 1,247 |
| Upcoding flags issued | 2,914 |
| Peer groups benchmarked | 1,200+ |
| Duplicate classifier accuracy | ~96% |

---

## Why This Is Useful

- **Investigators** get a ranked shortlist instead of millions of raw rows — Critical and High priority cases surface first
- **Analysts** can drill into specific anomaly types: Z-score outliers, isolation forest flags, upcoding scores, or LEIE matches independently
- **Researchers** can adapt the peer-group benchmarking and composite scoring approach to other domains (insurance, procurement, financial services)

> **Important:** This system flags statistically unusual billing patterns. It does not prove fraud. All outputs are investigative leads for human review, not definitive conclusions.

---

## System Workflow

<img src=".\Assests\workflow.jpeg"/>

## Pipeline Workflow

<img src=".\Assests\pipeline.jpeg"/>
---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/abeeraahmad666-ship-it/healthcare-fraud-detection.git
cd healthcare-fraud-detection
```

### 2. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib plotly joblib requests
```

### 3. Download the data

**CMS Medicare data** — download manually from [CMS.gov](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners):

```
Save to: data/raw/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv
```

**LEIE exclusion list** — auto-downloaded by the notebook (Step 1.2), or manually from [OIG HHS](https://oig.hhs.gov/exclusions/exclusions_list.asp):

```
Save to: data/raw/leie_exclusions.csv
```

### 4. Run the notebook

Open `Fraud_detection.ipynb` and run all cells in order. The notebook is organized into 6 sequential day-sections.

**Google Colab users** — mount your Drive and update the base path at the top:

```python
BASE = '/content/drive/MyDrive/fraud_detection'
cms_path = '/content/drive/MyDrive/MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv'
```

---

## Project Structure

```
healthcare-fraud-detection/
├── data/
│   ├── raw/
│   │   ├── MUP_PHY_R25_P05_V20_D23_Prov_Svc.csv   # Download separately (CMS)
│   │   └── leie_exclusions.csv                       # Auto-downloaded (OIG)
│   └── processed/
│       ├── provider_fraud_features_final.csv
│       ├── outlier_flags.csv
│       ├── upcoding_scores.csv
│       ├── critical_priority_providers.csv
│       └── active_leie_matches_YYYYMMDD.csv
├── models/
│   ├── isolation_forest.pkl
│   ├── scaler.pkl
│   ├── duplicate_classifier.pkl
│   └── duplicate_classifier_threshold_info.csv
├── reports/
│   ├── final_project_summary.csv
│   ├── final_top100_suspicious_providers.csv
│   ├── final_high_priority_providers.csv
│   └── [graphs 5–13: charts and visualizations]
├── Fraud_detection.ipynb
└── README.md
```

---

## Methodology

### Feature engineering

Five provider-level features are computed from raw CMS data:

| Feature | Formula |
|---|---|
| `Payment_Per_Bene` | Total Medicare payment ÷ total beneficiaries |
| `Services_Per_Bene` | Total services ÷ total beneficiaries |
| `Payment_Per_Service` | Total payment ÷ total services |
| `Charge_To_Payment_Ratio` | Avg submitted charge ÷ avg Medicare payment |
| `Unique_HCPCS_Count` | Distinct billing codes used |

### Peer group benchmarking

Providers are grouped by `Provider Type × State` (1,200+ groups). Z-scores are computed within each group. A provider is flagged if any Z-score exceeds ±3 standard deviations.

### Fraud risk score

```
Score = count of features where Z-score > 2 (positive deviation only)

High Risk    →  Score ≥ 3
Medium Risk  →  Score 1–2
Low Risk     →  Score = 0
```

### Anomaly detection

`IsolationForest(contamination=0.05, n_estimators=200)` trained on scaled provider features. Providers predicted as `-1` are anomaly-flagged.

Combined flag:

```
combined_model_risk = outlier_zscore OR iso_flag
```

### Upcoding detection

A weighted upcoding score is computed for E&M consultation codes (HCPCS 99211–99215). Providers significantly above their peer mean are flagged.

### Investigation priority

```
Critical  →  High Risk + LEIE match
High      →  High Risk  OR  (Medium Risk + LEIE match)
Medium    →  Medium Risk
Standard  →  Low Risk
```

---

## Models

| Model | Type | Purpose | Config |
|---|---|---|---|
| Isolation Forest | Unsupervised | Anomaly detection | `contamination=0.05`, `n_estimators=200` |
| Random Forest | Supervised | Duplicate claim detection | `n_estimators=100`, `random_state=42` |
| Z-score analysis | Statistical | Peer-group outlier detection | Threshold `> 3σ` |
| E&M upcoding score | Rule-based | High-value code overbilling | HCPCS 99211–99215 |

---

## Output Files

| File | Description |
|---|---|
| `outlier_flags.csv` | All providers with Z-score, Isolation Forest, and combined risk flags |
| `upcoding_scores.csv` | E&M upcoding scores per provider |
| `active_leie_matches_YYYYMMDD.csv` | Providers matched against active LEIE exclusions |
| `critical_priority_providers.csv` | High-risk + LEIE matched providers |
| `final_top100_suspicious_providers.csv` | Top 100 by Fraud Risk Score |
| `final_high_priority_providers.csv` | All High + Critical priority providers |
| `final_project_summary.csv` | Aggregate summary statistics |
| `duplicate_classifier_final_report.csv` | Classification report for the Random Forest model |

---

## Tech Stack

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Core language |
| Pandas | 2.x | Data wrangling |
| NumPy | 1.x | Numerical operations |
| scikit-learn | 1.3+ | ML models (Isolation Forest, Random Forest, scaling) |
| Matplotlib | 3.x | Static charts |
| Plotly | 5.x | Interactive charts |
| Joblib | — | Model serialization |
| Google Colab | — | Cloud execution environment |

---

## Getting Help

- **Issues:** Open a [GitHub Issue](https://github.com/abeeraahmad666-ship-it/healthcare-fraud-detection/issues) to report bugs or ask questions
- **CMS data documentation:** [CMS.gov Medicare Physician & Other Practitioners](https://data.cms.gov/provider-summary-by-type-of-service/medicare-physician-other-practitioners)
- **LEIE exclusion list:** [OIG HHS Exclusions](https://oig.hhs.gov/exclusions/exclusions_list.asp)
- **scikit-learn docs:** [scikit-learn.org](https://scikit-learn.org/stable/)

---

## Contributing

Contributions are welcome. To contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a pull request

Please open an issue first for major changes so we can discuss the approach.

---

## Author & Maintainer

**Abeera Ahmad**
BS Computer Science · AI & Data Science specialization
Riphah International University, Faisalabad
AI Intern @ Well Mind

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/abeera-ahmad-a26983363/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/abeeraahmad666-ship-it)

---

## License

This project is licensed under the [MIT License](LICENSE).

---

<p align="center">Built to support healthcare integrity — detecting anomalies, protecting Medicare.</p>
