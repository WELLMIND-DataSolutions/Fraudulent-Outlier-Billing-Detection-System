# 🔬 Research & Development Document (RND)
### Healthcare Fraud Detection System

---

## What is an RND Document?

> An **RND (Research & Development) document** is a technical deep-dive that explains the *"why"* and *"how"* behind every decision in your project.
>
> While a **README** tells someone *how to use* your project, the **RND** tells them *how you built it* — what you researched, what you experimented with, what failed, what worked, and why you made each technical choice.
>
> Think of it as your **researcher's notebook made public** — it shows that you didn't just run code, you actually understood and made deliberate decisions.

**Commonly used in:**
- ML/AI research projects
- Industry technical documentation
- Internship & final year project portfolios
- Open-source repos that want to be taken seriously

---

## 📋 Table of Contents

1. [Research Background](#1-research-background)
2. [Problem Formulation](#2-problem-formulation)
3. [Related Work & Prior Art](#3-related-work--prior-art)
4. [Data Research](#4-data-research)
5. [Feature Engineering Decisions](#5-feature-engineering-decisions)
6. [Model Selection & Rationale](#6-model-selection--rationale)
7. [Experimental Results](#7-experimental-results)
8. [Challenges & Solutions](#8-challenges--solutions)
9. [Limitations](#9-limitations)
10. [Future Work](#10-future-work)
11. [References](#11-references)

---

## 1. Research Background

### 1.1 The Problem Domain

Medicare fraud is a serious and persistent problem in the U.S. healthcare system. The **Office of Inspector General (OIG)** estimates that improper Medicare payments — including fraud, waste, and abuse — account for approximately **$60–100 billion annually**.

Common fraud types this project targets:

| Fraud Type | Description |
|-----------|-------------|
| **Billing for services not rendered** | Provider charges Medicare for procedures never performed |
| **Upcoding** | Billing for a higher-complexity code than the service actually provided |
| **Duplicate billing** | Same service billed multiple times |
| **Unbundling** | Splitting one procedure into multiple separate codes for higher reimbursement |
| **Excluded provider billing** | A provider barred from Medicare continues to bill |

### 1.2 Why Machine Learning?

Traditional fraud detection relies on manual audits, which are:
- **Slow** — auditors can review only a small fraction of claims
- **Inconsistent** — different auditors flag different things
- **Reactive** — fraud is caught months or years after it happened

An ML-based approach enables:
- **Scale** — analyze 40,000+ providers simultaneously
- **Consistency** — same statistical rules applied to every provider
- **Pattern discovery** — detect anomalies humans wouldn't notice in raw data

---

## 2. Problem Formulation

### 2.1 Task Definition

This is a **semi-supervised anomaly detection problem** with an additional supervised component:

| Component | Task Type | Goal |
|-----------|-----------|------|
| Z-score outlier flagging | Unsupervised (statistical) | Flag providers statistically deviant from peers |
| Isolation Forest | Unsupervised (ML) | Detect anomalous billing patterns |
| Upcoding detection | Rule-based | Identify E&M code abuse |
| Duplicate classifier | Supervised (ML) | Classify duplicate-like billing patterns |
| LEIE matching | Deterministic lookup | Flag federally excluded providers |

### 2.2 Why Not a Simple Classification Problem?

A supervised fraud classifier requires **labeled fraud data** — i.e., known fraudulent claims.  
We deliberately avoided this approach because:

1. **No reliable labeled dataset** — Confirmed fraud cases in Medicare are rare, asymmetric, and rarely made public
2. **Class imbalance is extreme** — Actual fraud is < 1% of claims; training a classifier on this would be unreliable
3. **Anomaly detection is more appropriate** — We don't need labeled fraud; we need to detect *statistical deviation from expected behavior*

### 2.3 Success Criteria

The system was considered successful if it could:
- Assign a meaningful risk tier to every provider
- Identify known risk patterns (upcoding, extreme Z-scores) reliably
- Match excluded providers from LEIE with high NPI recall
- Produce actionable, ranked priority lists for investigators

---

## 3. Related Work & Prior Art

### 3.1 Existing Approaches in Healthcare Fraud Detection

| Approach | Description | Limitation |
|----------|-------------|-----------|
| **Rule-based systems** | Hard-coded thresholds (e.g. "flag anyone billing > $X") | Too rigid, misses novel patterns |
| **Supervised classification** | Train on labeled fraud cases | Requires ground truth labels, hard to get |
| **Network analysis** | Graph-based detection of collusion | Complex to implement, needs entity relationships |
| **Peer-group benchmarking** | Compare providers to similar peers | Our primary approach — statistically sound |
| **Isolation Forest** | Anomaly detection on billing features | Effective for high-dimensional unsupervised detection |

### 3.2 How This Project Differs

Most academic work uses synthetic or highly restricted datasets. This project:
- Uses **real, publicly available CMS data** (~2–3 GB)
- Combines **statistical + ML + rule-based + lookup methods** into one unified pipeline
- Produces **prioritized investigation lists** rather than just raw flags
- Cross-references the **official OIG exclusion database** — a step most research papers skip

---

## 4. Data Research

### 4.1 Why CMS Medicare Data?

Several public healthcare datasets were considered:

| Dataset | Considered? | Reason for Choice / Rejection |
|---------|-------------|-------------------------------|
| CMS Medicare Physician & Practitioner | ✅ **Selected** | Largest public Medicare billing dataset, NPI-level granularity |
| CMS Part D Drug Prescriptions | ❌ Rejected | Drug-only scope, different fraud patterns |
| AHRQ Hospital Data | ❌ Rejected | Facility-level only, no individual provider NPIs |
| State Medicaid Data | ❌ Rejected | Fragmented across states, no unified national format |

### 4.2 Why OIG LEIE?

The LEIE (List of Excluded Individuals/Entities) is the **official federal registry** of healthcare providers barred from Medicare and Medicaid participation. Using it:
- Provides **ground truth on known bad actors** for validation
- Directly actionable — a LEIE match is a confirmed regulatory violation
- Freely downloadable and regularly updated

### 4.3 Data Quality Issues Encountered

| Issue | Discovery | Resolution |
|-------|-----------|-----------|
| `Rndrng_Prvdr_First_Name` had nulls for organizations | Step 2.1 | Filled with `'Organization'` |
| Numeric columns stored as strings | Step 2.3 | `pd.to_numeric(errors='coerce')` |
| LEIE NPI column had `.0` float suffix | Step 5 | Strip `.0` + regex filter for 10-digit NPIs |
| Invalid LEIE NPIs (`0`, `nan`, `0000000000`) | Step 5 | Replaced with `NaN` before matching |
| CMS file too large for single `pd.read_csv()` | Step 2.1 | Chunked loading (`chunksize=100_000`) |
| Duplicate provider rows after groupby | Step 2 | Rebuild with `dropna=False` on groupby |

---

## 5. Feature Engineering Decisions

### 5.1 Why Provider-Level Aggregation?

The raw CMS file is at the **provider × HCPCS code level** — one row per provider per billing code. This means a single provider may appear 100+ times.

We aggregated to **provider level** because:
- Fraud is a provider behavior, not a per-code behavior
- Peer-group comparison requires one vector per provider
- Reduces dataset from millions of rows to ~44,000 providers

### 5.2 Feature Rationale

| Feature | Formula | Why It Matters |
|---------|---------|----------------|
| `Payment_Per_Bene` | `Total Payment / Total Beneficiaries` | High value = excessive billing per patient |
| `Services_Per_Bene` | `Total Services / Total Beneficiaries` | Unusually high = service overutilization |
| `Payment_Per_Service` | `Total Payment / Total Services` | High-value billing per individual service |
| `Charge_To_Payment_Ratio` | `Avg Submitted Charge / Avg Medicare Payment` | Very high ratio = inflated billing attempts |
| `Unique_HCPCS_Count` | `nunique(HCPCS_Cd)` | Extreme breadth may indicate code farming |

### 5.3 Why Peer Groups Instead of Global Comparison?

Initial testing showed that global Z-scores flagged **the wrong providers**. A cardiologist performing complex surgeries will always have higher payment-per-service than a general practitioner — flagging them globally is a false positive.

**Solution:** Group by `Provider Type × State` before computing Z-scores.

```
Peer Group = "Internal Medicine_CA"  vs  "Cardiology_CA"
```

This ensures each provider is only compared to **clinically similar peers in the same market**.

Result: 1,200+ peer groups created, dramatically reducing false positives.

---

## 6. Model Selection & Rationale

### 6.1 Anomaly Detection — Why Isolation Forest?

Several unsupervised anomaly detection methods were evaluated:

| Method | Considered | Verdict |
|--------|-----------|---------|
| **Z-Score** | ✅ Used as first layer | Simple, interpretable, peer-group aware |
| **Isolation Forest** | ✅ Used as second layer | Best for high-dimensional tabular data, fast on large datasets |
| **Local Outlier Factor (LOF)** | Considered | Slow on 44K+ providers, density-based (not ideal here) |
| **One-Class SVM** | Considered | High memory requirement, sensitive to scaling |
| **Autoencoder** | Considered | Overkill for tabular data, harder to explain |

**Why Isolation Forest specifically:**
- Works well with **mixed-scale numerical features** (no strong distribution assumptions)
- `contamination=0.05` — assumes ~5% of providers are anomalous (conservative, aligns with domain estimates)
- `n_estimators=200` — more trees = more stable anomaly scores
- `random_state=42` — reproducibility

### 6.2 Duplicate Classifier — Why Random Forest?

For the duplicate billing classifier:

| Method | Verdict |
|--------|---------|
| **Logistic Regression** | Rejected — linear boundary insufficient for complex billing patterns |
| **Decision Tree** | Rejected — overfits, not stable |
| **Random Forest** | ✅ Selected — handles non-linearity, built-in feature importance, robust to noise |
| **XGBoost** | Considered — would likely perform similarly, but RF sufficient here |
| **SVM** | Rejected — doesn't scale well to 40K+ providers |

**Key hyperparameters chosen:**
```python
RandomForestClassifier(
    n_estimators=100,   # 100 trees — sufficient stability without excessive compute
    random_state=42,    # reproducibility
    n_jobs=-1           # use all CPU cores — needed for large CMS data
)
```

Threshold tuning: Default `0.5` was evaluated using precision-recall curve. `0.5` gave the best balance for this use case.

### 6.3 Threshold Selection for Risk Scoring

The composite Fraud Risk Score is the count of features with Z-score > 2 (positive side only):

```
Score = 0 → Low Risk    (no anomalous features)
Score 1-2 → Medium Risk (1-2 features anomalous)
Score ≥ 3 → High Risk   (3+ features simultaneously anomalous)
```

**Rationale:** A single anomalous feature could be legitimate (e.g., a specialist seeing complex patients). Requiring **3+ simultaneous anomalies** reduces false positives significantly.

---

## 7. Experimental Results

### 7.1 Risk Distribution

| Risk Level | Count | % of Total |
|-----------|-------|-----------|
| High Risk | 3,842 | 8.6% |
| Medium Risk | 11,276 | 25.3% |
| Low Risk | 29,410 | 66.1% |
| **Total** | **44,528** | **100%** |

### 7.2 Duplicate Classifier Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~96% |
| Threshold | 0.5 |
| True Positives (Duplicate correctly identified) | 4,788 |
| True Negatives (Real correctly identified) | 4,813 |
| False Positives | 187 |
| False Negatives | 212 |

### 7.3 Upcoding Detection

| Metric | Value |
|--------|-------|
| Providers with E&M data analyzed | Subset of total |
| Upcoding flags raised | 2,914 |
| HCPCS codes monitored | 99211, 99212, 99213, 99214, 99215 |

### 7.4 Feature Importance (Duplicate Classifier)

Top features identified by the Random Forest model:
1. `Charge_To_Payment_Ratio` — Highest discriminative power
2. `Services_Per_Bene` — Second most important
3. `Payment_Per_Bene` — Strong indicator
4. `Unique_HCPCS_Count` — Code diversity matters
5. `Payment_Per_Service` — Useful but less discriminative

---

## 8. Challenges & Solutions

### 8.1 Memory — CMS File Too Large to Load at Once

**Problem:** The CMS file is 2–3 GB. A single `pd.read_csv()` call caused memory errors.

**Solution:** Chunked loading with `chunksize=100_000`, then `pd.concat()`:
```python
chunks = []
for chunk in pd.read_csv(cms_path, chunksize=100_000, low_memory=False):
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)
```

### 8.2 LEIE NPI Cleaning — Float Suffix & Invalid Values

**Problem:** LEIE NPIs were stored with `.0` float suffix (e.g., `1234567890.0`). Direct NPI matching returned zero matches.

**Solution:** Strip `.0`, then filter with regex for valid 10-digit NPIs only:
```python
leie['NPI_clean'] = leie['NPI'].astype(str).str.replace('.0', '', regex=False).str.strip()
leie.loc[~leie['NPI_clean'].str.match(r'^\d{10}$', na=False), 'NPI_clean'] = np.nan
```

### 8.3 Peer Group Too Small for Reliable Z-Scores

**Problem:** Some rare specialty × state combinations had only 1–2 providers. Z-scores for these are unreliable (std dev = 0 or undefined).

**Solution:** Added `Reliable_Peer_Group` flag:
```python
provider_df['Reliable_Peer_Group'] = provider_df['Provider_Count'] >= 10
```
Providers in small peer groups are still included but their risk scores are treated with lower confidence.

### 8.4 Provider Name — Individuals vs Organizations

**Problem:** The CMS file uses separate `First_Name` + `Last_Name` columns for individuals, but `Last_Org_Name` for organizations (like hospitals). Naive concatenation produced garbage names.

**Solution:**
```python
df['Provider_Name'] = np.where(
    df['Rndrng_Prvdr_Ent_Cd'] == 'O',
    df['Rndrng_Prvdr_Last_Org_Name'],                        # Organization
    df['Rndrng_Prvdr_First_Name'] + ' ' + df['Rndrng_Prvdr_Last_Org_Name']  # Individual
)
```

### 8.5 Division by Zero in Feature Engineering

**Problem:** A small number of providers had `Tot_Benes = 0` (data entry issues), causing `inf` values in ratio features.

**Solution:** `pd.to_numeric(errors='coerce')` converts all problematic values to `NaN`, which are then safely ignored by the Z-score and ML models.

---

## 9. Limitations

| Limitation | Impact | Possible Mitigation |
|-----------|--------|---------------------|
| **No confirmed fraud labels** | Cannot compute true precision/recall against real fraud | Obtain validated OIG investigation outcomes |
| **Annual data only** | Temporal trends (billing increases/decreases over time) not captured | Use multi-year CMS data |
| **NPI-level only** | Does not detect collusion between providers | Add network/graph analysis layer |
| **LEIE NPI coverage gaps** | ~40% of LEIE records have no valid NPI — matched by name only if at all | Use name + address fuzzy matching |
| **Synthetic duplicate training data** | RF classifier was trained on simulated duplicates, not confirmed ones | Obtain CMS duplicate claim investigation data |
| **No claim-level analysis** | Aggregated to provider level — individual fraudulent claims not visible | Extend pipeline to claim-level anomaly detection |
| **No temporal drift handling** | The Isolation Forest is trained once on 2023 data | Retrain periodically as billing patterns shift |

---

## 10. Future Work

### Short-Term (Next 3 Months)
- [ ] Add **multi-year comparison** (2021 vs 2022 vs 2023 CMS data) to detect sudden billing spikes
- [ ] Improve LEIE matching with **fuzzy name + address matching** for records without valid NPIs
- [ ] Build a **Streamlit dashboard** for interactive provider investigation

### Medium-Term (3–6 Months)
- [ ] Incorporate **CMS Part D (drug prescriptions)** data to detect pharmacy-linked fraud
- [ ] Add **graph/network analysis** to detect provider collusion rings
- [ ] Implement **model drift monitoring** — alert when billing patterns shift significantly

### Long-Term
- [ ] Connect to a **live CMS data feed** for near-real-time detection
- [ ] Train a **true supervised fraud classifier** if labeled data becomes available from OIG settlements
- [ ] Package as a **REST API** consumable by compliance teams

---

## 11. References

| # | Source |
|---|--------|
| 1 | CMS Medicare Physician & Other Practitioners Dataset — https://data.cms.gov |
| 2 | OIG LEIE Exclusion List — https://oig.hhs.gov/exclusions/exclusions_list.asp |
| 3 | Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest.* ICDM. |
| 4 | Breiman, L. (2001). *Random Forests.* Machine Learning, 45(1), 5–32. |
| 5 | OIG Work Plan — https://oig.hhs.gov/reports-and-publications/workplan/ |
| 6 | CMS Fraud Prevention — https://www.cms.gov/priorities/innovation/key-concept/fraud-prevention |
| 7 | Scikit-learn Documentation — https://scikit-learn.org/stable/ |

---

<p align="center">
  <i>This RND document reflects the research decisions, experiments, and findings of the Healthcare Fraud Detection project.</i><br/>
  <i>For usage instructions, see <a href="README.md">README.md</a></i>
</p>
