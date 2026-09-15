# CreditPulse — Marketing & Revenue Analytics Platform

> **Which leads are most likely to convert, when will they convert, how valuable are they, and which interventions should we prioritize?**

CreditPulse is an **end-to-end Marketing / RevOps Analytics Engineering project** that combines data engineering, machine learning, survival analysis, customer lifetime value (CLV), and campaign analytics into a single decision-support platform.

The project simulates a B2B CRM environment and takes data through:

**Synthetic CRM Data → Medallion ETL → Analytics Warehouse → ML → Survival Analysis → CLV → Campaign Analysis → Dashboard**

---

## Project Overview

CreditPulse was originally designed as a fintech credit-risk project and evolved into a **Marketing / RevOps analytics platform**.

The underlying engineering architecture remains the same, while the business problem shifted from financial default prediction to revenue and customer analytics.

| Original Fintech Concept     | CreditPulse Use Case       |
| ---------------------------- | -------------------------- |
| Credit default prediction    | Lead conversion prediction |
| Time-to-default              | Time-to-conversion         |
| Credit risk + customer value | Lead score + CLV           |
| Loan intervention            | Campaign intervention      |
| Credit risk reporting        | RevOps decision support    |

The goal is to demonstrate how a data professional can connect:

**Data Engineering + Machine Learning + Statistics + Marketing Analytics + Business Decision-Making**

---

## Important: Synthetic Data

All data in CreditPulse is **synthetically generated**.

The dataset is created using:

* Python
* Faker
* NumPy
* Pandas

Faker generates realistic multi-locale customer and company information, while the data-generation process introduces realistic relationships between:

* Lead source
* Industry
* Job-title seniority
* Company size
* Region
* Engagement
* Campaign exposure
* Campaign response
* Conversion

The conversion outcome is generated using a logistic probability model with controlled signal and noise.

This allows the project to demonstrate a realistic ML workflow without exposing real customer or company data.

---

# Architecture

CreditPulse follows a **Bronze → Silver → Gold medallion architecture**.

```text
                    SYNTHETIC CRM DATA
                           │
                           ▼
                  ┌─────────────────┐
                  │     BRONZE      │
                  │   Raw ingestion │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │     SILVER      │
                  │ Cleaning & DQ   │
                  │ Deduplication   │
                  │ Quarantine      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │      GOLD       │
                  │  Star Schema    │
                  │ Lead Summary    │
                  └────────┬────────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
          ML Scoring   Survival       CLV
              │         Analysis        │
              └────────────┼────────────┘
                           ▼
                  Campaign / Uplift
                      Analysis
                           │
                           ▼
                  ┌─────────────────┐
                  │   Streamlit     │
                  │   Dashboard     │
                  └─────────────────┘
```

The architecture is intentionally similar to patterns used in modern analytics platforms and cloud data warehouses.

---

# Data Model

The synthetic dataset contains four primary source tables.

| Table                  |    Rows | Description                                       |
| ---------------------- | ------: | ------------------------------------------------- |
| `raw_leads`            |  50,871 | Lead, person, company and acquisition information |
| `raw_opportunities`    |  30,401 | Opportunity and deal information                  |
| `raw_activities`       | 249,186 | Calls, emails, meetings and other engagement      |
| `raw_campaign_touches` |  22,889 | Campaign treatment/control interactions           |

### Gold Layer

The Gold layer contains a dimensional model consisting of:

* `dim_time`
* `dim_leads`
* `fact_activities`
* `fact_opportunities`
* `fact_campaign_touches`

It also contains a business-ready:

### `lead_summary`

This table brings together the features required by downstream analytics and ML.

Key fields include:

* Lead attributes
* Industry
* Company size
* Lead source
* Region
* Job title
* Activity counts
* Campaign exposure
* Treatment group
* Campaign response
* Opportunity information
* Deal value
* Conversion status
* Time-to-conversion

---

# Data Quality & ETL

The ETL pipeline intentionally introduces common CRM data-quality problems so that they can be handled as part of the engineering workflow.

### Examples

| Problem           | Example                | Handling                     |
| ----------------- | ---------------------- | ---------------------------- |
| Missing values    | Missing company size   | Preserve NULL + missing flag |
| Incorrect types   | `"163 emp"`            | Regex conversion             |
| Duplicate records | Duplicate activity ID  | Deduplication                |
| Orphan records    | Invalid `lead_id`      | Quarantine                   |
| Extreme values    | Unrealistic deal value | Flag as outlier              |

### Silver Layer Principles

* Raw data is never silently overwritten.
* Nulls are preserved during cleaning.
* Duplicates are removed.
* Invalid foreign keys are quarantined.
* Outliers are flagged.
* Modeling decisions are kept separate from data cleaning.

Example:

```text
Bronze
   │
   ├── Valid records ──────► Silver
   │
   └── Invalid records ────► Rejection / Quarantine
```

This creates an auditable data-quality process rather than simply deleting bad records.

---

# Machine Learning — Lead Scoring

### Business Question

> **Which leads are most likely to convert?**

The ML pipeline compares:

* Logistic Regression
* Random Forest
* XGBoost

### Workflow

```text
Gold lead_summary
       │
       ▼
Feature Selection
       │
       ▼
Train / Test Split
       │
       ▼
Preprocessing
       │
       ▼
Model Training
       │
       ▼
Model Comparison
       │
       ▼
Best Model
       │
       ▼
Conversion Probability
       │
       ▼
Lead Tier
```

### Features

**Numeric**

* `total_activities`
* `company_size`
* `received_campaign`

**Categorical**

* `industry`
* `lead_source`
* `region`
* `job_title`
* `treatment_group`
* `campaign_response`

### Target

```text
event_converted
```

### Feature Leakage Prevention

The following fields are deliberately excluded:

* `deal_value`
* `duration_days`
* `status`
* `opportunity_stage`

These fields contain information that would not legitimately be available at prediction time and could leak the outcome into the model.

---

## Model Results

| Model               |        AUC |   Accuracy |         F1 |  Precision |     Recall |
| ------------------- | ---------: | ---------: | ---------: | ---------: | ---------: |
| Logistic Regression |     0.7418 |     0.6614 |     0.4688 |     0.3571 |     0.6822 |
| **Random Forest**   | **0.7667** | **0.6948** | **0.4939** | **0.3878** | **0.6800** |
| XGBoost             |     0.7660 |     0.6904 |     0.4979 |     0.3861 |     0.7011 |

### Selected Model

**Random Forest — AUC 0.7667**

Random Forest was selected because it achieved the highest holdout AUC, with XGBoost performing almost identically.

The dataset intentionally contains noise, so an AUC around 0.77 is more meaningful than artificially optimizing the synthetic data toward an unrealistically high score.

---

# Lead Risk Tiers

Instead of using arbitrary probability thresholds, CreditPulse uses **quantile-based tiers**.

```text
Top 20%       → High Priority
Middle 60%    → Medium Priority
Bottom 20%    → Low Priority
```

This approach allows the tiers to adapt when the underlying score distribution changes.

The thresholds are stored in model metadata for reproducibility.

---

# Survival Analysis

### Business Question

> **Not only "will this lead convert?" — but "when will it convert?"**

Classification predicts the probability of conversion.

Survival analysis adds the **time dimension**.

CreditPulse uses:

* Kaplan-Meier analysis
* Cox Proportional Hazards

It also handles **right-censoring**, meaning leads that have not converted by the dataset's snapshot date are not incorrectly treated as failures.

---

## Kaplan-Meier Results

| Lead Source  | Median Time | Interpretation             |
| ------------ | ----------: | -------------------------- |
| **Referral** | ~3,787 days | Fastest converting segment |
| Webinar      | ~5,430 days | Relatively faster          |
| Trade Show   | ~5,652 days | Moderate                   |
| LinkedIn     | ~6,031 days | Slower                     |
| Website      | Not reached | Fewer than 50% converted   |
| Cold Call    | Not reached | Fewer than 50% converted   |

---

## Cox Proportional Hazards

| Feature               | Hazard Ratio | Interpretation                |
| --------------------- | -----------: | ----------------------------- |
| **Referral**          |     **3.38** | Converts substantially faster |
| Webinar               |         2.36 | Faster conversion             |
| SaaS                  |         2.15 | Higher conversion speed       |
| **Received campaign** |     **1.52** | 52% higher conversion hazard  |
| FinServ               |         1.88 | Faster than baseline          |
| Total activities      |         1.03 | Small positive effect         |
| EMEA                  |         0.91 | Slightly slower               |

### Model Result

**C-index: 0.6753**

The C-index answers a different question from AUC:

* **AUC:** Can the model rank converters above non-converters?
* **C-index:** Can the model correctly rank which lead converts earlier?

---

# Customer Lifetime Value

### Business Question

> **Which leads or customers are worth the most?**

CreditPulse combines:

1. Conversion probability
2. Deal value
3. Time-to-conversion

into a survival-adjusted CLV estimate.

### Formula

```text
CLV =
P(convert)
× effective deal value
× exp(-r × t)
```

Where:

* `P(convert)` = ML conversion probability
* `effective deal value` = observed deal value or estimated industry median
* `r` = annual discount rate
* `t` = predicted median time-to-conversion

The model therefore accounts for both **probability and timing**.

A lead expected to generate the same revenue several years later receives a lower present value than a lead expected to convert sooner.

---

## CLV Results

| Metric                          |        Value |
| ------------------------------- | -----------: |
| **Total Portfolio CLV**         | **$170.44M** |
| **Median CLV / Lead**           |   **$1,936** |
| **Mean CLV / Lead**             |   **$3,351** |
| **P90 CLV**                     |   **$7,517** |
| **Maximum CLV**                 | **$131,689** |
| Discount Rate                   |   10% / year |
| Leads with estimated deal value |       20,473 |

### Highest-value segments

**Industry:** SaaS
**Lead Source:** Referral

This creates an important cross-model insight:

> Referral leads convert faster and also generate higher expected value.

---

## CLV Data Limitation

Approximately 40% of leads do not have an associated opportunity.

For these leads, the model uses the **industry median deal value** as an estimate.

These records are explicitly flagged using:

```text
used_imputed_deal
```

This allows the dashboard to distinguish between observed and estimated CLV rather than presenting all values as equally certain.

---

# Campaign & Uplift Analysis

The Gold layer contains:

* `treatment_group`
* `campaign_response`

This enables the next stage of the project:

### Uplift Modeling

Instead of asking:

> "Who is likely to convert?"

uplift modeling asks:

> **"Who is likely to convert because of the intervention?"**

The planned model will identify four segments:

| Segment             | Meaning                                   | Action                 |
| ------------------- | ----------------------------------------- | ---------------------- |
| **Persuadables**    | Campaign increases conversion probability | Target                 |
| **Sure Things**     | Likely to convert anyway                  | Reduce campaign spend  |
| **Lost Causes**     | Unlikely to convert regardless            | Avoid excessive effort |
| **Do Not Disturbs** | Intervention may reduce conversion        | Avoid                  |

Planned approaches include:

* T-Learner
* X-Learner
* Uplift scores
* Qini curve
* Incremental conversion analysis

This is intended to move the project from **predictive analytics to prescriptive marketing analytics**.

---

# Streamlit Dashboard

The project includes an interactive Streamlit dashboard.

### Dashboard Sections

| Page                   | Purpose                                |
| ---------------------- | -------------------------------------- |
| 🏠 Overview            | Portfolio KPIs and key insights        |
| 👥 Customer Analytics  | Lead search and customer 360           |
| ⚠️ Risk Analysis       | Lead priority and risk segmentation    |
| 🧪 Interventions       | Treatment vs control analysis          |
| 🧠 ML Models           | Model comparison and calibration       |
| 📈 Survival Analysis   | Conversion timing and hazard ratios    |
| 💎 CLV                 | Customer value and value-risk analysis |
| 🗄️ Data & ETL         | Pipeline and data-quality monitoring   |
| 📊 Executive Dashboard | End-to-end decision view               |
| ⚙️ Settings            | Application configuration              |

### ML Dashboard

Includes:

* AUC
* Accuracy
* F1
* Precision
* Recall
* Model comparison
* Conversion probability distribution
* Lead tiers
* Calibration analysis

### Survival Dashboard

Includes:

* C-index
* Kaplan-Meier curves
* Median conversion time
* Cox hazard ratios
* Confidence intervals
* Per-lead predicted conversion timing

### CLV Dashboard

Includes:

* Portfolio CLV
* CLV distribution
* CLV tiers
* Value vs risk analysis
* Top-value leads
* CLV by industry
* CLV by lead source

---

# Technology Stack

| Area                  | Technology            |
| --------------------- | --------------------- |
| Programming           | Python 3.13           |
| Data Processing       | Pandas, NumPy, SciPy  |
| Synthetic Data        | Faker                 |
| Database              | PostgreSQL 18.6       |
| SQL Connectivity      | SQLAlchemy, psycopg2  |
| Machine Learning      | Scikit-learn, XGBoost |
| Survival Analysis     | Lifelines             |
| Dashboard             | Streamlit, Plotly     |
| Configuration         | python-dotenv         |
| Version Control       | Git, GitHub           |
| Architecture Diagrams | draw.io               |
| Planned Orchestration | Apache Airflow        |

---

# Repository Structure

```text
CreditPulse/
│
├── data/
│   └── raw/
│       ├── raw_leads.csv
│       ├── raw_opportunities.csv
│       ├── raw_activities.csv
│       └── raw_campaign_touches.csv
│
├── models/
│   ├── random_forest_pipeline.pkl
│   ├── model_metadata.json
│   ├── survival_cox_summary.json
│   ├── survival_km_by_segment.json
│   └── clv_summary.json
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── load_to_postgres.py
│   ├── clean_bronze_to_silver.py
│   ├── run_sql_file.py
│   ├── create_survival_view.py
│   ├── train_ml_models.py
│   ├── survival_analysis.py
│   ├── clv_modeling.py
│   └── check_disk.py
│
├── sql/
│   ├── create_bronze_tables.sql
│   ├── create_silver_tables.sql
│   ├── create_gold_tables.sql
│   ├── create_survival_tables.sql
│   └── create_ml_tables.sql
│
├── app.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# Running the Project

## Prerequisites

* Python 3.11+
* PostgreSQL 18+
* Git

## Installation

```bash
git clone <repository-url>
cd CreditPulse

python -m venv .venv
```

### Windows

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Configure Database

Create a `.env` file:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=creditpulse
DB_USER=postgres
DB_PASSWORD=your_password

DATABASE_URL=postgresql+psycopg2://postgres:your_password@localhost:5432/creditpulse
```

Create the PostgreSQL database:

```sql
CREATE DATABASE creditpulse;
```

---

# Run the Pipeline

### 1. Generate synthetic data

```bash
python scripts/generate_synthetic_data.py
```

### 2. Create and load Bronze

```bash
python scripts/run_sql_file.py sql/create_bronze_tables.sql
python scripts/load_to_postgres.py
```

### 3. Create and populate Silver

```bash
python scripts/run_sql_file.py sql/create_silver_tables.sql
python scripts/clean_bronze_to_silver.py
```

### 4. Build Gold

```bash
python scripts/run_sql_file.py sql/create_gold_tables.sql
```

### 5. Train lead-scoring models

```bash
python scripts/train_ml_models.py
```

### 6. Run survival analysis

```bash
python scripts/create_survival_view.py
python scripts/survival_analysis.py
```

### 7. Calculate CLV

```bash
python scripts/clv_modeling.py
```

### 8. Launch dashboard

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

# Engineering Decisions

## 1. Reproducible snapshot date

The synthetic dataset uses a fixed snapshot date rather than the current system date.

This ensures that the dataset and model results remain reproducible.

---

## 2. Nulls are preserved during cleaning

Missing values are not automatically imputed in the Silver layer.

Imputation is treated as a **modeling decision**, allowing different downstream models to use different strategies.

---

## 3. Invalid records are quarantined

Orphan records are moved into rejection tables with a reason instead of being silently deleted.

This maintains an audit trail.

---

## 4. Conversion is a lead-level outcome

`event_converted` is sourced from the lead's status rather than opportunity stage.

A lead can convert without necessarily becoming an opportunity, so using opportunity stage as the target would incorrectly classify some conversions.

---

## 5. Avoiding ML leakage

Fields such as:

```text
deal_value
duration_days
status
opportunity_stage
```

are excluded from the lead-scoring model because they contain information about the outcome.

---

## 6. Quantile-based segmentation

Lead priority and CLV tiers use quantiles rather than arbitrary fixed thresholds.

This makes segmentation more robust to changes in score distributions.

---

## 7. Idempotent pipeline

Each major pipeline stage can be safely re-run.

Target tables are cleared before new outputs are written, allowing failures to be recovered without manually cleaning partial results.

---

# Infrastructure Lesson

During development, PostgreSQL was running on Windows with its data directory configured on `D:\pgdata`.

A pipeline run generated enough temporary/database data to severely reduce free space on the system drive, eventually causing PostgreSQL to fail with:

```text
No space left on device
```

The issue led to:

* Moving PostgreSQL data to a dedicated drive
* Reinstalling PostgreSQL with the correct data directory
* Adding disk-space checks to pipeline scripts
* Improving failure diagnostics

A disk-space guard now prevents the pipeline from running when insufficient system storage is available.

This was an important engineering lesson: **data pipelines need operational safeguards, not just correct transformations.**

---

# Current Project Status

| Phase                                   | Status          |
| --------------------------------------- | --------------- |
| Synthetic Data Generation               | ✅ Complete      |
| Bronze / Silver / Gold ETL              | ✅ Complete      |
| Data Quality Handling                   | ✅ Complete      |
| Lead Scoring                            | ✅ Complete      |
| Random Forest — AUC 0.77                | ✅ Complete      |
| Survival Analysis — C-index 0.675       | ✅ Complete      |
| Survival-adjusted CLV — $170M portfolio | ✅ Complete      |
| Streamlit Dashboard                     | 🔶 MVP Complete |
| Uplift Modeling                         | ⬜ Next          |
| Airflow Orchestration                   | ⬜ Planned       |
| Dashboard Polish                        | 🔶 Ongoing      |
| Deployment                              | ⬜ Planned       |

---

# Roadmap

### Phase 9 — Uplift Modeling

* T-Learner / X-Learner
* Individual treatment effect
* Uplift segmentation
* Qini curve
* Incremental conversion analysis
* Streamlit uplift dashboard

### Phase 10 — Pipeline Orchestration

* Apache Airflow
* DAG-based ETL
* Dependency management
* Pipeline monitoring
* Automated model execution

### Phase 11 — Deployment

* Production-style deployment
* Environment configuration
* Documentation
* Reproducible setup
* Dashboard deployment

---

# Key Takeaway

CreditPulse demonstrates an end-to-end analytics workflow rather than a standalone ML model.

```text
DATA
  ↓
ENGINEERING
  ↓
QUALITY
  ↓
WAREHOUSE
  ↓
PREDICTION
  ↓
TIME-TO-EVENT
  ↓
CUSTOMER VALUE
  ↓
INTERVENTION
  ↓
BUSINESS DECISION
```

The project is designed to demonstrate the intersection of:

**Marketing Analytics × Revenue Operations × Data Engineering × Machine Learning × Statistics × Customer Value**

---

## Acknowledgements

CreditPulse uses:

* Faker for synthetic data generation
* Pandas / NumPy / SciPy for data processing
* Scikit-learn / XGBoost for machine learning
* Lifelines for survival analysis
* PostgreSQL for data storage
* Streamlit / Plotly for visualization

---

*CreditPulse is a portfolio project using entirely synthetic data.*

*Last updated: September 2026*
