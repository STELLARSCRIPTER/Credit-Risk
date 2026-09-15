# CreditPulse — End-to-End Credit Risk & Customer Value Analytics Platform

> A portfolio-grade analytics engineering project that transforms synthetic CRM data into actionable lead scoring, survival analysis, customer lifetime value, and causal uplift insights.

CreditPulse was originally designed as a **credit-risk analytics platform** and has been intentionally adapted into a **Marketing / RevOps Analytics Engineering** project.

The project demonstrates an end-to-end workflow:

**Synthetic CRM Data → PostgreSQL → Medallion ETL → ML Lead Scoring → Survival Analysis → CLV → Uplift Modeling → Streamlit Dashboard**

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Credit Risk → Marketing / RevOps Mapping](#credit-risk--marketing--revops-mapping)
4. [Synthetic Data](#synthetic-data)
5. [Architecture](#architecture)
6. [Data Model](#data-model)
7. [Data Quality & ETL](#data-quality--etl)
8. [ML Lead Scoring](#ml-lead-scoring)
9. [Customer Lifetime Value](#customer-lifetime-value)
10. [Uplift Modeling](#uplift-modeling)
11. [Dashboard](#dashboard)
12. [Infrastructure Notes](#infrastructure-notes)
13. [Setup & Running Locally](#setup--running-locally)
14. [Key Design Decisions](#key-design-decisions)
15. [Project Status](#project-status)

---

## Project Overview

CreditPulse is an end-to-end analytics platform built to demonstrate how CRM and marketing data can be transformed into predictive and decision-oriented analytics.

The project covers:

* Synthetic CRM data generation
* PostgreSQL data warehousing
* Bronze → Silver → Gold medallion architecture
* Data cleaning and deduplication
* Lead conversion prediction
* Survival / time-to-conversion analysis
* Customer lifetime value estimation
* Causal uplift modeling
* Executive analytics through Streamlit
* Reproducible local infrastructure

The project is intentionally designed around **business questions**, rather than simply demonstrating individual machine-learning algorithms.

---

## Business Problem

A CRM team does not only need to know:

> **"Which leads are likely to convert?"**

It also needs to understand:

1. **Will the lead convert?**
2. **When is the lead likely to convert?**
3. **How valuable is the lead?**
4. **Will a campaign actually cause the lead to convert?**

CreditPulse answers these questions through four complementary analytics layers:

| Phase   | Business Question                   | Technique         |
| ------- | ----------------------------------- | ----------------- |
| Phase 6 | Will this lead convert?             | Classification    |
| Phase 7 | When will this lead convert?        | Survival Analysis |
| Phase 8 | What is this lead worth?            | CLV               |
| Phase 9 | Will the campaign cause conversion? | Uplift Modeling   |

---

## Credit Risk → Marketing / RevOps Mapping

The original analytical concepts were mapped into a CRM and marketing context.

| Original Credit Concept        | CreditPulse Marketing / RevOps Equivalent |
| ------------------------------ | ----------------------------------------- |
| Customer default risk          | Lead conversion probability               |
| Survival time-to-default       | Time-to-conversion                        |
| Credit risk + CLV              | Lead score + CLV                          |
| Loan offer intervention        | Campaign intervention                     |
| Treatment / control experiment | Campaign A/B test                         |
| Credit risk memo               | RevOps / Marketing decision support       |
| Risk tiers                     | Lead priority tiers                       |

This makes the project relevant to **Marketing Analytics, RevOps, CRM Analytics, Customer Analytics, and Marketing-Finance Analytics** roles.

---

## Synthetic Data

The project uses **synthetic CRM data** generated using:

* Faker
* NumPy
* Pandas

The generator creates realistic relationships between:

* Lead source
* Industry
* Region
* Job title
* Seniority
* Company size
* Campaign exposure
* Campaign response
* Activities
* Opportunities
* Deal value
* Conversion status

The synthetic dataset intentionally contains both signal and noise so that the downstream models behave more like a real-world analytics problem.

### Dataset Size

| Dataset          | Records |
| ---------------- | ------: |
| Leads            |  50,871 |
| Opportunities    |  30,401 |
| Activities       | 249,186 |
| Campaign touches |  22,889 |

> **Important:** All business data in this project is synthetic and does not represent real customers or companies.

---

## Architecture

CreditPulse follows a **Bronze → Silver → Gold medallion architecture**.

```text
                    ┌──────────────────────┐
                    │  Synthetic CRM Data  │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      BRONZE          │
                    │      Raw Data        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      SILVER          │
                    │ Cleaned / Validated  │
                    │ Deduplicated Data    │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │       GOLD           │
                    │ Business-Ready Data │
                    │ Star Schema + Views  │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
        Lead Scoring      Survival         CLV
           Model          Analysis       Modeling
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Uplift Modeling    │
                    │ Campaign Causal ML   │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Streamlit Dashboard  │
                    └──────────────────────┘
```

---

## Data Model

The Gold layer follows a dimensional model.

### Dimensions

* `dim_time`
* `dim_leads`

### Facts

* `fact_activities`
* `fact_opportunities`
* `fact_campaign_touches`

### Business-ready table

* `lead_summary`

`lead_summary` brings together the main lead-level attributes required by the ML and analytics layers.

---

## Data Quality & ETL

The ETL pipeline follows a deterministic Bronze → Silver → Gold process.

### Bronze

Raw generated data is loaded without attempting to overwrite the source values.

### Silver

The cleaning layer handles:

* Deduplication
* Data-type normalization
* Invalid records
* Referential integrity
* Null preservation
* Rejected activity records

Invalid activity records are moved into a quarantine table with an explicit rejection reason instead of being silently discarded.

### Gold

The Gold layer creates business-ready analytical structures for:

* Lead scoring
* Survival analysis
* CLV
* Uplift modeling
* Dashboard reporting

The pipeline is designed to be **idempotent**, allowing target tables to be rebuilt safely.

---

## ML Lead Scoring

### Business Question

> **Which leads are most likely to convert?**

Three classification approaches were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

### Model Performance

| Model               |      AUC |
| ------------------- | -------: |
| Logistic Regression |    ~0.70 |
| Random Forest       | **0.77** |
| XGBoost             |    ~0.75 |

The **Random Forest** model was selected as the final lead-scoring model.

### Final Model

**Random Forest AUC: 0.7667**

The model produces a probability of conversion for each lead.

These probabilities are converted into three operational tiers using quantile-based thresholds:

| Tier            | Population |
| --------------- | ---------: |
| High Priority   |        20% |
| Medium Priority |        60% |
| Low Priority    |        20% |

The thresholds are stored in `models/model_metadata.json` so that scoring remains reproducible.

### Leakage Prevention

The following fields are deliberately excluded from the model:

* `deal_value`
* `duration_days`
* `status`
* `opportunity_stage`

These fields contain information that would only become available after or during conversion and therefore could leak the target into the model.

---

## Survival Analysis

### Business Question

> **Not only "will this lead convert?" — but "when will it convert?"**

CreditPulse uses survival analysis to model time-to-conversion while correctly handling right-censored observations.

### Methods

* Kaplan-Meier survival curves
* Cox Proportional Hazards model

A fixed snapshot date is used for reproducibility rather than dynamically using the current date.

### Cox Model

**C-index: 0.6753**

Selected hazard-ratio findings include:

| Variable          | Hazard Ratio |
| ----------------- | -----------: |
| Referral          |         3.38 |
| Webinar           |         2.36 |
| SaaS              |         2.15 |
| FinServ           |         1.88 |
| Campaign received |         1.52 |
| Total activities  |         1.03 |
| EMEA              |         0.91 |

A hazard ratio above 1 indicates faster conversion relative to the baseline, while a value below 1 indicates slower conversion.

### Kaplan-Meier Findings

Median time-to-conversion varies substantially by lead source.

Examples:

* Referral: ~3,787 days
* Webinar: ~5,430 days
* Trade Show: ~5,652 days
* LinkedIn: ~6,031 days
* Website: median not reached
* Cold Call: median not reached

The survival model also produces a predicted median time-to-conversion for individual leads.

---

## Customer Lifetime Value

### Business Question

> **What is each lead worth after accounting for both conversion probability and time?**

CreditPulse uses a survival-integrated and discounted CLV approach:

```text
CLV = P(convert) × effective deal value × exp(-r × t)
```

Where:

* `P(convert)` = predicted conversion probability
* `effective deal value` = observed or imputed deal value
* `r` = annual discount rate
* `t` = expected time to conversion

### Portfolio Results

| Metric              |            Value |
| ------------------- | ---------------: |
| Total portfolio CLV | **$170,444,069** |
| Median CLV / lead   |       **$1,936** |
| Mean CLV / lead     |       **$3,351** |
| P90 CLV             |       **$7,517** |
| Maximum CLV         |     **$131,689** |

The model uses a **10% annual discount rate**.

### Imputed Deal Values

20,473 leads have no recorded opportunity.

For these leads, deal value is imputed using the industry median.

The `used_imputed_deal` flag is retained so the source of the valuation remains auditable.

### CLV Segmentation

| Tier   | Population |
| ------ | ---------: |
| High   |    Top 20% |
| Medium | Middle 30% |
| Low    | Bottom 50% |

The dashboard also combines CLV with lead risk to create a **value × risk quadrant**.

---

## Uplift Modeling

**Script:** `scripts/uplift_modeling.py` — Phase 9
**Table:** `ml_uplift_predictions`

### Why uplift modeling

Phase 6 answered *"will this lead convert?"*
Phase 7 answered *"when will it convert?"*
Phase 8 answered *"what is this lead worth?"*

Phase 9 answers a fundamentally different question: **"will this lead convert *because* of the campaign?"** — not just *"does the campaign work on average?"*

Standard A/B testing only reveals the **average treatment effect** across all leads. Uplift modeling reveals the **conditional treatment effect** — which specific leads benefit, which are unaffected, and which are actively harmed by intervention.

### The four segments

| Segment            | Control behaviour | Treatment behaviour | Action                        |
| ------------------ | ----------------- | ------------------- | ----------------------------- |
| **Persuadable**    | Would not convert | Converts            | **Target these**              |
| **Sure Thing**     | Would convert     | Converts            | Skip — spend is wasted        |
| **Lost Cause**     | Would not convert | Would not convert   | Skip — no campaign helps      |
| **Do Not Disturb** | Would convert     | Would not convert   | **Suppress — campaign hurts** |

Only **Persuadables** justify the campaign spend. **Do Not Disturbs** are where the campaign actively backfires.

### Methodology — T-learner

Two independent XGBoost classifiers are trained:

1. **Treatment model** — fit only on leads that received the campaign
2. **Control model** — fit only on leads that did not

For each lead, predicted uplift is:

```text
uplift = P(convert | treatment) − P(convert | control)
```

This is the classic T-learner approach. It is simple, interpretable, and well-suited to the balanced arm sizes we have (11,552 Treatment vs. 11,337 Control).

### Features used

**Numeric:** `total_activities`, `company_size`
**Categorical:** `industry`, `lead_source`, `region`, `job_title`

**Deliberately excluded:**

* `treatment_group` — that's the split variable, not a predictor
* `campaign_response` — a post-treatment variable that would leak the treatment effect

### Model quality

| Arm       | AUC (holdout) |
| --------- | ------------- |
| Treatment | 0.7020        |
| Control   | 0.7024        |

The two arms perform almost identically, which validates that the models are learning genuine per-lead conversion probability in each condition — not spuriously distinguishing the arms.

### Aggregate results

Baseline conversion:

* **Control:** 27.65% (11,337 leads)
* **Treatment:** 33.00% (11,552 leads)
* **Raw average uplift:** +5.35pp

Per-lead predicted uplift:

| Metric        | Value      |
| ------------- | ---------- |
| Mean uplift   | **+4.82%** |
| Median uplift | **+4.37%** |
| P90 uplift    | +16.21%    |
| P10 uplift    | −5.62%     |

The model's mean predicted uplift closely matches the observed Treatment-minus-Control difference — a strong sanity check that the T-learner is not drifting from the ground truth.

### Segment distribution

| Segment            |      Count | % of population | Mean uplift |
| ------------------ | ---------: | --------------: | ----------: |
| **Persuadable**    | **10,730** |       **46.9%** | **+12.31%** |
| Lost Cause         |      5,191 |           22.7% |      +0.67% |
| Sure Thing         |      4,356 |           19.0% |      +0.43% |
| **Do Not Disturb** |  **2,612** |       **11.4%** | **−10.37%** |

### Key findings

**1. Almost half the portfolio is Persuadable.** The campaign is broad-based; it produces meaningful uplift on 10,730 leads. This is the segment to prioritise.

**2. Over 2,600 leads are actively harmed by the campaign.** These are overwhelmingly high-baseline leads (Referral source, SaaS industry) that already convert at 77–83% without intervention, but drop to 22–28% when the campaign runs on them. **Suppressing the campaign on this segment saves budget AND improves outcomes.**

**3. A/B testing cannot find the Do Not Disturb segment.** It only reports the average. Uplift modeling identifies the specific 2,612 leads where intervention is counter-productive.

### Evaluating the model — the Qini curve

The Qini curve is the uplift modeling equivalent of the ROC curve. It plots cumulative incremental conversions as we target leads in predicted-uplift order:

* **Steeper initial slope** = the top of the ranking captures the strongest persuadables
* **Curve above the diagonal** = the model's ranking beats random targeting
* **Peak Qini** = maximum incremental conversions available from prioritising by this model

The model's Qini curve sits well above the random-targeting diagonal, peaking around **+1,200 incremental conversions** at roughly 60% of the population.

### Outputs

* `ml_uplift_predictions` table — per-lead P_treatment, P_control, uplift, segment
* `models/uplift_summary.json` — aggregate stats, arm metrics, segment counts

---

## Dashboard

The Streamlit dashboard provides an executive view of the complete analytics pipeline.

### Pages

1. **Overview**
2. **Customer Analytics**
3. **Risk Analysis**
4. **Interventions**
5. **ML Models**
6. **Survival Analysis**
7. **CLV**
8. **Data & ETL**
9. **Dashboard / Executive View**
10. **Settings**

### ML Models Page

Displays:

* Model AUC
* Accuracy
* F1
* Precision
* Recall
* Logistic Regression vs Random Forest vs XGBoost
* Lead tier distribution
* Calibration chart

### Survival Analysis Page

Displays:

* C-index
* Event count
* Total leads
* Kaplan-Meier curves
* Median time-to-conversion by lead source
* Cox hazard ratios
* 95% confidence intervals
* Predicted median time-to-conversion
* Fastest-converting leads

### CLV Page

Displays:

* Total CLV
* Median CLV
* Mean CLV
* CLV distribution
* CLV tier distribution
* Value × risk quadrant
* Top 25 leads
* CLV by industry
* CLV by lead source

### Uplift Modeling

The Phase 9 outputs are available for analytical use through:

* `ml_uplift_predictions`
* Per-lead treatment probability
* Per-lead control probability
* Predicted uplift
* Uplift segment
* Qini evaluation

---

## Infrastructure Notes

The project was developed locally using **PostgreSQL 18.6 on Windows**.

The PostgreSQL data directory was moved from the default system drive to:

```text
D:\pgdata
```

### The problem

Earlier pipeline runs wrote approximately 250 MB to the C: drive.

Eventually the system reached approximately:

```text
0.42 GB free
```

PostgreSQL subsequently failed with:

```text
No space left on device
```

The Windows service wrapper initially obscured the underlying PostgreSQL error.

### Resolution

The PostgreSQL installation was rebuilt using the EDB Windows installer with the data directory configured on `D:\pgdata`.

A disk-space guard was also added to the pipeline to reduce the risk of future failures.

### Lesson

For local data engineering projects, **storage planning is part of the engineering problem**, particularly when repeatedly rebuilding analytical databases.

---

## Setup & Running Locally

### Requirements

* Python 3.13
* PostgreSQL 18+
* Git

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure environment

Copy:

```text
.env.example
```

to:

```text
.env
```

and configure the PostgreSQL connection.

### Generate synthetic data

```bash
python scripts/generate_synthetic_data.py
```

### Create Bronze tables

```bash
python scripts/run_sql_file.py sql/create_bronze_tables.sql
```

### Load raw data

```bash
python scripts/load_to_postgres.py
```

### Create Silver tables

```bash
python scripts/run_sql_file.py sql/create_silver_tables.sql
```

### Clean Bronze → Silver

```bash
python scripts/clean_bronze_to_silver.py
```

### Create Gold tables

```bash
python scripts/run_sql_file.py sql/create_gold_tables.sql
```

### Train ML models

```bash
python scripts/train_ml_models.py
```

### Run survival analysis

```bash
python scripts/survival_analysis.py
```

### Run CLV modeling

```bash
python scripts/clv_modeling.py
```

### Run uplift modeling

```bash
python scripts/uplift_modeling.py
```

### Launch dashboard

```bash
streamlit run app.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

---

## Key Design Decisions

### 1. Fixed snapshot date

The project uses:

```python
AS_OF_DATE = 2026-09-06
```

rather than dynamically using today's date.

This makes the survival and CLV calculations reproducible.

### 2. Preserve nulls in Silver

Missing values are not automatically replaced during the cleaning layer.

Imputation is treated as a modeling decision rather than a generic ETL operation.

### 3. Quarantine invalid records

Invalid activity records are moved into:

```text
activities_rejected
```

with an explicit rejection reason.

### 4. Conversion definition

`event_converted` is sourced from:

```text
leads_clean.status
```

rather than opportunity stage.

A lead can convert without becoming an opportunity, so using opportunity stage as the conversion definition would silently misclassify true conversions.

### 5. Quantile-based risk tiers

Lead risk tiers use quantile thresholds to create a usable:

```text
20 / 60 / 20
```

population split.

The thresholds are stored in model metadata for reproducibility.

### 6. Idempotent loads

Target tables are truncated before writes so that failed or partial pipeline executions can be safely rebuilt.

### 7. Feature leakage prevention

Post-outcome variables are excluded from ML models.

Examples:

```text
deal_value
duration_days
status
opportunity_stage
```

For uplift modeling, post-treatment variables such as:

```text
campaign_response
```

are also excluded.

---

## Project Status

| Phase                                        | Status                                                                  |
| -------------------------------------------- | ----------------------------------------------------------------------- |
| 1. Synthetic Data Generation                 | ✅ Complete                                                              |
| 2. PostgreSQL / Bronze Layer                 | ✅ Complete                                                              |
| 3. Silver Cleaning & Validation              | ✅ Complete                                                              |
| 4. Gold Data Model                           | ✅ Complete                                                              |
| 5. Airflow Orchestration                     | ⬜ Not started                                                           |
| 6. ML Lead Scoring                           | ✅ Complete — Random Forest, AUC 0.77                                    |
| 7. Survival Analysis                         | ✅ Complete — Cox PH, C-index 0.675                                      |
| 8. Customer Lifetime Value                   | ✅ Complete — $170M portfolio CLV                                        |
| 9. Intervention Experiment (Causal / Uplift) | ✅ **Complete — T-learner, 10,730 Persuadables / 2,612 Do Not Disturbs** |
| 10. Dashboard Polish                         | ⬜ Pending                                                               |
| 11. Deployment / Documentation               | ⬜ Pending                                                               |

---

## What's Next

All ML / analytics phases (6 through 9) are now complete:

* ✅ **Phase 6** — Lead scoring (Random Forest, AUC 0.77)
* ✅ **Phase 7** — Survival analysis (Cox PH, C-index 0.675)
* ✅ **Phase 8** — Customer lifetime value (survival-integrated, $170M portfolio)
* ✅ **Phase 9** — Uplift modeling (T-learner, 10,730 Persuadables / 2,612 Do Not Disturbs)

The remaining work is operational polish:

* **Phase 5 — Airflow orchestration.** A DAG that runs the full pipeline
  (`load_to_postgres → clean_bronze_to_silver → create_gold → train_ml_models →
  survival_analysis → clv_modeling → uplift_modeling`) on a schedule, with retries
  and alerting.

* **Phase 10 — Dashboard polish.** Replace the deprecated `use_container_width`
  argument with `width='stretch'` throughout `app.py`.

* **Phase 11 — Deployment notes.** A Docker Compose setup for local
  reproduction, plus a `CONTRIBUTING.md` documenting the full pipeline run
  sequence.

The ML and analytics stack is production-quality. What remains is orchestration
and packaging.

---

## Key Takeaway

CreditPulse demonstrates a complete progression from raw CRM data to business decision-making:

```text
Raw CRM Data
     ↓
Data Engineering
     ↓
Lead Scoring
     ↓
Survival Analysis
     ↓
Customer Lifetime Value
     ↓
Causal Uplift Modeling
     ↓
Business Action
```

The important distinction is that each analytical layer answers a different business question.

**Classification** identifies likely converters.

**Survival analysis** estimates when they will convert.

**CLV** estimates their economic value.

**Uplift modeling** determines who should actually receive an intervention.

Together, these create a decision-oriented **Marketing / RevOps Analytics Engineering** portfolio project rather than a standalone machine-learning exercise.

---

## Tech Stack

### Data & Engineering

* Python 3.13
* PostgreSQL 18.6
* SQLAlchemy
* psycopg2-binary
* python-dotenv
* Pandas
* NumPy
* SciPy

### Machine Learning

* scikit-learn
* XGBoost
* lifelines

### Analytics & Visualization

* Streamlit
* Plotly

### Development

* Git
* GitHub
* draw.io

---

## Repository Structure

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
│   ├── clv_summary.json
│   └── uplift_summary.json
│
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── load_to_postgres.py
│   ├── clean_bronze_to_silver.py
│   ├── train_ml_models.py
│   ├── survival_analysis.py
│   ├── clv_modeling.py
│   ├── uplift_modeling.py
│   └── disk_space_check.py
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

## Acknowledgements

CreditPulse uses and builds upon:

* Faker
* NumPy
* Pandas
* scikit-learn
* XGBoost
* lifelines
* Streamlit
* Plotly

---

## Disclaimer

CreditPulse uses **fully synthetic data** generated for educational and portfolio purposes.

No real customer, financial, campaign, or company data is used.

The project is intended to demonstrate analytics engineering, machine learning, causal modeling, and business decision-making workflows.

---

**Last updated: September 2026**
