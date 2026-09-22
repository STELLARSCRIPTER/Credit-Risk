# CreditPulse — End-to-End Credit Risk & Customer Value Analytics Platform

> **A production-style analytics platform combining credit risk, customer value, survival analysis, and causal intervention modeling — built to demonstrate the bridge between Finance Analytics and Marketing / RevOps.**

**🔗 Live demo:** [creditpulse-analytics.streamlit.app](https://creditpulse-analytics.streamlit.app)

CreditPulse is an end-to-end analytics platform built on **synthetic customer and credit data**.

The project takes a traditional credit-risk problem and extends it into a broader customer analytics system:

**Who is likely to convert? → When will they convert? → How valuable are they? → Will an intervention actually change their behavior?**

The result is a complete analytical pipeline covering data engineering, machine learning, survival analysis, customer lifetime value, causal/uplift modeling, orchestration, and executive dashboards.

---

## Table of Contents

1. [Business Problem](#business-problem)
2. [Credit Risk → Marketing / RevOps Mapping](#credit-risk--marketing--revops-mapping)
3. [Synthetic Dataset](#synthetic-dataset)
4. [Architecture](#architecture)
5. [Data Model](#data-model)
6. [ETL & Data Quality](#etl--data-quality)
7. [ML Lead Scoring](#ml-lead-scoring)
8. [Survival Analysis](#survival-analysis)
9. [Customer Lifetime Value](#customer-lifetime-value)
10. [Uplift Modeling](#uplift-modeling)
11. [Pipeline Orchestration](#pipeline-orchestration)
12. [Dashboard](#dashboard)
13. [Infrastructure Notes](#infrastructure-notes)
14. [Setup & Running Locally](#setup--running-locally)
15. [Key Design Decisions](#key-design-decisions)
16. [Project Status](#project-status)

---

## Business Problem

Traditional credit-risk systems answer questions such as:

* Which customers are likely to default?
* What is their estimated risk?
* How long until an event occurs?

CreditPulse extends the same analytical thinking into customer and marketing analytics.

Instead of stopping at risk prediction, the platform asks:

1. **Which leads are likely to convert?**
2. **When are they likely to convert?**
3. **What is their expected customer lifetime value?**
4. **Which customers should receive an intervention?**
5. **Who should *not* receive the intervention because it may have little benefit or even a negative effect?**

This creates a bridge between:

**Finance Analytics → Customer Analytics → Marketing Analytics → RevOps Decisioning**

---

# Credit Risk → Marketing / RevOps Mapping

The project deliberately maps financial-risk concepts to customer and revenue problems.

| Finance / Credit Concept | CreditPulse Equivalent             |
| ------------------------ | ---------------------------------- |
| Default probability      | Conversion probability             |
| Risk score               | Lead score                         |
| Time-to-default          | Time-to-conversion                 |
| Survival probability     | Customer/lead survival probability |
| Expected loss            | Expected customer value            |
| Portfolio value          | Customer lifetime value            |
| Risk intervention        | Marketing intervention             |
| Treatment effect         | Campaign uplift                    |
| Risk segmentation        | Customer segmentation              |

This allows the project to demonstrate both **financial analytics reasoning** and **customer/revenue analytics engineering**.

---

# Synthetic Dataset

CreditPulse uses synthetic data so the complete pipeline can be reproduced without exposing real customer information.

The dataset contains:

* Customer / lead information
* Company attributes
* Industry
* Region
* Lead source
* Job title
* Customer activity
* Conversion outcomes
* Treatment assignment
* Campaign response
* Revenue-related variables
* Time-to-event information

The final dataset contains approximately **50K+ lead records**.

The synthetic dataset is intentionally designed to contain realistic relationships between:

* Customer behavior
* Conversion probability
* Time-to-conversion
* Customer value
* Intervention response

---

## On Scale and Generalizability

The dataset is deliberately sized at **~50,871 leads** (~250K activities, ~23K campaign touches, ~30K opportunities) — large enough for four production-grade models to produce statistically meaningful metrics (AUC 0.77, C-index 0.675, T-learner AUC ~0.70 per arm), but small enough that the entire medallion pipeline runs end-to-end on a laptop in under 5 minutes and can be publicly deployed on free-tier infrastructure.

Real enterprise CRM systems process millions of leads. The **methodology is identical at that scale** — the same medallion pattern (Bronze → Silver → Gold), the same quantile-based risk tiers, the same survival and uplift techniques. Scaling further would mean changing the *infrastructure layer* (a cloud data warehouse such as Fabric or Snowflake, orchestrated incremental loads, partitioned fact tables) rather than the analytical code itself. In my day job at Frost & Sullivan, I build medallion pipelines on Microsoft Fabric against much larger volumes — this project demonstrates the same methodology in a self-contained, reproducible form.

Why synthetic data specifically:

* Real CRM lead-to-conversion datasets with outcome labels are proprietary; no company publishes them.
* The generator mimics the statistical structure of a real B2B lead-scoring dataset — Referral leads convert at ~45%, Cold Call leads at ~9%, campaign responses carry realistic effect sizes, and irreducible noise is injected via a Gaussian term on the logit.
* The resulting AUC (~0.77) sits squarely in the range real production lead-scoring systems achieve, which is the point: the methodology transfers one-to-one, only the numbers would change.

---

# Architecture

CreditPulse follows a **Medallion Architecture**:

```text
                    ┌──────────────────────┐
                    │   Synthetic Data     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Bronze Layer       │
                    │   Raw PostgreSQL     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Silver Layer      │
                    │ Cleaning & Validation │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │     Gold Layer       │
                    │ Business Data Model  │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                ▼              ▼              ▼
         Lead Scoring    Survival Model      CLV
                │              │              │
                └──────────────┼──────────────┘
                               ▼
                       Uplift Modeling
                               │
                               ▼
                     Streamlit Dashboard
```

The complete workflow is orchestrated using **Prefect 3.x**.

---

# Data Model

The Gold layer provides analytics-ready tables for downstream modeling.

Major analytical outputs include:

```text
gold_customer
gold_company
gold_activity
gold_conversion
gold_campaign
ml_lead_scores
ml_survival_predictions
ml_clv_predictions
ml_uplift_predictions
```

The model separates:

* Customer attributes
* Company attributes
* Behavioral activity
* Conversion outcomes
* Campaign treatment
* Machine-learning predictions

This allows individual analytical components to be developed without coupling the entire system together.

---

# ETL & Data Quality

The ETL pipeline follows:

```text
Raw Data
   ↓
Bronze
   ↓
Silver
   ↓
Gold
```

### Bronze

Raw source data is loaded into PostgreSQL with minimal transformation.

### Silver

The cleaning layer handles:

* Missing values
* Type conversion
* Duplicate detection
* Invalid values
* Standardization
* Feature preparation

### Gold

The Gold layer creates business-ready analytical tables using SQL.

Data-quality checks are currently implemented directly in SQL and Python.

### Great Expectations

Formal Great Expectations validation is currently treated as **manual / optional** rather than a required production component.

The existing SQL and Python checks already validate the key assumptions needed by the downstream models.

---

# ML Lead Scoring

**Script:** `scripts/train_ml_models.py`

Phase 6 predicts the probability that a lead will convert.

Models evaluated:

* Logistic Regression
* Random Forest
* XGBoost

The final model uses **Random Forest**.

### Model Performance

**ROC-AUC: 0.7667**

The model output is converted into three operational lead tiers:

```text
Top 20%     → High Priority
Middle 60%  → Medium Priority
Bottom 20%  → Low Priority
```

This transforms a statistical probability into a practical CRM / RevOps workflow.

### Leakage Prevention

Features that would only become available after conversion were excluded from model training.

This ensures the model represents a realistic pre-conversion scoring scenario.

---

# Survival Analysis

**Script:** `scripts/survival_analysis.py`

Phase 7 moves beyond:

> "Will this lead convert?"

and asks:

> "When is this lead likely to convert?"

Two approaches are used:

### Kaplan-Meier

Used to visualize the overall survival probability over time.

### Cox Proportional Hazards

Used to estimate how customer attributes influence the rate of conversion.

Model performance:

**C-index: 0.6753**

The analysis produces:

* Survival probability
* Estimated median conversion time
* Hazard ratios
* Kaplan-Meier curves
* Time-to-event predictions

This adds a temporal dimension to the lead-scoring problem.

---

# Customer Lifetime Value

**Script:** `scripts/clv_modeling.py`

Phase 8 combines:

* Conversion probability
* Survival information
* Revenue
* Customer duration
* Discounting

to estimate expected customer lifetime value.

### CLV Formula

The conceptual calculation is:

```text
CLV =
Expected Revenue
× Survival Probability
× Conversion Probability
× Discount Factor
```

The implementation integrates survival probabilities over the expected customer lifetime.

### Portfolio Results

| Metric              |            Value |
| ------------------- | ---------------: |
| Total Portfolio CLV | **$170,444,069** |
| Median CLV          |       **$1,936** |
| Mean CLV            |       **$3,351** |
| P90 CLV             |       **$7,517** |
| Maximum CLV         |     **$131,689** |
| Discount Rate       |          **10%** |

Approximately **20,473 records** required imputed customer-lifetime information.

The output turns individual lead/customer predictions into a portfolio-level value estimate.

---

# Uplift Modeling

**Script:** `scripts/uplift_modeling.py`

Phase 9 moves from prediction into **intervention decisioning**.

Previous phases answer:

* **Phase 6:** Who will convert?
* **Phase 7:** When will they convert?
* **Phase 8:** How valuable are they?

Uplift modeling asks:

> **Will this lead convert because we intervene?**

This distinction is critical.

A normal A/B test estimates the **average treatment effect** across a population.

Uplift modeling estimates the **conditional treatment effect for individual customers**.

## Four Uplift Segments

Each customer is classified into one of four behavioral groups:

| Segment            | Control           | Treatment         | Marketing Action              |
| ------------------ | ----------------- | ----------------- | ----------------------------- |
| **Persuadable**    | Would not convert | Converts          | Target                        |
| **Sure Thing**     | Converts          | Converts          | Skip unnecessary intervention |
| **Lost Cause**     | Would not convert | Would not convert | Skip                          |
| **Do Not Disturb** | Converts          | Does not convert  | Suppress                      |

The objective is therefore not simply:

> "Who has the highest conversion probability?"

but:

> "Who has the highest incremental response to intervention?"

---

## T-Learner Architecture

CreditPulse uses a **T-learner**.

Two separate XGBoost classifiers are trained:

```text
                  Customer Features
                         │
              ┌──────────┴──────────┐
              ▼                     ▼
       Treatment Model       Control Model
         XGBoost               XGBoost
              │                     │
              ▼                     ▼
       P(convert | T)        P(convert | C)
              │                     │
              └──────────┬──────────┘
                         ▼
              Uplift = P(T) - P(C)
```

The core uplift calculation is:

```text
Uplift =
P(convert | treatment)
-
P(convert | control)
```

### Treatment / Control Population

The dataset contains approximately balanced treatment arms:

| Arm       | Records |
| --------- | ------: |
| Treatment |  11,552 |
| Control   |  11,337 |

### Features

Numeric:

* `total_activities`
* `company_size`

Categorical:

* `industry`
* `lead_source`
* `region`
* `job_title`

The following treatment variables are excluded from the feature set to avoid leakage:

* `treatment_group`
* `campaign_response`

---

## Model Performance

Treatment-arm model:

**AUC: 0.7020**

Control-arm model:

**AUC: 0.7024**

Observed conversion:

| Group     | Conversion |
| --------- | ---------: |
| Control   |     27.65% |
| Treatment |     33.00% |

Raw average uplift:

**+5.35 percentage points**

However, the individual uplift distribution is much more informative than the average treatment effect.

| Metric | Predicted Uplift |
| ------ | ---------------: |
| Mean   |           +4.82% |
| Median |           +4.37% |
| P90    |          +16.21% |
| P10    |           -5.62% |

---

## Uplift Segment Distribution

| Segment            | Records | Share | Mean Uplift |
| ------------------ | ------: | ----: | ----------: |
| **Persuadable**    |  10,730 | 46.9% |     +12.31% |
| **Lost Cause**     |   5,191 | 22.7% |      +0.67% |
| **Sure Thing**     |   4,356 | 19.0% |      +0.43% |
| **Do Not Disturb** |   2,612 | 11.4% |     -10.37% |

The important business insight is that the average treatment effect hides substantial customer-level variation.

Almost half of the population is classified as **Persuadable**, while more than **2,600 customers** fall into the **Do Not Disturb** segment.

The latter group has negative predicted treatment effects.

For example, some high-baseline segments such as Referral / SaaS customers already have very high conversion rates without intervention, while campaign exposure can reduce their predicted response.

This creates a practical marketing action:

```text
High positive uplift
        ↓
Target

Low / zero uplift
        ↓
Avoid unnecessary spend

Negative uplift
        ↓
Suppress intervention
```

---

## Qini Curve

The project also evaluates cumulative incremental conversions through a **Qini-style uplift curve**.

The curve reaches a peak of approximately:

**+1,200 incremental conversions**

around the **60% population mark**.

This provides a way to evaluate whether ranking customers by predicted uplift produces more incremental conversions than indiscriminate targeting.

### Outputs

The model writes:

```text
ml_uplift_predictions
models/uplift_summary.json
```

---

# Pipeline Orchestration

**Flow:** `orchestration/pipeline_flow.py` — Phase 5
**Framework:** **Prefect 3.x** — Python-native workflow orchestration

Prefect provides flow/task orchestration, retries, timeouts, dependency tracking, scheduling, and run-level observability for the CreditPulse pipeline.

## Why Prefect, not Airflow

Airflow was not selected for the local Windows implementation because it would require an additional Linux-compatible environment such as WSL2 or Docker.

Prefect is Python-native and runs directly in the existing Python environment, while still providing the workflow concepts required here: tasks, dependencies, retries, timeouts, scheduling, and UI-based monitoring.

The underlying pipeline logic remains portable because each stage is already implemented as an independent Python/SQL operation.

---

## What the Flow Does

The complete CreditPulse pipeline runs in dependency order:

```text
load_bronze
    ↓
clean_to_silver
    ↓
build_gold
    ↓
create_survival_view
    ↓
train_ml_models
    ↓
survival_analysis
    ↓
clv_modeling
    ↓
uplift_modeling
```

The flow is defined in:

```text
orchestration/pipeline_flow.py
```

and exposed as:

```text
creditpulse_pipeline
```

Prefect automatically tracks task state and dependencies, allowing individual tasks to be monitored independently.

---

## Production-Grade Features

| Feature                  | Implementation                                                     |
| ------------------------ | ------------------------------------------------------------------ |
| **Retry policy**         | `@task(retries=2, retry_delay_seconds=30)` on data tasks           |
| **Task timeouts**        | `timeout_seconds=600` on every task                                |
| **VenV-safe subprocess** | Explicitly uses `.venv/Scripts/python.exe`                         |
| **Structured logging**   | Script stdout/stderr streamed to the Prefect UI with `│` prefix    |
| **Failure isolation**    | A failed task stops the pipeline rather than continuing downstream |
| **Run history**          | Flow runs, task states, logs, and execution graphs are retained    |

Prefect supports task-level retries and timeout configuration directly through task decorators.

---

## Verified Run

**Flow run: `industrious-yak`**

**Status: Completed**

**Total runtime: 5 minutes 11 seconds**

All 8 tasks passed.

| Task                   | Duration |
| ---------------------- | -------: |
| `load_bronze`          |  ~45 sec |
| `clean_to_silver`      |  ~50 sec |
| `build_gold`           |   ~5 sec |
| `create_survival_view` |   ~3 sec |
| `train_ml_models`      |  ~75 sec |
| `survival_analysis`    |  ~60 sec |
| `clv_modeling`         |  ~10 sec |
| `uplift_modeling`      |  ~25 sec |

The Prefect UI provides the corresponding flow graph, task states, execution timings, and logs.

---

## Running the Flow

### Manually

```bash
python orchestration/pipeline_flow.py
```

### With the Prefect UI

Start the local Prefect server:

```bash
prefect server start
```

Then, in a second terminal:

```bash
python orchestration/pipeline_flow.py
```

Open the local Prefect UI at:

```text
http://127.0.0.1:4200
```

The UI provides visibility into flow runs, task states, execution logs, and the dependency graph.

### Optional Scheduled Deployment

A nightly deployment can be configured with:

```bash
prefect deployment build orchestration/pipeline_flow.py:creditpulse_pipeline \
    --name "nightly" --cron "0 6 * * *" --apply

prefect worker start --pool "default-agent-pool"
```

This allows the same pipeline to move from manual execution toward scheduled refreshes.

---

## Design Note — The `survival_analysis` Hang

The first two complete flow runs stalled at:

```text
survival_analysis
```

Prefect's task-level observability made the bottleneck immediately visible.

### Root Cause

The original implementation used:

```python
cph.predict_median()
```

for approximately **50,858 leads**.

The calculation evaluated survival information independently across leads and took more than **20 minutes**.

### Optimization

The implementation was rewritten to:

1. Generate the survival function once across all leads.
2. Use a fixed time grid.
3. Derive individual median survival times using a NumPy mask.

This reduced the runtime to approximately:

**~30 seconds**

representing roughly a:

**40× speedup**

The issue demonstrated one of the practical benefits of orchestration: the pipeline did not simply "run slowly"; task-level observability identified exactly which stage required investigation.

---

# Dashboard

CreditPulse includes a Streamlit dashboard providing both executive and analytical views.

**🔗 Live demo:** [creditpulse-analytics.streamlit.app](https://creditpulse-analytics.streamlit.app)

### Dashboard Pages

* **Overview**
* **Customer Analytics**
* **Risk Analysis**
* **Interventions**
* **ML Models**
* **Survival Analysis**
* **CLV**
* **Data & ETL**
* **Executive View**
* **Settings**

The dashboard combines:

* Lead scoring
* Conversion probability
* Survival predictions
* CLV
* Uplift segments
* Customer characteristics
* Data-pipeline information

The goal is to convert model outputs into business-readable decision support rather than exposing raw model objects alone.

### Deployment

The dashboard is deployed on **Streamlit Community Cloud**, with the Gold-layer analytical tables hosted on **Supabase Postgres**. The app reads the connection string from Streamlit's secrets system via a `get_db_url()` helper that falls back to a local `.env` file for offline development.

---

# Infrastructure Notes

## PostgreSQL

CreditPulse uses PostgreSQL locally on Windows.

The project initially encountered a storage issue because PostgreSQL data was consuming space on the C: drive.

At one point the available disk space dropped to approximately:

```text
0.42 GB
```

This resulted in:

```text
No space left on device
```

The PostgreSQL data directory was subsequently moved to:

```text
D:\pgdata
```

The PostgreSQL installation was rebuilt through EDB with the new data location.

A disk-space guard was also added to prevent the same failure from silently affecting future pipeline runs.

## Cloud Deployment

Two infrastructure issues surfaced during the Streamlit Cloud deployment:

1. **Python version.** Streamlit Cloud initially defaulted to Python 3.14, which has no prebuilt wheels for `psycopg2-binary` or `pandas` — pip fell back to compiling from source and failed. Fixed by setting the app's Python version to **3.11** in the Streamlit Cloud dashboard.
2. **Dependency surface.** The full `requirements.txt` (with `prefect`, `xgboost`, `lifelines`, `scikit-learn`) adds several minutes to the build. A minimal `requirements-cloud.txt` was introduced that contains only the runtime-serving dependencies, and is used as the deployed `requirements.txt`. The full list is preserved as `requirements-full.txt` for local development.

---

# Setup & Running Locally

## 1. Clone the Repository

```bash
git clone <repository-url>
cd CreditPulse
```

## 2. Create Virtual Environment

```bash
python -m venv .venv
```

Activate it:

### Windows

```bash
.venv\Scripts\activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure PostgreSQL

Create the required PostgreSQL database and update the project's database configuration.

## 5. Generate / Load Data

Run the relevant data-generation and loading scripts.

## 6. Execute the Pipeline

The individual stages can be executed manually:

```text
Bronze
  ↓
Silver
  ↓
Gold
  ↓
ML
  ↓
Survival
  ↓
CLV
  ↓
Uplift
```

or through the Prefect orchestration flow:

```bash
python orchestration/pipeline_flow.py
```

## 7. Launch Dashboard

Run:

```bash
streamlit run app.py
```

---

# Key Design Decisions

## Why Synthetic Data?

The project is designed as a public portfolio project.

Synthetic data allows the complete architecture and modeling process to be demonstrated without exposing confidential customer information.

---

## Why This Dataset Size?

The dataset is intentionally sized at ~50K leads. Large enough for four ML models to produce honest, non-trivial metrics; small enough to run end-to-end on a laptop in under 5 minutes and be publicly deployed on free-tier infrastructure. The methodology is identical at enterprise scale — only the infrastructure layer changes.

---

## Why PostgreSQL?

PostgreSQL provides a realistic relational database environment for:

* ETL
* SQL transformations
* Analytical tables
* Feature preparation
* Model outputs

---

## Why Medallion Architecture?

The Bronze → Silver → Gold structure separates:

* Raw ingestion
* Cleaning / validation
* Business-ready analytical data

This makes downstream modeling easier to maintain and debug.

---

## Why Multiple ML Techniques?

Each model answers a different business question:

```text
Lead Scoring
     ↓
Who is likely to convert?

Survival Analysis
     ↓
When are they likely to convert?

CLV
     ↓
How valuable are they?

Uplift Modeling
     ↓
Will intervention change the outcome?
```

Together they create a more complete decision system than a single predictive model.

---

## Why Uplift Modeling?

A customer with a high probability of conversion is not necessarily a customer who needs marketing intervention.

For example:

```text
Customer A
P(convert without campaign) = 80%
P(convert with campaign)    = 82%

Uplift = +2%
```

versus:

```text
Customer B
P(convert without campaign) = 30%
P(convert with campaign)    = 55%

Uplift = +25%
```

Customer B may provide substantially more incremental value from intervention even though Customer A has the higher absolute conversion probability.

This distinction is central to the project's Marketing / RevOps positioning.

---

# Project Status

| Phase                               | Status                                                      |
| ----------------------------------- | ----------------------------------------------------------- |
| 1. Setup & Synthetic Data           | ✅ Complete                                                  |
| 2. ETL Pipeline                     | ✅ Complete                                                  |
| 3. Bronze → Silver → Gold           | ✅ Complete                                                  |
| 4. Great Expectations               | 🔶 Manual only                                              |
| **5. Pipeline Orchestration**       | ✅ **Complete — Prefect flow, 8 tasks, 5m 11s verified run** |
| 6. ML Lead Scoring                  | ✅ Complete                                                  |
| 7. Survival Analysis                | ✅ Complete                                                  |
| 8. Customer Lifetime Value          | ✅ Complete                                                  |
| 9. Intervention Experiment / Uplift | ✅ Complete                                                  |
| 10. Dashboard Polish                | 🔶 Cosmetic only                                            |
| 11. Deployment                      | ✅ **Live on Streamlit Community Cloud**                    |

---

# What's Next

Every functional phase is complete, and the app is publicly deployed.

Remaining items are cosmetic or optional:

* **Phase 4 — Great Expectations.** Formalize the ad-hoc data-quality checks currently done in SQL into a declarative expectations suite. Nice-to-have; the current checks already demonstrate the concept.

* **Phase 10 — Dashboard polish.** Replace the deprecated `use_container_width` argument with `width='stretch'` throughout `app.py`. Non-breaking, just removes console warnings.

* **Optional extensions** — a dbt-based transformation layer, GitHub Actions CI, or a second portfolio project using a real-data source such as Lending Club or Home Credit.

---

# Key Takeaway

CreditPulse is designed to demonstrate an end-to-end analytical workflow:

```text
Synthetic Customer Data
          ↓
      PostgreSQL
          ↓
Bronze → Silver → Gold
          ↓
   Lead Scoring
          ↓
 Survival Analysis
          ↓
         CLV
          ↓
  Uplift Modeling
          ↓
  Marketing Decisioning
          ↓
      Dashboard
```

The project therefore demonstrates more than predictive modeling.

It combines:

* **SQL**
* **PostgreSQL**
* **Python**
* **ETL**
* **Data Modeling**
* **Machine Learning**
* **Survival Analysis**
* **Customer Lifetime Value**
* **Causal / Uplift Modeling**
* **Workflow Orchestration**
* **Streamlit**
* **Business Decisioning**

The central idea is:

> **Predict the customer. Understand the timing. Quantify the value. Measure the intervention.**

---

# Tech Stack

| Layer             | Technology              |
| ----------------- | ----------------------- |
| Language          | Python                  |
| Database          | PostgreSQL              |
| ETL               | Python + SQL            |
| Data Architecture | Bronze / Silver / Gold  |
| ML                | Scikit-learn + XGBoost  |
| Survival Analysis | Lifelines               |
| CLV               | Python / NumPy / Pandas |
| Uplift Modeling   | XGBoost                 |
| Orchestration     | Prefect 3.x             |
| Dashboard         | Streamlit               |
| Visualization     | Plotly                  |
| Data Quality      | SQL / Python            |
| Version Control   | Git / GitHub            |
| Deployment        | Streamlit Community Cloud + Supabase Postgres |

---

# Repository Structure

```text
CreditPulse/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── models/
│   ├── lead_scoring/
│   ├── survival/
│   ├── clv/
│   └── uplift/
│
├── orchestration/
│   └── pipeline_flow.py
│
├── scripts/
│   ├── generate_data.py
│   ├── load_bronze.py
│   ├── clean_to_silver.py
│   ├── create_gold_tables.py
│   ├── train_ml_models.py
│   ├── survival_analysis.py
│   ├── clv_modeling.py
│   └── uplift_modeling.py
│
├── sql/
│   ├── bronze/
│   ├── silver/
│   └── gold/
│
├── app.py
├── requirements.txt
├── requirements-full.txt
├── requirements-cloud.txt
└── README.md
```

---

# Acknowledgements

CreditPulse was built as an independent portfolio project to demonstrate the application of:

* Financial analytics concepts
* Marketing analytics
* Revenue operations
* Customer analytics
* Machine learning
* Data engineering
* Causal inference

The project intentionally uses synthetic data and does not represent the financial performance or credit decisions of any real organization.

---

# Disclaimer

**CreditPulse is an educational and portfolio project.**

All customer, financial, behavioral, and campaign data used in the project is synthetic.

The model outputs should not be used for real-world credit decisions, customer targeting, financial decisions, or production marketing campaigns without appropriate validation, governance, fairness assessment, and domain review.

---

**Last Updated:** September 2026