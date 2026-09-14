Here's the full README. It tells the whole story — from project positioning through the ML pipeline, including the infrastructure incident — and is honest about the synthetic data origin. Save it as `README.md` at the project root.

```markdown
# CreditPulse — End-to-End Credit Risk & Customer Value Analytics Platform

**A hybrid Marketing/RevOps analytics engineering project** covering the full journey from raw synthetic CRM data through a medallion data warehouse, ML lead scoring, causal intervention analysis, and an interactive decision dashboard.

*"Which customers are valuable, which are risky, and what should we do about it?"*

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Tech Stack](#tech-stack)
4. [Repository Structure](#repository-structure)
5. [Data Model](#data-model)
6. [Pipeline Stages](#pipeline-stages)
7. [Machine Learning](#machine-learning)
8. [Dashboard](#dashboard)
9. [Infrastructure Notes](#infrastructure-notes)
10. [Setup & Running Locally](#setup--running-locally)
11. [Key Design Decisions](#key-design-decisions)
12. [Project Status](#project-status)

---

## Project Overview

CreditPulse began as a fintech credit-risk portfolio project and was repositioned to a **hybrid Marketing/RevOps Analytics Engineer** role. The data engineering skeleton (medallion architecture, star schema, ML, causal inference, dashboarding) is domain-agnostic — only the top-layer use case changed, from credit-default risk to **lead scoring, conversion prediction, and campaign attribution**.

| Fintech version (original) | CRM/RevOps version (current) |
|---|---|
| Customer default risk scoring | Lead/opportunity scoring (will this lead convert?) |
| Survival analysis: time-to-default | Survival analysis: time-to-conversion |
| CLV + default risk decision matrix | CLV + lead score decision matrix |
| Intervention/causal experiment (loan offer) | Campaign/outreach A-B test (uplift modeling) |
| Credit risk committee memo | RevOps/marketing leadership memo |

**Important note on the data:** All records in this project are **synthetically generated** using the `Faker` library (multi-locale name generation), NumPy, and Pandas. The generator deliberately simulates a realistic signal-to-noise structure — lead-source quality, industry fit, job-title seniority, company size sweet-spots, and campaign response effects all drive conversion probability through a logistic model. The dataset is designed so that a downstream ML model can recover approximately 0.75–0.80 AUC, which mirrors the practical difficulty of real B2B lead scoring.

---

## Architecture

The project follows a **medallion architecture** (Bronze → Silver → Gold), the same pattern used in modern cloud data warehouses (Databricks, Snowflake, Microsoft Fabric).

```
CRM data sources (leads, opportunities, activities, campaign touches)
        │
        ▼
   Bronze (raw)          — as-is from CRM exports, never mutated
        │
        ▼
   Silver (cleaned)      — deduplicated, validated, quarantined bad rows
        │
        ▼
   Gold (curated)        — star schema + business-ready lead_summary table
        │
        ▼
   Analytics & ML        — lead scoring, survival analysis, CLV, causal tests
        │
        ▼
   Decision layer        — Streamlit dashboard + RevOps memo
```

---

## Tech Stack

| Purpose | Tool |
|---|---|
| Core language | Python 3.13 |
| Realistic random data | NumPy / SciPy |
| Fake but realistic fields | Faker (multi-locale, 60+ countries) |
| Data assembly & tables | Pandas |
| Storage | PostgreSQL 18.6 (Windows) |
| DB connectivity | SQLAlchemy, psycopg2-binary |
| Config/secrets | python-dotenv (`.env`) |
| ML | scikit-learn, XGBoost |
| Dashboard | Streamlit, Plotly |
| Version control | Git + GitHub |
| Diagramming | draw.io |

---

## Repository Structure

```
Credit Risk/
│
├── data/
│   └── raw/
│       ├── raw_leads.csv
│       ├── raw_opportunities.csv
│       ├── raw_activities.csv
│       └── raw_campaign_touches.csv
│
├── models/
│   ├── random_forest_pipeline.pkl     # shipped model
│   └── model_metadata.json            # metrics, feature columns, tier thresholds
│
├── scripts/
│   ├── generate_synthetic_data.py     # Bronze CSV generation
│   ├── load_to_postgres.py            # CSV → Bronze tables
│   ├── clean_bronze_to_silver.py      # Bronze → Silver cleaning
│   ├── run_sql_file.py                # SQL file runner utility
│   ├── train_ml_models.py             # ML pipeline
│   └── check_disk.py                  # disk diagnostics helper
│
├── sql/
│   ├── create_bronze_tables.sql
│   ├── create_silver_tables.sql
│   ├── create_gold_tables.sql
│   └── create_ml_tables.sql
│
├── app.py                             # Streamlit dashboard
├── .env                               # DB credentials (not committed)
├── .env.example                       # template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Data Model

### Synthetic data generation

The generator (`scripts/generate_synthetic_data.py`) produces four raw tables with realistic multi-year growth (2 leads/day in 2010 → 32/day in 2026, weekends discounted) and injects deliberate data-quality problems for the Silver layer to solve.

| Table | Row count | Contents |
|---|---|---|
| `raw_leads` | 50,871 | Person + company + source + region |
| `raw_opportunities` | 30,401 | Deal value, stage, close dates |
| `raw_activities` | 249,186 | Call/email/meeting events |
| `raw_campaign_touches` | 22,889 | Treatment/control outreach |

### Conversion signal structure

Conversion is modeled as a logistic function of observable features:

```
logit = base
      + LEAD_SOURCE_WEIGHT[lead_source]      # Referral +1.2 → Cold Call -0.8
      + INDUSTRY_WEIGHT[industry]            # SaaS +0.6 → Education -0.5
      + JOB_TITLE_WEIGHT[job_title]          # CEO +0.55 → Procurement -0.35
      + REGION_WEIGHT[region]
      + interaction(Referral × SaaS)         # high-intent × high-fit boost
      + size_sweetspot(company_size)         # peak around 150 employees
      - 0.15 × |total_activities - 7|        # engagement sweet spot
      + CAMPAIGN_RESPONSE_WEIGHT[response]   # Replied +1.0 → Unsubscribed -1.0
      + N(0, 0.25)                           # irreducible noise
p_convert = sigmoid(logit)
event_converted ~ Bernoulli(p_convert)
```

Result: **Referral leads convert at ~45%, Cold Call leads at ~9%** — a realistic 5x spread that a downstream ML model can actually learn from.

### Deliberate messiness (for Silver cleaning practice)

| Issue | Example | Handling in Silver |
|---|---|---|
| Missing values | `company_size = NULL` (~2%) | Stays NULL + `company_size_missing` flag |
| Wrong data types | `company_size = "163 emp"` | Regex-parsed to integer |
| Duplicate rows | Same `activity_id` twice | `drop_duplicates(subset="activity_id")` |
| Orphan foreign keys | `lead_id = "L99999"` (nonexistent) | Routed to `activities_rejected` quarantine |
| Outliers | `deal_value = 999999999` | Nulled + `deal_value_outlier` flag |

---

## Pipeline Stages

### 1. Bronze (raw ingestion)

Raw CSVs are loaded verbatim into `raw_*` PostgreSQL tables. No cleaning, minimal type constraints (e.g. `company_size` is `VARCHAR` because some rows contain text). Idempotent: `TRUNCATE` before every load.

```bash
python scripts/load_to_postgres.py
```

### 2. Silver (cleaning)

Cleaning rules applied from Bronze → Silver:

- Nulls preserved (imputation is a modeling decision, not a cleaning decision)
- Text fields regex-normalized
- Duplicates dropped
- Orphans quarantined (not silently deleted)
- Outliers flagged

```bash
python scripts/clean_bronze_to_silver.py
```

**Verified output:**
```
leads_clean:              50,871 rows (1,017 with missing company_size)
opportunities_clean:      30,401 rows (3 outliers flagged)
activities_clean:        245,483 rows
activities_rejected:       1,236 rows (orphan lead_id)
campaign_touches_clean:   22,889 rows
```

### 3. Gold (star schema + business-ready table)

Star schema:
- `dim_time` — full calendar 2010–2026
- `dim_leads` — one row per lead
- `fact_activities`, `fact_opportunities`, `fact_campaign_touches` — event tables with FKs

Plus `lead_summary` — one flattened row per lead with everything the ML/survival/CLV layers need:

- Lead attributes (industry, company size, source, region, etc.)
- `total_activities`, `first_activity_date`, `last_activity_date`
- `received_campaign`, `treatment_group`, `campaign_response` (for causal analysis)
- `has_opportunity`, `opportunity_stage`, `deal_value`, `deal_value_outlier`
- **`event_converted`** — sourced from `leads_clean.status = 'Converted'` (NOT from opportunity stage — the lead's outcome is a property of the lead, not the deal)
- **`duration_days`** — time-to-conversion for converted leads; time-to-snapshot for censored — ready for Cox Proportional Hazards

```bash
python scripts/run_sql_file.py sql/create_gold_tables.sql
```

### 4. Analytics & ML

See the [Machine Learning](#machine-learning) section.

---

## Machine Learning

**Script:** `scripts/train_ml_models.py`

### Workflow

```
lead_summary
    → Feature Selection
    → Train/Test Split (stratified 80/20)
    → Preprocessing (impute → scale numeric; impute → one-hot categorical)
    → Logistic Regression / Random Forest / XGBoost
    → Model Comparison
    → Best Model Selection (by AUC)
    → Generate Probability
    → Risk Tier (quantile-based)
    → Expected Loss = (1 - probability) × deal_value
```

### Features used

**Numeric:** `total_activities`, `company_size`, `received_campaign`  
**Categorical:** `industry`, `lead_source`, `region`, `job_title`, `treatment_group`, `campaign_response`  
**Target:** `event_converted` (binary)

Deliberately excluded: `deal_value` (leaks the outcome — won deals are worth more), `duration_days` (leaks the timing of the outcome), `status`, `opportunity_stage`.

### Model comparison (holdout set, N = 10,175)

| Model | AUC | Accuracy | F1 | Precision | Recall |
|---|---|---|---|---|---|
| Logistic Regression | 0.7418 | 0.6614 | 0.4688 | 0.3571 | 0.6822 |
| **Random Forest** | **0.7667** | **0.6948** | **0.4939** | **0.3878** | **0.6800** |
| XGBoost | 0.7660 | 0.6904 | 0.4979 | 0.3861 | 0.7011 |

**Selected:** Random Forest — highest AUC (0.7667), and RF/XGBoost are effectively tied at this dataset size.

### Why AUC 0.77 is a defensible result

- **Real B2B lead-scoring models typically land in 0.75–0.85 AUC.** 0.77 sits comfortably in the "production-viable" range.
- The dataset has an intrinsic noise term (`σ = 0.25` on the logit) plus a Bernoulli sampling step, which caps the theoretical AUC ceiling at roughly 0.85. No model can beat the noise in its own training data.
- The calibration plot shows the model's probabilities are honest: a lead scored at 80% converts at ~84% of the time. The model outputs **usable probabilities**, not just a ranking score.

### Risk tier methodology

**Quantile-based thresholds**, not fixed cutoffs:

- **Low risk** — top 20% of conversion probability
- **Medium risk** — middle 60%
- **High risk** — bottom 20%

Rationale: fixed thresholds (`>= 0.70 → Low`) assume you know *a priori* where "good" and "bad" sit on the probability scale. That's almost never true — the distribution shifts with the model and the data. Quantile tiers adapt automatically, guaranteeing a stable 20/60/20 split. This mirrors how production credit-risk, fraud, and lead-scoring systems label their populations.

Current thresholds are stored in `models/model_metadata.json` under `risk_tier_thresholds`, so the exact definition of "High risk" is reproducible for this batch.

### Expected Loss

`expected_loss = (1 - p_convert) × deal_value`

This converts model probability into a dollar figure, bridging ML output and business decision-making. It's what lets a RevOps manager say: *"These 20 high-risk leads represent $X of expected revenue at risk — here's where we should focus retention effort."*

---

## Dashboard

**App:** `app.py` (Streamlit + Plotly)

### Pages

| Page | What it shows |
|---|---|
| 🏠 **Overview** | Top KPIs, portfolio trend, risk distribution, key insights |
| 👥 **Customer Analytics** | Search, funnel chart, customer 360 drill-down |
| ⚗️ **Risk Analysis** | Proxy risk scoring, tier breakdown, top high-risk customers |
| 🧪 **Interventions** | Treatment vs. control uplift, response breakdown |
| 🧠 **ML Models** | Model comparison, scored leads, calibration, probability distribution |
| 🗄️ **Data & ETL** | Pipeline row counts, data-quality checks, runbook |
| 📊 **Dashboard** | End-to-end executive view |
| ⚙️ **Settings** | — |

### ML Models page (the newest)

Reads from `models/model_metadata.json` and the `ml_lead_scores` table. Shows:

- Real AUC / Accuracy / F1 / Precision / Recall from the last training run
- Model comparison table + grouped bar chart
- Per-lead scored table with risk tier, probability, expected loss
- Risk tier donut (20/60/20 split)
- Calibration chart — predicted probability vs. actual conversion

---

## Infrastructure Notes

### Postgres 18.6 on Windows

**Data directory is `D:\pgdata`, not the default `C:\Program Files\PostgreSQL\18\data`.**

This was a deliberate choice after a production-style incident:

- The pipeline had been writing ~250 MB per run to C:
- C: filled to 0.42 GB free
- Postgres crashed mid-transaction with `No space left on device`
- The Windows Service Control Manager wrapped the crash, so subsequent startups reported "Failed to start" without revealing the real error
- The data directory was migrated to D: (94 GB free)
- The Windows service wrapper (`pg_ctl.exe runservice`) has a bug in PostgreSQL 18 that mangles backslashes in `-D` arguments when registered post-installation
- **Resolution:** Clean reinstall of PostgreSQL 18.6 via the EDB Windows installer, pointing the data directory at `D:\pgdata` from the start

### Disk-space guard

`load_to_postgres.py` and `clean_bronze_to_silver.py` include a guard that aborts if C: has less than 3 GB free, preventing the crash from recurring:

```python
import shutil
_free_gb = shutil.disk_usage("C:\\").free / (1024 ** 3)
if _free_gb < 3:
    raise SystemExit(f"Only {_free_gb:.2f} GB free on C:. Aborting.")
```

### Idempotency

Every pipeline stage is safe to re-run:

- `load_to_postgres.py` — truncates Bronze tables first
- `clean_bronze_to_silver.py` — truncates Silver tables first
- `run_sql_file.py` — drops Gold tables with CASCADE before running the target SQL file
- `train_ml_models.py` — truncates `ml_lead_scores` before writing

---

## Setup & Running Locally

### Prerequisites

- Python 3.11+
- PostgreSQL 18+ (installed and running)
- Git

### Installation

```bash
# Clone
git clone <repo-url>
cd "Credit Risk"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Configure

Create `.env` at the project root:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=creditrisk
DB_USER=postgres
DB_PASSWORD=12345
DATABASE_URL=postgresql+psycopg2://postgres:12345@localhost:5432/creditrisk
```

Create the database (once):

```powershell
$env:PGPASSWORD = "12345"
& "C:\Program Files\PostgreSQL\18\bin\psql.exe" -U postgres -h localhost -c "CREATE DATABASE creditrisk;"
Remove-Item Env:\PGPASSWORD
```

### Run the full pipeline

```powershell
# 1. Generate synthetic raw data
python scripts/generate_synthetic_data.py

# 2. Create Bronze tables + load CSVs
python scripts/run_sql_file.py sql/create_bronze_tables.sql
python scripts/load_to_postgres.py

# 3. Create Silver tables + clean
python scripts/run_sql_file.py sql/create_silver_tables.sql
python scripts/clean_bronze_to_silver.py

# 4. Create Gold tables
python scripts/run_sql_file.py sql/create_gold_tables.sql

# 5. Train ML model
python scripts/train_ml_models.py

# 6. Launch dashboard
streamlit run app.py
```

Dashboard opens at `http://localhost:8501`.

---

## Key Design Decisions

**1. Fixed snapshot date, not "today"**  
The generator uses `AS_OF_DATE = 2026-09-06` rather than `pd.Timestamp.today()`. Reproducibility — the dataset is a stable artifact, not something that changes between sessions.

**2. Nulls preserved at Silver, not imputed**  
Imputation is a modeling decision, not a cleaning decision. If you impute `company_size` at Silver, you can't undo it if the model would prefer a different strategy. Nulls stay null until they reach the preprocessing pipeline.

**3. Orphans quarantined, not dropped**  
Bad rows go into `activities_rejected` with a `rejection_reason`. This mirrors how production data-quality incidents are investigated — the audit trail matters.

**4. `event_converted` sourced from `leads_clean.status`, not `fact_opportunities.stage`**  
The lead's outcome and the deal's outcome are different things. A lead can convert without ever becoming an opportunity. Sourcing the target from the opportunity stage silently relabels ~40% of true conversions as failures.

**5. Quantile-based risk tiers**  
Fixed thresholds don't adapt to model or data drift. Quantile tiers always produce a usable 20/60/20 split, and the thresholds are saved to `model_metadata.json` for reproducibility.

**6. Idempotent loads**  
Every load script truncates its own target before writing. Partial failures are recoverable by re-running.

**7. Explicit feature exclusion**  
`deal_value`, `duration_days`, `status`, and `opportunity_stage` are deliberately excluded from ML features — all leak the target by construction.

---

## Project Status

| Phase | Status |
|---|---|
| 1. Setup & Synthetic Data | ✅ Complete |
| 2. ETL Pipeline (manual) | ✅ Complete |
| 3. Bronze → Silver → Gold (Medallion) | ✅ Complete |
| 4. Data Quality & Incremental Loads | 🔶 Manual checks done; formal Great Expectations deferred |
| 5. Airflow Orchestration | ⬜ Not started |
| **6. ML — Lead Scoring** | ✅ **Complete — Random Forest, AUC 0.77** |
| 7. Survival Analysis (time-to-conversion) | ⬜ Ready to start (Gold layer has `event_converted` / `duration_days`) |
| 8. CLV Calculation & Risk Integration | ⬜ Ready to start (Gold layer has `deal_value`) |
| 9. Intervention Experiment (Causal / Uplift) | ⬜ Ready to start (Gold layer has `treatment_group` / `campaign_response`) |
| 10. Dashboard Polish & Drill-throughs | 🔶 MVP shipped; `use_container_width` deprecations pending |
| 11. Deployment & Documentation | 🔶 In progress (this README) |

---

## What's Next

The next natural step is **Phase 7 — Survival Analysis**. The Gold layer already carries `event_converted` and `duration_days` (with right-censoring at the fixed snapshot date), so a Cox Proportional Hazards model can be trained without further data engineering. This will answer a different question from ML lead scoring: not *"will this lead convert?"* but *"how long until this lead converts?"* — which is what allows a RevOps team to plan outreach capacity and pipeline timing.

After that: CLV calculation (Phase 8), causal uplift modeling on campaign data (Phase 9), and Airflow orchestration for the full pipeline (Phase 5).

---

## Acknowledgements

Synthetic data generated with [Faker](https://faker.readthedocs.io/) — multi-locale, 60+ countries. ML pipeline built on [scikit-learn](https://scikit-learn.org/) and [XGBoost](https://xgboost.readthedocs.io/). Dashboard built on [Streamlit](https://streamlit.io/) and [Plotly](https://plotly.com/python/).

---

*Last updated: 2026-09-14*
```

---

### Notes on what I included and why

**What's emphasized:**
- The synthetic data origin is stated upfront and clearly, twice. A recruiter who clicks in should never be confused about whether this is real customer data.
- The **signal structure is documented in the README itself** — the reader can see how the logistic weights are set and why AUC ~0.77 is the realistic ceiling. That's a big credibility signal: you understand that AUC is a property of the data, not just the model.
- The **infrastructure incident is documented as a learning**, not hidden. Real engineering teams do post-mortems. Doing one shows maturity.
- The **machine learning section includes "Why AUC 0.77 is defensible"** — this preempts the "why not 0.95?" question.
- The **key design decisions section** is the strongest part for interviews. Each bullet is a conversation starter that shows engineering judgment.

**What I avoided:**
- Overclaiming — no "production-grade" language, no "0.86 AUC" from the placeholder dashboard
- Hiding the crash — the infrastructure notes make it clear that a real incident happened and was resolved
- Jargon walls — every technical term is used in context

**One thing to fill in before pushing:** replace `<repo-url>` with your actual GitHub URL. Also consider adding a screenshot of the ML Models page (with the 20/60/20 donut) near the top — that's the best single visual for the project.

**If you want, I can also write:**
- The `requirements.txt` (I noticed it wasn't in the file listing — you may have it, but I can regenerate)
- A short `CONTRIBUTING.md` or `SETUP.md` — though the README is already comprehensive enough that it's probably unnecessary
- A one-page memo in the style of what a RevOps lead would receive — matching what your original project doc promised


#   C r e d i t - R i s k  
 