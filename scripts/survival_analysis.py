# scripts/survival_analysis.py
"""
CreditPulse: Survival Analysis
Answers the question: NOT just "will this lead convert?" but
"WHEN will it convert, and how does timing differ by segment?"

Uses:
  - Kaplan-Meier estimator (non-parametric survival curves)
  - Cox Proportional Hazards (semi-parametric model with hazard ratios)

Outputs:
  - models/survival_km_by_segment.json
  - models/survival_cox_summary.json
  - postgres: ml_survival_predictions
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from lifelines import KaplanMeierFitter, CoxPHFitter

warnings.filterwarnings("ignore")

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL not found in .env")

COX_NUMERIC = ["company_size", "total_activities", "received_campaign"]
COX_CATEGORICAL = ["lead_source", "industry", "region"]
COX_ALL = COX_NUMERIC + COX_CATEGORICAL


def load_survival_data():
    print("Loading vw_survival_input...")
    engine = create_engine(DB_URL)
    df = pd.read_sql("SELECT * FROM vw_survival_input", engine)
    print(f"  Loaded {len(df):,} rows")
    print(f"  Events (converted): {int(df['event_converted'].sum()):,} "
          f"({df['event_converted'].mean():.1%})")
    print(f"  Median duration: {df['duration_days'].median():.0f} days")
    return df, engine


def fit_km_by_segment(df, segment_col, top_n=6):
    print(f"\nFitting Kaplan-Meier by {segment_col}...")
    top = df[segment_col].value_counts().head(top_n).index.tolist()
    subset = df[df[segment_col].isin(top)]

    results = {}
    for category in top:
        group = subset[subset[segment_col] == category]
        kmf = KaplanMeierFitter()
        kmf.fit(group["duration_days"], group["event_converted"])
        median = kmf.median_survival_time_
        results[category] = {
            "median_days": float(median) if not np.isinf(median) else None,
            "n": int(len(group)),
            "events": int(group["event_converted"].sum()),
        }
        median_str = f"{median:.0f}" if not np.isinf(median) else "not reached"
        print(f"  {category:20s}  n={len(group):>6,}  "
              f"events={group['event_converted'].sum():>5,}  median={median_str}")
    return results


def fit_cox(df):
    print("\nFitting Cox Proportional Hazards...")
    cox_df = df[COX_ALL + ["duration_days", "event_converted"]].copy()
    cox_df["company_size"] = cox_df["company_size"].fillna(cox_df["company_size"].median())
    cox_encoded = pd.get_dummies(cox_df, columns=COX_CATEGORICAL, drop_first=True, dtype=float)

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(cox_encoded, duration_col="duration_days", event_col="event_converted")

    print(f"  Concordance index (C-index): {cph.concordance_index_:.4f}")
    print(f"  Log-likelihood: {cph.log_likelihood_:.2f}")

    summary = cph.summary[["exp(coef)", "exp(coef) lower 95%",
                            "exp(coef) upper 95%", "p"]].copy()
    summary.columns = ["hazard_ratio", "hr_lower_95", "hr_upper_95", "p_value"]
    summary = summary.sort_values("hazard_ratio", ascending=False)
    return cph, summary


def save_artifacts(km_results, cox_summary, cox_c_index):
    os.makedirs("models", exist_ok=True)
    with open("models/survival_km_by_segment.json", "w") as f:
        json.dump(km_results, f, indent=4)
    print("\nSaved models/survival_km_by_segment.json")

    cox_dict = {
        "concordance_index": float(cox_c_index),
        "features": cox_summary.to_dict(orient="index"),
    }
    with open("models/survival_cox_summary.json", "w") as f:
        json.dump(cox_dict, f, indent=4)
    print("Saved models/survival_cox_summary.json")


def write_predictions_to_db(cph, df, engine):
    """
    Save per-lead median survival prediction to Postgres.

    Uses a VECTORIZED call to predict_survival_function() across all leads
    at once, then derives the median per lead with numpy. This is ~50x
    faster than calling cph.predict_median() row-by-row, which is what
    previously caused the pipeline to appear to hang.
    """
    print("\nWriting per-lead survival predictions (vectorized)...")

    cox_df = df[COX_ALL + ["lead_id", "duration_days", "event_converted"]].copy()
    cox_df["company_size"] = cox_df["company_size"].fillna(cox_df["company_size"].median())

    encoded = pd.get_dummies(
        cox_df.drop(columns=["lead_id", "duration_days", "event_converted"]),
        columns=COX_CATEGORICAL, drop_first=True, dtype=float,
    )
    encoded = encoded.reindex(columns=cph.params_.index, fill_value=0)

    # ---- Vectorized median survival ----
    # Evaluate survival function at a fixed time grid for ALL leads at once.
    # Then for each lead, find the first time where S(t) <= 0.5.
    time_grid = np.linspace(1, 6500, 500)  # days; covers our data range
    print(f"  Evaluating survival curves at {len(time_grid)} time points "
          f"for {len(encoded):,} leads...")

    survival_curves = cph.predict_survival_function(encoded, times=time_grid)
    survival_arr = survival_curves.values  # shape (len(time_grid), n_leads)

    # For each lead: first time where S(t) <= 0.5
    below = survival_arr <= 0.5
    first_below = below.argmax(axis=0)
    reached = below.any(axis=0)
    median_days = np.where(reached, time_grid[first_below], np.nan)

    print(f"  Median reached for {int(reached.sum()):,} leads "
          f"({reached.mean():.1%}); censored-only for the rest")

    out = pd.DataFrame({
        "lead_id": cox_df["lead_id"].values,
        "predicted_median_days": median_days,
        "survival_model": "CoxPH",
    })

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_survival_predictions (
                lead_id VARCHAR(50) PRIMARY KEY,
                predicted_median_days FLOAT,
                survival_model VARCHAR(50),
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("TRUNCATE TABLE ml_survival_predictions"))

    out.to_sql("ml_survival_predictions", engine, if_exists="append",
               index=False, method="multi", chunksize=1000)
    print(f"  Wrote {len(out):,} rows to ml_survival_predictions")


if __name__ == "__main__":
    print("=" * 60)
    print("CREDITPULSE SURVIVAL ANALYSIS")
    print("=" * 60)

    df, engine = load_survival_data()

    km_source = fit_km_by_segment(df, "lead_source", top_n=6)
    km_industry = fit_km_by_segment(df, "industry", top_n=7)

    cph, cox_summary = fit_cox(df)

    print("\n--- Cox PH: Top features by hazard ratio ---")
    print(cox_summary.round(4).to_string())

    save_artifacts(
        km_results={"by_lead_source": km_source, "by_industry": km_industry},
        cox_summary=cox_summary,
        cox_c_index=cph.concordance_index_,
    )
    write_predictions_to_db(cph, df, engine)

    print("\n" + "=" * 60)
    print("SURVIVAL ANALYSIS COMPLETE")
    print(f"C-index: {cph.concordance_index_:.4f}")
    print("=" * 60)