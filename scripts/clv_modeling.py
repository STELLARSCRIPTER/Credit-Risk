# scripts/clv_modeling.py
"""
CreditPulse: Customer Lifetime Value (CLV) — Phase 8

Survival-integrated CLV, Option B:
    CLV = P(convert) × deal_value × discount_factor(median_time_to_convert)

Where:
    - P(convert) comes from Phase 6 (ml_lead_scores)
    - median_time_to_convert comes from Phase 7 (ml_survival_predictions)
    - discount_factor(t) = exp(-r × t) with r = annual discount rate

Outputs:
    - Postgres: ml_clv_predictions
    - models/clv_summary.json
"""

import os
import json
import warnings

import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

# ==========================================
# CONFIG
# ==========================================
DISCOUNT_RATE = 0.10        # 10% annual discount rate (standard SaaS/RevOps)
FALLBACK_LIFETIME_YEARS = 5 # For leads with no Cox prediction, assume 5-year horizon
CLV_TIERS = {               # quantile cut points
    "Low":    0.50,
    "Medium": 0.80,
    "High":   1.00,
}

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL not found in .env")


# ==========================================
# 1. LOAD
# ==========================================
def load_inputs():
    print("Loading Phase 6 (probability), Phase 7 (survival), deal values...")
    engine = create_engine(DB_URL)

    scores = pd.read_sql("""
        SELECT lead_id, conversion_probability, risk_tier, expected_loss
        FROM ml_lead_scores
    """, engine)

    survival = pd.read_sql("""
        SELECT lead_id, predicted_median_days
        FROM ml_survival_predictions
    """, engine)

    leads = pd.read_sql("""
        SELECT lead_id, industry, lead_source, region, deal_value
        FROM lead_summary
    """, engine)

    print(f"  Scores:    {len(scores):,} rows")
    print(f"  Survival:  {len(survival):,} rows")
    print(f"  Leads:     {len(leads):,} rows")

    return scores, survival, leads, engine


# ==========================================
# 2. COMPUTE CLV
# ==========================================
def compute_clv(scores, survival, leads):
    print("\nJoining and computing CLV...")
    df = leads.merge(scores, on="lead_id", how="left")
    df = df.merge(survival, on="lead_id", how="left")

    # Use segment median deal value for missing deal_values
    # (industries with sparse data get the global median)
    global_median = df.loc[df["deal_value"] > 0, "deal_value"].median()
    segment_medians = (
        df[df["deal_value"] > 0]
        .groupby("industry")["deal_value"]
        .median()
    )
    df["segment_median_deal"] = df["industry"].map(segment_medians).fillna(global_median)
    df["effective_deal_value"] = df["deal_value"].where(
        df["deal_value"] > 0, df["segment_median_deal"]
    )
    df["used_imputed_deal"] = (df["deal_value"] <= 0) | df["deal_value"].isna()

    # --- Survival-integrated discount factor ---
    # Convert predicted_median_days -> years. If missing, use fallback.
    df["median_years"] = df["predicted_median_days"] / 365.25
    df["median_years"] = df["median_years"].fillna(FALLBACK_LIFETIME_YEARS)
    # Guard against 0 or negative
    df.loc[df["median_years"] <= 0, "median_years"] = FALLBACK_LIFETIME_YEARS

    # discount_factor(t) = exp(-r * t)
    df["discount_factor"] = np.exp(-DISCOUNT_RATE * df["median_years"])

    # Fill missing probability with the global base rate
    base_rate = df["conversion_probability"].mean()
    df["conversion_probability"] = df["conversion_probability"].fillna(base_rate)

    # --- CLV ---
    # CLV = P(convert) × effective_deal_value × discount_factor
    df["clv_discounted"] = (
        df["conversion_probability"]
        * df["effective_deal_value"]
        * df["discount_factor"]
    )

    # Also compute undiscounted for comparison
    df["clv_undiscounted"] = df["conversion_probability"] * df["effective_deal_value"]

    # --- CLV tiers via quantiles ---
    valid = df["clv_discounted"].dropna()
    q50 = valid.quantile(0.50)
    q80 = valid.quantile(0.80)

    def tier(x):
        if pd.isna(x):    return "Unscored"
        if x >= q80:      return "High"
        if x >= q50:      return "Medium"
        return "Low"

    df["clv_tier"] = df["clv_discounted"].apply(tier)

    print(f"  Discounted CLV — median: ${df['clv_discounted'].median():,.0f}")
    print(f"  Discounted CLV — mean:   ${df['clv_discounted'].mean():,.0f}")
    print(f"  Discounted CLV — P90:    ${df['clv_discounted'].quantile(0.9):,.0f}")
    print(f"  Discounted CLV — max:    ${df['clv_discounted'].max():,.0f}")
    print(f"  Quantile thresholds: q50=${q50:,.0f}  q80=${q80:,.0f}")
    print(f"  Imputed deal values: {df['used_imputed_deal'].sum():,} leads")

    return df, {"q50": float(q50), "q80": float(q80)}


# ==========================================
# 3. SAVE
# ==========================================
def save_artifacts(df, thresholds, engine):
    # Save summary JSON
    os.makedirs("models", exist_ok=True)
    summary = {
        "model_version": "CLV v1 (survival-integrated)",
        "discount_rate_annual": DISCOUNT_RATE,
        "fallback_lifetime_years": FALLBACK_LIFETIME_YEARS,
        "clv_quantile_thresholds": thresholds,
        "aggregate_stats": {
            "total_leads": int(len(df)),
            "leads_with_probability": int(df["conversion_probability"].notna().sum()),
            "total_clv_discounted": float(df["clv_discounted"].sum()),
            "median_clv_discounted": float(df["clv_discounted"].median()),
            "mean_clv_discounted": float(df["clv_discounted"].mean()),
            "tier_counts": df["clv_tier"].value_counts().to_dict(),
        },
    }
    with open("models/clv_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print("\nSaved models/clv_summary.json")

    # Save to Postgres
    print("\nWriting ml_clv_predictions to Postgres...")
    out = df[[
        "lead_id", "conversion_probability", "effective_deal_value",
        "median_years", "discount_factor", "clv_discounted",
        "clv_undiscounted", "clv_tier", "industry", "lead_source", "region",
    ]].copy()
    out["discount_rate"] = DISCOUNT_RATE
    out["model_version"] = "CLV v1 (survival-integrated)"

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_clv_predictions (
                lead_id VARCHAR(50) PRIMARY KEY,
                conversion_probability FLOAT,
                effective_deal_value FLOAT,
                median_years FLOAT,
                discount_factor FLOAT,
                clv_discounted FLOAT,
                clv_undiscounted FLOAT,
                clv_tier VARCHAR(20),
                industry VARCHAR(100),
                lead_source VARCHAR(50),
                region VARCHAR(50),
                discount_rate FLOAT,
                model_version VARCHAR(50),
                scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("TRUNCATE TABLE ml_clv_predictions"))

    out.to_sql("ml_clv_predictions", engine, if_exists="append",
               index=False, method="multi", chunksize=1000)
    print(f"  Wrote {len(out):,} rows to ml_clv_predictions")


# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("CREDITPULSE CLV MODELING — PHASE 8")
    print("=" * 60)

    scores, survival, leads, engine = load_inputs()
    df, thresholds = compute_clv(scores, survival, leads)
    save_artifacts(df, thresholds, engine)

    print("\n" + "=" * 60)
    print("CLV MODELING COMPLETE")
    print(f"Total CLV across portfolio: ${df['clv_discounted'].sum():,.0f}")
    print("=" * 60)