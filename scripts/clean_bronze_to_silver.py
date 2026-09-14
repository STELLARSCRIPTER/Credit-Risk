"""
Clean Bronze-layer tables into Silver-layer tables.

Prerequisites:
    1. sql/create_bronze_tables.sql and sql/create_silver_tables.sql both run
    2. Bronze tables loaded (scripts/load_to_postgres.py already run)
    3. .env filled in with your DB credentials

What this script fixes, matching the deliberate messiness in Bronze:
    - company_size: strips text like "163 emp" down to a clean integer;
      genuinely missing values stay NULL (never imputed here - imputation
      is a modeling decision, not a cleaning decision) with a
      company_size_missing flag so it's still visible downstream.
    - duplicate activity rows: dropped (kept first occurrence).
    - orphan activities (lead_id with no matching lead): NOT silently
      dropped - moved into activities_rejected so the issue stays
      traceable and analyzable.
    - deal_value outliers (implausibly large values): nulled out and
      flagged via deal_value_outlier, not silently overwritten.

Safe to re-run: truncates all Silver tables before reloading.
"""

import os
import re
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import shutil

# Fail fast if disk is low — Postgres crashes uncleanly at 0 space and can
# leave the database in an unrecoverable state. 3 GB gives Postgres room for
# WAL writes during normal operation.
_free_gb = shutil.disk_usage("C:\\").free / (1024 ** 3)
if _free_gb < 3:
    raise SystemExit(
        f"Only {_free_gb:.2f} GB free on C:. Aborting to avoid Postgres "
        f"crash. Free up space and re-run."
    )


load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "creditrsik")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

# A deal_value this large can only be the injected outlier (999999999),
# never a real value - max plausible is roughly company_size(50000) * 300 = 15,000,000
DEAL_VALUE_OUTLIER_THRESHOLD = 50_000_000


def parse_company_size(value):
    """Extract a clean integer from company_size, which in Bronze may be
    a plain number, a float-as-string ("23.0"), text like "163 emp", or NULL."""
    if pd.isna(value):
        return np.nan
    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else np.nan


def truncate_silver_tables(conn):
    conn.execute(text(
        "TRUNCATE TABLE activities_rejected, campaign_touches_clean, "
        "activities_clean, opportunities_clean, leads_clean;"
    ))
    print("Truncated existing Silver tables (clean slate before reload).\n")


def clean_leads(raw_leads):
    df = raw_leads.copy()
    df["company_size"] = df["company_size"].apply(parse_company_size)
    df["company_size_missing"] = df["company_size"].isna()
    # Keep as plain object dtype with Python None (not pandas Int64/pd.NA) -
    # this avoids very slow per-value type coercion when SQLAlchemy builds
    # large multi-row INSERT statements.
    df["company_size"] = df["company_size"].astype(object)
    df["company_size"] = df["company_size"].where(df["company_size"].notna(), None)
    df["company_size"] = pd.Series(
        [int(x) if x is not None else None for x in df["company_size"]],
        index=df.index, dtype=object,
    )
    return df[[
        "lead_id", "contact_name", "company_name", "industry",
        "company_size", "company_size_missing", "lead_source",
        "job_title", "country", "region", "status", "created_date",
    ]]


def clean_opportunities(raw_opps, valid_lead_ids):
    df = raw_opps.copy()
    df["deal_value_outlier"] = df["deal_value"] > DEAL_VALUE_OUTLIER_THRESHOLD
    df.loc[df["deal_value_outlier"], "deal_value"] = np.nan

    # keep only opportunities pointing to a real lead
    df = df[df["lead_id"].isin(valid_lead_ids)]

    return df[[
        "opportunity_id", "lead_id", "deal_value", "deal_value_outlier",
        "stage", "owner", "open_date", "expected_close_date", "actual_close_date",
    ]]


def clean_activities(raw_activities, valid_lead_ids):
    df = raw_activities.copy()

    # drop exact duplicate rows (same activity_id repeated)
    df = df.drop_duplicates(subset="activity_id", keep="first")

    is_orphan = ~df["lead_id"].isin(valid_lead_ids)
    valid = df[~is_orphan].copy()
    rejected = df[is_orphan].copy()
    rejected["rejection_reason"] = "orphan_lead_id"

    valid = valid[["activity_id", "lead_id", "activity_date", "activity_type", "channel", "outcome"]]
    rejected = rejected[["activity_id", "lead_id", "activity_date", "activity_type",
                          "channel", "outcome", "rejection_reason"]]
    return valid, rejected


def clean_campaign_touches(raw_touches, valid_lead_ids):
    df = raw_touches.copy()
    df = df.drop_duplicates(subset="touch_id", keep="first")
    df = df[df["lead_id"].isin(valid_lead_ids)]
    return df[["touch_id", "lead_id", "campaign_id", "touch_date", "response", "treatment_group"]]


if __name__ == "__main__":
    print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT} ...")
    with engine.connect() as conn:
        print("Connection successful.\n")

    print("Reading Bronze tables...")
    raw_leads = pd.read_sql("SELECT * FROM raw_leads", engine)
    raw_opps = pd.read_sql("SELECT * FROM raw_opportunities", engine)
    raw_activities = pd.read_sql("SELECT * FROM raw_activities", engine)
    raw_touches = pd.read_sql("SELECT * FROM raw_campaign_touches", engine)
    print(f"  raw_leads:             {len(raw_leads):>7,} rows")
    print(f"  raw_opportunities:     {len(raw_opps):>7,} rows")
    print(f"  raw_activities:        {len(raw_activities):>7,} rows")
    print(f"  raw_campaign_touches:  {len(raw_touches):>7,} rows\n")

    print("Cleaning...")
    leads_clean = clean_leads(raw_leads)
    valid_lead_ids = set(leads_clean["lead_id"])

    opportunities_clean = clean_opportunities(raw_opps, valid_lead_ids)
    activities_clean, activities_rejected = clean_activities(raw_activities, valid_lead_ids)
    campaign_touches_clean = clean_campaign_touches(raw_touches, valid_lead_ids)

    print(f"  leads_clean:             {len(leads_clean):>7,} rows "
          f"({leads_clean['company_size_missing'].sum():,} with missing company_size)")
    print(f"  opportunities_clean:     {len(opportunities_clean):>7,} rows "
          f"({opportunities_clean['deal_value_outlier'].sum():,} outliers flagged)")
    print(f"  activities_clean:        {len(activities_clean):>7,} rows")
    print(f"  activities_rejected:     {len(activities_rejected):>7,} rows (orphan lead_id)")
    print(f"  campaign_touches_clean:  {len(campaign_touches_clean):>7,} rows\n")

    print("Writing Silver tables to Postgres...")
    with engine.begin() as conn:
        truncate_silver_tables(conn)

    leads_clean.to_sql("leads_clean", engine, if_exists="append", index=False, method="multi", chunksize=500)
    opportunities_clean.to_sql("opportunities_clean", engine, if_exists="append", index=False, method="multi", chunksize=500)
    activities_clean.to_sql("activities_clean", engine, if_exists="append", index=False, method="multi", chunksize=500)
    activities_rejected.to_sql("activities_rejected", engine, if_exists="append", index=False, method="multi", chunksize=500)
    campaign_touches_clean.to_sql("campaign_touches_clean", engine, if_exists="append", index=False, method="multi", chunksize=500)

    print("\nDone. Verify with: SELECT COUNT(*) FROM leads_clean;")

    