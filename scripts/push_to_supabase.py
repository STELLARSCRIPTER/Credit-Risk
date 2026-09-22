# scripts/push_to_supabase.py
"""
One-time helper: push the dashboard-relevant tables from local Postgres
to a remote Supabase Postgres instance.

Usage:
    # Set the Supabase connection string in an environment variable
    $env:SUPABASE_URL = "postgresql+psycopg2://postgres:<password>@db.<project>.supabase.co:5432/postgres?sslmode=require"

    python scripts/push_to_supabase.py

Notes:
    - Only pushes the tables the dashboard actually reads. See TABLES below.
    - Idempotent: drops & recreates each table on the target.
    - Preserves column types by reading from local with pandas and using
      to_sql() on the target.
"""

import os
import sys
import time

import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

LOCAL_URL = os.getenv("DATABASE_URL")
REMOTE_URL = os.getenv("SUPABASE_URL")

if not LOCAL_URL:
    sys.exit("DATABASE_URL is not set in .env")
if not REMOTE_URL:
    sys.exit(
        "SUPABASE_URL is not set.\n"
        "  Set it in PowerShell: "
        '$env:SUPABASE_URL = "postgresql+psycopg2://..."'
    )

# Tables the deployed dashboard actually queries.
# Order matters — smaller tables first so a failure surfaces early.
TABLES = [
    "lead_summary",
    "ml_lead_scores",
    "ml_survival_predictions",
    "ml_clv_predictions",
    "ml_uplift_predictions",
    "leads_clean",
]


def push():
    print("=" * 60)
    print("PUSHING DASHBOARD TABLES TO SUPABASE")
    print("=" * 60)

    local = create_engine(LOCAL_URL)
    remote = create_engine(REMOTE_URL)

    print("\nTesting remote connection...")
    with remote.begin() as conn:
        conn.execute(text("SELECT 1"))
    print("  ✅ Remote connection OK")

    for table in TABLES:
        t0 = time.time()
        print(f"\nPushing {table}...")

        # Read from local
        df = pd.read_sql(f"SELECT * FROM {table}", local)
        print(f"  Read {len(df):,} rows from local")

        # Drop & recreate on remote for a clean, idempotent push
        with remote.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))

        # Write to remote
        df.to_sql(
            table,
            remote,
            if_exists="replace",
            index=False,
            method="multi",
            chunksize=1000,
        )

        elapsed = time.time() - t0
        print(f"  ✅ Wrote {len(df):,} rows in {elapsed:.1f}s")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    push()