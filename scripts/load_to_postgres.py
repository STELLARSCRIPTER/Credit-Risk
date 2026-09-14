"""
Load synthetic CRM CSVs into PostgreSQL Bronze-layer tables.

Prerequisites:
    1. PostgreSQL running locally, database created (e.g. CREATE DATABASE creditrsik;)
    2. Bronze tables already created by running sql/create_bronze_tables.sql
       (e.g. psql -U postgres -d creditrsik -f sql/create_bronze_tables.sql)
    3. .env file filled in with your DB credentials (see .env.example)
    4. CSVs present in data/raw/ (run scripts/generate_synthetic_data.py first)

This script is SAFE TO RE-RUN: it truncates each Bronze table before
reloading, so row counts always match the CSVs exactly - running it
twice will never double your data.
"""

import os
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

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

engine = create_engine(
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

TABLES = {
    "raw_leads.csv": "raw_leads",
    "raw_opportunities.csv": "raw_opportunities",
    "raw_activities.csv": "raw_activities",
    "raw_campaign_touches.csv": "raw_campaign_touches",
}

def truncate_all_tables():
    """Empty every Bronze table before reloading, so this script never
    doubles up data when run more than once."""
    with engine.begin() as conn:
        table_list = ", ".join(TABLES.values())
        conn.execute(text(f"TRUNCATE TABLE {table_list};"))
    print("Truncated existing Bronze tables (clean slate before reload).\n")

def load_csv_to_table(csv_filename, table_name):
    path = os.path.join(DATA_DIR, csv_filename)
    if not os.path.exists(path):
        print(f"  SKIPPED: {csv_filename} not found at {path}")
        return
    df = pd.read_csv(path)
    df.to_sql(table_name, engine, if_exists="append", index=False, method="multi", chunksize=5000)
    print(f"  Loaded {len(df):>6} rows -> {table_name}")

if __name__ == "__main__":
    print(f"Connecting to {DB_NAME} at {DB_HOST}:{DB_PORT} ...")
    with engine.connect() as conn:
        print("Connection successful.\n")

    truncate_all_tables()

    print("Loading CSVs into Bronze tables:")
    for csv_file, table in TABLES.items():
        load_csv_to_table(csv_file, table)

    print("\nDone. Verify with: SELECT COUNT(*) FROM raw_leads;")