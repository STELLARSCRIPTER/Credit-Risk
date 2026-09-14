# scripts/run_sql_file.py
"""
Run a .sql file against the Postgres database using SQLAlchemy.

Usage:
    python scripts/run_sql_file.py sql/create_gold_tables.sql

Before running the file, this script drops the Gold-layer tables with
CASCADE so that foreign-key dependencies don't block a rebuild.
"""
import os
import sys
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise SystemExit("DATABASE_URL not found in .env")

if len(sys.argv) < 2:
    raise SystemExit("Usage: python scripts/run_sql_file.py <path-to-sql-file>")

sql_path = sys.argv[1]
if not os.path.exists(sql_path):
    raise SystemExit(f"SQL file not found: {sql_path}")

# Drop Gold tables (and their dependent FKs) before rebuilding.
# Order does not matter because we use CASCADE.
GOLD_TABLES = [
    "fact_campaign_touches",
    "fact_opportunities",
    "fact_activities",
    "dim_leads",
    "lead_summary",
    "dim_time",
]

engine = create_engine(DB_URL)

with engine.begin() as conn:
    print("Dropping existing Gold tables (CASCADE)...")
    for t in GOLD_TABLES:
        conn.execute(text(f"DROP TABLE IF EXISTS {t} CASCADE"))
        print(f"  dropped {t} (if existed)")

with open(sql_path, "r", encoding="utf-8") as f:
    sql = f.read()

# Split on semicolons to run statements one at a time.
# Note: this is a simple splitter — it's fine for DDL scripts without
# stored procedures, but won't handle functions with embedded semicolons.
statements = [s.strip() for s in sql.split(";") if s.strip()]

print(f"\nRunning {sql_path} against {engine.url.database} ({len(statements)} statements)...")

with engine.begin() as conn:
    for i, stmt in enumerate(statements, 1):
        try:
            conn.execute(text(stmt))
        except Exception as e:
            print(f"\n[ERROR] Statement {i} failed:")
            print(stmt[:500])
            print(f"\n{type(e).__name__}: {e}")
            raise

print(f"Done. Ran {len(statements)} statements.")