import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

sql = open("sql/create_survival_tables.sql", encoding="utf-8").read()
statements = [s.strip() for s in sql.split(";") if s.strip()]

with engine.begin() as conn:
    for stmt in statements:
        conn.execute(text(stmt))

print(f"Ran {len(statements)} statements. Survival view created.")
