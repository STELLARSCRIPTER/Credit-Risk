import os
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()
engine = create_engine(os.getenv("DATABASE_URL"))

q = """
SELECT treatment_group, event_converted, COUNT(*) AS n
FROM lead_summary
WHERE received_campaign = TRUE
  AND treatment_group IN ('Treatment', 'Control')
GROUP BY treatment_group, event_converted
ORDER BY treatment_group, event_converted
"""
print(pd.read_sql(q, engine).to_string(index=False))

q2 = """
SELECT treatment_group,
       AVG(event_converted::int) AS conv_rate,
       COUNT(*) AS n
FROM lead_summary
WHERE received_campaign = TRUE
  AND treatment_group IN ('Treatment', 'Control')
GROUP BY treatment_group
"""
print()
print("Baseline conversion by arm:")
print(pd.read_sql(q2, engine).to_string(index=False))
