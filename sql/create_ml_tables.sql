-- sql/create_ml_tables.sql
-- Stores the output of the ML scoring pipeline for the dashboard.

CREATE TABLE IF NOT EXISTS ml_lead_scores (
    lead_id VARCHAR(50) PRIMARY KEY,
    conversion_probability FLOAT,
    risk_tier VARCHAR(20),          -- 'High', 'Medium', 'Low'
    expected_loss FLOAT,            -- (1 - probability) * deal_value
    model_version VARCHAR(50),
    scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Optional: A view to join scores back to the lead_summary for easy dashboarding
CREATE OR REPLACE VIEW vw_lead_summary_scored AS
SELECT 
    ls.*,
    ms.conversion_probability,
    ms.risk_tier,
    ms.expected_loss
FROM lead_summary ls
LEFT JOIN ml_lead_scores ms ON ls.lead_id = ms.lead_id;

