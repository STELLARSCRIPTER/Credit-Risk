-- Survival analysis preparation.
-- We build a view that exposes the exact columns Cox PH needs, with the
-- 13 rows that have duration_days <= 0 filtered out (lifelines cannot
-- handle non-positive durations).
--
-- WHY A VIEW (not a table):
-- lead_summary is already the source of truth. The survival view just
-- projects + filters it. No need to materialize a second copy that could
-- drift out of sync.

DROP VIEW IF EXISTS vw_survival_input CASCADE;

CREATE VIEW vw_survival_input AS
SELECT
    lead_id,
    duration_days::FLOAT          AS duration_days,
    event_converted::INT          AS event_converted,

    -- Covariates for the Cox model.
    -- Keep this list SHORT and interpretable - Cox coefficients are
    -- reported as hazard ratios, so every feature must have a clean story.
    industry,
    lead_source,
    region,
    job_title,
    CASE
        WHEN company_size IS NULL THEN NULL
        ELSE company_size::FLOAT
    END                           AS company_size,
    total_activities::INT         AS total_activities,
    COALESCE(received_campaign, false)::INT AS received_campaign
FROM lead_summary
WHERE duration_days > 0;