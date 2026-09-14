-- Silver layer: cleaned, validated, deduplicated tables.
-- Missing/invalid data is never silently dropped without a trace:
--   - company_size stays NULL when genuinely missing, with a flag column
--     so downstream modeling can decide how to handle it (not baked in here).
--   - deal_value outliers are nulled out and flagged, so the flag itself
--     is visible for audit purposes.
--   - orphan activities (lead_id with no matching lead) are quarantined into
--     activities_rejected instead of being silently discarded.

DROP TABLE IF EXISTS leads_clean;
CREATE TABLE leads_clean (
    lead_id                 VARCHAR(20) PRIMARY KEY,
    contact_name            VARCHAR(150),
    company_name            VARCHAR(150),
    industry                VARCHAR(50),
    company_size            INTEGER,          -- NULL when genuinely missing
    company_size_missing    BOOLEAN NOT NULL, -- TRUE if company_size is NULL (i.e. was missing in Bronze)
    lead_source             VARCHAR(50),
    job_title               VARCHAR(100),
    country                 VARCHAR(100),
    region                  VARCHAR(50),
    status                  VARCHAR(50),
    created_date            DATE
);

DROP TABLE IF EXISTS opportunities_clean;
CREATE TABLE opportunities_clean (
    opportunity_id       VARCHAR(20) PRIMARY KEY,
    lead_id              VARCHAR(20) REFERENCES leads_clean(lead_id),
    deal_value           NUMERIC,          -- NULL when flagged as an outlier
    deal_value_outlier   BOOLEAN NOT NULL, -- TRUE if the original Bronze value was an implausible outlier
    stage                VARCHAR(50),
    owner                VARCHAR(150),
    open_date            DATE,
    expected_close_date  DATE,
    actual_close_date    DATE
);

DROP TABLE IF EXISTS activities_clean;
CREATE TABLE activities_clean (
    activity_id     VARCHAR(20) PRIMARY KEY,
    lead_id         VARCHAR(20) REFERENCES leads_clean(lead_id),
    activity_date   DATE,
    activity_type   VARCHAR(50),
    channel         VARCHAR(50),
    outcome         VARCHAR(50)
);

-- Quarantine table: activities that failed validation (e.g. orphan lead_id)
-- are kept here rather than silently dropped, so data-quality issues remain
-- traceable and analyzable (e.g. "how many rejects per source, per week").
DROP TABLE IF EXISTS activities_rejected;
CREATE TABLE activities_rejected (
    activity_id       VARCHAR(20),
    lead_id           VARCHAR(20),
    activity_date     DATE,
    activity_type     VARCHAR(50),
    channel           VARCHAR(50),
    outcome           VARCHAR(50),
    rejection_reason  VARCHAR(100)
);

DROP TABLE IF EXISTS campaign_touches_clean;
CREATE TABLE campaign_touches_clean (
    touch_id          VARCHAR(20) PRIMARY KEY,
    lead_id           VARCHAR(20) REFERENCES leads_clean(lead_id),
    campaign_id       VARCHAR(20),
    touch_date        DATE,
    response          VARCHAR(50),
    treatment_group   VARCHAR(20)
);

