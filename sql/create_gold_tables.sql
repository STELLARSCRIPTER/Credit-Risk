-- Gold layer: business-ready tables built from Silver.
-- Two things live here:
--   1. A star schema (DIM_TIME, DIM_LEADS, FACT_ACTIVITIES,
--      FACT_OPPORTUNITIES, FACT_CAMPAIGN_TOUCHES) for BI/dashboard querying.
--   2. lead_summary: one flattened row per lead with derived features -
--      this is the table ML scoring, survival analysis, and CLV will read from.
--
-- Design note: DIM_TIME uses the calendar date itself as its primary key
-- (date_id DATE), rather than a generated surrogate integer key. This is a
-- common simplified variant of a date dimension - it keeps fact-table joins
-- simple (join directly on the date value) while still demonstrating the
-- star-schema pattern (dimension table with descriptive attributes, fact
-- tables referencing it by key).
--
-- AS_OF_DATE below (2026-09-06) must match the AS_OF_DATE constant in
-- scripts/generate_synthetic_data.py - it's the fixed snapshot date used
-- to calculate "how long has this lead been open" for still-open leads
-- (right-censoring, used later in survival analysis).

-- ============================================================
-- 1. DIM_TIME
-- ============================================================
DROP TABLE IF EXISTS dim_time;
CREATE TABLE dim_time AS
SELECT
    d::date                          AS date_id,
    EXTRACT(YEAR FROM d)::INT        AS year,
    EXTRACT(QUARTER FROM d)::INT     AS quarter,
    EXTRACT(MONTH FROM d)::INT       AS month,
    TO_CHAR(d, 'Month')              AS month_name,
    EXTRACT(DAY FROM d)::INT         AS day,
    EXTRACT(DOW FROM d)::INT         AS day_of_week,   -- 0=Sunday ... 6=Saturday
    TO_CHAR(d, 'Day')                AS day_name,
    (EXTRACT(DOW FROM d) IN (0, 6))  AS is_weekend
FROM generate_series(DATE '2010-01-01', DATE '2026-12-31', INTERVAL '1 day') AS d;

ALTER TABLE dim_time ADD PRIMARY KEY (date_id);

-- ============================================================
-- 2. DIM_LEADS  (materialized copy of leads_clean)
-- ============================================================
DROP TABLE IF EXISTS dim_leads;
CREATE TABLE dim_leads AS
SELECT
    lead_id,
    contact_name,
    company_name,
    industry,
    company_size,
    company_size_missing,
    lead_source,
    job_title,
    country,
    region,
    status,
    created_date
FROM leads_clean;

ALTER TABLE dim_leads ADD PRIMARY KEY (lead_id);
ALTER TABLE dim_leads ADD CONSTRAINT fk_dim_leads_created_date
    FOREIGN KEY (created_date) REFERENCES dim_time(date_id);

-- ============================================================
-- 3. FACT_ACTIVITIES
-- ============================================================
DROP TABLE IF EXISTS fact_activities;
CREATE TABLE fact_activities AS
SELECT
    activity_id,
    lead_id,
    activity_date AS date_id,
    activity_type,
    channel,
    outcome
FROM activities_clean;

ALTER TABLE fact_activities ADD PRIMARY KEY (activity_id);
ALTER TABLE fact_activities ADD CONSTRAINT fk_fact_activities_lead
    FOREIGN KEY (lead_id) REFERENCES dim_leads(lead_id);
ALTER TABLE fact_activities ADD CONSTRAINT fk_fact_activities_date
    FOREIGN KEY (date_id) REFERENCES dim_time(date_id);

-- ============================================================
-- 4. FACT_OPPORTUNITIES
-- ============================================================
DROP TABLE IF EXISTS fact_opportunities;
CREATE TABLE fact_opportunities AS
SELECT
    opportunity_id,
    lead_id,
    deal_value,
    deal_value_outlier,
    stage,
    owner,
    open_date        AS open_date_id,
    expected_close_date AS expected_close_date_id,
    actual_close_date   AS actual_close_date_id
FROM opportunities_clean;

ALTER TABLE fact_opportunities ADD PRIMARY KEY (opportunity_id);
ALTER TABLE fact_opportunities ADD CONSTRAINT fk_fact_opps_lead
    FOREIGN KEY (lead_id) REFERENCES dim_leads(lead_id);
ALTER TABLE fact_opportunities ADD CONSTRAINT fk_fact_opps_open_date
    FOREIGN KEY (open_date_id) REFERENCES dim_time(date_id);
ALTER TABLE fact_opportunities ADD CONSTRAINT fk_fact_opps_expected_close
    FOREIGN KEY (expected_close_date_id) REFERENCES dim_time(date_id);
-- actual_close_date_id is nullable (opportunity may still be open) - no FK
-- enforced with NOT NULL, Postgres allows NULL to satisfy a FK constraint.
ALTER TABLE fact_opportunities ADD CONSTRAINT fk_fact_opps_actual_close
    FOREIGN KEY (actual_close_date_id) REFERENCES dim_time(date_id);

-- ============================================================
-- 5. FACT_CAMPAIGN_TOUCHES
-- ============================================================
DROP TABLE IF EXISTS fact_campaign_touches;
CREATE TABLE fact_campaign_touches AS
SELECT
    touch_id,
    lead_id,
    campaign_id,
    touch_date AS date_id,
    response,
    treatment_group
FROM campaign_touches_clean;

ALTER TABLE fact_campaign_touches ADD PRIMARY KEY (touch_id);
ALTER TABLE fact_campaign_touches ADD CONSTRAINT fk_fact_touches_lead
    FOREIGN KEY (lead_id) REFERENCES dim_leads(lead_id);
ALTER TABLE fact_campaign_touches ADD CONSTRAINT fk_fact_touches_date
    FOREIGN KEY (date_id) REFERENCES dim_time(date_id);

-- ============================================================
-- 6. lead_summary  (business-ready, one row per lead - feeds ML/survival/CLV)
-- ============================================================
DROP TABLE IF EXISTS lead_summary;
CREATE TABLE lead_summary AS
WITH activity_stats AS (
    SELECT
        lead_id,
        COUNT(*)              AS total_activities,
        MIN(activity_date)    AS first_activity_date,
        MAX(activity_date)    AS last_activity_date
    FROM activities_clean
    GROUP BY lead_id
),
opp AS (
    -- each lead has at most one opportunity in this dataset
    SELECT
        lead_id,
        opportunity_id,
        stage,
        deal_value,
        deal_value_outlier,
        open_date,
        expected_close_date,
        actual_close_date
    FROM opportunities_clean
),
touch AS (
    -- each lead received at most one campaign touch in this dataset
    SELECT lead_id, campaign_id, touch_date, response, treatment_group
    FROM campaign_touches_clean
)
SELECT
    l.lead_id,
    l.contact_name,
    l.company_name,
    l.industry,
    l.company_size,
    l.company_size_missing,
    l.lead_source,
    l.job_title,
    l.country,
    l.region,
    l.status,
    l.created_date,

    COALESCE(a.total_activities, 0)   AS total_activities,
    a.first_activity_date,
    a.last_activity_date,

    (t.lead_id IS NOT NULL)           AS received_campaign,
    t.touch_date                      AS campaign_touch_date,
    t.response                        AS campaign_response,
    t.treatment_group,

    (o.lead_id IS NOT NULL)           AS has_opportunity,
    o.opportunity_id,
    o.stage                           AS opportunity_stage,
    o.deal_value,
    o.deal_value_outlier,
    o.open_date,
    o.expected_close_date,
    o.actual_close_date,

    -- event_converted: 1 if the LEAD converted (source of truth = leads_clean.status).
    -- Note: this is intentionally decoupled from opportunity stage. A lead
    -- can convert without ever having an opportunity, and an opportunity
    -- can exist without the lead being marked Converted (e.g. still open).
    -- Design choice: "Closed-Lost" and "still open" opportunities are both
    -- treated as censored (event_converted = 0) for survival analysis - i.e.
    -- we are modeling time-to-conversion specifically, not time-to-any-outcome.
    CASE WHEN l.status = 'Converted' THEN 1 ELSE 0 END AS event_converted,

    -- duration_days: time from lead creation to the conversion event, or to
    -- the fixed snapshot date for leads that never converted (right-censored
    -- duration - required for Cox Proportional Hazards).
    CASE
        WHEN l.status = 'Converted'
             AND o.actual_close_date IS NOT NULL
            THEN (o.actual_close_date - l.created_date)
        ELSE (DATE '2026-09-06' - l.created_date)
    END AS duration_days

FROM dim_leads l
LEFT JOIN activity_stats a ON a.lead_id = l.lead_id
LEFT JOIN opp o             ON o.lead_id = l.lead_id
LEFT JOIN touch t           ON t.lead_id = l.lead_id;

ALTER TABLE lead_summary ADD PRIMARY KEY (lead_id);