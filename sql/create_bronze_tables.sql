
DROP TABLE IF EXISTS raw_leads;
CREATE TABLE raw_leads (
    lead_id         VARCHAR(20),
    contact_name    VARCHAR(150),
    company_name    VARCHAR(150),
    industry        VARCHAR(50),
    company_size    VARCHAR(50),   -- text type on purpose: some rows contain "163 emp" style values
    lead_source     VARCHAR(50),
    job_title       VARCHAR(100),
    country         VARCHAR(100),
    region          VARCHAR(50),
    status          VARCHAR(50),
    created_date    DATE
);

DROP TABLE IF EXISTS raw_opportunities;
CREATE TABLE raw_opportunities (
    opportunity_id      VARCHAR(20),
    lead_id             VARCHAR(20),   -- FK to raw_leads.lead_id (not enforced in Bronze)
    deal_value          NUMERIC,
    stage               VARCHAR(50),
    owner               VARCHAR(150),
    open_date           DATE,
    expected_close_date DATE,
    actual_close_date   DATE
);

DROP TABLE IF EXISTS raw_activities;
CREATE TABLE raw_activities (
    activity_id     VARCHAR(20),
    lead_id         VARCHAR(20),   -- FK to raw_leads.lead_id (a few rows are orphans on purpose)
    activity_date   DATE,
    activity_type   VARCHAR(50),
    channel         VARCHAR(50),
    outcome         VARCHAR(50)
);

DROP TABLE IF EXISTS raw_campaign_touches;
CREATE TABLE raw_campaign_touches (
    touch_id         VARCHAR(20),
    lead_id          VARCHAR(20),   -- FK to raw_leads.lead_id
    campaign_id      VARCHAR(20),
    touch_date       DATE,
    response         VARCHAR(50),
    treatment_group  VARCHAR(20)
);