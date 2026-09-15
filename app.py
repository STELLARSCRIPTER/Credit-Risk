import os
import json
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

st.set_page_config(page_title="CreditPulse", page_icon="〽", layout="wide")
load_dotenv()

st.markdown("""
<style>
    .stApp { background: #f6f9fd; color: #0b2855; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #061b37, #0b315d);
    }
    [data-testid="stSidebar"] * { color: #eef5ff !important; }
    [data-testid="stSidebar"] .stRadio label {
        padding: 9px 8px; border-radius: 7px;
    }
    .top-title { font-size: 30px; font-weight: 800; margin-top: 8px; }
    .top-subtitle { font-size: 18px; font-weight: 700; }
    .description { color: #587097; margin: 10px 0 18px; }
    .metric-card {
        background: white; border: 1px solid #dce7f5; border-radius: 9px;
        padding: 16px; min-height: 100px;
    }
    .metric-label { color: #3e5d88; font-size: 12px; font-weight: 600; }
    .metric-value { color: #0a2959; font-size: 25px; font-weight: 800; margin-top: 6px; }
    .metric-note { color: #12a56d; font-size: 12px; margin-top: 7px; }
    .panel {
        background: white; border: 1px solid #dce7f5; border-radius: 9px;
        padding: 12px; margin-bottom: 12px;
    }
    .panel-title { color: #0a2959; font-size: 16px; font-weight: 800; }
    .insight {
        border-bottom: 1px solid #e7edf7; padding: 13px 0;
    }
    .insight-title { font-size: 14px; font-weight: 800; color: #102e5b; }
    .insight-text { font-size: 12px; color: #587097; margin-top: 5px; }
</style>
""", unsafe_allow_html=True)


def card(icon, label, value, note):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} &nbsp; {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">● {note}</div>
    </div>
    """, unsafe_allow_html=True)


@st.cache_data(ttl=300)
def load_data():
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        st.error("DATABASE_URL is missing from your .env file.")
        st.stop()
    engine = create_engine(db_url)
    return pd.read_sql("""
        SELECT
            lead_id, company_name, industry, company_size, job_title, lead_source, country, region,
            status, created_date, total_activities, received_campaign,
            campaign_response, treatment_group, opportunity_stage,
            deal_value, actual_close_date, duration_days
        FROM lead_summary
    """, engine)


df = load_data()
df["deal_value"] = df["deal_value"].fillna(0)
df["created_date"] = pd.to_datetime(df["created_date"])


# ---------------------------------------------------------------------------
# Customer Analytics helpers
# ---------------------------------------------------------------------------

STAGE_ORDER = ["New", "Contacted", "Qualified", "Converted"]


def _build_funnel_stages(customer_df: pd.DataFrame) -> pd.DataFrame:
    stage_index = {s: i for i, s in enumerate(STAGE_ORDER)}
    stage_df = customer_df.copy()
    stage_df["_stage_rank"] = stage_df["status"].map(stage_index)

    rows = []
    for i, stage in enumerate(STAGE_ORDER):
        reached = (stage_df["_stage_rank"] >= i).sum()
        rows.append({"stage": stage, "count": int(reached)})

    funnel_df = pd.DataFrame(rows)
    funnel_df["prev_count"] = funnel_df["count"].shift(1)
    funnel_df["dropoff_pct"] = (
        (funnel_df["prev_count"] - funnel_df["count"]) / funnel_df["prev_count"] * 100
    ).round(1)
    funnel_df.loc[0, "dropoff_pct"] = 0.0

    return funnel_df.drop(columns="prev_count")


def render_funnel_chart(customer_df: pd.DataFrame):
    st.markdown('<div class="panel"><div class="panel-title">Lead Conversion Funnel</div>', unsafe_allow_html=True)
    if customer_df.empty:
        st.caption("No leads match the current filter.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    funnel_df = _build_funnel_stages(customer_df)
    fig = px.funnel(funnel_df, x="count", y="stage", color_discrete_sequence=["#1769e0"])
    fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    cols = st.columns(len(funnel_df) - 1)
    for i, col in enumerate(cols, start=1):
        with col:
            st.metric(label=f"Drop-off into '{funnel_df.loc[i, 'stage']}'",
                      value=f"{funnel_df.loc[i, 'dropoff_pct']:.1f}%")
    exited = customer_df[customer_df["status"].isin(["Lost", "Unqualified"])]
    if not exited.empty:
        st.caption(f"{len(exited):,} leads ({len(exited) / len(customer_df):.1%} of total) "
                   f"exited as Lost or Unqualified and are excluded from the funnel above.")
        exit_breakdown = exited["status"].value_counts().reset_index()
        exit_breakdown.columns = ["status", "count"]
        st.dataframe(exit_breakdown, hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)


def render_customer_drilldown(customer_df: pd.DataFrame):
    st.markdown('<div class="panel"><div class="panel-title">Customer 360 — Drill-Down</div>', unsafe_allow_html=True)
    if customer_df.empty:
        st.caption("No leads match the current filter.")
        st.markdown("</div>", unsafe_allow_html=True)
        return
    lookup_df = customer_df.copy()
    lookup_df["_label"] = lookup_df["lead_id"] + " — " + lookup_df["company_name"]
    selected_label = st.selectbox("Select a customer",
                                  options=lookup_df["_label"].sort_values(),
                                  index=0, key="drilldown_select")
    lead_id = selected_label.split(" — ")[0]
    record = lookup_df.loc[lookup_df["lead_id"] == lead_id].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Industry", record.get("industry") or "—")
    c2.metric("Region", record.get("region") or "—")
    c3.metric("Country", record.get("country") or "—")
    c4.metric("Company Size", record.get("company_size") or "—")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Job Title", record.get("job_title") or "—")
    c6.metric("Lead Source", record.get("lead_source") or "—")
    c7.metric("Status", record.get("status") or "—")
    c8.metric("Opportunity Stage", record.get("opportunity_stage") or "—")
    st.write("")
    e1, e2, e3, e4 = st.columns(4)
    avg_activities = customer_df["total_activities"].mean()
    e1.metric("Total Activities", f"{int(record.get('total_activities', 0))}",
              delta=f"{record.get('total_activities', 0) - avg_activities:+.1f} vs. avg")
    e2.metric("Received Campaign", "Yes" if record.get("received_campaign") else "No")
    e3.metric("Campaign Response", record.get("campaign_response") or "—")
    e4.metric("Treatment Group", record.get("treatment_group") or "—")
    st.write("")
    o1, o2, o3, o4 = st.columns(4)
    deal_value = record.get("deal_value", 0)
    o1.metric("Deal Value", f"${deal_value:,.0f}" if pd.notnull(deal_value) else "—")
    created = record.get("created_date")
    o2.metric("Created Date", created.strftime("%b %d, %Y") if pd.notnull(created) else "—")
    closed = record.get("actual_close_date")
    o3.metric("Closed Date", pd.to_datetime(closed).strftime("%b %d, %Y") if pd.notnull(closed) else "—")
    duration = record.get("duration_days")
    o4.metric("Duration (days)", f"{duration:.0f}" if pd.notnull(duration) else "—")
    st.markdown("</div>", unsafe_allow_html=True)


def render_customer_analytics(customer_df):
    st.markdown('<div class="top-title">Customer Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Explore customer profiles, segments, engagement, and conversion performance.</div>', unsafe_allow_html=True)

    search = st.text_input("Search customer", placeholder="Search by Lead ID or company name...")
    analytics_df = customer_df.copy()
    if search:
        analytics_df = analytics_df[
            analytics_df["lead_id"].str.contains(search, case=False, na=False)
            | analytics_df["company_name"].str.contains(search, case=False, na=False)
        ]

    total_customers = len(analytics_df)
    converted = (analytics_df["status"] == "Converted").sum()
    conversion_rate = converted / total_customers * 100 if total_customers else 0
    pipeline_value = analytics_df["deal_value"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total_customers:,}")
    c2.metric("Converted Customers", f"{converted:,}")
    c3.metric("Conversion Rate", f"{conversion_rate:.2f}%")
    c4.metric("Pipeline Value", f"${pipeline_value:,.0f}")

    st.subheader("Customer Portfolio")
    st.dataframe(analytics_df[[
        "lead_id", "company_name", "industry", "company_size",
        "job_title", "region", "lead_source", "total_activities",
        "status", "opportunity_stage", "deal_value"
    ]], hide_index=True, use_container_width=True, height=450)
    st.write("")
    render_funnel_chart(analytics_df)
    render_customer_drilldown(analytics_df)


# ---------------------------------------------------------------------------
# Risk Analysis page
# ---------------------------------------------------------------------------

STATUS_RISK_WEIGHT = {
    "Lost": 1.0, "Unqualified": 0.85, "New": 0.55,
    "Contacted": 0.45, "Qualified": 0.30, "Converted": 0.10,
}


def _pct_rank_inverse(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return 1 - series.rank(pct=True, na_option="bottom")


def _pct_rank(series: pd.Series) -> pd.Series:
    if series.nunique(dropna=True) <= 1:
        return pd.Series(0.5, index=series.index)
    return series.rank(pct=True, na_option="bottom")


def _compute_proxy_risk_scores(source_df: pd.DataFrame) -> pd.DataFrame:
    risk_df = source_df.copy()
    status_component = risk_df["status"].map(STATUS_RISK_WEIGHT).fillna(0.5)
    activity_component = _pct_rank_inverse(risk_df["total_activities"].fillna(0))
    duration_component = _pct_rank(risk_df["duration_days"].fillna(risk_df["duration_days"].median()))
    no_response_penalty = (
        (risk_df["received_campaign"] == True) & (risk_df["campaign_response"].isin([None, "No Response", "None"]))
    ).astype(float)
    risk_df["risk_score"] = (
        0.45 * status_component + 0.20 * activity_component
        + 0.20 * duration_component + 0.15 * no_response_penalty
    ).clip(0, 1)
    risk_df["risk_tier"] = pd.cut(risk_df["risk_score"],
                                  bins=[-0.01, 0.4, 0.7, 1.0],
                                  labels=["Low", "Medium", "High"])
    avg_deal_value = risk_df.loc[risk_df["deal_value"] > 0, "deal_value"].mean() or 0
    risk_df["exposure"] = risk_df["deal_value"].where(risk_df["deal_value"] > 0, avg_deal_value)
    risk_df["expected_loss"] = risk_df["risk_score"] * risk_df["exposure"]
    return risk_df


def render_risk_analysis(source_df: pd.DataFrame):
    st.markdown('<div class="top-title">Risk Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Proxy risk scoring based on engagement and status signals. '
                'Swap in real model predictions once the risk model is trained.</div>', unsafe_allow_html=True)

    risk_df = _compute_proxy_risk_scores(source_df)
    f1, f2, f3 = st.columns(3)
    regions = f1.multiselect("Region", sorted(risk_df["region"].dropna().unique()), key="risk_region")
    industries = f2.multiselect("Industry", sorted(risk_df["industry"].dropna().unique()), key="risk_industry")
    tiers = f3.multiselect("Risk Tier", ["Low", "Medium", "High"], key="risk_tier_filter")

    filtered = risk_df.copy()
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if industries:
        filtered = filtered[filtered["industry"].isin(industries)]
    if tiers:
        filtered = filtered[filtered["risk_tier"].isin(tiers)]

    total_customers = len(filtered)
    high_risk = (filtered["risk_tier"] == "High").sum()
    high_risk_rate = high_risk / total_customers * 100 if total_customers else 0
    total_exposure = filtered["exposure"].sum()
    total_expected_loss = filtered["expected_loss"].sum()
    avg_risk_score = filtered["risk_score"].mean() if total_customers else 0

    cards = [
        ("👥", "Total Customers", f"{total_customers:,}", "In selected view"),
        ("💼", "Total Exposure", f"${total_exposure:,.0f}", "Actual + estimated deal value"),
        ("⚠️", "High-Risk Customers", f"{high_risk:,} ({high_risk_rate:.1f}%)", "Risk tier = High"),
        ("💸", "Expected Loss", f"${total_expected_loss:,.0f}", "Risk score × exposure"),
        ("📊", "Avg. Risk Score", f"{avg_risk_score:.2f}", "0 (low) – 1 (high)"),
    ]
    for col, (icon, label, value, note) in zip(st.columns(5), cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} &nbsp; {label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">● {note}</div>
            </div>
            """, unsafe_allow_html=True)
    st.write("")

    if filtered.empty:
        st.info("No customers match the current filters.")
        return

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.markdown('<div class="panel"><div class="panel-title">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="risk_score", nbins=25, color_discrete_sequence=["#1769e0"])
        fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Risk Score", yaxis_title="Customers")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown('<div class="panel"><div class="panel-title">Risk Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = filtered["risk_tier"].value_counts().reindex(["Low", "Medium", "High"]).reset_index()
        tier_counts.columns = ["risk_tier", "count"]
        fig = px.pie(tier_counts, values="count", names="risk_tier", hole=.6, color="risk_tier",
                     color_discrete_map={"Low": "#1eb27b", "Medium": "#ffc21a", "High": "#ef5350"})
        fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        st.markdown('<div class="panel"><div class="panel-title">Avg. Risk Score by Industry</div>', unsafe_allow_html=True)
        industry_risk = (filtered.groupby("industry", as_index=False)
                         .agg(avg_risk=("risk_score", "mean"))
                         .sort_values("avg_risk", ascending=False).head(8))
        fig = px.bar(industry_risk, x="avg_risk", y="industry", orientation="h",
                     color="avg_risk", color_continuous_scale="Reds")
        fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5),
                          coloraxis_showscale=False, xaxis_title="Avg. Risk Score")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col4:
        st.markdown('<div class="panel"><div class="panel-title">Expected Loss by Region</div>', unsafe_allow_html=True)
        region_loss = (filtered.groupby("region", as_index=False)
                       .agg(expected_loss=("expected_loss", "sum"))
                       .sort_values("expected_loss", ascending=False))
        fig = px.bar(region_loss, x="region", y="expected_loss", color_discrete_sequence=["#ef5350"])
        fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis_title="Expected Loss ($)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Top High-Risk Customers</div>', unsafe_allow_html=True)
    top_risk = filtered.sort_values("risk_score", ascending=False).head(15)
    st.dataframe(top_risk[[
        "lead_id", "company_name", "industry", "region", "status",
        "total_activities", "duration_days", "risk_score", "risk_tier", "expected_loss"
    ]].style.format({"risk_score": "{:.2f}", "expected_loss": "${:,.0f}"}),
        hide_index=True, use_container_width=True, height=450)
    st.markdown("</div>", unsafe_allow_html=True)


def render_interventions(source_df: pd.DataFrame):
    st.markdown('<div class="top-title">Interventions</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Measure campaign treatment impact and identify the customer segments where an intervention performs best.</div>', unsafe_allow_html=True)

    campaign_df = source_df[(source_df["received_campaign"] == True)
                            & (source_df["treatment_group"].isin(["Control", "Treatment"]))].copy()

    f1, f2, f3 = st.columns(3)
    regions = f1.multiselect("Region", sorted(campaign_df["region"].dropna().unique()), key="intervention_region")
    sources = f2.multiselect("Lead Source", sorted(campaign_df["lead_source"].dropna().unique()), key="intervention_source")
    responses = f3.multiselect("Campaign Response", sorted(campaign_df["campaign_response"].dropna().unique()), key="intervention_response")

    if regions:
        campaign_df = campaign_df[campaign_df["region"].isin(regions)]
    if sources:
        campaign_df = campaign_df[campaign_df["lead_source"].isin(sources)]
    if responses:
        campaign_df = campaign_df[campaign_df["campaign_response"].isin(responses)]
    if campaign_df.empty:
        st.info("No campaign records match the selected filters.")
        return

    summary = campaign_df.groupby("treatment_group", as_index=False).agg(
        leads=("lead_id", "count"),
        conversions=("status", lambda value: (value == "Converted").sum()),
        won_revenue=("deal_value", lambda value: value[campaign_df.loc[value.index, "opportunity_stage"] == "Closed-Won"].sum()),
    )
    summary["conversion_rate"] = summary["conversions"] / summary["leads"] * 100

    control = summary[summary["treatment_group"] == "Control"]
    treatment = summary[summary["treatment_group"] == "Treatment"]
    control_rate = control["conversion_rate"].iloc[0] if not control.empty else 0
    treatment_rate = treatment["conversion_rate"].iloc[0] if not treatment.empty else 0
    treatment_leads = treatment["leads"].iloc[0] if not treatment.empty else 0
    treatment_conversions = treatment["conversions"].iloc[0] if not treatment.empty else 0
    control_revenue = control["won_revenue"].iloc[0] if not control.empty else 0
    treatment_revenue = treatment["won_revenue"].iloc[0] if not treatment.empty else 0
    uplift_pp = treatment_rate - control_rate
    additional_conversions = treatment_conversions - treatment_leads * control_rate / 100

    cards = [
        ("🧪", "Treatment Leads", f"{treatment_leads:,}", "Customers receiving treatment"),
        ("🎯", "Treatment Conversion", f"{treatment_rate:.2f}%", f"Control: {control_rate:.2f}%"),
        ("📈", "Treatment Uplift", f"{uplift_pp:+.2f} pp", "Compared with control"),
        ("👥", "Additional Conversions", f"{additional_conversions:+.0f}", "Estimated treatment effect"),
        ("💰", "Revenue Difference", f"${treatment_revenue - control_revenue:+,.0f}", "Treatment minus control"),
    ]
    for col, (icon, label, value, note) in zip(st.columns(5), cards):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{icon} &nbsp; {label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">● {note}</div>
            </div>
            """, unsafe_allow_html=True)

    st.write("")
    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Treatment vs Control Conversion</div>', unsafe_allow_html=True)
        fig = px.bar(summary, x="treatment_group", y="conversion_rate", text="conversion_rate",
                     color="treatment_group",
                     color_discrete_map={"Control": "#3478df", "Treatment": "#1eb27b"},
                     labels={"treatment_group": "Campaign Group", "conversion_rate": "Conversion Rate (%)"})
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5), showlegend=False,
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Campaign Response by Group</div>', unsafe_allow_html=True)
        response_df = campaign_df.groupby(["campaign_response", "treatment_group"], as_index=False).agg(leads=("lead_id", "count"))
        fig = px.bar(response_df, x="campaign_response", y="leads", color="treatment_group", barmode="group",
                     color_discrete_map={"Control": "#3478df", "Treatment": "#1eb27b"},
                     labels={"campaign_response": "Response", "leads": "Leads", "treatment_group": "Campaign Group"})
        fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Treatment Uplift by Lead Source</div>', unsafe_allow_html=True)
    source_summary = campaign_df.groupby(["lead_source", "treatment_group"], as_index=False).agg(
        leads=("lead_id", "count"),
        conversions=("status", lambda value: (value == "Converted").sum()),
    )
    source_summary["conversion_rate"] = source_summary["conversions"] / source_summary["leads"] * 100
    pivot = source_summary.pivot(index="lead_source", columns="treatment_group", values="conversion_rate").reset_index()
    if {"Control", "Treatment"}.issubset(pivot.columns):
        pivot["uplift_pp"] = pivot["Treatment"] - pivot["Control"]
        pivot = pivot.dropna(subset=["uplift_pp"])
        fig = px.bar(pivot.sort_values("uplift_pp", ascending=False),
                     x="lead_source", y="uplift_pp", text="uplift_pp",
                     color="uplift_pp", color_continuous_scale=["#ef5350", "#ffc21a", "#1eb27b"],
                     labels={"lead_source": "Lead Source", "uplift_pp": "Treatment Uplift (percentage points)"})
        fig.update_traces(texttemplate="%{text:+.2f}", textposition="outside")
        fig.update_layout(height=330, margin=dict(l=5, r=5, t=10, b=5),
                          coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        if not pivot.empty:
            best = pivot.loc[pivot["uplift_pp"].idxmax()]
            st.success(f"Recommendation: prioritize Treatment for **{best['lead_source']}** leads; the observed uplift is **{best['uplift_pp']:+.2f} percentage points**.")
    else:
        st.info("Both Treatment and Control records are needed for source-level uplift analysis.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Intervention Results</div>', unsafe_allow_html=True)
    st.dataframe(summary[["treatment_group", "leads", "conversions", "conversion_rate", "won_revenue"]].style.format(
        {"conversion_rate": "{:.2f}%", "won_revenue": "${:,.0f}"}),
        hide_index=True, use_container_width=True)
    st.caption("This page shows observed outcomes from your synthetic campaign data and demonstrates a treatment-versus-control intervention analysis.")
    st.markdown('</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# ML Models page
# ---------------------------------------------------------------------------

MODEL_META_PATH = "models/model_metadata.json"


@st.cache_data(ttl=300)
def load_model_metadata():
    if not os.path.exists(MODEL_META_PATH):
        return None
    with open(MODEL_META_PATH) as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_scored_leads():
    engine = create_engine(os.getenv("DATABASE_URL"))
    try:
        return pd.read_sql("""
            SELECT
                s.lead_id, l.company_name, l.industry, l.region, l.lead_source,
                l.job_title, l.company_size, l.status, l.deal_value,
                l.event_converted, l.duration_days,
                s.conversion_probability, s.risk_tier, s.expected_loss, s.model_version
            FROM ml_lead_scores s
            LEFT JOIN lead_summary l ON l.lead_id = s.lead_id
        """, engine)
    except Exception:
        return pd.DataFrame()


def render_ml_models():
    st.markdown('<div class="top-title">ML Models — Lead Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Model comparison, scoring distribution, and per-lead '
                'conversion probability produced by the training pipeline.</div>', unsafe_allow_html=True)

    meta = load_model_metadata()
    if meta is None:
        st.warning("No trained model found. Run `python scripts/train_ml_models.py` "
                   "to generate `models/model_metadata.json`.")
        return

    best = meta.get("best_model", "—")
    metrics = meta.get("best_model_metrics", {})
    st.caption(f"Model currently shipped: **{best}**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("AUC (ROC)", f"{metrics.get('AUC', 0):.2f}")
    c2.metric("Accuracy", f"{metrics.get('Accuracy', 0):.2f}")
    c3.metric("F1 Score", f"{metrics.get('F1', 0):.2f}")
    c4.metric("Precision", f"{metrics.get('Precision', 0):.2f}")
    c5.metric("Recall", f"{metrics.get('Recall', 0):.2f}")

    if metrics.get("AUC", 0) < 0.65:
        st.warning("⚠️ AUC is only slightly above random (0.5).")
    else:
        st.success("Model is performing within a usable range.")

    st.markdown('<div class="panel"><div class="panel-title">Model Comparison</div>', unsafe_allow_html=True)
    comp = pd.DataFrame(meta.get("all_model_results", {})).T
    if not comp.empty:
        comp = comp[["AUC", "Accuracy", "F1", "Precision", "Recall"]]
        st.dataframe(comp, use_container_width=True)
        melted = comp.reset_index().melt(id_vars="index", var_name="Metric", value_name="Score")
        fig = px.bar(melted, x="index", y="Score", color="Metric", barmode="group",
                     labels={"index": "Model"},
                     color_discrete_sequence=["#1769e0", "#1eb27b", "#ffc21a", "#865fd3", "#ef5350"])
        fig.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    scored = load_scored_leads()
    if scored.empty:
        st.info("No rows in `ml_lead_scores` yet. Run the training script.")
        return

    st.markdown('<div class="panel"><div class="panel-title">Scored Leads</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3)
    tiers = f1.multiselect("Risk Tier", ["Low", "Medium", "High"],
                           default=["Low", "Medium", "High"], key="ml_tiers")
    industries = f2.multiselect("Industry", sorted(scored["industry"].dropna().unique()), key="ml_industries")
    sources = f3.multiselect("Lead Source", sorted(scored["lead_source"].dropna().unique()), key="ml_sources")

    filtered = scored[scored["risk_tier"].isin(tiers)]
    if industries:
        filtered = filtered[filtered["industry"].isin(industries)]
    if sources:
        filtered = filtered[filtered["lead_source"].isin(sources)]
    if filtered.empty:
        st.info("No leads match the current filters.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Leads shown", f"{len(filtered):,}")
    k2.metric("Avg conversion prob", f"{filtered['conversion_probability'].mean():.1%}")
    k3.metric("Total expected loss", f"${filtered['expected_loss'].sum():,.0f}")
    actual = filtered["event_converted"].dropna()
    k4.metric("Actual converted", f"{actual.mean():.1%}" if len(actual) else "—")

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel-title">Risk Tier Distribution</div>', unsafe_allow_html=True)
        tier_counts = (filtered["risk_tier"].value_counts()
                       .reindex(["Low", "Medium", "High"]).fillna(0).reset_index())
        tier_counts.columns = ["risk_tier", "count"]
        fig = px.pie(tier_counts, names="risk_tier", values="count", hole=0.6, color="risk_tier",
                     color_discrete_map={"Low": "#1eb27b", "Medium": "#ffc21a", "High": "#ef5350"})
        fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">Conversion Probability Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="conversion_probability", nbins=40, color="risk_tier",
                           color_discrete_map={"Low": "#1eb27b", "Medium": "#ffc21a", "High": "#ef5350"})
        fig.update_layout(height=300, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Conversion Probability", yaxis_title="Leads",
                          legend_title_text="Risk Tier")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel-title">Calibration — Predicted vs. Actual Conversion</div>', unsafe_allow_html=True)
    cal = filtered.dropna(subset=["event_converted"]).copy()
    if not cal.empty:
        cal["prob_bin"] = pd.cut(cal["conversion_probability"],
                                 bins=[0, .2, .4, .6, .8, 1.0],
                                 labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
        cal_df = cal.groupby("prob_bin", observed=True, as_index=False).agg(
            predicted=("conversion_probability", "mean"),
            actual=("event_converted", "mean"),
            leads=("lead_id", "count"))
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cal_df["prob_bin"], y=cal_df["actual"],
                             name="Actual conversion rate", marker_color="#1eb27b"))
        fig.add_trace(go.Scatter(x=cal_df["prob_bin"], y=cal_df["predicted"],
                                 mode="lines+markers", name="Predicted probability",
                                 line=dict(color="#1769e0", width=3)))
        fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis_title="Rate", xaxis_title="Predicted Probability Bin",
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel-title">Top Leads by Conversion Probability</div>', unsafe_allow_html=True)
    top = filtered.sort_values("conversion_probability", ascending=False).head(25)
    st.dataframe(top[[
        "lead_id", "company_name", "industry", "region", "lead_source",
        "job_title", "deal_value", "conversion_probability",
        "risk_tier", "expected_loss", "event_converted"
    ]].style.format({"conversion_probability": "{:.2%}",
                     "expected_loss": "${:,.0f}",
                     "deal_value": "${:,.0f}"}),
        hide_index=True, use_container_width=True, height=460)
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Survival Analysis page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_survival_km():
    if not os.path.exists("models/survival_km_by_segment.json"):
        return None
    with open("models/survival_km_by_segment.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_survival_cox():
    if not os.path.exists("models/survival_cox_summary.json"):
        return None
    with open("models/survival_cox_summary.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_survival_predictions():
    engine = create_engine(os.getenv("DATABASE_URL"))
    try:
        return pd.read_sql("""
            SELECT p.lead_id, p.predicted_median_days,
                   s.lead_source, s.industry, s.region,
                   s.event_converted, s.duration_days, s.total_activities
            FROM ml_survival_predictions p
            LEFT JOIN lead_summary s ON s.lead_id = p.lead_id
        """, engine)
    except Exception:
        return pd.DataFrame()


def render_survival_page():
    st.markdown('<div class="top-title">Survival Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="description">Not just <em>whether</em> a lead converts — '
        '<em>when</em> it converts, and how timing differs by segment. '
        'Kaplan-Meier curves + Cox Proportional Hazards.</div>',
        unsafe_allow_html=True,
    )

    km = load_survival_km()
    cox = load_survival_cox()
    if km is None or cox is None:
        st.warning("No survival model found. Run `python scripts/survival_analysis.py` first.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("C-index (concordance)", f"{cox['concordance_index']:.3f}")
    c2.metric("Events observed", "11,138")
    c3.metric("Total leads analysed", "50,858")

    if cox["concordance_index"] < 0.60:
        st.warning("C-index is weak — model has limited discriminative power.")
    else:
        st.success("Survival model is performing within a usable range.")

    st.markdown('<div class="panel"><div class="panel-title">'
                'Time-to-Conversion by Lead Source</div>', unsafe_allow_html=True)

    km_src = km.get("by_lead_source", {})
    if km_src:
        rows = []
        for source, vals in km_src.items():
            rows.append({
                "lead_source": source,
                "median_days": vals["median_days"] if vals["median_days"] else None,
                "n": vals["n"],
                "events": vals["events"],
                "conversion_rate": vals["events"] / vals["n"] if vals["n"] else 0,
            })
        km_df = pd.DataFrame(rows)
        km_df["median_years"] = km_df["median_days"] / 365.25

        plotted = km_df.dropna(subset=["median_days"]).sort_values("median_days")
        fig = px.bar(plotted, x="lead_source", y="median_years",
                     color="median_years",
                     color_continuous_scale=["#1eb27b", "#ffc21a", "#ef5350"],
                     labels={"median_years": "Median time to convert (years)",
                             "lead_source": "Lead source"},
                     text=plotted["median_years"].round(1).astype(str) + " yrs")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          coloraxis_showscale=False,
                          yaxis_range=[0, plotted["median_years"].max() * 1.15])
        st.plotly_chart(fig, use_container_width=True)

        not_reached = km_df[km_df["median_days"].isna()]
        if not not_reached.empty:
            st.caption("Sources below 50% conversion are not shown: "
                       + ", ".join(not_reached["lead_source"].tolist())
                       + " — fewer than half of their leads ever convert.")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Cox Proportional Hazards — Hazard Ratios</div>', unsafe_allow_html=True)

    feats = cox.get("features", {})
    if feats:
        hr_df = pd.DataFrame(feats).T.reset_index().rename(columns={"index": "feature"})
        hr_df = hr_df[hr_df["p_value"] < 0.05].copy()
        hr_df = hr_df[(hr_df["hazard_ratio"] > 1.02) | (hr_df["hazard_ratio"] < 0.98)].copy()
        hr_df = hr_df.sort_values("hazard_ratio", ascending=True)

        label_map = {
            "lead_source_Referral": "Lead source: Referral",
            "lead_source_Webinar": "Lead source: Webinar",
            "lead_source_Trade Show": "Lead source: Trade Show",
            "lead_source_LinkedIn": "Lead source: LinkedIn",
            "lead_source_Website": "Lead source: Website",
            "industry_SaaS": "Industry: SaaS",
            "industry_FinServ": "Industry: FinServ",
            "industry_Healthcare": "Industry: Healthcare",
            "industry_Manufacturing": "Industry: Manufacturing",
            "industry_Logistics": "Industry: Logistics",
            "region_North America": "Region: North America",
            "region_Europe": "Region: Europe",
            "region_EMEA": "Region: EMEA",
            "received_campaign": "Received campaign",
        }
        hr_df["label"] = hr_df["feature"].map(label_map).fillna(hr_df["feature"])

        colors = ["#ef5350" if hr < 1 else "#1eb27b" for hr in hr_df["hazard_ratio"]]
        error_plus = (hr_df["hr_upper_95"] - hr_df["hazard_ratio"]).clip(lower=0)
        error_minus = (hr_df["hazard_ratio"] - hr_df["hr_lower_95"]).clip(lower=0)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_df["hazard_ratio"],
            y=hr_df["label"],
            mode="markers",
            marker=dict(size=14, color=colors, line=dict(color="white", width=1.5)),
            error_x=dict(type="data", symmetric=False,
                         array=error_plus, arrayminus=error_minus,
                         color="#888", thickness=1.5),
            hovertemplate="<b>%{y}</b><br>Hazard ratio: %{x:.2f}<br><extra></extra>",
        ))
        fig.add_vline(x=1.0, line_dash="dash", line_color="#666", line_width=1.5)

        all_vals = list(hr_df["hr_lower_95"]) + list(hr_df["hr_upper_95"]) + [1.0]
        lo = max(0.1, min(all_vals) * 0.9)
        hi = max(all_vals) * 1.1

        fig.update_layout(
            height=max(420, len(hr_df) * 32),
            margin=dict(l=5, r=5, t=10, b=5),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Hazard ratio (log scale) — right = converts faster",
            xaxis=dict(type="log", range=[np.log10(lo), np.log10(hi)],
                       tickvals=[0.2, 0.5, 1, 2, 3, 5],
                       ticktext=["0.2", "0.5", "1.0", "2.0", "3.0", "5.0"]),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**How to read this:** HR > 1 (green) = converts *faster* than the baseline. "
                   "HR < 1 (red) = converts *slower*. Dashed line at 1.0 = no effect. "
                   "Error bars are 95% confidence intervals. "
                   "Only statistically significant features (p < 0.05) with visible effect are shown.")
    st.markdown("</div>", unsafe_allow_html=True)

    preds = load_survival_predictions()
    if not preds.empty:
        st.markdown('<div class="panel"><div class="panel-title">'
                    'Per-Lead Predicted Median Time-to-Convert</div>', unsafe_allow_html=True)

        preds = preds.dropna(subset=["predicted_median_days"])
        preds["predicted_years"] = preds["predicted_median_days"] / 365.25

        k1, k2, k3 = st.columns(3)
        k1.metric("Leads scored", f"{len(preds):,}")
        k2.metric("Median predicted time", f"{preds['predicted_years'].median():.1f} yrs")
        k3.metric("Fastest 10%", f"{preds['predicted_years'].quantile(0.10):.1f} yrs")

        fig = px.histogram(preds, x="predicted_years", nbins=50, color="lead_source",
                           color_discrete_sequence=px.colors.qualitative.Set2,
                           labels={"predicted_years": "Predicted median years to convert",
                                   "lead_source": "Lead source"})
        fig.update_layout(height=380, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white", legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Fastest 10 leads to convert (predicted)**")
        fast = preds.sort_values("predicted_median_days").head(10)
        st.dataframe(fast[["lead_id", "lead_source", "industry", "region",
                           "predicted_years", "duration_days", "event_converted"]]
                     .style.format({"predicted_years": "{:.1f} yrs"}),
                     hide_index=True, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CLV page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_clv_summary():
    if not os.path.exists("models/clv_summary.json"):
        return None
    with open("models/clv_summary.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_clv_predictions():
    engine = create_engine(os.getenv("DATABASE_URL"))
    try:
        return pd.read_sql("""
            SELECT c.lead_id, c.clv_discounted, c.clv_undiscounted, c.clv_tier,
                   c.conversion_probability, c.effective_deal_value,
                   c.median_years, c.discount_factor,
                   c.industry, c.lead_source, c.region,
                   s.risk_tier, l.event_converted
            FROM ml_clv_predictions c
            LEFT JOIN ml_lead_scores s ON s.lead_id = c.lead_id
            LEFT JOIN lead_summary l ON l.lead_id = c.lead_id
        """, engine)
    except Exception:
        return pd.DataFrame()


def render_clv_page():
    st.markdown('<div class="top-title">Customer Lifetime Value</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="description">Discounted CLV per lead — '
        'combining conversion probability (Phase 6), time-to-convert (Phase 7), '
        'and deal value into a single dollar figure.</div>',
        unsafe_allow_html=True,
    )

    summary = load_clv_summary()
    if summary is None:
        st.warning("No CLV model found. Run `python scripts/clv_modeling.py` first.")
        return

    agg = summary.get("aggregate_stats", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Portfolio CLV", f"${agg.get('total_clv_discounted', 0):,.0f}")
    c2.metric("Median CLV per lead", f"${agg.get('median_clv_discounted', 0):,.0f}")
    c3.metric("Mean CLV per lead", f"${agg.get('mean_clv_discounted', 0):,.0f}")
    c4.metric("Discount rate", f"{summary.get('discount_rate_annual', 0):.0%}/yr")

    st.success(
        f"CLV computed for {agg.get('total_leads', 0):,} leads. "
        f"Median time-to-convert of {summary.get('fallback_lifetime_years', 5)} years "
        f"used as fallback horizon when Cox prediction was unavailable."
    )

    df = load_clv_predictions()
    if df.empty:
        st.info("No rows in `ml_clv_predictions` yet.")
        return

    left, right = st.columns(2)
    with left:
        st.markdown('<div class="panel-title">CLV Distribution (log scale)</div>', unsafe_allow_html=True)
        plot_df = df[df["clv_discounted"] > 0].copy()
        fig = px.histogram(plot_df, x="clv_discounted", nbins=60, color="clv_tier",
                           color_discrete_map={"High": "#1eb27b", "Medium": "#ffc21a", "Low": "#ef5350"},
                           log_y=True)
        fig.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Discounted CLV ($)", yaxis_title="Leads (log)",
                          legend_title_text="CLV Tier")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">CLV Tier Breakdown</div>', unsafe_allow_html=True)
        tier_counts = (df["clv_tier"].value_counts()
                       .reindex(["High", "Medium", "Low"]).fillna(0).reset_index())
        tier_counts.columns = ["clv_tier", "count"]
        fig = px.pie(tier_counts, names="clv_tier", values="count", hole=0.6, color="clv_tier",
                     color_discrete_map={"High": "#1eb27b", "Medium": "#ffc21a", "Low": "#ef5350"})
        fig.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Value × Risk Quadrant — Where to Focus</div>', unsafe_allow_html=True)
    st.caption(
        "Each point is a segment (industry × lead source). "
        "X-axis = mean conversion probability. Y-axis = mean discounted CLV. "
        "Bubble size = number of leads. Top-right = high-value × high-probability segments."
    )
    seg = df.groupby(["industry", "lead_source"], as_index=False).agg(
        mean_prob=("conversion_probability", "mean"),
        mean_clv=("clv_discounted", "mean"),
        n_leads=("lead_id", "count"),
    )
    seg = seg[seg["n_leads"] >= 100]
    fig = px.scatter(seg, x="mean_prob", y="mean_clv", size="n_leads",
                     color="mean_clv",
                     color_continuous_scale=["#ef5350", "#ffc21a", "#1eb27b"],
                     hover_data={"industry": True, "lead_source": True, "n_leads": True},
                     labels={"mean_prob": "Mean conversion probability",
                             "mean_clv": "Mean discounted CLV ($)"})
    fig.update_layout(height=460, margin=dict(l=5, r=5, t=10, b=5),
                      plot_bgcolor="white", paper_bgcolor="white",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Top 25 Leads by Discounted CLV</div>', unsafe_allow_html=True)
    top = df.sort_values("clv_discounted", ascending=False).head(25)
    st.dataframe(top[["lead_id", "industry", "lead_source", "region",
                      "conversion_probability", "effective_deal_value",
                      "median_years", "discount_factor",
                      "clv_discounted", "clv_tier", "risk_tier", "event_converted"]]
                 .style.format({
                     "conversion_probability": "{:.1%}",
                     "effective_deal_value": "${:,.0f}",
                     "median_years": "{:.1f} yrs",
                     "discount_factor": "{:.3f}",
                     "clv_discounted": "${:,.0f}",
                 }),
                 hide_index=True, use_container_width=True, height=460)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Average CLV by Segment</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**By industry**")
        by_ind = df.groupby("industry", as_index=False).agg(
            mean_clv=("clv_discounted", "mean"),
            total_clv=("clv_discounted", "sum"),
            n=("lead_id", "count"),
        ).sort_values("mean_clv", ascending=False)
        fig = px.bar(by_ind, x="mean_clv", y="industry", orientation="h",
                     color="mean_clv", color_continuous_scale=["#b9d8ff", "#1769e0"])
        fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                          coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Mean discounted CLV ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**By lead source**")
        by_src = df.groupby("lead_source", as_index=False).agg(
            mean_clv=("clv_discounted", "mean"),
            total_clv=("clv_discounted", "sum"),
            n=("lead_id", "count"),
        ).sort_values("mean_clv", ascending=False)
        fig = px.bar(by_src, x="mean_clv", y="lead_source", orientation="h",
                     color="mean_clv", color_continuous_scale=["#b9d8ff", "#1769e0"])
        fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                          coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white",
                          xaxis_title="Mean discounted CLV ($)", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if not by_ind.empty and not by_src.empty:
        best_ind = by_ind.iloc[0]
        best_src = by_src.iloc[0]
        st.markdown(
            f"""
            <div class="panel" style="background: #ecfdf5; border-color: #1eb27b;">
                <div class="panel-title">Key Finding</div>
                <p style="color: #065f46; margin-top: 6px;">
                    The highest-value industry segment is
                    <b>{best_ind['industry']}</b>
                    (mean discounted CLV <b>${best_ind['mean_clv']:,.0f}</b>)
                    and the highest-value lead source is
                    <b>{best_src['lead_source']}</b>
                    (mean discounted CLV <b>${best_src['mean_clv']:,.0f}</b>).
                    These are the segments where acquisition spend has the
                    largest expected return.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Uplift Modeling page (Phase 9)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_uplift_summary():
    if not os.path.exists("models/uplift_summary.json"):
        return None
    with open("models/uplift_summary.json") as f:
        return json.load(f)


@st.cache_data(ttl=300)
def load_uplift_predictions():
    engine = create_engine(os.getenv("DATABASE_URL"))
    try:
        return pd.read_sql("""
            SELECT u.lead_id, u.treatment_group, u.event_converted,
                   u.p_treatment, u.p_control, u.uplift, u.uplift_segment,
                   u.industry, u.lead_source, u.region,
                   l.deal_value
            FROM ml_uplift_predictions u
            LEFT JOIN lead_summary l ON l.lead_id = u.lead_id
        """, engine)
    except Exception:
        return pd.DataFrame()


def _compute_qini_curve(df, n_bins=25):
    df = df.copy().sort_values("uplift", ascending=False)
    treat = df[df["treatment_group"] == "Treatment"]
    ctrl = df[df["treatment_group"] == "Control"]
    n = len(df)
    fracs = np.linspace(0.05, 1.0, n_bins)
    treat_ratio = len(treat) / len(ctrl) if len(ctrl) else 1.0
    qini_points = []
    for f in fracs:
        k = int(f * n)
        top_k = df.iloc[:k]
        y_t = top_k[top_k["treatment_group"] == "Treatment"]["event_converted"].sum()
        y_c = top_k[top_k["treatment_group"] == "Control"]["event_converted"].sum()
        qini = y_t - y_c * treat_ratio
        qini_points.append({"fraction": f, "qini": qini, "n": k})
    return pd.DataFrame(qini_points)


def render_uplift_page():
    st.markdown('<div class="top-title">Uplift Modeling</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="description">Not just "does the campaign work?" but '
        '"<em>which leads</em> does the campaign actually change?" '
        'T-learner uplift model on campaign-targeted leads.</div>',
        unsafe_allow_html=True,
    )

    summary = load_uplift_summary()
    if summary is None:
        st.warning("No uplift model found. Run `python scripts/uplift_modeling.py` first.")
        return

    agg = summary.get("aggregate", {})
    seg_counts = summary.get("segment_counts", {})
    arm_metrics = summary.get("arm_metrics", {})

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean uplift", f"{agg.get('mean_uplift', 0):+.2%}")
    c2.metric("Median uplift", f"{agg.get('median_uplift', 0):+.2%}")
    c3.metric("Persuadable leads",
              f"{seg_counts.get('Persuadable', 0):,} "
              f"({agg.get('share_persuadable', 0):.1%})")
    c4.metric("Do Not Disturb leads",
              f"{seg_counts.get('Do Not Disturb', 0):,} "
              f"({agg.get('share_do_not_disturb', 0):.1%})")

    st.success(
        f"T-learner arms both at ~{arm_metrics.get('treatment_auc', 0):.2f} AUC. "
        f"Mean uplift of {agg.get('mean_uplift', 0):+.2%} matches the raw "
        f"Treatment minus Control conversion difference."
    )

    df = load_uplift_predictions()
    if df.empty:
        st.info("No rows in `ml_uplift_predictions` yet.")
        return

    color_map = {
        "Persuadable": "#1eb27b",
        "Sure Thing": "#3478df",
        "Lost Cause": "#94a3b8",
        "Do Not Disturb": "#ef5350",
    }

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown('<div class="panel-title">Segment Breakdown</div>', unsafe_allow_html=True)
        seg_df = pd.DataFrame({
            "segment": ["Persuadable", "Sure Thing", "Lost Cause", "Do Not Disturb"],
            "count": [seg_counts.get(s, 0) for s in
                      ["Persuadable", "Sure Thing", "Lost Cause", "Do Not Disturb"]],
        })
        fig = px.pie(seg_df, names="segment", values="count", hole=0.6,
                     color="segment", color_discrete_map=color_map)
        fig.update_layout(height=340, margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown('<div class="panel-title">Uplift Distribution</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x="uplift", nbins=60, color="uplift_segment",
                           color_discrete_map=color_map)
        fig.add_vline(x=0, line_dash="dash", line_color="#666")
        fig.update_layout(
            height=340, margin=dict(l=5, r=5, t=10, b=5),
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Predicted uplift (P_treatment − P_control)",
            yaxis_title="Leads", legend_title_text="",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Qini Curve — How Well the Model Ranks Uplift</div>',
                unsafe_allow_html=True)
    st.caption(
        "The Qini curve shows cumulative incremental conversions as we "
        "target leads in predicted-uplift order. A model that ranks correctly "
        "produces a curve above the diagonal."
    )
    qini_df = _compute_qini_curve(df, n_bins=25)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=qini_df["fraction"], y=qini_df["qini"],
        mode="lines+markers", name="Model",
        line=dict(color="#1eb27b", width=3),
    ))
    max_q = qini_df["qini"].iloc[-1]
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, max_q],
        mode="lines", name="Random targeting",
        line=dict(color="#94a3b8", dash="dash"),
    ))
    fig.update_layout(
        height=400, margin=dict(l=5, r=5, t=10, b=5),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Fraction of population targeted (ranked by predicted uplift)",
        yaxis_title="Cumulative incremental conversions",
        legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Uplift and Baseline by Segment</div>', unsafe_allow_html=True)
    seg_stats = df.groupby("uplift_segment", as_index=False).agg(
        n=("lead_id", "count"),
        mean_uplift=("uplift", "mean"),
        mean_p_treatment=("p_treatment", "mean"),
        mean_p_control=("p_control", "mean"),
    ).sort_values("mean_uplift", ascending=False)
    st.dataframe(
        seg_stats.style.format({
            "n": "{:,}",
            "mean_uplift": "{:+.2%}",
            "mean_p_treatment": "{:.1%}",
            "mean_p_control": "{:.1%}",
        }),
        hide_index=True, use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Top 25 Persuadable Leads — Target These</div>',
                unsafe_allow_html=True)
    st.caption(
        "Leads with the highest predicted uplift. Running the campaign on "
        "these leads produces the largest expected incremental conversions."
    )
    top_pers = (df[df["uplift_segment"] == "Persuadable"]
                .sort_values("uplift", ascending=False).head(25))
    st.dataframe(
        top_pers[["lead_id", "industry", "lead_source", "region",
                  "deal_value", "p_control", "p_treatment",
                  "uplift", "event_converted"]]
        .style.format({
            "deal_value": "${:,.0f}",
            "p_control": "{:.1%}",
            "p_treatment": "{:.1%}",
            "uplift": "{:+.2%}",
        }),
        hide_index=True, use_container_width=True, height=460,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">'
                'Top 15 Do Not Disturb Leads — Suppress These</div>',
                unsafe_allow_html=True)
    st.caption(
        "Leads where the campaign reduces conversion probability. "
        "Suppressing the campaign for these leads saves budget AND improves "
        "their expected outcome. A/B testing alone cannot identify this segment."
    )
    top_dnd = (df[df["uplift_segment"] == "Do Not Disturb"]
               .sort_values("uplift", ascending=True).head(15))
    st.dataframe(
        top_dnd[["lead_id", "industry", "lead_source", "region",
                 "deal_value", "p_control", "p_treatment",
                 "uplift", "event_converted"]]
        .style.format({
            "deal_value": "${:,.0f}",
            "p_control": "{:.1%}",
            "p_treatment": "{:.1%}",
            "uplift": "{:+.2%}",
        }),
        hide_index=True, use_container_width=True, height=340,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    n_pers = seg_counts.get("Persuadable", 0)
    n_dnd = seg_counts.get("Do Not Disturb", 0)
    n_sure = seg_counts.get("Sure Thing", 0)
    n_lost = seg_counts.get("Lost Cause", 0)
    st.markdown(
        f"""
        <div class="panel" style="background: #ecfdf5; border-color: #1eb27b;">
            <div class="panel-title">Business Takeaway</div>
            <p style="color: #065f46; margin-top: 6px;">
                Of {len(df):,} campaign-targeted leads:<br>
                <b>• {n_pers:,} are Persuadable</b> — the campaign materially
                increases their conversion. Target these first.<br>
                <b>• {n_dnd:,} are Do Not Disturb</b> — the campaign
                <b>lowers</b> their conversion. Suppressing them saves budget
                and improves outcomes.<br>
                <b>• {n_sure:,} are Sure Things</b> — they'd convert anyway,
                so campaign spend on them is wasted.<br>
                <b>• {n_lost:,} are Lost Causes</b> — no campaign will help.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# ETL / Data quality page
# ---------------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_etl_metrics():
    engine = create_engine(os.getenv("DATABASE_URL"))
    row_counts = pd.read_sql("""
        SELECT 'Bronze' AS layer, 'raw_leads' AS table_name, COUNT(*) AS row_count FROM raw_leads
        UNION ALL SELECT 'Bronze', 'raw_opportunities', COUNT(*) FROM raw_opportunities
        UNION ALL SELECT 'Bronze', 'raw_activities', COUNT(*) FROM raw_activities
        UNION ALL SELECT 'Bronze', 'raw_campaign_touches', COUNT(*) FROM raw_campaign_touches
        UNION ALL SELECT 'Silver', 'leads_clean', COUNT(*) FROM leads_clean
        UNION ALL SELECT 'Silver', 'opportunities_clean', COUNT(*) FROM opportunities_clean
        UNION ALL SELECT 'Silver', 'activities_clean', COUNT(*) FROM activities_clean
        UNION ALL SELECT 'Silver', 'campaign_touches_clean', COUNT(*) FROM campaign_touches_clean
        UNION ALL SELECT 'Gold', 'dim_leads', COUNT(*) FROM dim_leads
        UNION ALL SELECT 'Gold', 'fact_opportunities', COUNT(*) FROM fact_opportunities
        UNION ALL SELECT 'Gold', 'fact_activities', COUNT(*) FROM fact_activities
        UNION ALL SELECT 'Gold', 'fact_campaign_touches', COUNT(*) FROM fact_campaign_touches
        UNION ALL SELECT 'Gold', 'lead_summary', COUNT(*) FROM lead_summary
    """, engine)
    quality = pd.read_sql("""
        SELECT
            COUNT(*) AS total_leads,
            COUNT(*) FILTER (WHERE lead_id IS NULL) AS missing_lead_ids,
            COUNT(*) FILTER (WHERE company_name IS NULL OR TRIM(company_name) = '') AS missing_company_names,
            COUNT(*) FILTER (WHERE region IS NULL OR TRIM(region) = '') AS missing_regions,
            COUNT(*) FILTER (WHERE status IS NULL OR TRIM(status) = '') AS missing_statuses,
            COUNT(*) FILTER (WHERE deal_value < 0) AS negative_deal_values
        FROM lead_summary
    """, engine)
    return row_counts, quality


def render_data_etl():
    st.markdown('<div class="top-title">Data & ETL</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Monitor the flow from raw source files through cleaning and curated analytics tables.</div>', unsafe_allow_html=True)

    if st.button("↻ Refresh Pipeline Metrics"):
        load_etl_metrics.clear()
        st.rerun()

    try:
        row_counts, quality = load_etl_metrics()
    except Exception as error:
        st.error(f"Could not load ETL metrics: {error}")
        return

    bronze_rows = row_counts.loc[row_counts["layer"] == "Bronze", "row_count"].sum()
    silver_rows = row_counts.loc[row_counts["layer"] == "Silver", "row_count"].sum()
    gold_rows = row_counts.loc[row_counts["layer"] == "Gold", "row_count"].sum()
    quality_row = quality.iloc[0]
    issues = (quality_row["missing_lead_ids"] + quality_row["missing_company_names"]
              + quality_row["missing_regions"] + quality_row["missing_statuses"]
              + quality_row["negative_deal_values"])

    metrics = [
        ("📥", "Bronze Rows", f"{bronze_rows:,}", "Raw source records"),
        ("✨", "Silver Rows", f"{silver_rows:,}", "Cleaned records"),
        ("🏆", "Gold Rows", f"{gold_rows:,}", "Analytics-ready records"),
        ("🛡️", "Quality Issues", f"{issues:,}", "Core lead-summary checks"),
        ("⏱️", "Pipeline Status", "Healthy", "Tables successfully queried"),
    ]
    for col, values in zip(st.columns(5), metrics):
        with col:
            card(*values)

    st.write("")
    st.markdown('<div class="panel"><div class="panel-title">End-to-End Data Pipeline</div>', unsafe_allow_html=True)
    p1, arrow1, p2, arrow2, p3, arrow3, p4 = st.columns([2, .45, 2, .45, 2, .45, 2])
    p1.info("**1. Data Sources**\n\nCSV / Faker-generated leads, activities, opportunities, and campaign touches")
    arrow1.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p2.success("**2. Bronze**\n\nRaw PostgreSQL tables\n\nAs-ingested data")
    arrow2.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p3.warning("**3. Silver**\n\nCleaned and validated tables\n\nQuality checks applied")
    arrow3.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p4.success("**4. Gold**\n\nFact, dimension, and lead-summary tables\n\nDashboard-ready data")
    st.markdown('</div>', unsafe_allow_html=True)

    left, right = st.columns([1.65, 1])
    with left:
        st.markdown('<div class="panel"><div class="panel-title">Pipeline Table Inventory</div>', unsafe_allow_html=True)
        fig = px.bar(row_counts, x="table_name", y="row_count", color="layer", barmode="group",
                     color_discrete_map={"Bronze": "#a66a2c", "Silver": "#8b929b", "Gold": "#d7a600"},
                     labels={"table_name": "PostgreSQL Table", "row_count": "Rows", "layer": "Layer"})
        fig.update_layout(height=360, margin=dict(l=5, r=5, t=10, b=70), plot_bgcolor="white", paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(row_counts, hide_index=True, use_container_width=True, height=280)
        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel"><div class="panel-title">Data Quality & Governance</div>', unsafe_allow_html=True)
        quality_data = pd.DataFrame({
            "Check": ["Missing Lead IDs", "Missing Company Names", "Missing Regions",
                      "Missing Statuses", "Negative Deal Values"],
            "Issues": [quality_row["missing_lead_ids"], quality_row["missing_company_names"],
                       quality_row["missing_regions"], quality_row["missing_statuses"],
                       quality_row["negative_deal_values"]],
        })
        st.dataframe(quality_data, hide_index=True, use_container_width=True)
        if issues == 0:
            st.success("All core data-quality checks passed.")
        else:
            st.warning(f"{issues:,} quality issue(s) need review.")
        st.caption(f"Lead summary records checked: {quality_row['total_leads']:,}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">ETL Runbook</div>', unsafe_allow_html=True)
        st.markdown("""
        1. Generate or receive raw data files.  
        2. Load them into Bronze (`raw_*`) tables.  
        3. Run `clean_bronze_to_silver.py`.  
        4. Build Gold dimensions, facts, and `lead_summary`.  
        5. Refresh this dashboard.  
        """)
        st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard(source_df: pd.DataFrame):
    st.markdown('<div class="top-title">End-to-End Decision Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">A single view from raw data ingestion through data quality, customer outcomes, risk signals, and campaign intervention results.</div>', unsafe_allow_html=True)

    try:
        row_counts, quality = load_etl_metrics()
        quality_row = quality.iloc[0]
    except Exception as error:
        st.error(f"Could not load pipeline metrics: {error}")
        return

    total_leads = len(source_df)
    converted = (source_df["status"] == "Converted").sum()
    conversion_rate = converted / total_leads * 100 if total_leads else 0
    won = source_df[source_df["opportunity_stage"] == "Closed-Won"]
    won_revenue = won["deal_value"].sum()
    at_risk = source_df["status"].isin(["Lost", "Unqualified"]).sum()
    risk_df = _compute_proxy_risk_scores(source_df)
    high_risk = (risk_df["risk_tier"] == "High").sum()
    quality_issues = (quality_row["missing_lead_ids"] + quality_row["missing_company_names"]
                      + quality_row["missing_regions"] + quality_row["missing_statuses"]
                      + quality_row["negative_deal_values"])

    metrics = [
        ("👥", "Total Leads", f"{total_leads:,}", "Gold lead-summary view"),
        ("🎯", "Conversion Rate", f"{conversion_rate:.2f}%", f"{converted:,} converted leads"),
        ("💰", "Closed-Won Revenue", f"${won_revenue:,.0f}", "Realized deal value"),
        ("⚠️", "At-Risk Leads", f"{at_risk:,}", "Lost or unqualified"),
        ("🛡️", "Data Quality", "Passed" if quality_issues == 0 else f"{quality_issues:,} issues", "Core data checks"),
    ]
    for col, values in zip(st.columns(5), metrics):
        with col:
            card(*values)

    st.write("")
    st.markdown('<div class="panel"><div class="panel-title">End-to-End Data Flow</div>', unsafe_allow_html=True)
    p1, a1, p2, a2, p3, a3, p4, a4, p5 = st.columns([1.8, .3, 1.8, .3, 1.8, .3, 1.8, .3, 1.8])
    p1.info("**Sources**\n\nFaker / CSV inputs\n\nLeads, activities, opportunities, campaigns")
    a1.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p2.info("**Bronze**\n\nRaw PostgreSQL tables\n\nAs-ingested records")
    a2.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p3.warning("**Silver**\n\nClean and validate\n\nQuality checks")
    a3.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p4.success("**Gold**\n\nFacts, dimensions\n\nLead summary")
    a4.markdown("<h2 style='text-align:center; padding-top:35px;'>→</h2>", unsafe_allow_html=True)
    p5.success("**Decisions**\n\nCustomer, risk,\n\nand intervention insights")
    st.markdown('</div>', unsafe_allow_html=True)

    main, side = st.columns([3.25, 1.15])
    with main:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="panel"><div class="panel-title">Pipeline Health — Rows by Layer</div>', unsafe_allow_html=True)
            fig = px.bar(row_counts, x="table_name", y="row_count", color="layer",
                         color_discrete_map={"Bronze": "#a66a2c", "Silver": "#8b929b", "Gold": "#d7a600"},
                         labels={"table_name": "Table", "row_count": "Rows", "layer": "Layer"})
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=70), plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="panel"><div class="panel-title">Campaign Intervention Impact</div>', unsafe_allow_html=True)
            campaign = source_df[(source_df["received_campaign"] == True) & source_df["treatment_group"].isin(["Control", "Treatment"])]
            campaign = campaign.groupby("treatment_group", as_index=False).agg(
                leads=("lead_id", "count"),
                conversions=("status", lambda value: (value == "Converted").sum()))
            campaign["conversion_rate"] = campaign["conversions"] / campaign["leads"] * 100
            fig = px.bar(campaign, x="treatment_group", y="conversion_rate", text="conversion_rate",
                         color="treatment_group",
                         color_discrete_map={"Control": "#3478df", "Treatment": "#1eb27b"},
                         labels={"treatment_group": "Campaign Group", "conversion_rate": "Conversion Rate (%)"})
            fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                              showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown('<div class="panel"><div class="panel-title">Risk Proxy by Industry</div>', unsafe_allow_html=True)
            risk_by_industry = (risk_df.groupby("industry", as_index=False)
                                .agg(avg_risk_score=("risk_score", "mean"))
                                .sort_values("avg_risk_score", ascending=False).head(8))
            fig = px.bar(risk_by_industry, x="avg_risk_score", y="industry", orientation="h",
                         color="avg_risk_score", color_continuous_scale="Reds",
                         labels={"avg_risk_score": "Average Risk Score", "industry": "Industry"})
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                              coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with c4:
            st.markdown('<div class="panel"><div class="panel-title">Customer Conversion by Lead Source</div>', unsafe_allow_html=True)
            source = source_df.groupby("lead_source", as_index=False).agg(
                leads=("lead_id", "count"),
                conversions=("status", lambda value: (value == "Converted").sum()))
            source["conversion_rate"] = source["conversions"] / source["leads"] * 100
            fig = px.bar(source.sort_values("conversion_rate", ascending=False),
                         x="lead_source", y="conversion_rate",
                         color="conversion_rate", color_continuous_scale=["#b9d8ff", "#1769e0"],
                         labels={"lead_source": "Lead Source", "conversion_rate": "Conversion Rate (%)"})
            fig.update_layout(height=320, margin=dict(l=5, r=5, t=10, b=5),
                              coloraxis_showscale=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="panel"><div class="panel-title">Priority Customers — Highest Risk Signals</div>', unsafe_allow_html=True)
        priority = risk_df.sort_values("risk_score", ascending=False).head(15)
        st.dataframe(priority[["lead_id", "company_name", "industry", "region", "status",
                               "total_activities", "risk_score", "risk_tier", "expected_loss"]]
                     .style.format({"risk_score": "{:.2f}", "expected_loss": "${:,.0f}"}),
                     hide_index=True, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with side:
        st.markdown('<div class="panel"><div class="panel-title">💡 Executive Insights</div>', unsafe_allow_html=True)
        bronze = row_counts.loc[row_counts["layer"] == "Bronze", "row_count"].sum()
        silver = row_counts.loc[row_counts["layer"] == "Silver", "row_count"].sum()
        source_rates = source.sort_values("conversion_rate", ascending=False)
        best_source = source_rates.iloc[0] if not source_rates.empty else None
        intervention_text = "No comparison available."
        if len(campaign) == 2:
            treatment_rate = campaign.loc[campaign["treatment_group"] == "Treatment", "conversion_rate"].iloc[0]
            control_rate = campaign.loc[campaign["treatment_group"] == "Control", "conversion_rate"].iloc[0]
            intervention_text = f"Treatment changed conversion by {treatment_rate - control_rate:+.2f} percentage points."
        best_text = f"{best_source['lead_source']} is strongest at {best_source['conversion_rate']:.2f}% conversion." if best_source is not None else "No conversion source data available."
        st.markdown(f"""
        <div class="insight"><div class="insight-title">🗄️ Pipeline health</div><div class="insight-text">{bronze:,} raw records flow through {silver:,} validated Silver records.</div></div>
        <div class="insight"><div class="insight-title">🛡️ Data quality</div><div class="insight-text">{'All core checks passed.' if quality_issues == 0 else f'{quality_issues:,} issue(s) require review.'}</div></div>
        <div class="insight"><div class="insight-title">🎯 Best customer segment</div><div class="insight-text">{best_text}</div></div>
        <div class="insight"><div class="insight-title">🧪 Intervention outcome</div><div class="insight-text">{intervention_text}</div></div>
        <div class="insight"><div class="insight-title">⚠️ Risk priority</div><div class="insight-text">{high_risk:,} customers are classified High Risk by the transparent proxy score.</div></div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# Sidebar
with st.sidebar:
    st.markdown("## 〽 CreditPulse")
    st.caption("Predict • Prioritize • Drive Value")
    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠  Overview",
            "👥  Customer Analytics",
            "⚗️  Risk Analysis",
            "🧪  Interventions",
            "🧠  ML Models",
            "📈  Survival Analysis",
            "💎  CLV",
            "🎯  Uplift",
            "🗄️  Data & ETL",
            "📊  Dashboard",
            "⚙️  Settings"
        ],
        label_visibility="collapsed"
    )

    st.divider()
    st.caption("End-to-End Credit Risk &")
    st.caption("Customer Value Analytics Platform")

if page == "👥  Customer Analytics":
    render_customer_analytics(df)
    st.stop()

if page == "⚗️  Risk Analysis":
    render_risk_analysis(df)
    st.stop()

if page == "🧪  Interventions":
    render_interventions(df)
    st.stop()

if page == "🧠  ML Models":
    render_ml_models()
    st.stop()

if page == "📈  Survival Analysis":
    render_survival_page()
    st.stop()

if page == "💎  CLV":
    render_clv_page()
    st.stop()

if page == "🎯  Uplift":
    render_uplift_page()
    st.stop()

if page == "🗄️  Data & ETL":
    render_data_etl()
    st.stop()

if page == "📊  Dashboard":
    render_dashboard(df)
    st.stop()


# Top toolbar
toolbar_1, toolbar_2, toolbar_3 = st.columns([4, 2, 1])

with toolbar_2:
    search_text = st.text_input("Search",
                                placeholder="Search customer, account, or transaction...",
                                label_visibility="collapsed")

with toolbar_3:
    st.selectbox("Period",
                 ["Jan 2025 – Dec 2026", "Last 12 Months", "All Time"],
                 label_visibility="collapsed")

if search_text:
    search_mask = (
        df["lead_id"].str.contains(search_text, case=False, na=False) |
        df["company_name"].str.contains(search_text, case=False, na=False)
    )
    df = df[search_mask]

st.markdown('<div class="top-title">CreditPulse</div>', unsafe_allow_html=True)
st.markdown('<div class="top-subtitle">End-to-End Credit Risk & Customer Value Analytics Platform</div>',
            unsafe_allow_html=True)
st.markdown('<div class="description">Transforming raw data into actionable insights for smarter credit decisions.</div>',
            unsafe_allow_html=True)

total_leads = len(df)
converted = (df["status"] == "Converted").sum()
conversion_rate = converted / total_leads * 100 if total_leads else 0
at_risk = df["status"].isin(["Lost", "Unqualified"]).sum()
at_risk_rate = at_risk / total_leads * 100 if total_leads else 0
won = df[df["opportunity_stage"] == "Closed-Won"]
won_revenue = won["deal_value"].sum()
avg_deal = won["deal_value"].mean() if len(won) else 0
pipeline = df["deal_value"].sum()

cards = [
    ("👥", "Total Customers", f"{total_leads:,}", "Active lead portfolio"),
    ("💼", "Total Exposure", f"${pipeline:,.0f}", "Pipeline value"),
    ("⚠️", "At Risk Customers", f"{at_risk:,} ({at_risk_rate:.1f}%)", "Lost or unqualified"),
    ("📈", "Avg. Deal Value", f"${avg_deal:,.0f}", "Closed-won deals"),
    ("🛡️", "Portfolio Health", f"{conversion_rate:.1f}%", "Lead conversion rate"),
]

for col, (icon, label, value, note) in zip(st.columns(5), cards):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">{icon} &nbsp; {label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">● {note}</div>
        </div>
        """, unsafe_allow_html=True)

st.write("")

main_col, insights_col = st.columns([3.25, 1.15])

with main_col:
    st.markdown('<div class="panel"><div class="panel-title">Risk Dashboard (Streamlit)</div>', unsafe_allow_html=True)

    f1, f2, f3, f4 = st.columns(4)
    regions = f1.multiselect("Time Period / Region", sorted(df["region"].dropna().unique()))
    sources = f2.multiselect("Customer Segment / Source", sorted(df["lead_source"].dropna().unique()))
    industries = f3.multiselect("Industry", sorted(df["industry"].dropna().unique()))
    groups = f4.multiselect("Risk Tier / Campaign", sorted(df["treatment_group"].dropna().unique()))

    filtered = df.copy()
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    if sources:
        filtered = filtered[filtered["lead_source"].isin(sources)]
    if industries:
        filtered = filtered[filtered["industry"].isin(industries)]
    if groups:
        filtered = filtered[filtered["treatment_group"].isin(groups)]

    filtered["month"] = filtered["created_date"].dt.strftime("%b %Y")
    monthly = (filtered.groupby("month", as_index=False)
               .agg(leads=("lead_id", "count"),
                    converted=("status", lambda x: (x == "Converted").sum())))
    monthly["conversion_rate"] = monthly["converted"] / monthly["leads"] * 100
    monthly = monthly.tail(12)

    chart1, chart2 = st.columns([1.65, 1])
    with chart1:
        st.markdown('<div class="panel-title">Customer Risk Vs. Portfolio Trend</div>', unsafe_allow_html=True)
        fig = px.bar(monthly, x="month", y="leads", color_discrete_sequence=["#2ab17d"],
                     labels={"month": "", "leads": "Leads"})
        fig.add_scatter(x=monthly["month"], y=monthly["conversion_rate"],
                        mode="lines+markers", name="Conversion rate (%)", yaxis="y2",
                        line=dict(color="#1769e0", width=3))
        fig.update_layout(height=310, margin=dict(l=5, r=5, t=10, b=5),
                          plot_bgcolor="white", paper_bgcolor="white",
                          yaxis2=dict(overlaying="y", side="right", title="Conversion %"),
                          legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        st.markdown('<div class="panel-title">Risk Distribution</div>', unsafe_allow_html=True)
        status_data = filtered.groupby("status", as_index=False).agg(leads=("lead_id", "count"))
        fig = px.pie(status_data, values="leads", names="status", hole=.62,
                     color_discrete_sequence=["#1eb27b", "#ffc21a", "#ef5350", "#3478df", "#865fd3"])
        fig.update_layout(height=310, margin=dict(l=5, r=5, t=10, b=5), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    chart3, chart4, chart5 = st.columns(3)
    with chart3:
        st.markdown('<div class="panel-title">Top Risk Segments</div>', unsafe_allow_html=True)
        segment_data = (filtered.groupby("industry", as_index=False)
                        .agg(leads=("lead_id", "count"))
                        .sort_values("leads", ascending=False).head(5))
        fig = px.bar(segment_data, x="leads", y="industry", orientation="h",
                     color="leads", color_continuous_scale="Blues")
        fig.update_layout(height=270, margin=dict(l=5, r=5, t=10, b=5), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with chart4:
        st.markdown('<div class="panel-title">Customer Lifetime Value (CLV)</div>', unsafe_allow_html=True)
        region_data = (filtered.groupby("region", as_index=False)
                       .agg(avg_value=("deal_value", "mean"))
                       .sort_values("avg_value", ascending=False))
        fig = px.line(region_data, x="region", y="avg_value", markers=True,
                      color_discrete_sequence=["#1769e0"])
        fig.update_layout(height=270, margin=dict(l=5, r=5, t=10, b=5), yaxis_title="Avg. Deal Value")
        st.plotly_chart(fig, use_container_width=True)

    with chart5:
        st.markdown('<div class="panel-title">Intervention Impact</div>', unsafe_allow_html=True)
        campaign_data = (filtered[filtered["received_campaign"] == True]
                         .groupby("treatment_group", as_index=False)
                         .agg(leads=("lead_id", "count"),
                              converted=("status", lambda x: (x == "Converted").sum())))
        campaign_data["conversion_rate"] = campaign_data["converted"] / campaign_data["leads"] * 100
        fig = px.bar(campaign_data, x="treatment_group", y="conversion_rate", color="treatment_group",
                     color_discrete_map={"Control": "#3478df", "Treatment": "#1eb27b"})
        fig.update_layout(height=270, margin=dict(l=5, r=5, t=10, b=5), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="panel-title">Recent Customers</div>', unsafe_allow_html=True)
    st.dataframe(filtered[["lead_id", "company_name", "industry", "region",
                           "status", "opportunity_stage", "deal_value"]].head(10),
                 hide_index=True, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with insights_col:
    st.markdown('<div class="panel"><div class="panel-title">💡 Key Insights</div>', unsafe_allow_html=True)
    best_source_data = (filtered.groupby("lead_source", as_index=False)
                        .agg(leads=("lead_id", "count"),
                             converted=("status", lambda x: (x == "Converted").sum())))
    best_source_data["rate"] = best_source_data["converted"] / best_source_data["leads"] * 100
    best_source = best_source_data.sort_values("rate", ascending=False).iloc[0]

    st.markdown(f"""
    <div class="insight">
        <div class="insight-title">📈 Customer portfolio overview</div>
        <div class="insight-text">{total_leads:,} leads are currently included in the selected portfolio.</div>
    </div>
    <div class="insight">
        <div class="insight-title">⚠️ High-risk segment needs attention</div>
        <div class="insight-text">{at_risk_rate:.1f}% of leads are lost or unqualified.</div>
    </div>
    <div class="insight">
        <div class="insight-title">🎯 Best converting source</div>
        <div class="insight-text">{best_source["lead_source"]} leads convert at {best_source["rate"]:.2f}%.</div>
    </div>
    <div class="insight">
        <div class="insight-title">💰 Portfolio revenue</div>
        <div class="insight-text">${won_revenue:,.0f} has been generated from closed-won opportunities.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">🧠 Model Performance</div>', unsafe_allow_html=True)
    _meta = load_model_metadata()
    if _meta:
        _m = _meta["best_model_metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC (ROC)", f"{_m['AUC']:.2f}")
        c2.metric("Accuracy", f"{_m['Accuracy']:.2f}")
        c3.metric("F1 Score", f"{_m['F1']:.2f}")
        if _m["AUC"] >= 0.70:
            st.success("Model healthy")
        else:
            st.warning("Model needs improvement — see ML Models page")
    else:
        st.info("Train the model to populate this card.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="panel"><div class="panel-title">Quick Actions</div>', unsafe_allow_html=True)
    a1, a2 = st.columns(2)
    a1.button("Run Risk Analysis", use_container_width=True)
    a2.button("Run Intervention", use_container_width=True)
    a1, a2 = st.columns(2)
    a1.button("Upload Data", use_container_width=True)
    a2.button("View Full Dashboard", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)