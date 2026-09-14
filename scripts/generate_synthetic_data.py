import os
import numpy as np
import pandas as pd
from faker import Faker
import random
from datetime import timedelta

# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
SEED = 42

START_DATE = pd.Timestamp("2010-01-01")
AS_OF_DATE = pd.Timestamp("2026-09-06")
END_DATE = AS_OF_DATE

np.random.seed(SEED)
random.seed(SEED)

fake = Faker()
Faker.seed(SEED)

INDUSTRIES = ["SaaS", "Manufacturing", "Retail", "Healthcare", "FinServ", "Education", "Logistics"]
LEAD_SOURCES = ["Website", "Referral", "Cold Call", "LinkedIn", "Trade Show", "Webinar"]
JOB_TITLES = ["VP Sales", "Marketing Manager", "CEO", "Procurement Lead", "Operations Director", "IT Manager"]
STAGES = ["Prospecting", "Qualification", "Proposal", "Negotiation", "Closed-Won", "Closed-Lost"]
ACTIVITY_TYPES = ["Email", "Call", "Meeting", "Demo", "Website Visit"]
CHANNELS = ["Email", "Phone", "LinkedIn", "In-person"]
OUTCOMES = ["Positive", "Neutral", "Negative", "No Response"]
RESPONSES = ["Opened", "Clicked", "Replied", "Unsubscribed", "No Action"]

def random_date(start, end):
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))

COUNTRY_LOCALE_REGION = {
    # ---- North America ----
    "United States":            ("en_US", "North America"),
    "Canada":                   ("en_CA", "North America"),

    # ---- Europe ----
    "United Kingdom":           ("en_GB", "Europe"),
    "Ireland":                  ("en_IE", "Europe"),
    "Germany":                  ("de_DE", "Europe"),
    "Austria":                  ("de_AT", "Europe"),
    "Switzerland":              ("de_CH", "Europe"),
    "France":                   ("fr_FR", "Europe"),
    "Belgium":                  ("fr_BE", "Europe"),
    "Netherlands":              ("nl_NL", "Europe"),
    "Luxembourg":               ("lb_LU", "Europe"),
    "Spain":                    ("es_ES", "Europe"),
    "Portugal":                 ("pt_PT", "Europe"),
    "Italy":                    ("it_IT", "Europe"),
    "Sweden":                   ("sv_SE", "Europe"),
    "Norway":                   ("no_NO", "Europe"),
    "Denmark":                  ("da_DK", "Europe"),
    "Finland":                  ("fi_FI", "Europe"),
    "Poland":                   ("pl_PL", "Europe"),
    "Czech Republic":           ("cs_CZ", "Europe"),
    "Slovakia":                 ("sk_SK", "Europe"),
    "Hungary":                  ("hu_HU", "Europe"),
    "Romania":                  ("ro_RO", "Europe"),
    "Bulgaria":                 ("bg_BG", "Europe"),
    "Greece":                   ("el_GR", "Europe"),
    "Croatia":                  ("hr_HR", "Europe"),
    "Slovenia":                 ("sl_SI", "Europe"),
    "Bosnia and Herzegovina":   ("bs_BA", "Europe"),
    "Albania":                  ("sq_AL", "Europe"),
    "Estonia":                  ("et_EE", "Europe"),
    "Latvia":                   ("lv_LV", "Europe"),
    "Lithuania":                ("lt_LT", "Europe"),
    "Ukraine":                  ("uk_UA", "Europe"),
    "Russia":                   ("ru_RU", "Europe"),
    "Malta":                    ("mt_MT", "Europe"),

    # ---- APAC ----
    "India":                    ("en_IN", "APAC"),
    "China":                    ("zh_CN", "APAC"),
    "Taiwan":                   ("zh_TW", "APAC"),
    "Japan":                    ("ja_JP", "APAC"),
    "South Korea":              ("ko_KR", "APAC"),
    "Singapore":                ("en_US", "APAC"),
    "Australia":                ("en_AU", "APAC"),
    "New Zealand":              ("en_NZ", "APAC"),
    "Indonesia":                ("id_ID", "APAC"),
    "Thailand":                 ("th_TH", "APAC"),
    "Vietnam":                  ("vi_VN", "APAC"),
    "Philippines":              ("en_PH", "APAC"),
    "Malaysia":                 ("en_MS", "APAC"),
    "Bangladesh":               ("bn_BD", "APAC"),
    "Pakistan":                 ("en_PK", "APAC"),
    "Nepal":                    ("ne_NP", "APAC"),
    "Kazakhstan":               ("ru_RU", "APAC"),
    "Uzbekistan":               ("uz_UZ", "APAC"),

    # ---- LATAM ----
    "Brazil":                   ("pt_BR", "LATAM"),
    "Mexico":                   ("es_MX", "LATAM"),
    "Argentina":                ("es_AR", "LATAM"),
    "Chile":                    ("es_CL", "LATAM"),
    "Colombia":                 ("es_CO", "LATAM"),
    "Peru":                     ("es_ES", "LATAM"),
    "Venezuela":                ("es_ES", "LATAM"),

    # ---- EMEA (Middle East & Africa) ----
    "United Arab Emirates":     ("ar_AA", "EMEA"),
    "Saudi Arabia":             ("ar_SA", "EMEA"),
    "Egypt":                    ("ar_EG", "EMEA"),
    "Jordan":                   ("ar_JO", "EMEA"),
    "Palestine":                ("ar_PS", "EMEA"),
    "Israel":                   ("he_IL", "EMEA"),
    "Turkey":                   ("tr_TR", "EMEA"),
    "Iran":                     ("fa_IR", "EMEA"),
    "Azerbaijan":               ("az_AZ", "EMEA"),
    "Armenia":                  ("hy_AM", "EMEA"),
    "Georgia":                  ("ka_GE", "EMEA"),
    "South Africa":             ("zu_ZA", "EMEA"),
    "Nigeria":                  ("yo_NG", "EMEA"),
    "Ghana":                    ("tw_GH", "EMEA"),
}

COUNTRIES = list(COUNTRY_LOCALE_REGION.keys())

_locale_fakers = {}
def faker_for_locale(locale):
    if locale not in _locale_fakers:
        f = Faker(locale)
        Faker.seed(SEED)
        _locale_fakers[locale] = f
    return _locale_fakers[locale]

# ---------------------------------------------------------
# CORPORATE-STYLE COMPANY NAME GENERATOR
# ---------------------------------------------------------
COMPANY_PREFIXES = ["Nex", "Blue", "Orbit", "Sky", "Vertex", "Prime", "Nova", "Zenith",
                    "Quantum", "Bright", "Silver", "Apex", "Northstar", "Crest", "Delta",
                    "Pioneer", "Summit", "Cobalt", "Lumen", "Vantage"]
COMPANY_ROOTS = ["ify", "wave", "core", "logic", "sphere", "path", "gate", "link",
                 "forge", "shift", "grid", "loop", "trail", "beam", "cove"]
COMPANY_SUFFIXES = ["Technologies", "Solutions", "Group", "Systems", "Industries",
                    "Global", "Dynamics", "Ventures", "Labs", "Networks",
                    "Enterprises", "Innovations", "Partners"]

def generate_company_name():
    style = random.random()
    if style < 0.6:
        return f"{random.choice(COMPANY_PREFIXES)}{random.choice(COMPANY_ROOTS)} {random.choice(COMPANY_SUFFIXES)}"
    else:
        return f"{random.choice(COMPANY_PREFIXES)}{random.choice(COMPANY_ROOTS).capitalize()} {random.choice(COMPANY_SUFFIXES)}"

# ---------------------------------------------------------
# SIGNAL WEIGHTS  (the actual fix — makes the ML problem learnable)
# ---------------------------------------------------------
LEAD_SOURCE_WEIGHT = {
    "Referral":   1.20,
    "Webinar":    0.75,
    "Trade Show": 0.60,
    "LinkedIn":   0.40,
    "Website":    0.10,
    "Cold Call": -0.80,
}

INDUSTRY_WEIGHT = {
    "SaaS":          +0.60,
    "FinServ":       +0.35,
    "Healthcare":    +0.15,
    "Manufacturing": -0.05,
    "Logistics":     -0.20,
    "Retail":        -0.35,
    "Education":     -0.50,
}

JOB_TITLE_WEIGHT = {
    "CEO":                +0.55,
    "VP Sales":           +0.45,
    "Operations Director": +0.20,
    "IT Manager":         +0.05,
    "Marketing Manager":  -0.15,
    "Procurement Lead":   -0.35,
}

REGION_WEIGHT = {
    "North America": +0.20,
    "Europe":        +0.10,
    "APAC":           0.00,
    "LATAM":         -0.10,
    "EMEA":          -0.15,
}

CAMPAIGN_RESPONSE_WEIGHT = {
    "Replied":      +1.00,
    "Clicked":      +0.40,
    "Opened":       +0.05,
    "No Action":    -0.20,
    "Unsubscribed": -1.00,
}


def logistic(x):
    # clip to avoid overflow warnings on extreme logits
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))

# ---------------------------------------------------------
# 1. raw_leads
# ---------------------------------------------------------
def get_daily_lead_volume(date):
    year = date.year
    if year <= 2012:
        base_volume = 2
    elif year <= 2015:
        base_volume = 4
    elif year <= 2018:
        base_volume = 7
    elif year <= 2020:
        base_volume = 10
    elif year <= 2022:
        base_volume = 14
    elif year <= 2024:
        base_volume = 20
    elif year == 2025:
        base_volume = 27
    else:
        base_volume = 32

    if date.weekday() >= 5:
        base_volume = int(base_volume * 0.45)

    variation = np.random.uniform(0.75, 1.25)
    return max(1, int(base_volume * variation))


leads = []
lead_counter = 10001
current_date = START_DATE

while current_date <= END_DATE:
    daily_volume = get_daily_lead_volume(current_date)

    for _ in range(daily_volume):
        lead_id = f"L{lead_counter}"

        company_size = int(np.random.lognormal(mean=4.5, sigma=1.2))
        company_size = max(5, min(company_size, 50000))

        source = random.choice(LEAD_SOURCES)
        country = random.choice(COUNTRIES)
        locale, macro_region = COUNTRY_LOCALE_REGION[country]
        locale_fake = faker_for_locale(locale)

        leads.append({
            "lead_id": lead_id,
            "contact_name": locale_fake.name(),
            "company_name": generate_company_name(),
            "industry": random.choice(INDUSTRIES),
            "company_size": company_size,
            "lead_source": source,
            "job_title": random.choice(JOB_TITLES),
            "country": country,
            "region": macro_region,
            "status": None,
            "created_date": current_date,
        })

        lead_counter += 1

    current_date += timedelta(days=1)


df_leads = pd.DataFrame(leads)
print(f"Generated {len(df_leads):,} leads from {START_DATE.date()} to {END_DATE.date()}")


# ---------------------------------------------------------
# 2. raw_activities (event log)
# ---------------------------------------------------------
activities = []
act_counter = 30001

for _, lead in df_leads.iterrows():
    n_acts = np.random.poisson(4) + 1
    for _ in range(n_acts):
        act_date = lead["created_date"] + timedelta(days=random.randint(0, 120))
        if act_date > END_DATE:
            continue
        activities.append({
            "activity_id": f"T{act_counter}",
            "lead_id": lead["lead_id"],
            "activity_date": act_date,
            "activity_type": random.choice(ACTIVITY_TYPES),
            "channel": random.choice(CHANNELS),
            "outcome": random.choices(OUTCOMES, weights=[0.3, 0.3, 0.2, 0.2])[0],
        })
        act_counter += 1

df_activities = pd.DataFrame(activities)
print(f"Generated {len(df_activities):,} activities")


# ---------------------------------------------------------
# 3. COMPUTE BASE CONVERSION PROBABILITY  (moved up: needs to run before
#    campaign targeting so that targeting can be quality-aware)
# ---------------------------------------------------------
print("Computing base conversion probabilities (pre-campaign)...")


def compute_base_p_convert(row):
    """
    Compute base P(convert) from lead-intrinsic features only.
    The campaign response effect is added later (after targeting),
    because a lead's campaign response depends on whether it was targeted.
    """
    logit = -1.2

    logit += LEAD_SOURCE_WEIGHT.get(row["lead_source"], 0.0)
    logit += INDUSTRY_WEIGHT.get(row["industry"], 0.0)
    logit += JOB_TITLE_WEIGHT.get(row["job_title"], 0.0)
    logit += REGION_WEIGHT.get(row["region"], 0.0)

    # Interaction: high-intent source × high-fit industry
    if row["lead_source"] == "Referral" and row["industry"] in ("SaaS", "FinServ"):
        logit += 0.60
    if row["lead_source"] == "Cold Call" and row["industry"] in ("Education", "Retail"):
        logit -= 0.40

    # Company size sweet spot (peak around 150 employees)
    cs = row["company_size"] if pd.notnull(row["company_size"]) else 150
    try:
        cs = float(cs)
    except (TypeError, ValueError):
        cs = 150.0
    logit += 0.004 * (cs - 150) - 0.00002 * (cs - 150) ** 2

    # Residual noise
    logit += np.random.normal(0, 0.25)

    return float(logistic(logit))


df_leads["_base_p_convert"] = df_leads.apply(compute_base_p_convert, axis=1)


# ---------------------------------------------------------
# 4. raw_campaign_touches  (quality-aware targeting)
# ---------------------------------------------------------
# Key change vs. v1: campaign targeting is no longer a random 50% sample.
# High-quality leads (per their base _p_convert) are much more likely to be
# targeted, mirroring how real marketing spend follows promising leads.
# This turns `received_campaign` and `treatment_group` from pure noise
# into real predictors the downstream model can learn from.
# ---------------------------------------------------------
touches = []
touch_counter = 40001

# Targeting probability scales with lead quality.
# _base_p_convert = 0.10 -> ~18% targeted
# _base_p_convert = 0.30 -> ~50% targeted
# _base_p_convert = 0.60 -> ~83% targeted
targeting_logit = (df_leads["_base_p_convert"] - 0.30) * 6.0
targeting_prob = 1.0 / (1.0 + np.exp(-targeting_logit))
df_leads["_campaign_targeted"] = np.random.rand(len(df_leads)) < targeting_prob

campaign_leads = df_leads[df_leads["_campaign_targeted"]]

for _, lead in campaign_leads.iterrows():
    treatment = random.choice(["Treatment", "Control"])
    touch_date = lead["created_date"] + timedelta(days=random.randint(5, 30))
    if touch_date > END_DATE:
        continue

    if treatment == "Treatment":
        response = random.choices(RESPONSES, weights=[0.35, 0.25, 0.15, 0.05, 0.20])[0]
    else:
        response = random.choices(RESPONSES, weights=[0.20, 0.10, 0.05, 0.05, 0.60])[0]

    touches.append({
        "touch_id": f"P{touch_counter}",
        "lead_id": lead["lead_id"],
        "campaign_id": "CMP001",
        "touch_date": touch_date,
        "response": response,
        "treatment_group": treatment,
    })
    touch_counter += 1

df_touches = pd.DataFrame(touches)
print(f"Generated {len(df_touches):,} campaign touches "
      f"({df_leads['_campaign_targeted'].mean():.1%} of leads targeted)")


# ---------------------------------------------------------
# 5. FINALIZE CONVERSION PROBABILITY  (add campaign response effect +
#    activity-count effect, then sample event_converted)
# ---------------------------------------------------------
print("Finalizing conversion probabilities (adding campaign + activity effects)...")

# Roll up activity count per lead
activity_counts = df_activities.groupby("lead_id").size().rename("total_activities")
df_leads = df_leads.merge(activity_counts, on="lead_id", how="left")
df_leads["total_activities"] = df_leads["total_activities"].fillna(0).astype(int)

# Merge campaign response
campaign_info = df_touches.set_index("lead_id")[["response", "treatment_group"]]
df_leads = df_leads.merge(
    campaign_info.rename(columns={"response": "campaign_response"}),
    on="lead_id", how="left",
)
df_leads["received_campaign"] = df_leads["campaign_response"].notnull()
df_leads["campaign_response"] = df_leads["campaign_response"].fillna("No Action")


def finalize_p_convert(row):
    """
    Take the base p_convert (already stored) and add the two remaining
    effects that were not available at base-time:
      - Campaign response effect (only when the lead was targeted)
      - Activity-count sweet-spot effect
    Then apply the same logit transform to get the final probability.
    """
    # Convert base probability back to logit
    base_p = float(row["_base_p_convert"])
    base_p = min(max(base_p, 1e-6), 1 - 1e-6)
    logit = np.log(base_p / (1 - base_p))

    # Engagement sweet spot at ~7 activities
    logit -= 0.15 * abs(row["total_activities"] - 7)

    # Campaign response effect (only if the lead was in a campaign)
    if row["received_campaign"]:
        logit += CAMPAIGN_RESPONSE_WEIGHT.get(row["campaign_response"], 0.0)

    return float(logistic(logit))


df_leads["_p_convert"] = df_leads.apply(finalize_p_convert, axis=1)
df_leads["event_converted"] = (np.random.rand(len(df_leads)) < df_leads["_p_convert"]).astype(int)

print(f"  Overall conversion rate: {df_leads['event_converted'].mean():.1%}")
print("  By lead_source:")
print(df_leads.groupby("lead_source")["event_converted"].mean().round(3).sort_values(ascending=False).to_string())
print("  By industry:")
print(df_leads.groupby("industry")["event_converted"].mean().round(3).sort_values(ascending=False).to_string())
print("  By campaign status:")
print(df_leads.groupby("received_campaign")["event_converted"].mean().round(3).to_string())


# ---------------------------------------------------------
# 6. raw_opportunities — driven by event_converted
# ---------------------------------------------------------
opportunities = []
opp_counter = 20001

for _, lead in df_leads.iterrows():
    if random.random() >= 0.6:
        continue  # not every lead becomes an opportunity

    deal_value = int(lead["company_size"] * np.random.uniform(50, 300))
    open_date = lead["created_date"] + timedelta(days=random.randint(1, 20))
    expected_close = open_date + timedelta(days=random.randint(20, 90))

    if open_date + timedelta(days=90) > END_DATE:
        stage = random.choice(["Prospecting", "Qualification", "Proposal", "Negotiation"])
        actual_close = None
    elif lead["event_converted"] == 1:
        stage = "Closed-Won"
        actual_close = expected_close + timedelta(days=random.randint(-10, 20))
    else:
        stage = "Closed-Lost"
        actual_close = expected_close + timedelta(days=random.randint(-5, 30))

    opportunities.append({
        "opportunity_id": f"O{opp_counter}",
        "lead_id": lead["lead_id"],
        "deal_value": deal_value,
        "stage": stage,
        "owner": fake.name(),
        "open_date": open_date,
        "expected_close_date": expected_close,
        "actual_close_date": actual_close,
    })
    opp_counter += 1

df_opps = pd.DataFrame(opportunities)
print(f"Generated {len(df_opps):,} opportunities")


# ---------------------------------------------------------
# 7. DERIVE lead STATUS from event_converted + opportunity stage
# ---------------------------------------------------------
status_map = df_opps.set_index("lead_id")["stage"].to_dict()

def derive_status(row):
    lead_id = row["lead_id"]
    if row["event_converted"] == 1:
        return "Converted"
    if lead_id not in status_map:
        return random.choices(["New", "Contacted", "Unqualified"], weights=[0.5, 0.3, 0.2])[0]
    stage = status_map[lead_id]
    if stage == "Closed-Lost":
        return "Lost"
    return "Qualified"

df_leads["status"] = df_leads.apply(derive_status, axis=1)


# ---------------------------------------------------------
# 8. INJECT DELIBERATE MESSINESS (unchanged behaviour)
# ---------------------------------------------------------

# 8a. Missing company_size for a few leads
null_idx = df_leads.sample(frac=0.02, random_state=SEED).index
df_leads.loc[null_idx, "company_size"] = np.nan

# 8b. Duplicate activity rows
dupes = df_activities.sample(frac=0.01, random_state=SEED)
df_activities = pd.concat([df_activities, dupes], ignore_index=True)

# 8c. Wrong data types — text in company_size
df_leads["company_size"] = df_leads["company_size"].astype(object)
text_idx = df_leads.dropna(subset=["company_size"]).sample(frac=0.01, random_state=SEED).index
df_leads.loc[text_idx, "company_size"] = df_leads.loc[text_idx, "company_size"].apply(lambda x: f"{int(x)} emp")

# 8d. Orphan foreign keys in activities
orphan_rows = df_activities.sample(frac=0.005, random_state=SEED).index
df_activities.loc[orphan_rows, "lead_id"] = "L99999"

# 8e. Outlier deal values
outlier_idx = df_opps.sample(n=min(3, len(df_opps)), random_state=SEED).index
df_opps.loc[outlier_idx, "deal_value"] = 999999999


# ---------------------------------------------------------
# 9. SAVE TO CSV (with the correct column order for Bronze/Silver)
# ---------------------------------------------------------
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Drop helper columns before saving
df_leads_out = df_leads.drop(columns=["total_activities", "received_campaign",
                                      "campaign_response", "treatment_group",
                                      "_base_p_convert", "_p_convert",
                                      "event_converted", "_campaign_targeted"],
                             errors="ignore")

# Ensure exact column order expected downstream
df_leads_out = df_leads_out[[
    "lead_id", "contact_name", "company_name", "industry", "company_size",
    "lead_source", "job_title", "country", "region", "status", "created_date",
]]

df_opps_out = df_opps[[
    "opportunity_id", "lead_id", "deal_value", "stage", "owner",
    "open_date", "expected_close_date", "actual_close_date",
]]

df_activities_out = df_activities[[
    "activity_id", "lead_id", "activity_date", "activity_type", "channel", "outcome",
]]

df_touches_out = df_touches[[
    "touch_id", "lead_id", "campaign_id", "touch_date", "response", "treatment_group",
]]

df_leads_out.to_csv(os.path.join(OUTPUT_DIR, "raw_leads.csv"), index=False)
df_opps_out.to_csv(os.path.join(OUTPUT_DIR, "raw_opportunities.csv"), index=False)
df_activities_out.to_csv(os.path.join(OUTPUT_DIR, "raw_activities.csv"), index=False)
df_touches_out.to_csv(os.path.join(OUTPUT_DIR, "raw_campaign_touches.csv"), index=False)

print("\n" + "=" * 60)
print("SYNTHETIC DATA GENERATION COMPLETE")
print("=" * 60)
print(f"  raw_leads.csv             -> {len(df_leads_out):,} rows")
print(f"  raw_opportunities.csv     -> {len(df_opps_out):,} rows")
print(f"  raw_activities.csv        -> {len(df_activities_out):,} rows")
print(f"  raw_campaign_touches.csv  -> {len(df_touches_out):,} rows")
print(f"\nFiles saved to: {os.path.abspath(OUTPUT_DIR)}")
print("\nSample - raw_leads:")
print(df_leads_out.head(5).to_string(index=False))