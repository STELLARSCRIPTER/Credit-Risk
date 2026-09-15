# scripts/uplift_modeling.py
"""
CreditPulse: Uplift Modeling — Phase 9

T-learner approach:
    1. Train one model on the Treatment arm (predicts P(convert | treatment))
    2. Train one model on the Control arm  (predicts P(convert | control))
    3. Uplift per lead = P_treatment - P_control

Then classify leads into the classic four segments:
    - Persuadables    : uplift > +threshold
    - Sure Things     : high baseline, low uplift
    - Lost Causes     : low baseline, low uplift
    - Do Not Disturbs : uplift < -threshold

Outputs:
    - Postgres: ml_uplift_predictions
    - models/uplift_summary.json
"""

import os
import json
import warnings

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
load_dotenv()
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("DATABASE_URL not found in .env")

RANDOM_STATE = 42

# Features used to model uplift — deliberately exclude treatment_group itself
# (that's the split variable) and exclude campaign_response (it's a POST-treatment
# variable that would leak the treatment effect).
NUMERIC_FEATURES = ["total_activities", "company_size"]
CATEGORICAL_FEATURES = ["industry", "lead_source", "region", "job_title"]
TARGET = "event_converted"


def load_campaign_data():
    print("Loading campaign leads (Treatment + Control arms)...")
    engine = create_engine(DB_URL)
    q = """
    SELECT lead_id, treatment_group, event_converted,
           total_activities, company_size, industry, lead_source,
           region, job_title
    FROM lead_summary
    WHERE received_campaign = TRUE
      AND treatment_group IN ('Treatment', 'Control')
    """
    df = pd.read_sql(q, engine)
    print(f"  {len(df):,} campaign leads")
    print(f"  Treatment: {(df['treatment_group'] == 'Treatment').sum():,}")
    print(f"  Control:   {(df['treatment_group'] == 'Control').sum():,}")
    return df, engine


def build_pipeline():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def train_t_learner(df):
    """
    T-learner: fit two separate models — one per treatment arm.
    Returns (treatment_pipeline, control_pipeline, test_metrics).
    """
    print("\nFitting T-learner (separate models per arm)...")

    treat_df = df[df["treatment_group"] == "Treatment"].copy()
    ctrl_df = df[df["treatment_group"] == "Control"].copy()

    X_t = treat_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_t = treat_df[TARGET].astype(int)
    X_c = ctrl_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_c = ctrl_df[TARGET].astype(int)

    # Small holdout to sanity-check each arm's model quality
    Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
        X_t, y_t, test_size=0.20, random_state=RANDOM_STATE, stratify=y_t
    )
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_c, y_c, test_size=0.20, random_state=RANDOM_STATE, stratify=y_c
    )

    preprocessor = build_pipeline()

    treat_model = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    ctrl_model = Pipeline([
        ("prep", preprocessor),
        ("clf", XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])

    # Fit on training split, evaluate on holdout
    treat_model.fit(Xt_tr, yt_tr)
    ctrl_model.fit(Xc_tr, yc_tr)

    from sklearn.metrics import roc_auc_score
    t_auc = roc_auc_score(yt_te, treat_model.predict_proba(Xt_te)[:, 1])
    c_auc = roc_auc_score(yc_te, ctrl_model.predict_proba(Xc_te)[:, 1])
    print(f"  Treatment-arm AUC: {t_auc:.4f}")
    print(f"  Control-arm AUC:   {c_auc:.4f}")

    # Refit on the FULL arm data for production scoring
    treat_model.fit(X_t, y_t)
    ctrl_model.fit(X_c, y_c)

    return treat_model, ctrl_model, {"treatment_auc": float(t_auc), "control_auc": float(c_auc)}


def score_uplift(df, treat_model, ctrl_model):
    print("\nScoring per-lead uplift...")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    df = df.copy()
    df["p_treatment"] = treat_model.predict_proba(X)[:, 1]
    df["p_control"] = ctrl_model.predict_proba(X)[:, 1]
    df["uplift"] = df["p_treatment"] - df["p_control"]

    print(f"  Mean uplift: {df['uplift'].mean():+.4f}")
    print(f"  Median uplift: {df['uplift'].median():+.4f}")
    print(f"  P90 uplift: {df['uplift'].quantile(0.90):+.4f}")
    print(f"  P10 uplift: {df['uplift'].quantile(0.10):+.4f}")

    return df


def classify_segments(df):
    """
    Assign each lead to one of the four canonical uplift segments.

    Thresholds:
      - Uplift threshold: ±0.05 (5 percentage points) for "meaningful" effect
      - Baseline threshold: median p_control to split high/low baseline
    """
    print("\nClassifying into 4 uplift segments...")

    uplift_thresh = 0.05
    baseline_median = df["p_control"].median()

    def segment(row):
        up = row["uplift"]
        base = row["p_control"]
        if up >= uplift_thresh:
            return "Persuadable"
        if up <= -uplift_thresh:
            return "Do Not Disturb"
        if base >= baseline_median:
            return "Sure Thing"
        return "Lost Cause"

    df["uplift_segment"] = df.apply(segment, axis=1)
    counts = df["uplift_segment"].value_counts()
    print(counts.to_string())
    return df


def save_artifacts(df, engine, arm_metrics):
    # Summary JSON
    os.makedirs("models", exist_ok=True)
    summary = {
        "model_version": "Uplift T-learner v1",
        "arm_metrics": arm_metrics,
        "segment_counts": df["uplift_segment"].value_counts().to_dict(),
        "aggregate": {
            "n_leads": int(len(df)),
            "mean_uplift": float(df["uplift"].mean()),
            "median_uplift": float(df["uplift"].median()),
            "share_persuadable": float((df["uplift_segment"] == "Persuadable").mean()),
            "share_do_not_disturb": float((df["uplift_segment"] == "Do Not Disturb").mean()),
        },
    }
    with open("models/uplift_summary.json", "w") as f:
        json.dump(summary, f, indent=4, default=str)
    print("\nSaved models/uplift_summary.json")

    # Per-lead table
    out = df[[
        "lead_id", "treatment_group", "event_converted",
        "p_treatment", "p_control", "uplift", "uplift_segment",
        "industry", "lead_source", "region",
    ]].copy()

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ml_uplift_predictions (
                lead_id           VARCHAR(50) PRIMARY KEY,
                treatment_group   VARCHAR(20),
                event_converted   INTEGER,
                p_treatment       FLOAT,
                p_control         FLOAT,
                uplift            FLOAT,
                uplift_segment    VARCHAR(30),
                industry          VARCHAR(100),
                lead_source       VARCHAR(50),
                region            VARCHAR(50),
                scored_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.execute(text("TRUNCATE TABLE ml_uplift_predictions"))

    out.to_sql("ml_uplift_predictions", engine, if_exists="append",
               index=False, method="multi", chunksize=1000)
    print(f"  Wrote {len(out):,} rows to ml_uplift_predictions")


if __name__ == "__main__":
    print("=" * 60)
    print("CREDITPULSE UPLIFT MODELING — PHASE 9")
    print("=" * 60)

    df, engine = load_campaign_data()
    treat_model, ctrl_model, arm_metrics = train_t_learner(df)
    df = score_uplift(df, treat_model, ctrl_model)
    df = classify_segments(df)
    save_artifacts(df, engine, arm_metrics)

    print("\n" + "=" * 60)
    print("UPLIFT MODELING COMPLETE")
    print("=" * 60)