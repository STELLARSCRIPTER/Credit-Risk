# scripts/train_ml_models.py
"""
CreditPulse: Lead Scoring ML Pipeline
Follows the whiteboard workflow:
lead_summary -> Feature Selection -> Train/Test Split -> Preprocessing
-> Logistic Regression / Random Forest / XGBoost -> Model Comparison
-> Select Final Model -> Generate Probability -> Risk Tier -> Expected Loss
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, classification_report
)

# ==========================================
# CONFIG
# ==========================================
load_dotenv()
DB_CONNECTION_STR = os.getenv("DATABASE_URL")
if not DB_CONNECTION_STR:
    raise ValueError("DATABASE_URL not found in .env file")

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Quantile thresholds for risk tiers (see generate_business_outputs).
#   LOW_QUANTILE  -> bottom X% = "High" risk (worst leads)
#   HIGH_QUANTILE -> top (1-X)% = "Low" risk (best leads)
LOW_QUANTILE = 0.20
HIGH_QUANTILE = 0.80

# Feature lists — defined ONCE here so we can't get them out of sync
NUMERIC_FEATURES = [
    "total_activities",
    "company_size",
    "received_campaign",
]

CATEGORICAL_FEATURES = [
    "industry",
    "lead_source",
    "region",
    "job_title",
    "treatment_group",
    "campaign_response",
]

TARGET = "event_converted"


# ==========================================
# 1. LOAD DATA FROM POSTGRES
# ==========================================
def load_data():
    print("Connecting to Postgres and loading lead_summary...")
    engine = create_engine(DB_CONNECTION_STR)
    df = pd.read_sql("SELECT * FROM lead_summary", engine)
    print(f"Loaded {len(df)} rows.")
    print(f"Columns available: {list(df.columns)}")
    return df, engine


# ==========================================
# 2. FEATURE SELECTION & TARGET DEFINITION
# ==========================================
def prepare_features(df):
    print("\nSelecting features and defining target...")

    # Verify all required columns exist
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in lead_summary: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int)

    # deal_value is needed later for Expected Loss (kept out of training features)
    deal_values = df["deal_value"].fillna(0).astype(float).values

    print(f"Numeric features:     {NUMERIC_FEATURES}")
    print(f"Categorical features: {CATEGORICAL_FEATURES}")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(4)}")

    return X, y, deal_values


# ==========================================
# 3. PREPROCESSING PIPELINE
# ==========================================
def build_preprocessor():
    print("\nBuilding preprocessing pipeline...")

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


# ==========================================
# 4. MODEL TRAINING & COMPARISON
# ==========================================
def train_and_evaluate(X, y, preprocessor):
    print(f"\nSplitting data into Train/Test ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # scale_pos_weight helps XGBoost deal with the class imbalance
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE,
            n_jobs=-1, class_weight="balanced"
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            use_label_encoder=False, eval_metric="logloss",
            scale_pos_weight=pos_weight, random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    results = {}
    best_model_name = None
    best_auc = -1
    best_pipeline = None

    for name, model in models.items():
        print(f"\nTraining {name}...")

        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model),
        ])

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        results[name] = {
            "AUC": round(auc, 4),
            "Accuracy": round(acc, 4),
            "F1": round(f1, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
        }

        print(f"  -> AUC: {auc:.4f} | Accuracy: {acc:.4f} | F1: {f1:.4f} "
              f"| Precision: {prec:.4f} | Recall: {rec:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model_name = name
            best_pipeline = pipeline

    print(f"\nBest model: {best_model_name} (AUC = {best_auc:.4f})")
    return best_model_name, best_pipeline, results, X_test, y_test


# ==========================================
# 5. GENERATE PROBABILITY, RISK TIER, & EXPECTED LOSS
# ==========================================
def generate_business_outputs(df, best_pipeline, deal_values):
    print("\nGenerating final business outputs for all leads...")

    X_full = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    probabilities = best_pipeline.predict_proba(X_full)[:, 1]

    # Quantile-based tiers: guarantees a usable distribution regardless of
    # how the model's probabilities are calibrated. Thresholds come from the
    # empirical distribution itself, not from arbitrary fixed cutoffs.
    #   Low    = top 20% by conversion probability (best leads)
    #   Medium = middle 60%
    #   High   = bottom 20% by conversion probability (worst leads)
    #
    # NOTE: these thresholds are specific to THIS batch of leads. If you
    # later score a new batch and want consistent definitions of "High risk",
    # save q20/q80 to the model_metadata.json and reuse them.
    q20, q80 = np.quantile(probabilities, [LOW_QUANTILE, HIGH_QUANTILE])
    print(f"  Quantile thresholds — Low >= {q80:.4f}, High < {q20:.4f}")

    def get_risk_tier(p):
        if p >= q80:
            return "Low"
        elif p >= q20:
            return "Medium"
        else:
            return "High"

    risk_tiers = [get_risk_tier(p) for p in probabilities]

    # Expected Loss = (1 - probability of conversion) * deal value
    expected_loss = (1 - probabilities) * deal_values

    output_df = pd.DataFrame({
        "lead_id": df["lead_id"],
        "conversion_probability": probabilities.round(6),
        "risk_tier": risk_tiers,
        "expected_loss": expected_loss.round(2),
    })

    print(f"Scored {len(output_df)} leads.")
    print("Risk Tier distribution:")
    print(output_df["risk_tier"].value_counts().to_string())

    return output_df, {"q20": float(q20), "q80": float(q80)}


# ==========================================
# 6. SAVE MODEL & METADATA
# ==========================================
def save_artifacts(best_model_name, best_pipeline, results, tier_thresholds):
    os.makedirs("models", exist_ok=True)

    safe_name = best_model_name.lower().replace(" ", "_")
    model_path = f"models/{safe_name}_pipeline.pkl"
    joblib.dump(best_pipeline, model_path)
    print(f"\nBest model pipeline saved to {model_path}")

    metadata = {
        "best_model": best_model_name,
        "model_path": model_path,
        "feature_columns": {
            "numeric": NUMERIC_FEATURES,
            "categorical": CATEGORICAL_FEATURES,
            "target": TARGET,
        },
        "risk_tier_thresholds": {
            "method": "quantile",
            "low_quantile": LOW_QUANTILE,
            "high_quantile": HIGH_QUANTILE,
            "q20": tier_thresholds["q20"],
            "q80": tier_thresholds["q80"],
        },
        "all_model_results": results,
        "best_model_metrics": results[best_model_name],
    }
    with open("models/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
    print("Metadata saved to models/model_metadata.json")


# ==========================================
# 7. WRITE SCORES BACK TO POSTGRES
# ==========================================
def write_scores_to_db(output_df, engine, model_name):
    print("\nWriting ML scores to Postgres (table: ml_lead_scores)...")
    output_df = output_df.copy()
    output_df["model_version"] = model_name

    with engine.begin() as conn:
        # Create if not exists, then truncate for idempotency
        conn.execute(text("CREATE TABLE IF NOT EXISTS ml_lead_scores ("
                          "lead_id VARCHAR(50) PRIMARY KEY, "
                          "conversion_probability FLOAT, "
                          "risk_tier VARCHAR(20), "
                          "expected_loss FLOAT, "
                          "model_version VARCHAR(50), "
                          "scored_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"))
        conn.execute(text("TRUNCATE TABLE ml_lead_scores"))

    output_df.to_sql(
        "ml_lead_scores", engine, if_exists="append",
        index=False, method="multi", chunksize=1000,
    )
    print(f"Wrote {len(output_df)} rows to ml_lead_scores.")


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("CREDITPULSE ML PIPELINE - LEAD SCORING")
    print("=" * 60)

    df, engine = load_data()
    X, y, deal_values = prepare_features(df)
    preprocessor = build_preprocessor()

    best_model_name, best_pipeline, results, X_test, y_test = train_and_evaluate(
        X, y, preprocessor
    )

    print("\n--- Detailed report for best model on holdout set ---")
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = best_pipeline.predict(X_test)
    print(classification_report(y_test, y_pred, digits=4))

    output_df, tier_thresholds = generate_business_outputs(
        df, best_pipeline, deal_values
    )
    save_artifacts(best_model_name, best_pipeline, results, tier_thresholds)
    write_scores_to_db(output_df, engine, best_model_name)

    print("\n" + "=" * 60)
    print("ML PIPELINE COMPLETE")
    print(f"Best Model: {best_model_name}")
    print(f"Metrics: {results[best_model_name]}")
    print("=" * 60)