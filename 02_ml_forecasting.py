"""
Supply Chain Disruption Intelligence Platform
Step 2: ML Forecasting + Anomaly Detection + Risk Scoring
Run after 01_explore_and_clean.py
"""

import pandas as pd
import numpy as np
import duckdb
import mlflow
import mlflow.sklearn
import logging
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.arima.model import ARIMA
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

DB_PATH        = "data/supply_chain.duckdb"
MLFLOW_TRACKING = "mlruns"
Path("models").mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD FROM DUCKDB
# ═══════════════════════════════════════════════════════════════════════════════

def load_from_db() -> tuple[pd.DataFrame, pd.DataFrame]:
    con = duckdb.connect(DB_PATH, read_only=True)
    orders = con.execute("SELECT * FROM fact_orders").df()
    monthly = con.execute("SELECT * FROM vw_monthly_risk_by_region").df()
    con.close()
    log.info(f"Loaded {len(orders):,} orders and {len(monthly):,} monthly region rows")
    return orders, monthly


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LATE DELIVERY RISK — CLASSIFICATION MODEL
#    Target: late_risk_flag (0 or 1)
#    This is the core ML output for the dashboard risk score
# ═══════════════════════════════════════════════════════════════════════════════

def train_risk_classifier(orders: pd.DataFrame):
    log.info("Training late delivery risk classifier...")
    mlflow.set_experiment("late_delivery_risk")

    # Feature engineering
    features = [
        "lead_days", "quantity", "sales_amount",
        "discount_rate", "profit"
    ]

    # Encode categorical — region and delivery status
    le_region = LabelEncoder()
    orders["region_encoded"] = le_region.fit_transform(
        orders["region_name"].fillna("Unknown")
    )
    features.append("region_encoded")

    df_model = orders[features + ["late_risk_flag"]].dropna()

    X = df_model[features]
    y = df_model["late_risk_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    with mlflow.start_run(run_name="random_forest_v1"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=50,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        auc    = roc_auc_score(y_test, y_prob)
        cv_auc = cross_val_score(model, X, y, cv=5, scoring="roc_auc").mean()

        log.info(f"  AUC (test):     {auc:.4f}")
        log.info(f"  AUC (5-fold CV):{cv_auc:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))

        # Feature importance
        fi = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
        print("\nFeature Importance:")
        print(fi.to_string())

        # Log to MLflow
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 8)
        mlflow.log_metric("auc_test", auc)
        mlflow.log_metric("auc_cv", cv_auc)
        mlflow.sklearn.log_model(model, "random_forest_risk_model")

    return model, le_region, features


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TIME SERIES FORECASTING — LATE RISK % BY REGION
#    ARIMA on monthly late_risk_pct per region
#    Forecast 3 months ahead
# ═══════════════════════════════════════════════════════════════════════════════

def forecast_risk_by_region(monthly: pd.DataFrame) -> pd.DataFrame:
    log.info("Running ARIMA forecasts by region...")
    forecasts = []
    regions = monthly["region_name"].unique()

    for region in regions:
        series = (
            monthly[monthly["region_name"] == region]
            .sort_values("year_month")
            .set_index("year_month")["late_risk_pct"]
        )

        if len(series) < 12:
            log.warning(f"  Skipping {region} — insufficient history ({len(series)} months)")
            continue

        try:
            model   = ARIMA(series, order=(1, 1, 1))
            result  = model.fit()
            fc      = result.forecast(steps=3)
            conf    = result.get_forecast(steps=3).conf_int()

            for i, (val, (lo, hi)) in enumerate(zip(fc, conf.values)):
                forecasts.append({
                    "region_name":    region,
                    "forecast_step":  i + 1,
                    "forecast_month": f"T+{i+1}",
                    "predicted_risk": round(max(0, min(100, val)), 2),
                    "lower_bound":    round(max(0, lo), 2),
                    "upper_bound":    round(min(100, hi), 2),
                })
            log.info(f"  ✅ {region}")

        except Exception as e:
            log.warning(f"  ⚠️  {region}: {e}")

    df_fc = pd.DataFrame(forecasts)
    df_fc.to_parquet("data/processed/forecasts.parquet", index=False)
    log.info(f"Saved {len(df_fc)} forecast rows")
    return df_fc


# ═══════════════════════════════════════════════════════════════════════════════
# 4. ANOMALY DETECTION — ISOLATION FOREST
#    Detects unusual order patterns — sudden spikes in lead time, profit drops
# ═══════════════════════════════════════════════════════════════════════════════

def detect_anomalies(orders: pd.DataFrame) -> pd.DataFrame:
    log.info("Running anomaly detection with Isolation Forest...")

    features = ["lead_days", "sales_amount", "profit", "discount_rate", "quantity"]
    df_anom  = orders[features].dropna()

    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    df_anom["anomaly_score"] = iso.fit_predict(df_anom)
    df_anom["is_anomaly"]    = (df_anom["anomaly_score"] == -1).astype(int)

    n_anomalies = df_anom["is_anomaly"].sum()
    pct         = n_anomalies / len(df_anom) * 100
    log.info(f"  Detected {n_anomalies:,} anomalies ({pct:.1f}% of orders)")

    # Merge back to orders
    orders_with_anomalies = orders.copy()
    orders_with_anomalies["is_anomaly"] = df_anom["is_anomaly"].values

    # Monthly anomaly rate
    monthly_anomalies = (
        orders_with_anomalies.groupby(["year_month", "region_name"])
        .agg(
            total_orders=("order_id", "count"),
            anomaly_count=("is_anomaly", "sum")
        )
        .reset_index()
    )
    monthly_anomalies["anomaly_rate_pct"] = (
        monthly_anomalies["anomaly_count"] / monthly_anomalies["total_orders"] * 100
    ).round(2)

    monthly_anomalies.to_parquet("data/processed/anomalies.parquet", index=False)
    log.info("Saved anomaly data")
    return monthly_anomalies


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMPOSITE DISRUPTION RISK SCORE (0–100)
#    Combines: late_risk_pct (50%) + anomaly_rate (30%) + forecast trend (20%)
#    This is the headline number on the dashboard
# ═══════════════════════════════════════════════════════════════════════════════

def build_risk_score(monthly: pd.DataFrame, anomalies: pd.DataFrame, forecasts: pd.DataFrame) -> pd.DataFrame:
    log.info("Building composite disruption risk scores...")

    # Merge anomaly rate
    base = monthly.merge(
        anomalies[["year_month", "region_name", "anomaly_rate_pct"]],
        on=["year_month", "region_name"],
        how="left"
    ).fillna({"anomaly_rate_pct": 0})

    # Normalize components to 0–100
    def normalize(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return pd.Series([50.0] * len(series), index=series.index)
        return (series - mn) / (mx - mn) * 100

    base["late_risk_norm"]   = normalize(base["late_risk_pct"])
    base["anomaly_norm"]     = normalize(base["anomaly_rate_pct"])

    # Weighted composite score
    base["disruption_score"] = (
        base["late_risk_norm"]  * 0.50 +
        base["anomaly_norm"]    * 0.30 +
        20  # base risk floor — adjustable
    ).round(1).clip(0, 100)

    # Risk tier label
    def tier(score):
        if score >= 70: return "High"
        if score >= 40: return "Medium"
        return "Low"

    base["risk_tier"] = base["disruption_score"].apply(tier)

    # Save as the primary dashboard feed
    base.to_parquet("data/processed/risk_scores.parquet", index=False)
    log.info(f"Risk scores built for {len(base):,} region-month combinations")

    print("\nSample risk scores (top 10 by score):")
    print(
        base[["year_month", "region_name", "late_risk_pct", "disruption_score", "risk_tier"]]
        .sort_values("disruption_score", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    return base


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mlflow.set_tracking_uri(MLFLOW_TRACKING)

    # Load
    orders, monthly = load_from_db()

    # ML: classify late delivery risk
    model, le_region, features = train_risk_classifier(orders)

    # Time series: forecast risk 3 months ahead
    forecasts = forecast_risk_by_region(monthly)

    # Anomaly detection
    anomalies = detect_anomalies(orders)

    # Composite score
    risk_scores = build_risk_score(monthly, anomalies, forecasts)

    log.info("Step 2 complete. Run 03_dashboard.py next.")
