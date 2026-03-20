"""
Supply Chain Disruption Intelligence Platform
Step 1: Data Exploration, Cleaning & Ontology Model
Dataset: DataCo Smart Supply Chain Dataset
Download: https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
"""

import pandas as pd
import numpy as np
import duckdb
import os
import logging
from pathlib import Path
from datetime import datetime

# ── Logging setup ────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_DATA_PATH = "DataCoSupplyChainDataset.csv"
DB_PATH       = "data/supply_chain.duckdb"
Path("data/raw").mkdir(parents=True, exist_ok=True)
Path("data/processed").mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LOAD RAW DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_raw(path: str) -> pd.DataFrame:
    """Load the DataCo dataset — encoding must be latin-1 or it errors."""
    log.info(f"Loading raw data from {path}")
    df = pd.read_csv(path, encoding="latin-1")
    log.info(f"Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EXPLORE — run this once to understand what you have
# ═══════════════════════════════════════════════════════════════════════════════

def explore(df: pd.DataFrame):
    print("\n" + "═"*60)
    print("SHAPE:", df.shape)
    print("═"*60)

    print("\n── Columns ──")
    for col in df.columns:
        print(f"  {col:<45} {df[col].dtype}")

    print("\n── Nulls (top 15) ──")
    nulls = df.isnull().sum().sort_values(ascending=False).head(15)
    for col, n in nulls.items():
        pct = n / len(df) * 100
        print(f"  {col:<45} {n:>6,}  ({pct:.1f}%)")

    print("\n── Target column: Late_delivery_risk ──")
    print(df["Late_delivery_risk"].value_counts())

    print("\n── Order Regions ──")
    print(df["Order Region"].value_counts())

    print("\n── Delivery Status ──")
    print(df["Delivery Status"].value_counts())

    print("\n── Date range ──")
    df["order_date"] = pd.to_datetime(df["order date (DateOrders)"], errors="coerce")
    print(f"  From: {df['order_date'].min()}")
    print(f"  To:   {df['order_date'].max()}")
    print("═"*60 + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CLEAN
# ═══════════════════════════════════════════════════════════════════════════════

def clean(df: pd.DataFrame) -> pd.DataFrame:
    log.info("Cleaning data...")

    # Standardize column names — lowercase, underscores
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Parse dates
    df["order_date"]    = pd.to_datetime(df["order_date_dateorders"],   errors="coerce")
    df["shipping_date"] = pd.to_datetime(df["shipping_date_dateorders"], errors="coerce")

    # Derive useful columns
    df["actual_lead_days"]    = (df["shipping_date"] - df["order_date"]).dt.days
    df["order_year"]          = df["order_date"].dt.year
    df["order_month"]         = df["order_date"].dt.month
    df["order_quarter"]       = df["order_date"].dt.quarter
    df["order_year_month"]    = df["order_date"].dt.to_period("M").astype(str)

    # Flag late deliveries as binary int (already exists but ensure int)
    df["late_delivery_risk"] = df["late_delivery_risk"].fillna(0).astype(int)

    # Clean profit — can be negative (returns/discounts)
    df["order_profit_per_order"] = pd.to_numeric(
        df["order_profit_per_order"], errors="coerce"
    ).fillna(0)

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates()
    log.info(f"Removed {before - len(df):,} duplicate rows")

    # Drop columns with >50% nulls
    null_pct = df.isnull().mean()
    drop_cols = null_pct[null_pct > 0.5].index.tolist()
    if drop_cols:
        log.info(f"Dropping high-null columns: {drop_cols}")
        df = df.drop(columns=drop_cols)

    log.info(f"Clean dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATA QUALITY CHECKS  (Great Expectations style — manual version)
# ═══════════════════════════════════════════════════════════════════════════════

def run_quality_checks(df: pd.DataFrame) -> dict:
    log.info("Running data quality checks...")
    results = {}

    checks = {
        "no_null_order_date":       df["order_date"].isnull().sum() == 0,
        "no_null_region":           df["order_region"].isnull().sum() == 0,
        "late_risk_binary":         df["late_delivery_risk"].isin([0, 1]).all(),
        "positive_sales":           (df["sales"] >= 0).mean() > 0.95,
        "reasonable_lead_days":     df["actual_lead_days"].between(-5, 365).mean() > 0.95,
        "row_count_above_100k":     len(df) > 100_000,
    }

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        log.info(f"  {status}  {check_name}")
        results[check_name] = passed
        if not passed:
            all_passed = False

    results["all_passed"] = all_passed
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. BUILD ONTOLOGY-STYLE DATA MODEL IN DUCKDB
#    Entities: Customers · Products · Orders · Regions · Suppliers (inferred)
#    This mirrors Palantir Foundry Ontology Manager design
# ═══════════════════════════════════════════════════════════════════════════════

def build_ontology(df: pd.DataFrame, db_path: str):
    log.info(f"Building ontology model in DuckDB at {db_path}")
    con = duckdb.connect(db_path)

    # ── Entity: dim_region ───────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE dim_region AS
        SELECT DISTINCT
            order_region            AS region_name,
            market                  AS market_name,
            order_country           AS country
        FROM df
        WHERE order_region IS NOT NULL
        ORDER BY market_name, region_name
    """)
    log.info(f"  dim_region: {con.execute('SELECT COUNT(*) FROM dim_region').fetchone()[0]} rows")

    # ── Entity: dim_product ──────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE dim_product AS
        SELECT DISTINCT
            product_card_id         AS product_id,
            product_name            AS product_name,
            category_name           AS category_name,
            department_name         AS department_name
        FROM df
        WHERE product_card_id IS NOT NULL
        ORDER BY department_name, category_name
    """)
    log.info(f"  dim_product: {con.execute('SELECT COUNT(*) FROM dim_product').fetchone()[0]} rows")

    # ── Entity: dim_customer ─────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE dim_customer AS
        SELECT DISTINCT
            customer_id             AS customer_id,
            customer_segment        AS segment,
            customer_country        AS country,
            customer_city           AS city,
            customer_state          AS state
        FROM df
        WHERE customer_id IS NOT NULL
        ORDER BY segment
    """)
    log.info(f"  dim_customer: {con.execute('SELECT COUNT(*) FROM dim_customer').fetchone()[0]} rows")

    # ── Entity: dim_date ─────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE dim_date AS
        SELECT DISTINCT
            order_year_month        AS year_month,
            order_year              AS year,
            order_month             AS month,
            order_quarter           AS quarter
        FROM df
        WHERE order_year_month IS NOT NULL
        ORDER BY year_month
    """)
    log.info(f"  dim_date: {con.execute('SELECT COUNT(*) FROM dim_date').fetchone()[0]} rows")

    # ── Fact: fact_orders ────────────────────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE TABLE fact_orders AS
        SELECT
            order_id                        AS order_id,
            customer_id                     AS customer_id,
            product_card_id                 AS product_id,
            order_region                    AS region_name,
            order_year_month                AS year_month,
            order_date                      AS order_date,
            shipping_date                   AS shipping_date,
            actual_lead_days                AS lead_days,
            late_delivery_risk              AS late_risk_flag,
            delivery_status                 AS delivery_status,
            sales                           AS sales_amount,
            order_profit_per_order          AS profit,
            order_item_discount_rate        AS discount_rate,
            order_item_quantity             AS quantity
        FROM df
        WHERE order_id IS NOT NULL
    """)
    log.info(f"  fact_orders: {con.execute('SELECT COUNT(*) FROM fact_orders').fetchone()[0]:,} rows")

    # ── KPI view: monthly disruption risk by region ───────────────────────────
    # This is the core view that feeds the Streamlit dashboard
    con.execute("""
        CREATE OR REPLACE VIEW vw_monthly_risk_by_region AS
        SELECT
            year_month,
            region_name,
            COUNT(*)                                    AS total_orders,
            SUM(late_risk_flag)                         AS late_orders,
            ROUND(AVG(late_risk_flag) * 100, 2)         AS late_risk_pct,
            ROUND(AVG(lead_days), 1)                    AS avg_lead_days,
            ROUND(SUM(sales_amount), 2)                 AS total_sales,
            ROUND(SUM(profit), 2)                       AS total_profit,
            ROUND(AVG(discount_rate) * 100, 2)          AS avg_discount_pct
        FROM fact_orders
        GROUP BY year_month, region_name
        ORDER BY year_month, region_name
    """)

    # ── KPI view: department performance ─────────────────────────────────────
    con.execute("""
        CREATE OR REPLACE VIEW vw_department_kpis AS
        SELECT
            p.department_name,
            f.year_month,
            COUNT(*)                                    AS total_orders,
            ROUND(AVG(f.late_risk_flag) * 100, 2)       AS late_risk_pct,
            ROUND(SUM(f.sales_amount), 2)               AS total_sales,
            ROUND(SUM(f.profit), 2)                     AS total_profit
        FROM fact_orders f
        JOIN dim_product p ON f.product_id = p.product_id
        GROUP BY p.department_name, f.year_month
        ORDER BY f.year_month, p.department_name
    """)

    # Quick validation
    sample = con.execute("""
        SELECT year_month, region_name, late_risk_pct, total_orders
        FROM vw_monthly_risk_by_region
        ORDER BY late_risk_pct DESC
        LIMIT 5
    """).df()
    print("\nTop 5 highest risk region-months:")
    print(sample.to_string(index=False))

    con.close()
    log.info("Ontology model built successfully ✅")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SAVE PROCESSED CSV FOR ML STEP
# ═══════════════════════════════════════════════════════════════════════════════

def save_processed(df: pd.DataFrame):
    out = "data/processed/supply_chain_clean.parquet"
    df.to_parquet(out, index=False)
    log.info(f"Saved cleaned data to {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 1. Load
    df = load_raw(RAW_DATA_PATH)

    # 2. Explore — comment out after first run
    explore(df)

    # 3. Clean
    df_clean = clean(df)

    # 4. Quality checks
    qc = run_quality_checks(df_clean)
    if not qc["all_passed"]:
        log.warning("Some quality checks failed — review before proceeding")

    # 5. Build ontology model
    build_ontology(df_clean, DB_PATH)

    # 6. Save processed
    save_processed(df_clean)

    log.info("Step 1 complete. Run 02_ml_forecasting.py next.")