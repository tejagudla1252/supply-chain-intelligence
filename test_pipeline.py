"""
Unit tests for the supply chain pipeline
Run: pytest tests/ -v
"""
import pytest
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, "..")

from supply_chain_project_01_explore_and_clean import clean, run_quality_checks


@pytest.fixture
def sample_df():
    """Minimal synthetic dataframe mimicking DataCo structure."""
    return pd.DataFrame({
        "order date (DateOrders)":    ["2022-01-15", "2022-02-10", "2022-03-05"],
        "shipping date (DateOrders)": ["2022-01-20", "2022-02-15", "2022-03-12"],
        "Late_delivery_risk":          [1, 0, 1],
        "Order Region":               ["Western Europe", "Central America", "Western Europe"],
        "Market":                     ["Europe", "LATAM", "Europe"],
        "Order Country":              ["France", "Mexico", "Germany"],
        "Product Card Id":            [101, 102, 103],
        "Product Name":               ["Widget A", "Widget B", "Widget A"],
        "Category Name":              ["Electronics", "Clothing", "Electronics"],
        "Department Name":            ["Tech", "Fashion", "Tech"],
        "Customer Id":                [1001, 1002, 1003],
        "Customer Segment":           ["Consumer", "Corporate", "Consumer"],
        "Customer Country":           ["France", "Mexico", "Germany"],
        "Customer City":              ["Paris", "CDMX", "Berlin"],
        "Customer State":             ["IDF", "CDMX", "Berlin"],
        "Order Id":                   [5001, 5002, 5003],
        "Sales":                      [120.5, 85.0, 200.0],
        "Order Profit Per Order":     [30.0, -5.0, 45.0],
        "Order Item Discount Rate":   [0.05, 0.10, 0.0],
        "Order Item Quantity":        [2, 1, 3],
    })


def test_clean_creates_order_date(sample_df):
    df_clean = clean(sample_df)
    assert "order_date" in df_clean.columns
    assert df_clean["order_date"].notna().all()


def test_clean_creates_lead_days(sample_df):
    df_clean = clean(sample_df)
    assert "actual_lead_days" in df_clean.columns
    assert (df_clean["actual_lead_days"] >= 0).all()


def test_clean_late_risk_is_binary(sample_df):
    df_clean = clean(sample_df)
    assert df_clean["late_delivery_risk"].isin([0, 1]).all()


def test_quality_check_row_count_fails_on_small_df(sample_df):
    df_clean = clean(sample_df)
    results  = run_quality_checks(df_clean)
    # Small sample should fail the row count check
    assert results["row_count_above_100k"] == False


def test_quality_check_binary_risk_passes(sample_df):
    df_clean = clean(sample_df)
    results  = run_quality_checks(df_clean)
    assert results["late_risk_binary"] == True


def test_clean_removes_duplicates():
    df_dup = pd.DataFrame({
        "order date (DateOrders)":    ["2022-01-15", "2022-01-15"],
        "shipping date (DateOrders)": ["2022-01-20", "2022-01-20"],
        "Late_delivery_risk":          [1, 1],
        "Order Region":               ["Western Europe", "Western Europe"],
        "Market":                     ["Europe", "Europe"],
        "Order Country":              ["France", "France"],
        "Sales":                      [120.5, 120.5],
        "Order Profit Per Order":     [30.0, 30.0],
        "Order Item Discount Rate":   [0.05, 0.05],
        "Order Item Quantity":        [2, 2],
        "Product Card Id":            [101, 101],
        "Product Name":               ["Widget A", "Widget A"],
        "Category Name":              ["Electronics", "Electronics"],
        "Department Name":            ["Tech", "Tech"],
        "Customer Id":                [1001, 1001],
        "Customer Segment":           ["Consumer", "Consumer"],
        "Customer Country":           ["France", "France"],
        "Customer City":              ["Paris", "Paris"],
        "Customer State":             ["IDF", "IDF"],
        "Order Id":                   [5001, 5001],
    })
    df_clean = clean(df_dup)
    assert len(df_clean) == 1
