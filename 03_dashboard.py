"""
Supply Chain Disruption Intelligence Platform
Step 3: Streamlit Dashboard
Run: streamlit run 03_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import duckdb
import plotly.express as px
import plotly.graph_objects as go
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

DB_PATH = "data/supply_chain.duckdb"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_risk_scores():
    return pd.read_parquet("data/processed/risk_scores.parquet")

@st.cache_data
def load_forecasts():
    return pd.read_parquet("data/processed/forecasts.parquet")

@st.cache_data
def load_monthly():
    con = duckdb.connect(DB_PATH, read_only=True)
    df  = con.execute("SELECT * FROM vw_monthly_risk_by_region").df()
    dep = con.execute("SELECT * FROM vw_department_kpis").df()
    con.close()
    return df, dep

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.title("🔗 Supply Chain Intelligence")
st.sidebar.markdown("---")

risk_scores = load_risk_scores()
forecasts   = load_forecasts()
monthly, dept_kpis = load_monthly()

all_regions = sorted(risk_scores["region_name"].dropna().unique())
selected_regions = st.sidebar.multiselect(
    "Filter by Region",
    options=all_regions,
    default=all_regions[:5]
)

year_months = sorted(risk_scores["year_month"].unique())
date_range  = st.sidebar.select_slider(
    "Date Range",
    options=year_months,
    value=(year_months[0], year_months[-1])
)

risk_tier_filter = st.sidebar.multiselect(
    "Risk Tier",
    options=["High", "Medium", "Low"],
    default=["High", "Medium", "Low"]
)

# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────
mask = (
    (risk_scores["region_name"].isin(selected_regions)) &
    (risk_scores["year_month"] >= date_range[0]) &
    (risk_scores["year_month"] <= date_range[1]) &
    (risk_scores["risk_tier"].isin(risk_tier_filter))
)
df_filtered = risk_scores[mask]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.title("🔗 Supply Chain Disruption Intelligence Platform")
st.caption("Built on DataCo Smart Supply Chain dataset · 180K+ orders · Powered by ARIMA + Isolation Forest + Random Forest")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

avg_risk     = df_filtered["disruption_score"].mean()
high_risk_n  = (df_filtered["risk_tier"] == "High").sum()
avg_late_pct = df_filtered["late_risk_pct"].mean()
total_sales  = df_filtered["total_sales"].sum()
avg_lead     = df_filtered["avg_lead_days"].mean()

col1.metric("Avg Disruption Score", f"{avg_risk:.1f}/100",
            delta=None, help="Composite weighted risk score")
col2.metric("High Risk Periods",    f"{high_risk_n}",
            delta=None, help="Region-months with score ≥ 70")
col3.metric("Avg Late Risk %",      f"{avg_late_pct:.1f}%")
col4.metric("Total Sales",          f"${total_sales/1e6:.1f}M")
col5.metric("Avg Lead Days",        f"{avg_lead:.1f} days")

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# ROW 1: Risk trend + Regional heatmap
# ─────────────────────────────────────────────────────────────────────────────
row1_left, row1_right = st.columns([2, 1])

with row1_left:
    st.subheader("📈 Disruption Risk Score — Trend by Region")
    pivot = (
        df_filtered.pivot_table(
            index="year_month", columns="region_name",
            values="disruption_score", aggfunc="mean"
        )
        .reset_index()
        .melt(id_vars="year_month", var_name="region_name", value_name="disruption_score")
        .dropna()
    )
    fig_trend = px.line(
        pivot, x="year_month", y="disruption_score",
        color="region_name", markers=False,
        labels={"disruption_score": "Risk Score", "year_month": "Month"},
        height=340
    )
    fig_trend.add_hline(y=70, line_dash="dash", line_color="red",
                        annotation_text="High Risk Threshold")
    fig_trend.add_hline(y=40, line_dash="dash", line_color="orange",
                        annotation_text="Medium Risk Threshold")
    fig_trend.update_layout(margin=dict(t=20, b=20), legend_title="Region")
    st.plotly_chart(fig_trend, use_container_width=True)

with row1_right:
    st.subheader("🗺️ Latest Risk by Region")
    latest_month = df_filtered["year_month"].max()
    latest_df    = df_filtered[df_filtered["year_month"] == latest_month]
    color_map    = {"High": "#E24B4A", "Medium": "#EF9F27", "Low": "#639922"}

    fig_bar = px.bar(
        latest_df.sort_values("disruption_score", ascending=True),
        x="disruption_score", y="region_name",
        color="risk_tier", color_discrete_map=color_map,
        orientation="h", height=340,
        labels={"disruption_score": "Risk Score", "region_name": ""}
    )
    fig_bar.update_layout(margin=dict(t=20, b=20), showlegend=False)
    st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 2: Late delivery trend + Anomaly rate + Forecasts
# ─────────────────────────────────────────────────────────────────────────────
row2_left, row2_mid, row2_right = st.columns(3)

with row2_left:
    st.subheader("🚚 Late Delivery Risk % Over Time")
    avg_monthly = (
        df_filtered.groupby("year_month")["late_risk_pct"]
        .mean().reset_index()
    )
    fig_late = px.area(
        avg_monthly, x="year_month", y="late_risk_pct",
        labels={"late_risk_pct": "Late Risk %", "year_month": "Month"},
        height=280, color_discrete_sequence=["#E24B4A"]
    )
    fig_late.update_layout(margin=dict(t=10, b=10))
    st.plotly_chart(fig_late, use_container_width=True)

with row2_mid:
    st.subheader("⚠️ Anomaly Rate by Region")
    if "anomaly_rate_pct" in df_filtered.columns:
        anom_region = (
            df_filtered.groupby("region_name")["anomaly_rate_pct"]
            .mean().reset_index()
            .sort_values("anomaly_rate_pct", ascending=False)
        )
        fig_anom = px.bar(
            anom_region, x="region_name", y="anomaly_rate_pct",
            height=280, color="anomaly_rate_pct",
            color_continuous_scale="OrRd",
            labels={"anomaly_rate_pct": "Anomaly %", "region_name": ""}
        )
        fig_anom.update_layout(margin=dict(t=10, b=10), showlegend=False)
        st.plotly_chart(fig_anom, use_container_width=True)
    else:
        st.info("Run Step 2 to generate anomaly data")

with row2_right:
    st.subheader("🔮 3-Month Risk Forecast")
    if len(forecasts) > 0:
        fc_region = st.selectbox("Select region", sorted(forecasts["region_name"].unique()))
        fc_df     = forecasts[forecasts["region_name"] == fc_region].sort_values("forecast_step")
        fig_fc    = go.Figure()
        fig_fc.add_trace(go.Bar(
            x=fc_df["forecast_month"],
            y=fc_df["predicted_risk"],
            error_y=dict(
                type="data",
                symmetric=False,
                array=fc_df["upper_bound"] - fc_df["predicted_risk"],
                arrayminus=fc_df["predicted_risk"] - fc_df["lower_bound"]
            ),
            marker_color=["#E24B4A" if v >= 70 else "#EF9F27" if v >= 40 else "#639922"
                          for v in fc_df["predicted_risk"]]
        ))
        fig_fc.update_layout(
            height=280, margin=dict(t=10, b=10),
            yaxis_title="Predicted Late Risk %", xaxis_title=""
        )
        st.plotly_chart(fig_fc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# ROW 3: Department KPIs + Natural Language Query
# ─────────────────────────────────────────────────────────────────────────────
row3_left, row3_right = st.columns([1, 1])

with row3_left:
    st.subheader("🏭 Department Performance")
    dept_latest = (
        dept_kpis.groupby("department_name")
        .agg(total_sales=("total_sales","sum"),
             total_profit=("total_profit","sum"),
             late_risk_pct=("late_risk_pct","mean"))
        .reset_index()
        .sort_values("total_sales", ascending=False)
    )
    fig_dept = px.scatter(
        dept_latest,
        x="total_sales", y="late_risk_pct",
        size="total_profit", color="department_name",
        hover_name="department_name",
        labels={"total_sales": "Total Sales ($)", "late_risk_pct": "Avg Late Risk %"},
        height=300
    )
    fig_dept.update_layout(margin=dict(t=10, b=10), showlegend=False)
    st.plotly_chart(fig_dept, use_container_width=True)

with row3_right:
    st.subheader("💬 Ask a Question (NL Query)")
    st.caption("Powered by OpenAI API — translates plain English to SQL")

    openai_key = os.environ.get("OPENAI_API_KEY", "")
    user_q = st.text_input(
        "Ask about the supply chain data",
        placeholder="Which region has the highest late delivery rate?"
    )

    example_qs = [
        "Which region has the highest average disruption score?",
        "What months had the most anomalies?",
        "Which department generated the most profit?",
        "Show late risk trend for Western Europe",
    ]
    st.caption("Examples: " + " · ".join(f"`{q}`" for q in example_qs))

    if user_q and openai_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=openai_key)

            schema = """
            Tables:
            - fact_orders(order_id, customer_id, product_id, region_name, year_month, 
                          order_date, lead_days, late_risk_flag, delivery_status,
                          sales_amount, profit, discount_rate, quantity)
            - vw_monthly_risk_by_region(year_month, region_name, total_orders, late_orders,
                          late_risk_pct, avg_lead_days, total_sales, total_profit, avg_discount_pct)
            - dim_product(product_id, product_name, category_name, department_name)
            - dim_customer(customer_id, segment, country, city, state)
            """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"You are a SQL expert. Given this schema: {schema}\nWrite a single DuckDB SQL query to answer the question. Return ONLY the SQL, no explanation."},
                    {"role": "user",   "content": user_q}
                ],
                max_tokens=200
            )
            sql = response.choices[0].message.content.strip()

            st.code(sql, language="sql")

            con    = duckdb.connect(DB_PATH, read_only=True)
            result = con.execute(sql).df()
            con.close()
            st.dataframe(result, use_container_width=True)

        except Exception as e:
            st.error(f"Query error: {e}")

    elif user_q and not openai_key:
        st.warning("Set OPENAI_API_KEY environment variable to enable NL queries")

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Built by Tejamanikanta Gudla · "
    "Stack: Python · DuckDB · ARIMA · Isolation Forest · Random Forest · MLflow · Streamlit · Plotly · OpenAI API · "
    "Data: DataCo Smart Supply Chain (180K+ orders)"
)
