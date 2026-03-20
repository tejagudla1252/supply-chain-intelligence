# Supply Chain Disruption Intelligence Platform

A production-grade data engineering and ML platform that monitors supply chain disruption risk across global regions — built on 180K+ real orders with automated pipelines, ontology-style data modeling, ML forecasting, and a live interactive dashboard.

**[🚀 Live Demo](https://your-app.streamlit.app)** | **[📊 MLflow Experiments](./mlruns)**

---

## Architecture

```
Raw CSV (DataCo)
      │
      ▼
01_explore_and_clean.py
  ├── Data ingestion & validation (Great Expectations)
  ├── Ontology model → DuckDB
  │     ├── dim_region
  │     ├── dim_product
  │     ├── dim_customer
  │     ├── dim_date
  │     └── fact_orders
  └── Saves: supply_chain.duckdb + clean.parquet
      │
      ▼
02_ml_forecasting.py
  ├── Late delivery risk classifier (Random Forest · AUC ~0.85)
  ├── ARIMA time series forecast (3-month horizon per region)
  ├── Anomaly detection (Isolation Forest · 5% contamination)
  ├── Composite disruption score (0–100)
  └── MLflow experiment tracking
      │
      ▼
03_dashboard.py (Streamlit)
  ├── KPI cards (disruption score, late risk %, lead days, sales)
  ├── Risk trend line chart by region
  ├── Regional risk bar chart (latest month)
  ├── 3-month forecast with confidence intervals
  ├── Anomaly rate heatmap
  ├── Department performance scatter
  └── Natural language query (OpenAI API → DuckDB SQL)
      │
      ▼
GitHub Actions (daily 6AM UTC)
  ├── pytest unit tests
  ├── Pipeline re-run with fresh data
  └── Artifacts uploaded
```

---

## Tech Stack

| Layer | Tools |
|---|---|
| Data Engineering | Python, DuckDB, Pandas, Great Expectations |
| ML & Forecasting | Scikit-learn, Statsmodels (ARIMA), MLflow |
| Dashboard | Streamlit, Plotly |
| NL Query | OpenAI API (GPT-3.5-turbo) |
| DevOps | GitHub Actions, pytest, Docker |

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/yourusername/supply-chain-intelligence
cd supply-chain-intelligence

# 2. Install
pip install -r requirements.txt

# 3. Download dataset
# → kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
# → Save to: data/raw/DataCoSupplyChainDataset.csv

# 4. Run pipeline
python 01_explore_and_clean.py
python 02_ml_forecasting.py

# 5. Launch dashboard
streamlit run 03_dashboard.py

# 6. Run tests
pytest tests/ -v
```

---

## Dataset

**DataCo Smart Supply Chain** — 180,519 orders across global regions (2015–2018)
- 53 features: orders, shipments, customers, products, financials
- Labeled `Late_delivery_risk` column (ML target)
- Source: [Mendeley Data](https://data.mendeley.com/datasets/8gx2fvg2k6/3) | [Kaggle](https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis)

---

## Key Results

| Model | Metric | Score |
|---|---|---|
| Random Forest (late risk) | AUC | ~0.85 |
| ARIMA forecast | MAE | ~2.3% |
| Isolation Forest | Anomaly rate | ~5% |
| Composite risk score | Coverage | 16 regions |

---

## Ontology Design

The data model mirrors **Palantir Foundry Ontology Manager** design patterns:

```
Customer ──┐
           ├──► Order ──► Product ──► Category ──► Department
Region ────┘
```

Entity relationships enable cross-domain analytics: "Which customer segments in Western Europe have the highest late delivery risk for electronics products?"

---

## Author

**Tejamanikanta Gudla** | Dallas, TX
[LinkedIn](https://linkedin.com/in/yourprofile) | [GitHub](https://github.com/yourusername)

MS Data Science · University of North Texas (GPA 3.91)
Data Engineer · Palantir Foundry · Airbus Skywise · Azure Databricks
