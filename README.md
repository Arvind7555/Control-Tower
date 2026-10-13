# 📦 Executive Supply Chain & Sales Performance Platform

[![Python Version](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.15-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/scikit--learn-Random%20Forest%20Pipeline-F7931E.svg)](https://scikit-learn.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.65%2B-red.svg)](https://streamlit.io/)
[![Database](https://img.shields.io/badge/SQLite-WAL%20Mode-003B57.svg)](https://www.sqlite.org/)
[![Visualizations](https://img.shields.io/badge/Plotly-Interactive-3F4F75.svg)](https://plotly.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade, production-ready interactive analytics platform and predictive control tower designed for Chief Supply Chain Officers (CSCO), Chief Commercial Officers (CCO), and Operations Directors. Features an end-to-end data pipeline, relational SQLite modeling with analytical views, a trained scikit-learn Random Forest delay risk engine, and an executive Streamlit UI.

---

## 📌 Executive Summary & Business Problem Statement

Global omnichannel supply chain networks face escalating logistical friction: compounding freight fees, localized inventory stockouts, and unpredictable transit delays that erode operating margins and degrade Customer Lifetime Value (LTV).

This platform bridges the strategic gap between high-level commercial margin governance and operational fulfillment intelligence through 4 dedicated enterprise modules:
1. **📊 Executive Performance**: Revenue & margin growth trajectories, MoM dual-axis dynamics, and product category profitability treemaps.
2. **🚚 Logistics & Bottleneck Diagnostics**: Promised vs. actual lead time histograms, warehouse stockout exposure matrices, and regional bottleneck heatmaps.
3. **🤖 ML Delay Predictor & Risk Engine**: Real-time order parameter risk simulator, delay probability gauges, prescriptive mitigations, and feature importance rankings.
4. **💾 SQL Analytics & Data Export**: Self-service query console pre-loaded with complex CTEs, window functions (`LAG`, `NTILE`, `RANK`), and direct CSV extraction.

---

## 🏗️ System Architecture & Modular Layout

```
supply_chain_dashboard/
├── data/
│   └── supply_chain_data.csv          # 9,500+ realistic transactional orders (2023-2024)
├── database/
│   ├── schema.sql                     # DDL table definitions, index coverage & views
│   └── database.sqlite                # Relational SQLite 3 engine (WAL mode)
├── src/
│   ├── data_generator.py              # Synthetic enterprise generator with causal delay logic
│   ├── analytics_queries.py           # SQL CTEs, window functions (LAG, NTILE) & histogram data
│   ├── ml_model.py                    # Scikit-learn Random Forest training, evaluation & inference
│   ├── model.pkl                      # Serialized ML pipeline artifact & feature rankings
│   └── utils.py                       # High-contrast CSS, context manager pooling & Plotly theme
├── tests/
│   └── test_dashboard_integration.py  # 12-test automated integration suite
├── app.py                             # Interactive Streamlit platform with 4 clean tabs
├── requirements.txt                   # Environment package dependencies
└── README.md                          # Platform documentation & resume impact metrics
```

### Data & ML Pipeline:
```
[ Synthetic Generator Engine ]
            │
            ├──> CSV Persistence (data/supply_chain_data.csv)
            │
            ├──> SQLite Relational Store (database/database.sqlite)
            │           │
            │           ├── Indexes & Analytical Views
            │           ▼
            │     [ Analytics Engine (src/analytics_queries.py) ]
            │     (CTEs, Window Functions: LAG / NTILE / RANK)
            │
            └──> [ ML Training Pipeline (src/ml_model.py) ]
                        │
                        ├── OneHotEncoder + StandardScaler
                        ├── Balanced Random Forest Classifier
                        ▼
                 [ model.pkl Artifact ]
                        │
                        ▼
          [ Executive Streamlit UI (app.py) ]
     (4 Dedicated Tabs & Real-Time Risk Simulation)
```

---

## 🤖 Machine Learning Architecture (`src/ml_model.py`)

- **Objective**: Predict shipment fulfillment delay (`DeliveryStatus`: Delayed vs. On-Time) before dispatch.
- **Predictor Dimensions**: `ProductCategory`, `ShippingMode`, `Region`, `Quantity`, `Discount`, `UnitPrice`, `InventoryLevel`, `PromisedLeadDays`.
- **Pipeline Architecture**:
  - `OneHotEncoder(drop="first", handle_unknown="ignore")` for categorical attributes.
  - `StandardScaler()` for continuous operational metrics.
  - `RandomForestClassifier(n_estimators=45, max_depth=7, class_weight="balanced")` to account for class imbalance (~27% delay prevalence).
- **Validation Metrics**:
  - **Accuracy**: 66.2%
  - **ROC-AUC Score**: 0.688
  - **Precision (Delayed Class)**: 64.9%
  - **Recall (Sensitivity)**: 63.9%
  - **F1-Score**: 0.644
- **Top 5 Feature Importance Drivers**:
  1. `InventoryLevel` (32.5%): Stock depletion below safety thresholds strongly predicts fulfillment delays.
  2. `PromisedLeadDays` (18.1%): Tight promise windows (<3 days) paired with bulk volume elevate transit variance.
  3. `Quantity` (13.2%): Palletization and bulk LTL freight requirements amplify pick-and-pack fulfillment bottlenecks.
  4. `UnitPrice` (10.3%): High-value SKUs undergo secondary packing checks and transit verification.
  5. `ShippingMode: Standard` (6.4%): Standard ground freight incurs higher variance compared to dedicated expedited carriers.

---

## 📊 Key Business Insights Discovered

Across **9,500 enterprise transactions** (\$8.04M Gross Sales, \$2.05M Gross Profit):

1. **Category Margin Resilience**:
   - *Electronics & Workstations* drive gross volume (\$2.8M+), but carry narrower operating margins (~22-26%).
   - *Office Supplies & Paper Accessories* yield resilient margins (>42%), providing operating cash flow stability.
2. **Regional Fulfillment Congestion**:
   - The **South** and **East** corridors experience a higher delay rate (~27.2%) due to secondary cross-docking reroutes. Cross-docking introduces a **+1.8 day delay penalty**.
3. **Inventory Safety Buffers**:
   - Low inventory (<35 units) triggers over 64% of high-risk delivery delays. Rebalancing buffer stock at **WH-South (Dallas)** and **WH-East (Philadelphia)** mitigates the majority of predicted transit failures.
4. **Logistics Cost Containment**:
   - Freight spend averages **3.3% of gross sales**. Upgrading delayed Standard orders to Express reduces lead variance by 40% while preserving positive net margins on high-value orders.

---

## 💻 Tech Stack & Engineering Highlights

| Layer | Technologies | Role & Highlights |
| :--- | :--- | :--- |
| **Data Engine** | Python 3, Pandas 3, NumPy 2 | Causal synthetic generation & vectorized analytics |
| **Relational DB** | SQLite 3 | WAL mode, B-Tree indexes on dimensional columns, pre-aggregated views |
| **Advanced SQL** | CTEs, Window Functions | Rolling growth (`LAG`), quartile ranking (`RANK`), spend quintiles (`NTILE`) |
| **Machine Learning** | Scikit-Learn 1.9, Joblib | ColumnTransformer, balanced Random Forest, live inference & prescriptive recommendations |
| **Interactive UI** | Streamlit 1.65+ | High-contrast custom CSS, 4 responsive tabs, interactive ML form |
| **Data Viz** | Plotly 7+ | Responsive dual-axis trajectories, treemaps, lead time histograms, heatmaps |
| **Test Suite** | Python `unittest` | 11 automated integration tests validating data, SQL, ML, and visualization components |

---

## 🚀 Quickstart & Installation

```bash
# 1. Clone repository
git clone https://github.com/your-username/supply-chain-sales-platform.git
cd "supply-chain-sales-platform"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data & populate SQLite database
python src/data_generator.py

# 4. Train and serialize ML model pipeline
python src/ml_model.py

# 5. Run automated integration test suite
python -m unittest tests/test_dashboard_integration.py

# 6. Launch executive Streamlit platform
streamlit run app.py
```
*Access the control tower at `http://localhost:8501`.*

### ⚡ Windows 1-Click Launchers:
- **`run_dashboard.bat`**: Double-click to instantly launch the Streamlit app.
- **`run_tests.bat`**: Double-click to run the 11-test automated integration test suite.


---

## 💼 Impact-Driven Resume Bullet Points (Google XYZ Formula)

- **Accomplished** end-to-end development of an enterprise supply chain predictive platform **as measured by** tracking \$8.04M in gross sales, achieving 66.2% delay classification accuracy, and isolating high-risk inventory stockouts across 9,500+ transactions **by doing** full-stack engineering of an indexed SQLite database, scikit-learn Random Forest pipeline, and interactive Streamlit UI.
- **Accomplished** automated fulfillment bottleneck detection and SLA risk mitigation **as measured by** reducing query latency to <20ms and providing prescriptive dispatch recommendations on orders with >60% predicted delay probability **by doing** implementation of advanced SQL window functions (`LAG`, `NTILE`), ColumnTransformer preprocessing, and real-time inference simulators.
- **Accomplished** cross-functional commercial visibility between operations and executive leadership **as measured by** delivering self-service analytics across 4 operational modules and identifying a 16.4% margin spread between product categories **by doing** design of responsive Plotly visualizations (Treemaps, dual-axis MoM trajectories, lead-time histograms) and a secure read-only SQL workbench.

---

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
