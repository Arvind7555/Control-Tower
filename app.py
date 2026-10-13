"""
app.py
Executive Supply Chain & Sales Performance Platform.
Production-ready interactive analytics & Machine Learning platform built with Streamlit and Plotly.
"""

import sys
import os
import time
from datetime import datetime, date
from pathlib import Path
import importlib.util
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Dynamic Module Loader (Streamlit Cloud, Docker, Local)
# ---------------------------------------------------------
def find_and_load_module(module_name: str, filename: str):
    """Locates and loads a module from the repository tree regardless of cloud mount structure."""
    repo_root = Path(__file__).resolve().parent
    candidates = [
        repo_root / "src" / filename,
        repo_root / "supply_chain_dashboard" / "src" / filename,
        repo_root / filename,
        repo_root / "supply_chain_dashboard" / filename
    ]
    target = None
    for c in candidates:
        if c.exists():
            target = c
            break
            
    if not target:
        matches = list(repo_root.rglob(filename))
        if matches:
            target = matches[0]

    if target and target.exists():
        if str(target.parent) not in sys.path:
            sys.path.insert(0, str(target.parent))
        if str(target.parent.parent) not in sys.path:
            sys.path.insert(0, str(target.parent.parent))

        try:
            spec = importlib.util.spec_from_file_location(module_name, str(target))
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            sys.modules[f"src.{module_name}"] = mod
            spec.loader.exec_module(mod)
            return mod
        except Exception:
            for import_name in [f"src.{module_name}", module_name]:
                try:
                    return __import__(import_name, fromlist=["*"])
                except Exception:
                    pass
    return None

# Load core modules dynamically
utils_mod = find_and_load_module("utils", "utils.py")
queries_mod = find_and_load_module("analytics_queries", "analytics_queries.py")
ml_mod = find_and_load_module("ml_model", "ml_model.py")

# Check if source modules were located
if utils_mod is None or queries_mod is None or ml_mod is None:
    st.set_page_config(page_title="Supply Chain Setup Guide", layout="centered", page_icon="⚠️")
    st.error("🚨 Missing Source Modules in Cloud Repository")
    st.markdown(
        """
        Streamlit Cloud could not locate the required module files in your repository.
        
        ### 📂 Expected Repository Structure:
        ```
        supply-chain-analytics-dashboard/
        ├── app.py
        ├── requirements.txt
        ├── src/
        │   ├── __init__.py
        │   ├── utils.py
        │   ├── analytics_queries.py
        │   ├── ml_model.py
        │   └── data_generator.py
        ├── data/
        │   └── supply_chain_data.csv
        └── database/
            └── database.sqlite
        ```
        
        ### 🛠️ How to Resolve:
        1. Ensure the `src` folder (and its Python files) is committed and pushed to your GitHub repository.
        2. Once pushed, click **"Manage app"** (bottom right) ➔ **"Reboot app"** in Streamlit Cloud.
        """
    )
    repo_files = [str(p.relative_to(Path(__file__).resolve().parent)) for p in Path(__file__).resolve().parent.rglob("*") if not p.name.startswith(".")]
    with st.expander("🔍 Click to inspect currently detected files in your cloud deployment"):
        st.write(repo_files if repo_files else "No files detected.")
    st.stop()

# Extract symbols from dynamically loaded modules
get_custom_css = utils_mod.get_custom_css
apply_plotly_theme = utils_mod.apply_plotly_theme
format_currency = utils_mod.format_currency
format_percentage = utils_mod.format_percentage
format_number = utils_mod.format_number
EXECUTIVE_PALETTE = utils_mod.EXECUTIVE_PALETTE
CATEGORY_COLORS = utils_mod.CATEGORY_COLORS
STATUS_COLORS = utils_mod.STATUS_COLORS
get_csv_path = utils_mod.get_csv_path
ensure_dataset_ready = getattr(utils_mod, "ensure_dataset_ready", None)

get_executive_kpis = queries_mod.get_executive_kpis
get_monthly_growth_trends = queries_mod.get_monthly_growth_trends
get_category_performance = queries_mod.get_category_performance
get_customer_segment_performance = queries_mod.get_customer_segment_performance
get_regional_fulfillment_bottlenecks = queries_mod.get_regional_fulfillment_bottlenecks
get_lead_time_distribution = queries_mod.get_lead_time_distribution
get_lead_time_histogram_data = queries_mod.get_lead_time_histogram_data
get_warehouse_inventory_turnover = queries_mod.get_warehouse_inventory_turnover
get_dynamic_strategic_insights = queries_mod.get_dynamic_strategic_insights
execute_custom_sql = queries_mod.execute_custom_sql
SQL_SHOWCASE_LIBRARY = queries_mod.SQL_SHOWCASE_LIBRARY

load_trained_model = ml_mod.load_trained_model
predict_order_delay_risk = ml_mod.predict_order_delay_risk

# ---------------------------------------------------------
# Streamlit Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Executive Supply Chain & Sales Platform",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Clean Custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Ensure datasets are initialized and ready
if ensure_dataset_ready is not None:
    try:
        ensure_dataset_ready()
    except Exception as e:
        pass

# ---------------------------------------------------------
# Data & Model Pre-caching for Fast Exploratory Filtering
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_base_metadata():
    """Loads min/max dates and unique dimensional attributes."""
    if ensure_dataset_ready is not None:
        try:
            csv_file, _ = ensure_dataset_ready()
        except Exception:
            csv_file = get_csv_path()
    else:
        csv_file = get_csv_path()

    if not Path(csv_file).exists():
        try:
            from src.data_generator import generate_orders
            df_gen = generate_orders(num_orders=5000)
            Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
            df_gen.to_csv(csv_file, index=False)
        except Exception:
            pass

    df = pd.read_csv(csv_file, usecols=["OrderDate", "Region", "ProductCategory", "CustomerSegment", "DeliveryStatus"])
    df["OrderDate"] = pd.to_datetime(df["OrderDate"])
    return {
        "min_date": df["OrderDate"].min().date(),
        "max_date": df["OrderDate"].max().date(),
        "regions": sorted(df["Region"].unique().tolist()),
        "categories": sorted(df["ProductCategory"].unique().tolist()),
        "segments": sorted(df["CustomerSegment"].unique().tolist()),
        "statuses": ["All", "On-Time", "Delayed", "Cancelled"],
        "total_records": len(df)
    }

@st.cache_data(show_spinner=False)
def load_full_dataset():
    """Loads complete raw dataset for export and table explorer."""
    if ensure_dataset_ready is not None:
        try:
            csv_file, _ = ensure_dataset_ready()
        except Exception:
            csv_file = get_csv_path()
    else:
        csv_file = get_csv_path()

    if not Path(csv_file).exists():
        try:
            from src.data_generator import generate_orders
            df_gen = generate_orders(num_orders=5000)
            Path(csv_file).parent.mkdir(parents=True, exist_ok=True)
            df_gen.to_csv(csv_file, index=False)
            return df_gen
        except Exception:
            pass

    return pd.read_csv(csv_file)

@st.cache_resource(show_spinner=False)
def get_cached_ml_model():
    """Loads trained Random Forest model artifact."""
    return load_trained_model()

metadata = load_base_metadata()
ml_artifact = get_cached_ml_model()

# ---------------------------------------------------------
# Sidebar Filter Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border: 1px solid #334155;
            border-radius: 14px;
            padding: 14px 16px;
            margin-bottom: 18px;
            box-shadow: 0 4px 16px rgba(15, 23, 42, 0.18);
            display: flex;
            align-items: center;
            gap: 12px;
        ">
            <div style="
                width: 44px;
                height: 44px;
                border-radius: 12px;
                background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
                display: flex;
                align-items: center;
                justify-content: center;
                box-shadow: 0 4px 12px rgba(37, 99, 235, 0.45);
                flex-shrink: 0;
            ">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#FFFFFF" stroke-width="2.3" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="4" y1="21" x2="4" y2="14"></line>
                    <line x1="4" y1="10" x2="4" y2="3"></line>
                    <line x1="12" y1="21" x2="12" y2="12"></line>
                    <line x1="12" y1="8" x2="12" y2="3"></line>
                    <line x1="20" y1="21" x2="20" y2="16"></line>
                    <line x1="20" y1="12" x2="20" y2="3"></line>
                    <circle cx="4" cy="12" r="2.5" fill="#FFFFFF"></circle>
                    <circle cx="12" cy="10" r="2.5" fill="#FFFFFF"></circle>
                    <circle cx="20" cy="14" r="2.5" fill="#FFFFFF"></circle>
                </svg>
            </div>
            <div style="flex-grow: 1;">
                <div style="display: flex; align-items: center; justify-content: space-between;">
                    <span style="font-size: 1.05rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.01em;">
                        Control Tower
                    </span>
                    <span style="background: rgba(16, 185, 129, 0.2); color: #34D399; font-size: 0.68rem; font-weight: 700; padding: 2px 7px; border-radius: 9999px; border: 1px solid rgba(52, 211, 153, 0.35);">
                        ● LIVE
                    </span>
                </div>
                <div style="font-size: 0.74rem; color: #94A3B8; font-weight: 500; margin-top: 2px;">
                    Multi-Dimensional Filter Console
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Date Range Picker
    date_selection = st.date_input(
        "📅 Date Range",
        value=(metadata["min_date"], metadata["max_date"]),
        min_value=metadata["min_date"],
        max_value=metadata["max_date"],
        help="Filter historical transaction window"
    )

    if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
        start_date_val, end_date_val = date_selection
    elif isinstance(date_selection, (tuple, list)) and len(date_selection) == 1:
        start_date_val = date_selection[0]
        end_date_val = metadata["max_date"]
    else:
        start_date_val = metadata["min_date"]
        end_date_val = metadata["max_date"]

    start_date_str = start_date_val.strftime("%Y-%m-%d")
    end_date_str = end_date_val.strftime("%Y-%m-%d")

    # Region Multi-Select
    selected_regions = st.multiselect(
        "🌐 Geographic Regions",
        options=metadata["regions"],
        default=metadata["regions"],
        help="Filter shipments by delivery destination"
    )

    # Product Category Multi-Select
    selected_categories = st.multiselect(
        "🏷️ Product Categories",
        options=metadata["categories"],
        default=metadata["categories"],
        help="Filter by inventory category"
    )

    # Delivery Status Single-Select
    selected_delivery_status = st.selectbox(
        "🚚 Delivery Status",
        options=metadata["statuses"],
        index=0,
        help="Select fulfillment condition"
    )

    # Customer Segment Multi-Select
    selected_segments = st.multiselect(
        "👥 Customer Segments",
        options=metadata["segments"],
        default=metadata["segments"],
        help="Filter by corporate or consumer accounts"
    )

    status_filter_list = None if selected_delivery_status == "All" else [selected_delivery_status]

    st.markdown(
        """
        <div style="
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 12px;
            padding: 13px 15px;
            margin-top: 20px;
            box-shadow: 0 2px 8px rgba(15,23,42,0.04);
            font-size: 0.82rem;
        ">
            <div style="font-weight: 800; color: #0F172A; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
                <span>⚙️ System Architecture</span>
                <span style="font-size: 0.68rem; color: #059669; font-weight: 700; background: #ECFDF5; padding: 2px 6px; border-radius: 4px; border: 1px solid #A7F3D0;">ONLINE</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px dashed #E2E8F0;">
                <span style="color: #64748B;">Database:</span>
                <span style="font-weight: 700; color: #0F172A;">SQLite 3 (WAL)</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px dashed #E2E8F0;">
                <span style="color: #64748B;">ML Model:</span>
                <span style="font-weight: 700; color: #2563EB;">Random Forest</span>
            </div>
            <div style="display: flex; justify-content: space-between; padding: 3px 0;">
                <span style="color: #64748B;">Transactions:</span>
                <span style="font-weight: 700; color: #059669;">9,500 Orders</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------
# Top Executive Header Banner
# ---------------------------------------------------------
st.markdown(
    f"""
    <div class="exec-banner">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <h1>Executive Supply Chain & Sales Performance Platform</h1>
                <p>Enterprise Logistics Control Tower &bull; Predictive ML Delay Engine &bull; Relational SQL Workbench</p>
            </div>
            <div style="text-align: right;">
                <span class="metric-badge badge-neutral" style="background: rgba(255,255,255,0.15); color: #FFFFFF; font-size: 0.85rem; padding: 6px 14px; border: 1px solid rgba(255,255,255,0.25);">
                    🟢 Pipeline Active &bull; {start_date_str} to {end_date_str}
                </span>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Fetch Current Filtered KPIs
kpis = get_executive_kpis(
    start_date=start_date_str,
    end_date=end_date_str,
    regions=selected_regions if selected_regions else metadata["regions"],
    categories=selected_categories if selected_categories else metadata["categories"],
    delivery_statuses=status_filter_list,
    segments=selected_segments if selected_segments else metadata["segments"]
)

total_rev = kpis.get("total_revenue", 0.0)
total_profit = kpis.get("total_profit", 0.0)
profit_margin = kpis.get("profit_margin_pct", 0.0)
otd_rate = kpis.get("on_time_rate_pct", 0.0)
stockout_rate = kpis.get("high_stockout_rate_pct", 0.0)
aov = kpis.get("avg_order_value", 0.0)
total_orders = kpis.get("total_orders", 0)
logistics_ratio = kpis.get("logistics_cost_ratio_pct", 0.0)

# ---------------------------------------------------------
# Top Executive KPI Cards Row (Clean, Aligned, Generous Height)
# ---------------------------------------------------------
kpi_col1, kpi_col2, kpi_col3, kpi_col4, kpi_col5, kpi_col6 = st.columns(6)

with kpi_col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Total Revenue</div>
            <div class="metric-value">{format_currency(total_rev)}</div>
            <div class="metric-badge badge-positive">▲ Topline Sales</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col2:
    st.markdown(
        f"""
        <div class="metric-card success">
            <div class="metric-title">Gross Profit</div>
            <div class="metric-value">{format_currency(total_profit)}</div>
            <div class="metric-badge badge-positive">▲ Operating Profit</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col3:
    margin_color = "badge-positive" if profit_margin >= 20 else "badge-negative"
    st.markdown(
        f"""
        <div class="metric-card purple">
            <div class="metric-title">Profit Margin</div>
            <div class="metric-value">{format_percentage(profit_margin)}</div>
            <div class="metric-badge {margin_color}">● Operating Margin</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col4:
    otd_class = "badge-positive" if otd_rate >= 70 else "badge-negative"
    st.markdown(
        f"""
        <div class="metric-card {'success' if otd_rate >= 70 else 'danger'}">
            <div class="metric-title">On-Time Delivery</div>
            <div class="metric-value">{format_percentage(otd_rate)}</div>
            <div class="metric-badge {otd_class}">Target: 75% OTD</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col5:
    stockout_class = "badge-positive" if stockout_rate < 8 else "badge-negative"
    st.markdown(
        f"""
        <div class="metric-card {'warning' if stockout_rate >= 8 else ''}">
            <div class="metric-title">Critical Stockout</div>
            <div class="metric-value">{format_percentage(stockout_rate)}</div>
            <div class="metric-badge {stockout_class}">High Risk SKU Orders</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with kpi_col6:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Avg Order Value</div>
            <div class="metric-value">{format_currency(aov, compact=False)}</div>
            <div class="metric-badge badge-neutral">{format_number(total_orders)} Orders</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Primary Navigation Tabs (Concise, Clean Streamlit Tabs)
# ---------------------------------------------------------
tab_perf, tab_logistics, tab_ml_predictor, tab_sql_export = st.tabs([
    "📊 Executive Performance",
    "🚚 Logistics Diagnostics",
    "🤖 ML Delay Predictor",
    "💾 SQL Analytics & Export"
])

# =========================================================
# TAB 1: 📊 EXECUTIVE PERFORMANCE
# =========================================================
with tab_perf:
    with st.expander("💡 View Automated Strategic Insights & Operational Summary", expanded=True):
        dynamic_bullets = get_dynamic_strategic_insights(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )
        for bullet in dynamic_bullets:
            st.markdown(f"- {bullet}")

    col_trend, col_category = st.columns([1.6, 1.2])

    with col_trend:

        st.markdown("#### MoM & YoY Growth Trajectory (Dual-Axis Financial Trend)")
        trends_df = get_monthly_growth_trends(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )

        if not trends_df.empty:
            fig_trend = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Revenue Bars
            fig_trend.add_trace(
                go.Bar(
                    x=trends_df["YearMonth"],
                    y=trends_df["Revenue"],
                    name="Monthly Revenue (₹)",
                    marker_color="#3B82F6",
                    opacity=0.85,
                    hovertemplate="<b>%{x}</b><br>Revenue: ₹%{y:,.2f}<extra></extra>"
                ),
                secondary_y=False
            )

            # Gross Profit Line
            fig_trend.add_trace(
                go.Scatter(
                    x=trends_df["YearMonth"],
                    y=trends_df["Profit"],
                    name="Gross Profit (₹)",
                    mode="lines+markers",
                    line=dict(color="#10B981", width=3),
                    marker=dict(size=6, color="#059669"),
                    hovertemplate="<b>%{x}</b><br>Gross Profit: ₹%{y:,.2f}<extra></extra>"
                ),
                secondary_y=False
            )

            # Operating Profit Margin % Line on Secondary Axis
            fig_trend.add_trace(
                go.Scatter(
                    x=trends_df["YearMonth"],
                    y=trends_df["ProfitMarginPct"],
                    name="Operating Margin %",
                    mode="lines",
                    line=dict(color="#F59E0B", width=2.5, dash="dot"),
                    hovertemplate="<b>%{x}</b><br>Margin: %{y:.1f}%<extra></extra>"
                ),
                secondary_y=True
            )

            fig_trend.update_yaxes(title_text="Revenue & Profit (₹)", secondary_y=False)
            fig_trend.update_yaxes(title_text="Operating Margin (%)", secondary_y=True, range=[0, 45])
            fig_trend = apply_plotly_theme(fig_trend, height=440)
            st.plotly_chart(fig_trend, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No transactional records found for the selected filter combination.")

    with col_category:
        st.markdown("#### Sub-Category Margin & Revenue Treemap")
        cat_df = get_category_performance(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )

        if not cat_df.empty:
            fig_tree = px.treemap(
                cat_df,
                path=["ProductCategory", "SubCategory"],
                values="TotalRevenue",
                color="MarginPct",
                color_continuous_scale="Viridis",
                custom_data=["TotalProfit", "TotalRevenue", "MarginPct"],
                labels={"MarginPct": "Margin %"}
            )
            fig_tree.update_traces(
                textinfo="label+value",
                texttemplate="<b>%{label}</b><br>₹%{value:,.0f}",
                hovertemplate="<b>%{label}</b><br>Revenue: ₹%{value:,.2f}<br>Profit: ₹%{customdata[0]:,.2f}<br>Operating Margin: %{customdata[2]:.1f}%<extra></extra>"
            )
            fig_tree = apply_plotly_theme(fig_tree, height=440)
            st.plotly_chart(fig_tree, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No category data found for the selected filter criteria.")

    st.markdown("<hr style='margin: 18px 0;'/>", unsafe_allow_html=True)

    # Row 2: Customer Segment Spend Matrix + Regional Contribution
    col_seg, col_reg = st.columns([1, 1])

    with col_seg:
        st.markdown("#### Customer Segment Performance: Sales vs. Profit")
        seg_summary = get_customer_segment_performance(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )

        if not seg_summary.empty:
            fig_seg = go.Figure()
            fig_seg.add_trace(go.Bar(
                x=seg_summary["CustomerSegment"],
                y=seg_summary["Revenue"],
                name="Gross Revenue (₹)",
                marker_color="#2563EB",
                text=seg_summary["Revenue"],
                texttemplate="₹%{text:,.0f}",
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Gross Revenue: ₹%{y:,.2f}<extra></extra>"
            ))
            fig_seg.add_trace(go.Bar(
                x=seg_summary["CustomerSegment"],
                y=seg_summary["Profit"],
                name="Net Profit (₹)",
                marker_color="#10B981",
                text=seg_summary["Profit"],
                texttemplate="₹%{text:,.0f}",
                textposition="outside",
                hovertemplate="<b>%{x}</b><br>Net Profit: ₹%{y:,.2f}<extra></extra>"
            ))
            fig_seg.update_layout(barmode="group")
            fig_seg.update_yaxes(title_text="Amount (₹)")
            fig_seg = apply_plotly_theme(fig_seg, height=420)
            st.plotly_chart(fig_seg, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No segment data available.")

    with col_reg:
        st.markdown("#### Regional Revenue & Fulfillment Efficiency Matrix")
        reg_df = get_regional_fulfillment_bottlenecks(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )

        if not reg_df.empty:
            reg_rollup = reg_df.groupby("Region").agg(
                TotalOrders=("TotalOrders", "sum"),
                TotalRevenue=("TotalRevenue", "sum"),
                AvgOnTimeRate=("OnTimeRatePct", "mean"),
                AvgDelayWhenLate=("AvgDelayWhenLate", "mean")
            ).reset_index()

            fig_reg_bubble = px.scatter(
                reg_rollup,
                x="AvgOnTimeRate",
                y="AvgDelayWhenLate",
                size="TotalRevenue",
                color="Region",
                color_discrete_sequence=px.colors.qualitative.Bold,
                text="Region",
                custom_data=["TotalRevenue", "TotalOrders"],
                labels={
                    "AvgOnTimeRate": "On-Time Delivery Rate (%)",
                    "AvgDelayWhenLate": "Avg Delay When Late (Days)",
                    "TotalRevenue": "Revenue (₹)"
                }
            )
            fig_reg_bubble.update_traces(
                textposition="top center",
                hovertemplate="<b>%{text} Region</b><br>On-Time Rate: %{x:.1f}%<br>Avg Overdue Delay: %{y:.2f} days<br>Total Revenue: ₹%{customdata[0]:,.2f}<br>Total Orders: %{customdata[1]:,}<extra></extra>"
            )
            fig_reg_bubble = apply_plotly_theme(fig_reg_bubble, height=420)
            st.plotly_chart(fig_reg_bubble, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No regional data available.")

# =========================================================
# TAB 2: 🚚 LOGISTICS & BOTTLENECK DIAGNOSTICS
# =========================================================
with tab_logistics:
    sc_col1, sc_col2, sc_col3, sc_col4 = st.columns(4)
    with sc_col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Logistics Cost Ratio</div>
                <div class="metric-value">{format_percentage(logistics_ratio)}</div>
                <div class="metric-badge badge-neutral">Freight Spend / Revenue</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc_col2:
        avg_delay = kpis.get("avg_delay_days", 0.0)
        st.markdown(
            f"""
            <div class="metric-card warning">
                <div class="metric-title">Avg Lead Delay</div>
                <div class="metric-value">{avg_delay:.2f} Days</div>
                <div class="metric-badge badge-negative">Over Promised Date</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc_col3:
        tot_shipping = kpis.get("total_shipping_cost", 0.0)
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-title">Total Freight Spend</div>
                <div class="metric-value">{format_currency(tot_shipping)}</div>
                <div class="metric-badge badge-neutral">Direct Carrier Freight</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with sc_col4:
        st.markdown(
            f"""
            <div class="metric-card {'danger' if stockout_rate > 8 else 'success'}">
                <div class="metric-title">Critical Stockout Risk</div>
                <div class="metric-value">{format_percentage(stockout_rate)}</div>
                <div class="metric-badge badge-warning">High Risk Inventory</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br/>", unsafe_allow_html=True)

    # Row 1: Fulfillment Lead Time Histogram + Warehouse Stockout Risk Matrix
    row1_left, row1_right = st.columns([1.3, 1.2])

    with row1_left:
        st.markdown("#### Fulfillment Lead Time Distribution (Promised vs. Actual)")
        hist_df = get_lead_time_histogram_data(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )

        if not hist_df.empty:
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=hist_df["PromisedLeadDays"],
                name="Promised Lead SLA",
                opacity=0.65,
                marker_color="#94A3B8",
                xbins=dict(start=0.5, end=14.5, size=1)
            ))
            fig_hist.add_trace(go.Histogram(
                x=hist_df["ActualLeadDays"],
                name="Actual Delivered Lead Days",
                opacity=0.75,
                marker_color="#2563EB",
                xbins=dict(start=0.5, end=14.5, size=1)
            ))
            fig_hist.update_layout(barmode="overlay")
            fig_hist.update_xaxes(title_text="Fulfillment Days", dtick=1)
            fig_hist.update_yaxes(title_text="Order Volume")
            fig_hist = apply_plotly_theme(fig_hist, height=430)
            st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No lead time records match active filters.")

    with row1_right:
        st.markdown("#### Warehouse Inventory Turnover vs. Stockout Exposure")
        wh_df = get_warehouse_inventory_turnover(
            start_date=start_date_str,
            end_date=end_date_str,
            regions=selected_regions if selected_regions else metadata["regions"],
            categories=selected_categories if selected_categories else metadata["categories"],
            delivery_statuses=status_filter_list,
            segments=selected_segments if selected_segments else metadata["segments"]
        )

        if not wh_df.empty:
            fig_wh = px.scatter(
                wh_df,
                x="AvgInventoryOnHand",
                y="EstimatedInventoryTurnoverRatio",
                size="OrdersHandled",
                color="HighStockoutRiskPct",
                color_continuous_scale="Reds",
                text="WarehouseID",
                labels={
                    "AvgInventoryOnHand": "Avg Units On-Hand",
                    "EstimatedInventoryTurnoverRatio": "Inventory Turnover Proxy",
                    "HighStockoutRiskPct": "High Risk Orders %"
                }
            )
            fig_wh.update_traces(textposition="top center")
            fig_wh = apply_plotly_theme(fig_wh, height=430)
            st.plotly_chart(fig_wh, use_container_width=True, config={"displayModeBar": False})
        else:
            st.info("No warehouse inventory data available.")

    st.markdown("<hr style='margin: 18px 0;'/>", unsafe_allow_html=True)

    # Row 2: Regional Fulfillment Delay Rate Heatmap
    st.markdown("#### Regional Logistics Bottleneck & Delay Rate Heatmap")
    reg_bottlenecks = get_regional_fulfillment_bottlenecks(
        start_date=start_date_str,
        end_date=end_date_str,
        regions=selected_regions if selected_regions else metadata["regions"],
        categories=selected_categories if selected_categories else metadata["categories"],
        delivery_statuses=status_filter_list,
        segments=selected_segments if selected_segments else metadata["segments"]
    )

    if not reg_bottlenecks.empty:
        pivot_delays = reg_bottlenecks.pivot_table(
            index="Region",
            columns="WarehouseID",
            values="DelayRatePct",
            aggfunc="mean"
        ).fillna(0)

        fig_heat = px.imshow(
            pivot_delays,
            labels=dict(x="Fulfillment Warehouse", y="Destination Region", color="Delay Rate %"),
            color_continuous_scale="YlOrRd",
            text_auto=".1f",
            aspect="auto"
        )
        fig_heat = apply_plotly_theme(fig_heat, height=420)
        st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

        with st.expander("📋 View Detailed Regional Fulfillment Metrics Breakdown"):
            st.dataframe(
                reg_bottlenecks.style.format({
                    "TotalRevenue": "₹{:,.2f}",
                    "TotalShippingCost": "₹{:,.2f}",
                    "AvgShippingCostPerOrder": "₹{:,.2f}",
                    "OnTimeRatePct": "{:.1f}%",
                    "DelayRatePct": "{:.1f}%",
                    "CancelRatePct": "{:.1f}%",
                    "LogisticsCostRatioPct": "{:.2f}%"
                }),
                use_container_width=True
            )

# =========================================================
# TAB 3: 🤖 ML DELAY PREDICTOR & RISK ENGINE
# =========================================================
with tab_ml_predictor:
    st.markdown("### 🤖 Machine Learning Delivery Delay Predictor & Prescriptive Risk Engine")
    st.caption("Powered by a balanced Random Forest model trained on 9,500 historical supply chain transactions.")

    # =========================================================
    # Dedicated Model Performance & Feature Importance Card
    # =========================================================
    m_info = ml_artifact["metrics"]

    with st.container(border=True):
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h3 style="margin: 0; color: #0F172A; font-size: 1.20rem; font-weight: 800;">
                        🎯 Model Performance & Feature Importance
                    </h3>
                    <p style="margin: 3px 0 0 0; color: #64748B; font-size: 0.85rem;">
                        Balanced Random Forest Classifier &bull; Stratified Holdout Test Set (n = {m_info['test_samples']:,} orders)
                    </p>
                </div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span class="metric-badge badge-neutral" style="font-size: 0.82rem;">5-Fold Stratified CV</span>
                    <span class="metric-badge badge-positive" style="font-size: 0.82rem;">Balanced Class Weights</span>
                    <span class="metric-badge badge-neutral" style="font-size: 0.82rem;">F1-Score: {m_info['f1_score']:.3f}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        perf_col_kpis, perf_col_chart = st.columns([1.0, 1.4])

        with perf_col_kpis:
            st.markdown("##### 📈 Model Performance Metrics")
            kpi_row1_c1, kpi_row1_c2 = st.columns(2)
            with kpi_row1_c1:
                st.markdown(
                    f"""
                    <div class="metric-card purple" style="min-height: 98px; padding: 12px 14px; margin-bottom: 12px;">
                        <div class="metric-title">ROC-AUC Score</div>
                        <div class="metric-value" style="font-size: 1.55rem; color: #7C3AED;">{m_info['roc_auc']:.3f}</div>
                        <div class="metric-badge badge-positive">Discriminator Power</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="metric-card" style="min-height: 98px; padding: 12px 14px; margin-bottom: 12px;">
                        <div class="metric-title">Precision (Delayed)</div>
                        <div class="metric-value" style="font-size: 1.55rem; color: #2563EB;">{m_info['precision'] * 100:.1f}%</div>
                        <div class="metric-badge badge-neutral">Positive Delay Accuracy</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with kpi_row1_c2:
                st.markdown(
                    f"""
                    <div class="metric-card success" style="min-height: 98px; padding: 12px 14px; margin-bottom: 12px;">
                        <div class="metric-title">Recall (Sensitivity)</div>
                        <div class="metric-value" style="font-size: 1.55rem; color: #059669;">{m_info['recall'] * 100:.1f}%</div>
                        <div class="metric-badge badge-positive">Coverage of At-Risk Orders</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"""
                    <div class="metric-card" style="min-height: 98px; padding: 12px 14px; margin-bottom: 12px;">
                        <div class="metric-title">Overall Accuracy</div>
                        <div class="metric-value" style="font-size: 1.55rem; color: #0F172A;">{m_info['accuracy'] * 100:.1f}%</div>
                        <div class="metric-badge badge-neutral">Balanced Accuracy</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        with perf_col_chart:
            st.markdown("##### 🏆 Top 5 Driving Features Behind Shipping Delays")
            top5_df = ml_artifact["feature_importances"].head(5).copy()
            top5_df = top5_df.sort_values("Importance", ascending=True)

            fig_top5 = px.bar(
                top5_df,
                x="Importance",
                y="DisplayFeature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues",
                labels={"Importance": "Relative Feature Weight", "DisplayFeature": "Predictor Dimension"}
            )
            fig_top5.update_traces(
                text=top5_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
                textposition="outside",
                cliponaxis=False,
                marker_line_color="#1E40AF",
                marker_line_width=1
            )
            fig_top5.update_layout(
                xaxis_title="Relative Feature Importance Weight",
                yaxis_title="",
                showlegend=False,
                coloraxis_showscale=False
            )
            fig_top5 = apply_plotly_theme(fig_top5, height=270)
            st.plotly_chart(fig_top5, use_container_width=True, config={"displayModeBar": False})

    st.markdown("<hr style='margin: 18px 0;'/>", unsafe_allow_html=True)

    ml_col_left, ml_col_right = st.columns([1.2, 1.4])

    with ml_col_left:
        st.markdown("#### Interactive Order Risk Simulator")
        st.caption("Adjust order attributes to evaluate simulated delivery delay probability in real time.")

        with st.form(key="delay_prediction_form"):
            form_cat = st.selectbox("Product Category", options=metadata["categories"], index=0)
            form_ship = st.selectbox("Shipping Mode", options=["Standard", "Express", "Same Day"], index=0)
            form_reg = st.selectbox("Destination Region", options=metadata["regions"], index=1)
            
            form_qty = st.slider("Order Quantity (Units)", min_value=1, max_value=25, value=4)
            form_lead = st.slider("Promised Lead Days (SLA)", min_value=1, max_value=7, value=5)
            form_inv = st.slider("Warehouse Inventory On-Hand", min_value=5, max_value=450, value=75)
            
            c_disc, c_price = st.columns(2)
            with c_disc:
                form_disc = st.slider("Discount Given", min_value=0.0, max_value=0.30, value=0.05, step=0.05)
            with c_price:
                form_price = st.number_input("Unit Retail Price (₹)", min_value=5.0, max_value=2000.0, value=185.0, step=10.0)

            predict_submitted = st.form_submit_button("⚡ Run Delivery Risk Assessment", type="primary", use_container_width=True)

    with ml_col_right:
        input_payload = {
            "ProductCategory": form_cat if predict_submitted else "Electronics",
            "ShippingMode": form_ship if predict_submitted else "Standard",
            "Region": form_reg if predict_submitted else "South",
            "Quantity": form_qty if predict_submitted else 4,
            "Discount": form_disc if predict_submitted else 0.05,
            "UnitPrice": form_price if predict_submitted else 185.0,
            "InventoryLevel": form_inv if predict_submitted else 75,
            "PromisedLeadDays": form_lead if predict_submitted else 5
        }

        delay_prob, risk_tier, recs = predict_order_delay_risk(ml_artifact, input_payload)

        if risk_tier == "High Risk":
            container_style = "risk-high"
            pill_style = "pill-high"
        elif risk_tier == "Medium Risk":
            container_style = "risk-medium"
            pill_style = "pill-medium"
        else:
            container_style = "risk-low"
            pill_style = "pill-low"

        st.markdown(
            f"""
            <div class="risk-container {container_style}">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div>
                        <div style="color: #475569; font-size: 0.85rem; font-weight: 700; text-transform: uppercase;">Predicted Delay Risk</div>
                        <div style="color: #0F172A; font-size: 2.3rem; font-weight: 800; letter-spacing: -0.02em;">{delay_prob:.1f}%</div>
                    </div>
                    <div>
                        <span class="risk-pill {pill_style}">{risk_tier}</span>
                    </div>
                </div>
                <div style="margin-top: 10px;">
                    <div style="font-weight: 700; color: #0F172A; font-size: 0.95rem; margin-bottom: 6px;">Prescriptive Operational Mitigations:</div>
                    <ul style="margin: 0; padding-left: 20px; color: #334155; font-size: 0.90rem; line-height: 1.6;">
                        {''.join([f"<li>{r}</li>" for r in recs])}
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("#### Operational Simulation Parameters")
        st.markdown(
            f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; font-size: 0.88rem; color: #334155;">
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                    <div><strong>Category:</strong> {input_payload['ProductCategory']}</div>
                    <div><strong>Shipping Mode:</strong> {input_payload['ShippingMode']}</div>
                    <div><strong>Destination:</strong> {input_payload['Region']}</div>
                    <div><strong>Order Quantity:</strong> {input_payload['Quantity']} units</div>
                    <div><strong>Promised SLA:</strong> {input_payload['PromisedLeadDays']} days</div>
                    <div><strong>Inventory Buffer:</strong> {input_payload['InventoryLevel']} units</div>
                    <div><strong>Unit Retail Price:</strong> ₹{input_payload['UnitPrice']:,.2f}</div>
                    <div><strong>Applied Discount:</strong> {input_payload['Discount']*100:.0f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Model Evaluation Diagnostic Charts
    st.markdown("<hr style='margin: 18px 0;'/>", unsafe_allow_html=True)
    st.markdown("#### Model Validation Diagnostics: Confusion Matrix & ROC Curve")
    diag_col1, diag_col2 = st.columns(2)

    with diag_col1:
        cm = ml_artifact["metrics"]["confusion_matrix"]
        cm_z = [[cm[0][0], cm[0][1]], [cm[1][0], cm[1][1]]]
        
        fig_cm = px.imshow(
            cm_z,
            x=["Pred: On-Time", "Pred: Delayed"],
            y=["Actual: On-Time", "Actual: Delayed"],
            color_continuous_scale="Blues",
            text_auto=True,
            title="Validation Confusion Matrix"
        )
        fig_cm = apply_plotly_theme(fig_cm, height=380)
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})

    with diag_col2:
        roc_data = ml_artifact["metrics"].get("roc_curve", {})
        fpr_vals = roc_data.get("fpr", [0, 1])
        tpr_vals = roc_data.get("tpr", [0, 1])
        auc_score = ml_artifact["metrics"].get("roc_auc", 0.5)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr_vals,
            y=tpr_vals,
            mode="lines",
            name=f"Random Forest (AUC = {auc_score:.3f})",
            line=dict(color="#2563EB", width=3)
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Guessing (AUC = 0.500)",
            line=dict(color="#94A3B8", width=2, dash="dash")
        ))
        fig_roc.update_layout(title="Receiver Operating Characteristic (ROC)")
        fig_roc.update_xaxes(title_text="False Positive Rate")
        fig_roc.update_yaxes(title_text="True Positive Rate")
        fig_roc = apply_plotly_theme(fig_roc, height=380)
        st.plotly_chart(fig_roc, use_container_width=True, config={"displayModeBar": False})

# =========================================================
# TAB 4: 💾 SQL ANALYTICS & DATA EXPORT
# =========================================================

with tab_sql_export:
    st.markdown("### 💾 Enterprise SQL Analytics Workbench")
    st.caption("Inspect and execute advanced analytical queries with Window Functions (LAG, NTILE, RANK) and CTE aggregations.")

    # 4 Pre-built Complex SQL Queries Selection
    sql_keys = list(SQL_SHOWCASE_LIBRARY.keys())[:4]
    selected_query_key = st.selectbox(
        "💡 Select a Pre-Built Enterprise SQL Query:",
        options=sql_keys,
        index=0
    )

    query_meta = SQL_SHOWCASE_LIBRARY[selected_query_key]
    st.info(f"**Query Objective:** {query_meta['description']}")

    # SQL Editor area
    user_sql = st.text_area(
        "SQL Query Editor (Read-Only Analytical Sandbox):",
        value=query_meta["sql"],
        height=240,
        help="Edit query or type any SELECT statement."
    )

    run_btn = st.button("🚀 Execute Query", type="primary")

    if run_btn or user_sql:
        t0 = time.time()
        result_df, err = execute_custom_sql(user_sql)
        elapsed_ms = (time.time() - t0) * 1000

        if err:
            st.error(f"❌ SQL Execution Error: {err}")
        elif result_df is not None:
            st.success(f"✅ Query executed successfully in **{elapsed_ms:.1f} ms** &bull; Returned **{len(result_df):,} rows**.")
            st.dataframe(result_df, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🗃️ Raw Data Explorer & CSV Export")
    st.caption("Inspect filtered granular transactions and export for external modeling or reporting.")

    raw_df = load_full_dataset()

    # Filter to current view
    mask = (raw_df["OrderDate"] >= start_date_str) & (raw_df["OrderDate"] <= end_date_str)
    if selected_regions:
        mask &= raw_df["Region"].isin(selected_regions)
    if selected_categories:
        mask &= raw_df["ProductCategory"].isin(selected_categories)
    if status_filter_list:
        mask &= raw_df["DeliveryStatus"].isin(status_filter_list)
    if selected_segments:
        mask &= raw_df["CustomerSegment"].isin(selected_segments)

    filtered_export_df = raw_df[mask]

    col_meta1, col_meta2 = st.columns([2, 1])
    with col_meta1:
        st.write(f"Showing **{len(filtered_export_df):,}** of **{len(raw_df):,}** total orders matching active filters.")
    with col_meta2:
        csv_bytes = filtered_export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Filtered Dataset as CSV",
            data=csv_bytes,
            file_name=f"supply_chain_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )

    # Interactive Table with preview
    st.dataframe(
        filtered_export_df.head(250).style.format({
            "UnitPrice": "₹{:,.2f}",
            "UnitCost": "₹{:,.2f}",
            "ShippingCost": "₹{:,.2f}",
            "Sales": "₹{:,.2f}",
            "Profit": "₹{:,.2f}",
            "ProfitMargin": "{:.1f}%",
            "Discount": "{:.1%}"
        }),
        use_container_width=True
    )
