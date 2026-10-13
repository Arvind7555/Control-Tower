"""
src/analytics_queries.py
Production Analytics and SQL Query Module.
Provides optimized SQL queries using Common Table Expressions (CTEs),
Window Functions (LAG, NTILE, RANK), and aggregations to calculate
Executive and Supply Chain KPIs, trends, and bottlenecks.
"""

import sqlite3
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

try:
    from src.utils import get_db_connection
except ImportError:
    from utils import get_db_connection

def build_filter_clauses(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> Tuple[str, List[Any]]:
    """Builds parameterized WHERE clause fragments and parameter list."""
    clauses = ["1=1"]
    params = []

    if start_date:
        clauses.append("OrderDate >= ?")
        params.append(start_date)
    if end_date:
        clauses.append("OrderDate <= ?")
        params.append(end_date)
    if regions:
        placeholders = ",".join(["?"] * len(regions))
        clauses.append(f"Region IN ({placeholders})")
        params.extend(regions)
    if categories:
        placeholders = ",".join(["?"] * len(categories))
        clauses.append(f"ProductCategory IN ({placeholders})")
        params.extend(categories)
    if delivery_statuses and "All" not in delivery_statuses:
        placeholders = ",".join(["?"] * len(delivery_statuses))
        clauses.append(f"DeliveryStatus IN ({placeholders})")
        params.extend(delivery_statuses)
    if segments:
        placeholders = ",".join(["?"] * len(segments))
        clauses.append(f"CustomerSegment IN ({placeholders})")
        params.extend(segments)

    where_sql = " AND ".join(clauses)
    return where_sql, params

def get_executive_kpis(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Calculates overall executive financial and operational KPIs with delta comparisons."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)
    
    query = f"""
    WITH filtered_orders AS (
        SELECT *
        FROM orders
        WHERE {where_sql}
    )
    SELECT
        COUNT(OrderID) AS total_orders,
        COUNT(DISTINCT CustomerID) AS total_customers,
        COALESCE(SUM(Quantity), 0) AS total_units,
        COALESCE(SUM(Sales), 0.0) AS total_revenue,
        COALESCE(SUM(Profit), 0.0) AS total_profit,
        COALESCE(AVG(Sales), 0.0) AS avg_order_value,
        CASE 
            WHEN SUM(Sales) > 0 THEN (SUM(Profit) * 100.0 / SUM(Sales))
            ELSE 0.0 
        END AS profit_margin_pct,
        CASE 
            WHEN COUNT(OrderID) > 0 THEN (SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID))
            ELSE 0.0 
        END AS on_time_rate_pct,
        CASE 
            WHEN COUNT(OrderID) > 0 THEN (SUM(CASE WHEN StockoutRisk = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID))
            ELSE 0.0 
        END AS high_stockout_rate_pct,
        COALESCE(AVG(CASE WHEN ActualLeadDays > PromisedLeadDays THEN ActualLeadDays - PromisedLeadDays ELSE 0 END), 0.0) AS avg_delay_days,
        CASE 
            WHEN SUM(Sales) > 0 THEN (SUM(ShippingCost) * 100.0 / SUM(Sales))
            ELSE 0.0 
        END AS logistics_cost_ratio_pct,
        COALESCE(SUM(ShippingCost), 0.0) AS total_shipping_cost
    FROM filtered_orders;
    """
    
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        
    return dict(row) if row else {}

def get_monthly_growth_trends(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Calculates MoM and YoY revenue and profit growth trends using CTEs and Window Functions."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    WITH monthly_rollup AS (
        SELECT
            strftime('%Y-%m', OrderDate) AS YearMonth,
            strftime('%Y', OrderDate) AS OrderYear,
            strftime('%m', OrderDate) AS OrderMonth,
            COUNT(OrderID) AS OrderCount,
            ROUND(SUM(Sales), 2) AS Revenue,
            ROUND(SUM(Profit), 2) AS Profit,
            ROUND(SUM(Profit) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS ProfitMarginPct,
            ROUND(SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS OTD_Pct
        FROM orders
        WHERE {where_sql}
        GROUP BY strftime('%Y-%m', OrderDate)
    ),
    growth_calculations AS (
        SELECT
            YearMonth,
            OrderYear,
            OrderMonth,
            OrderCount,
            Revenue,
            Profit,
            ProfitMarginPct,
            OTD_Pct,
            -- Month-over-Month (MoM) Window Calculations
            LAG(Revenue, 1) OVER (ORDER BY YearMonth) AS PrevMonthRevenue,
            ROUND((Revenue - LAG(Revenue, 1) OVER (ORDER BY YearMonth)) * 100.0 / NULLIF(LAG(Revenue, 1) OVER (ORDER BY YearMonth), 0), 2) AS MoM_RevenueGrowthPct,
            LAG(Profit, 1) OVER (ORDER BY YearMonth) AS PrevMonthProfit,
            ROUND((Profit - LAG(Profit, 1) OVER (ORDER BY YearMonth)) * 100.0 / NULLIF(LAG(Profit, 1) OVER (ORDER BY YearMonth), 0), 2) AS MoM_ProfitGrowthPct,
            -- Year-over-Year (YoY) Window Calculations (12 months lag)
            LAG(Revenue, 12) OVER (ORDER BY YearMonth) AS PrevYearRevenue,
            ROUND((Revenue - LAG(Revenue, 12) OVER (ORDER BY YearMonth)) * 100.0 / NULLIF(LAG(Revenue, 12) OVER (ORDER BY YearMonth), 0), 2) AS YoY_RevenueGrowthPct
        FROM monthly_rollup
    )
    SELECT *
    FROM growth_calculations
    ORDER BY YearMonth ASC;
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df

def get_category_performance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Returns Category and SubCategory hierarchy breakdown."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    SELECT
        ProductCategory,
        SubCategory,
        COUNT(OrderID) AS TotalOrders,
        SUM(Quantity) AS TotalUnits,
        ROUND(SUM(Sales), 2) AS TotalRevenue,
        ROUND(SUM(Profit), 2) AS TotalProfit,
        ROUND(AVG(Discount) * 100.0, 1) AS AvgDiscountPct,
        ROUND(SUM(Profit) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS MarginPct
    FROM orders
    WHERE {where_sql}
    GROUP BY ProductCategory, SubCategory
    ORDER BY TotalRevenue DESC;
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df

def get_customer_segment_performance(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Returns Customer Segment sales, profit, and discount performance metrics."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    SELECT
        CustomerSegment,
        ROUND(SUM(Sales), 2) AS Revenue,
        ROUND(SUM(Profit), 2) AS Profit,
        COUNT(OrderID) AS Orders,
        ROUND(AVG(Discount) * 100.0, 1) AS AvgDiscountPct,
        ROUND(SUM(Profit) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS MarginPct
    FROM orders
    WHERE {where_sql}
    GROUP BY CustomerSegment
    ORDER BY Revenue DESC;
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df


def get_regional_fulfillment_bottlenecks(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Analyzes regional shipping bottlenecks, delay rates, and transit variance."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    SELECT
        Region,
        WarehouseID,
        COUNT(OrderID) AS TotalOrders,
        ROUND(SUM(Sales), 2) AS TotalRevenue,
        ROUND(AVG(PromisedLeadDays), 2) AS AvgPromisedDays,
        ROUND(AVG(ActualLeadDays), 2) AS AvgActualDays,
        ROUND(AVG(ActualLeadDays - PromisedLeadDays), 2) AS AvgLeadVariance,
        ROUND(AVG(CASE WHEN ActualLeadDays > PromisedLeadDays THEN ActualLeadDays - PromisedLeadDays ELSE 0 END), 2) AS AvgDelayWhenLate,
        ROUND(SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS OnTimeRatePct,
        ROUND(SUM(CASE WHEN DeliveryStatus = 'Delayed' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS DelayRatePct,
        ROUND(SUM(CASE WHEN DeliveryStatus = 'Cancelled' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS CancelRatePct,
        ROUND(SUM(ShippingCost), 2) AS TotalShippingCost,
        ROUND(AVG(ShippingCost), 2) AS AvgShippingCostPerOrder,
        ROUND(SUM(ShippingCost) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS LogisticsCostRatioPct
    FROM orders
    WHERE {where_sql}
    GROUP BY Region, WarehouseID
    ORDER BY Region, TotalOrders DESC;
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df

def get_lead_time_distribution(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Returns lead time comparisons broken down by Shipping Mode."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    SELECT
        ShippingMode,
        DeliveryStatus,
        COUNT(OrderID) AS OrderCount,
        ROUND(AVG(PromisedLeadDays), 1) AS AvgPromisedDays,
        ROUND(AVG(ActualLeadDays), 1) AS AvgActualDays,
        ROUND(AVG(ActualLeadDays - PromisedLeadDays), 2) AS AvgVarianceDays,
        ROUND(SUM(ShippingCost), 2) AS TotalShippingCost,
        ROUND(AVG(ShippingCost), 2) AS AvgShippingCost
    FROM orders
    WHERE {where_sql}
    GROUP BY ShippingMode, DeliveryStatus
    ORDER BY ShippingMode, DeliveryStatus;
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df

def get_lead_time_histogram_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Returns granular promised vs actual lead days for fulfillment distribution modeling."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    SELECT
        OrderID,
        ShippingMode,
        PromisedLeadDays,
        ActualLeadDays,
        (ActualLeadDays - PromisedLeadDays) AS LeadVarianceDays,
        DeliveryStatus,
        Region,
        WarehouseID
    FROM orders
    WHERE {where_sql};
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df

def get_warehouse_inventory_turnover(

    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> pd.DataFrame:
    """Analyzes Warehouse inventory levels, stockout risks, and throughput volume."""
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    query = f"""
    SELECT
        WarehouseID,
        COUNT(OrderID) AS OrdersHandled,
        SUM(Quantity) AS TotalUnitsDispatched,
        ROUND(AVG(InventoryLevel), 1) AS AvgInventoryOnHand,
        ROUND(SUM(Quantity) * 1.0 / NULLIF(AVG(InventoryLevel), 0), 2) AS EstimatedInventoryTurnoverRatio,
        ROUND(SUM(CASE WHEN StockoutRisk = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS HighStockoutRiskPct,
        ROUND(SUM(CASE WHEN StockoutRisk = 'Medium' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS MedStockoutRiskPct,
        ROUND(SUM(CASE WHEN StockoutRisk = 'Low' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS LowStockoutRiskPct,
        ROUND(SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS OnTimeRatePct
    FROM orders
    WHERE {where_sql}
    GROUP BY WarehouseID
    ORDER BY OrdersHandled DESC;
    """
    with get_db_connection() as conn:
        df = pd.read_sql_query(query, conn, params=params)
    return df

def execute_custom_sql(sql_query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Safely executes a read-only SQL query against the SQLite database.
    Guards against destructive write operations (DROP, DELETE, UPDATE, INSERT, ALTER).
    """
    cleaned_query = sql_query.strip().upper()
    destructive_keywords = ["DROP", "DELETE", "UPDATE", "INSERT", "ALTER", "TRUNCATE", "REPLACE"]
    for kw in destructive_keywords:
        if cleaned_query.startswith(kw) or f"; {kw}" in cleaned_query:
            return None, f"Query execution rejected: '{kw}' operations are disabled in read-only analytics mode."

    try:
        with get_db_connection() as conn:
            df = pd.read_sql_query(sql_query, conn)
            return df, None
    except Exception as e:
        return None, str(e)


def get_dynamic_strategic_insights(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    regions: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
    delivery_statuses: Optional[List[str]] = None,
    segments: Optional[List[str]] = None
) -> List[str]:
    """
    Computes 3 automated, dynamic strategic executive insights reflecting the currently active filter selection:
    1. Commercial Leadership & Product Margin Spread
    2. Regional Fulfillment Bottlenecks & SLA Delays
    3. Inventory Vulnerability & Stockout Risk
    """
    where_sql, params = build_filter_clauses(start_date, end_date, regions, categories, delivery_statuses, segments)

    with get_db_connection() as conn:
        df_cat = pd.read_sql_query(f"""
            SELECT ProductCategory, SubCategory,
                   SUM(Sales) as SubRevenue, SUM(Profit) as SubProfit,
                   ROUND(SUM(Profit) * 100.0 / NULLIF(SUM(Sales), 0), 2) as SubMarginPct
            FROM orders WHERE {where_sql}
            GROUP BY ProductCategory, SubCategory
            ORDER BY SubRevenue DESC;
        """, conn, params=params)

        df_reg = pd.read_sql_query(f"""
            SELECT Region, COUNT(OrderID) as OrderCount,
                   ROUND(SUM(CASE WHEN DeliveryStatus = 'Delayed' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 1) as DelayRatePct,
                   ROUND(AVG(CASE WHEN ActualLeadDays > PromisedLeadDays THEN ActualLeadDays - PromisedLeadDays ELSE 0 END), 2) as AvgDelayWhenLate
            FROM orders WHERE {where_sql}
            GROUP BY Region
            ORDER BY DelayRatePct DESC;
        """, conn, params=params)

        df_wh_bottlenecks = pd.read_sql_query(f"""
            SELECT Region, WarehouseID, COUNT(OrderID) as WhOrders,
                   ROUND(SUM(CASE WHEN DeliveryStatus = 'Delayed' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 1) as DelayRatePct
            FROM orders WHERE {where_sql}
            GROUP BY Region, WarehouseID
            ORDER BY DelayRatePct DESC;
        """, conn, params=params)

        df_cat_inv = pd.read_sql_query(f"""
            SELECT ProductCategory, COUNT(OrderID) as TotalOrders,
                   ROUND(SUM(CASE WHEN StockoutRisk = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 1) as HighRiskPct,
                   ROUND(AVG(InventoryLevel), 1) as AvgInventory
            FROM orders WHERE {where_sql}
            GROUP BY ProductCategory
            ORDER BY HighRiskPct DESC;
        """, conn, params=params)

        df_wh_inv = pd.read_sql_query(f"""
            SELECT WarehouseID, COUNT(OrderID) as TotalOrders,
                   ROUND(SUM(CASE WHEN StockoutRisk = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 1) as HighRiskPct,
                   ROUND(AVG(InventoryLevel), 1) as AvgInventory
            FROM orders WHERE {where_sql}
            GROUP BY WarehouseID
            ORDER BY HighRiskPct DESC;
        """, conn, params=params)

    if df_cat.empty:
        return [
            "**Commercial Leadership**: No commercial transaction data matches the currently active filter combination.",
            "**Regional Fulfillment**: No fulfillment records available for regional bottleneck evaluation.",
            "**Inventory Vulnerability**: No inventory turnover records available under active filter parameters."
        ]

    # 1. Commercial Topline & Margin Spread
    total_active_rev = float(df_cat["SubRevenue"].sum())
    total_active_profit = float(df_cat["SubProfit"].sum())
    active_margin_pct = (total_active_profit / total_active_rev * 100.0) if total_active_rev > 0 else 0.0

    cat_totals = df_cat.groupby("ProductCategory")["SubRevenue"].sum().sort_values(ascending=False)
    top_cat_name = cat_totals.index[0]
    top_cat_rev = float(cat_totals.iloc[0])
    top_cat_share = (top_cat_rev / total_active_rev * 100.0) if total_active_rev > 0 else 0.0

    best_sub = df_cat.sort_values("SubMarginPct", ascending=False).iloc[0]
    best_sub_name = best_sub["SubCategory"]
    best_sub_margin = float(best_sub["SubMarginPct"])
    best_sub_cat = best_sub["ProductCategory"]

    bullet_1 = (
        f"**Commercial Leadership & Margin Spread**: Active filters generate **₹{total_active_rev:,.2f}** in gross sales "
        f"and **₹{total_active_profit:,.2f}** in operating profit (**{active_margin_pct:.1f}%** operating margin). "
        f"**{top_cat_name}** is the primary topline engine with **₹{top_cat_rev:,.2f}** ({top_cat_share:.1f}% share), "
        f"while **{best_sub_name}** ({best_sub_cat}) leads operational profitability at **{best_sub_margin:.1f}%** margin."
    )

    # 2. Regional Fulfillment & SLA Bottlenecks
    if not df_reg.empty:
        worst_reg_row = df_reg.iloc[0]
        worst_reg_name = worst_reg_row["Region"]
        worst_reg_delay = float(worst_reg_row["DelayRatePct"])
        worst_reg_days = float(worst_reg_row["AvgDelayWhenLate"])

        worst_wh_match = df_wh_bottlenecks[df_wh_bottlenecks["Region"] == worst_reg_name]
        if not worst_wh_match.empty:
            bottleneck_wh = worst_wh_match.iloc[0]["WarehouseID"]
            bottleneck_wh_delay = float(worst_wh_match.iloc[0]["DelayRatePct"])
            wh_detail = f", driven by **{bottleneck_wh}** ({bottleneck_wh_delay:.1f}% outbound delay rate)"
        else:
            wh_detail = ""

        if len(df_reg) > 1:
            best_reg_row = df_reg.iloc[-1]
            best_reg_name = best_reg_row["Region"]
            best_reg_delay = float(best_reg_row["DelayRatePct"])
            comp_detail = f", compared to **{best_reg_name}** at a benchmark **{best_reg_delay:.1f}% delay rate**"
        else:
            comp_detail = ""

        bullet_2 = (
            f"**Regional Logistics Bottlenecks & SLA Delays**: **{worst_reg_name}** represents the highest fulfillment delay exposure "
            f"with a **{worst_reg_delay:.1f}% delay rate** (averaging **+{worst_reg_days:.2f} days** overdue on late shipments)"
            f"{wh_detail}{comp_detail}."
        )
    else:
        bullet_2 = "**Regional Logistics Bottlenecks & SLA Delays**: No regional delivery records match current filter scope."

    # 3. Inventory Vulnerability & Stockout Risk
    if not df_cat_inv.empty and not df_wh_inv.empty:
        worst_cat_row = df_cat_inv.iloc[0]
        worst_cat_name = worst_cat_row["ProductCategory"]
        worst_cat_risk = float(worst_cat_row["HighRiskPct"])
        worst_cat_inv_avg = float(worst_cat_row["AvgInventory"])

        worst_wh_row = df_wh_inv.iloc[0]
        worst_wh_name = worst_wh_row["WarehouseID"]
        worst_wh_risk = float(worst_wh_row["HighRiskPct"])
        worst_wh_inv_avg = float(worst_wh_row["AvgInventory"])

        bullet_3 = (
            f"**Inventory Vulnerability & Stockout Risk**: **{worst_cat_name}** exhibits the highest stockout exposure "
            f"with **{worst_cat_risk:.1f}%** of orders at critical levels (averaging {worst_cat_inv_avg:.0f} units on-hand), "
            f"while **{worst_wh_name}** is the most constrained facility with **{worst_wh_risk:.1f}%** high-risk order exposure "
            f"(avg {worst_wh_inv_avg:.0f} units buffer)."
        )
    else:
        bullet_3 = "**Inventory Vulnerability & Stockout Risk**: No inventory stock records match active filters."

    return [bullet_1, bullet_2, bullet_3]


# Pre-canned Showcase SQL Queries for Interviewers
SQL_SHOWCASE_LIBRARY = {
    "1. MoM & YoY Revenue & Profit Growth (Window Functions)": {
        "description": "Calculates monthly financial performance and computes Month-over-Month (MoM) and Year-over-Year (YoY) growth percentages using LAG() window functions.",
        "sql": """WITH monthly_financials AS (
    SELECT
        strftime('%Y-%m', OrderDate) AS MonthPeriod,
        ROUND(SUM(Sales), 2) AS Revenue,
        ROUND(SUM(Profit), 2) AS GrossProfit,
        COUNT(OrderID) AS OrderVolume
    FROM orders
    GROUP BY strftime('%Y-%m', OrderDate)
)
SELECT
    MonthPeriod,
    Revenue,
    LAG(Revenue, 1) OVER (ORDER BY MonthPeriod) AS PrevMonthRev,
    ROUND((Revenue - LAG(Revenue, 1) OVER (ORDER BY MonthPeriod)) * 100.0 / 
          NULLIF(LAG(Revenue, 1) OVER (ORDER BY MonthPeriod), 0), 2) AS MoM_Rev_Growth_Pct,
    GrossProfit,
    ROUND((GrossProfit - LAG(GrossProfit, 1) OVER (ORDER BY MonthPeriod)) * 100.0 / 
          NULLIF(LAG(GrossProfit, 1) OVER (ORDER BY MonthPeriod), 0), 2) AS MoM_Profit_Growth_Pct,
    LAG(Revenue, 12) OVER (ORDER BY MonthPeriod) AS SameMonthLastYearRev,
    ROUND((Revenue - LAG(Revenue, 12) OVER (ORDER BY MonthPeriod)) * 100.0 / 
          NULLIF(LAG(Revenue, 12) OVER (ORDER BY MonthPeriod), 0), 2) AS YoY_Rev_Growth_Pct
FROM monthly_financials
ORDER BY MonthPeriod ASC;"""
    },
    "2. Warehouse Fulfillment & Bottleneck Analysis (CTEs & Aggregations)": {
        "description": "Evaluates warehouse logistics efficiency by measuring On-Time Delivery rates, average delay variance, and logistics cost ratio to identify fulfillment bottlenecks.",
        "sql": """WITH warehouse_metrics AS (
    SELECT
        WarehouseID,
        Region,
        COUNT(OrderID) AS TotalDispatched,
        ROUND(AVG(PromisedLeadDays), 2) AS AvgPromisedDays,
        ROUND(AVG(ActualLeadDays), 2) AS AvgActualDays,
        ROUND(AVG(ActualLeadDays - PromisedLeadDays), 2) AS AvgLeadVariance,
        SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) AS OnTimeOrders,
        SUM(CASE WHEN DeliveryStatus = 'Delayed' THEN 1 ELSE 0 END) AS DelayedOrders,
        ROUND(SUM(ShippingCost), 2) AS TotalLogisticsCost,
        ROUND(SUM(Sales), 2) AS TotalRevenueDispatched
    FROM orders
    GROUP BY WarehouseID, Region
)
SELECT
    WarehouseID,
    Region,
    TotalDispatched,
    AvgPromisedDays,
    AvgActualDays,
    AvgLeadVariance,
    ROUND(OnTimeOrders * 100.0 / TotalDispatched, 2) AS OnTimeRatePct,
    ROUND(DelayedOrders * 100.0 / TotalDispatched, 2) AS DelayedRatePct,
    ROUND(TotalLogisticsCost * 100.0 / NULLIF(TotalRevenueDispatched, 0), 2) AS LogisticsCostRatioPct
FROM warehouse_metrics
ORDER BY TotalDispatched DESC;"""
    },
    "3. Product Pareto Profitability & Discount Elasticity (Window NTILE & RANK)": {
        "description": "Ranks product subcategories by revenue contribution and segments them into quintiles (NTILE) while tracking discount impact on operating margins.",
        "sql": """WITH subcat_summary AS (
    SELECT
        ProductCategory,
        SubCategory,
        ROUND(SUM(Sales), 2) AS SubCategoryRevenue,
        ROUND(SUM(Profit), 2) AS SubCategoryProfit,
        ROUND(SUM(Profit) * 100.0 / NULLIF(SUM(Sales), 0), 2) AS ProfitMarginPct,
        ROUND(AVG(Discount) * 100.0, 1) AS AvgDiscountGivenPct
    FROM orders
    GROUP BY ProductCategory, SubCategory
)
SELECT
    ProductCategory,
    SubCategory,
    SubCategoryRevenue,
    RANK() OVER (ORDER BY SubCategoryRevenue DESC) AS RevenueRank,
    NTILE(4) OVER (ORDER BY SubCategoryRevenue DESC) AS RevenueQuartile,
    SubCategoryProfit,
    ProfitMarginPct,
    AvgDiscountGivenPct
FROM subcat_summary
ORDER BY SubCategoryRevenue DESC;"""
    },
    "4. Customer Segment Cohort & Spend Quintiles (NTILE Windowing)": {
        "description": "Partitions customers into spend tiers using NTILE(5) quintiles to evaluate order frequency, revenue generation, and margin distribution per segment.",
        "sql": """WITH customer_aggregates AS (
    SELECT
        CustomerID,
        CustomerSegment,
        COUNT(OrderID) AS OrderCount,
        ROUND(SUM(Sales), 2) AS LifetimeSpend,
        ROUND(SUM(Profit), 2) AS LifetimeProfit,
        ROUND(AVG(Sales), 2) AS AvgSpendPerOrder
    FROM orders
    GROUP BY CustomerID, CustomerSegment
),
quintiled_customers AS (
    SELECT
        *,
        NTILE(5) OVER (ORDER BY LifetimeSpend DESC) AS SpendQuintile
    FROM customer_aggregates
)
SELECT
    SpendQuintile,
    CustomerSegment,
    COUNT(CustomerID) AS CustomerCount,
    ROUND(AVG(OrderCount), 1) AS AvgOrdersPerCustomer,
    ROUND(SUM(LifetimeSpend), 2) AS TotalQuintileRevenue,
    ROUND(AVG(AvgSpendPerOrder), 2) AS AvgOrderValue,
    ROUND(SUM(LifetimeProfit) * 100.0 / NULLIF(SUM(LifetimeSpend), 0), 2) AS QuintileProfitMarginPct
FROM quintiled_customers
GROUP BY SpendQuintile, CustomerSegment
ORDER BY SpendQuintile ASC, TotalQuintileRevenue DESC;"""
    },
    "5. Delivery Reliability vs. Stockout Exposure by Shipping Corridor": {
        "description": "Correlates shipping mode carrier performance with warehouse stockout vulnerability to pinpoint high-risk logistical failure points.",
        "sql": """SELECT
    ShippingMode,
    WarehouseID,
    COUNT(OrderID) AS ShipmentCount,
    ROUND(SUM(CASE WHEN DeliveryStatus = 'On-Time' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS OTD_Rate_Pct,
    ROUND(SUM(CASE WHEN DeliveryStatus = 'Delayed' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS Delay_Rate_Pct,
    ROUND(SUM(CASE WHEN StockoutRisk = 'High' THEN 1 ELSE 0 END) * 100.0 / COUNT(OrderID), 2) AS High_Stockout_Risk_Pct,
    ROUND(AVG(ActualLeadDays - PromisedLeadDays), 2) AS Mean_Delivery_Variance_Days,
    ROUND(AVG(ShippingCost), 2) AS Mean_Shipping_Cost
FROM orders
GROUP BY ShippingMode, WarehouseID
HAVING ShipmentCount > 50
ORDER BY ShippingMode, OTD_Rate_Pct DESC;"""
    }
}
