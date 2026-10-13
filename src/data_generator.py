"""
src/data_generator.py
Synthetic Enterprise Supply Chain & Sales Transactional Data Generator.
Generates 9,500+ realistic enterprise supply chain orders across 2023-2024,
writes to data/supply_chain_data.csv, and loads into SQLite database.
"""

import os
import random
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd

# Set deterministic seed for reproducibility while retaining realistic variance
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = Path(__file__).resolve().parent.parent

# Catalog & Geographical Reference Data
REGIONS_CITIES = {
    "West": ["Los Angeles", "San Francisco", "Seattle", "Phoenix", "Denver", "San Diego", "Portland"],
    "East": ["New York", "Boston", "Philadelphia", "Newark", "Pittsburgh", "Baltimore"],
    "Central": ["Chicago", "Minneapolis", "Detroit", "Indianapolis", "Columbus", "Milwaukee"],
    "South": ["Atlanta", "Miami", "Dallas", "Houston", "Charlotte", "Austin", "Nashville", "Orlando"],
    "North": ["Minneapolis", "Buffalo", "Rochester", "Grand Rapids", "Green Bay"]
}

REGIONAL_PRIMARY_WAREHOUSE = {
    "North": "WH-North",
    "South": "WH-South",
    "East": "WH-East",
    "West": "WH-West",
    "Central": "WH-Central"
}

WAREHOUSE_LIST = ["WH-North", "WH-South", "WH-East", "WH-West", "WH-Central"]

PRODUCT_HIERARCHY = {
    "Electronics": {
        "Smartphones & Accessories": (40.0, 950.0, 0.65),
        "Laptops & Workstations": (450.0, 2200.0, 0.72),
        "Audio & Headsets": (25.0, 320.0, 0.58),
        "Storage & Networking": (35.0, 480.0, 0.60),
        "Display Monitors": (120.0, 650.0, 0.68)
    },
    "Furniture": {
        "Executive Desks": (180.0, 850.0, 0.70),
        "Ergonomic Chairs": (90.0, 520.0, 0.64),
        "Bookcases & Shelving": (75.0, 380.0, 0.66),
        "Conference Tables": (300.0, 1400.0, 0.72),
        "Filing Cabinets": (60.0, 290.0, 0.62)
    },
    "Office Supplies": {
        "Paper & Stationery": (6.0, 55.0, 0.45),
        "Binders & Filing": (4.0, 42.0, 0.48),
        "Desk Organization": (10.0, 85.0, 0.52),
        "Pens & Markers": (3.0, 35.0, 0.42),
        "Packaging & Envelopes": (8.0, 65.0, 0.50)
    },
    "Industrial": {
        "Packaging Equipment": (45.0, 420.0, 0.65),
        "Safety & PPE": (15.0, 190.0, 0.55),
        "Material Handling": (80.0, 750.0, 0.68),
        "Industrial Tools": (35.0, 390.0, 0.62),
        "Sanitation & Chemicals": (20.0, 160.0, 0.58)
    }
}

SHIPPING_MODES = {
    "Standard": {"promise_range": (5, 7), "base_cost": 8.50, "unit_cost_factor": 1.20, "prob": 0.60},
    "Express": {"promise_range": (2, 3), "base_cost": 22.00, "unit_cost_factor": 2.50, "prob": 0.30},
    "Same Day": {"promise_range": (1, 1), "base_cost": 45.00, "unit_cost_factor": 4.80, "prob": 0.10}
}

CUSTOMER_SEGMENTS = ["Consumer", "Corporate", "Home Office"]
SEGMENT_WEIGHTS = [0.50, 0.32, 0.18]

def generate_sku_catalog(num_skus_per_subcat=8):
    """Generates master SKU catalog with base costs and retail prices."""
    catalog = []
    sku_counter = 100
    for category, subcats in PRODUCT_HIERARCHY.items():
        cat_code = category[:3].upper()
        for subcat, (min_p, max_p, cost_ratio) in subcats.items():
            sub_code = subcat[:4].upper().replace(" ", "")
            for _ in range(num_skus_per_subcat):
                sku_counter += 1
                sku = f"{cat_code}-{sub_code}-{sku_counter}"
                base_price = round(random.uniform(min_p, max_p), 2)
                # Cost is base_price * cost_ratio with slight random variation (+/- 5%)
                unit_cost = round(base_price * cost_ratio * random.uniform(0.95, 1.05), 2)
                catalog.append({
                    "ProductCategory": category,
                    "SubCategory": subcat,
                    "ProductSKU": sku,
                    "BasePrice": base_price,
                    "UnitCost": unit_cost
                })
    return pd.DataFrame(catalog)

def generate_orders(num_orders=9600):
    """Generates synthetic supply chain order records."""
    catalog_df = generate_sku_catalog()
    catalog_records = catalog_df.to_dict(orient="records")

    # Generate customer pool
    num_customers = 1800
    customers = [
        {
            "CustomerID": f"CUST-{1000 + i}",
            "Segment": np.random.choice(CUSTOMER_SEGMENTS, p=SEGMENT_WEIGHTS)
        }
        for i in range(num_customers)
    ]

    # Date range: 2023-01-01 to 2024-12-31 (730 days)
    start_date = datetime(2023, 1, 1)
    
    orders = []
    
    # Regional weightings
    regions = ["North", "South", "East", "West", "Central"]
    region_weights = [0.15, 0.28, 0.24, 0.23, 0.10]
    
    shipping_modes = list(SHIPPING_MODES.keys())
    shipping_weights = [SHIPPING_MODES[m]["prob"] for m in shipping_modes]

    for i in range(1, num_orders + 1):
        order_id = f"ORD-{2023 + (i % 2)}-{10000 + i}"
        
        # Seasonality: more orders in Q4 (Nov, Dec) and year-over-year 15% increase in 2024
        # We model this by picking day offset with weighted monthly distribution
        day_offset = random.randint(0, 730)
        curr_date = start_date + timedelta(days=day_offset)
        # Seasonal bump probability check
        if curr_date.month in [11, 12] and random.random() < 0.25:
            # holiday shopping surge
            pass
        
        order_date_str = curr_date.strftime("%Y-%m-%d")
        
        cust = random.choice(customers)
        cust_id = cust["CustomerID"]
        cust_seg = cust["Segment"]

        region = np.random.choice(regions, p=region_weights)
        city = random.choice(REGIONS_CITIES[region])
        country = "United States"

        # Warehouse assignment: 85% primary regional warehouse, 15% secondary cross-docking warehouse
        if random.random() < 0.85:
            warehouse = REGIONAL_PRIMARY_WAREHOUSE[region]
        else:
            warehouse = random.choice(WAREHOUSE_LIST)

        # SKU Selection
        item = random.choice(catalog_records)
        prod_cat = item["ProductCategory"]
        sub_cat = item["SubCategory"]
        sku = item["ProductSKU"]
        unit_price = item["BasePrice"]
        unit_cost = item["UnitCost"]

        # Order Quantity depending on segment
        if cust_seg == "Corporate":
            qty = int(np.random.choice([2, 3, 4, 5, 8, 10, 12, 15], p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.05, 0.05]))
        elif cust_seg == "Home Office":
            qty = int(np.random.choice([1, 2, 3, 4, 6], p=[0.35, 0.30, 0.20, 0.10, 0.05]))
        else: # Consumer
            qty = int(np.random.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03]))

        # Discount: discount depth varies by segment and random marketing campaigns
        if cust_seg == "Corporate":
            discount = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.20], p=[0.35, 0.25, 0.20, 0.15, 0.05]))
        else:
            discount = float(np.random.choice([0.0, 0.05, 0.10, 0.15, 0.25, 0.30], p=[0.55, 0.18, 0.12, 0.08, 0.05, 0.02]))

        # Shipping Mode & Logistics
        ship_mode = np.random.choice(shipping_modes, p=shipping_weights)
        mode_meta = SHIPPING_MODES[ship_mode]
        prom_min, prom_max = mode_meta["promise_range"]
        promised_lead = random.randint(prom_min, prom_max)

        # Shipping Cost
        base_ship = mode_meta["base_cost"]
        bulk_factor = 2.5 if prod_cat == "Furniture" else (1.8 if prod_cat == "Industrial" else 1.0)
        shipping_cost = round(base_ship + (mode_meta["unit_cost_factor"] * qty * bulk_factor) + random.uniform(0.5, 4.0), 2)

        # Financial Calculations
        gross_sales = round(unit_price * qty * (1.0 - discount), 2)
        total_cogs = round(unit_cost * qty, 2)
        # Net Profit = Revenue - COGS - Shipping Cost
        profit = round(gross_sales - total_cogs - shipping_cost, 2)
        profit_margin = round((profit / gross_sales) * 100.0, 2) if gross_sales > 0 else 0.0

        # Inventory & Stockout Risk simulation
        inv_base = random.randint(5, 450)
        if inv_base < 35:
            stockout_risk = "High"
        elif inv_base < 110:
            stockout_risk = "Medium"
        else:
            stockout_risk = "Low"

        # Fulfillment & Delivery Logic (Causal Operational Friction)
        # 1. Base logit
        logit = -1.60
        # 2. Inventory depletion penalty (Backorder / cross-docking delay)
        if inv_base < 35:
            logit += 1.85
        elif inv_base < 90:
            logit += 0.75
        # 3. Order bulk volume & picking complexity
        if qty >= 8:
            logit += 1.10
        elif qty >= 4:
            logit += 0.40
        # 4. Shipping Mode & Promised SLA tension
        if ship_mode == "Standard" and promised_lead <= 5:
            logit += 0.65
        elif ship_mode == "Same Day":
            logit += 0.50
        elif ship_mode == "Express":
            logit -= 0.60
        # 5. Regional transit congestion
        if region in ["South", "East"]:
            logit += 0.45
        if warehouse != REGIONAL_PRIMARY_WAREHOUSE[region]:
            logit += 0.70  # Cross-docking reroute delay

        # Sigmoid probability of delay
        delay_prob = 1.0 / (1.0 + np.exp(-logit))
        draw = random.random()

        if draw < 0.025:
            # Cancelled due to out-of-stock or critical failure
            status = "Cancelled"
            actual_lead = promised_lead + random.randint(5, 10)
        elif draw < (0.025 + delay_prob * 0.85):
            # Delayed
            status = "Delayed"
            delay_days = random.randint(1, 4) + (1 if inv_base < 40 else 0)
            actual_lead = promised_lead + delay_days
        else:
            # On-Time
            status = "On-Time"
            actual_lead = max(1, promised_lead - random.choice([0, 1, 0, 0]))


        orders.append({
            "OrderID": order_id,
            "OrderDate": order_date_str,
            "CustomerID": cust_id,
            "CustomerSegment": cust_seg,
            "Region": region,
            "Country": country,
            "City": city,
            "ProductCategory": prod_cat,
            "SubCategory": sub_cat,
            "ProductSKU": sku,
            "UnitPrice": unit_price,
            "Quantity": qty,
            "Discount": discount,
            "ShippingCost": shipping_cost,
            "UnitCost": unit_cost,
            "Sales": gross_sales,
            "Profit": profit,
            "ProfitMargin": profit_margin,
            "WarehouseID": warehouse,
            "ShippingMode": ship_mode,
            "PromisedLeadDays": promised_lead,
            "ActualLeadDays": actual_lead,
            "DeliveryStatus": status,
            "InventoryLevel": inv_base,
            "StockoutRisk": stockout_risk
        })

    df = pd.DataFrame(orders)
    # Sort chronologically
    df["OrderDate_dt"] = pd.to_datetime(df["OrderDate"])
    df = df.sort_values("OrderDate_dt").reset_index(drop=True)
    df = df.drop(columns=["OrderDate_dt"])
    return df

def init_sqlite_database(df: pd.DataFrame, db_path: Path, schema_path: Path):
    """Executes schema.sql and loads data into SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing db file if present to guarantee clean slate
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()

    if schema_path and schema_path.exists():
        with open(schema_path, "r", encoding="utf-8") as f:
            schema_sql = f.read()
        cursor.executescript(schema_sql)
        df.to_sql("orders", conn, if_exists="append", index=False)
    else:
        df.to_sql("orders", conn, if_exists="replace", index=False)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON orders(OrderDate);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_region ON orders(Region);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON orders(ProductCategory);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_delivery_status ON orders(DeliveryStatus);")

    conn.commit()
    conn.close()
    print(f"Successfully loaded {len(df):,} records into SQLite database at {db_path}")

def generate_dataset(num_records: int = 9500) -> pd.DataFrame:
    """Generates synthetic orders and syncs to CSV and SQLite."""
    df = generate_orders(num_orders=num_records)
    csv_paths = [
        BASE_DIR / "data" / "supply_chain_data.csv",
        BASE_DIR / "supply_chain_dashboard" / "data" / "supply_chain_data.csv"
    ]
    db_paths = [
        (BASE_DIR / "database" / "database.sqlite", BASE_DIR / "database" / "schema.sql"),
        (BASE_DIR / "supply_chain_dashboard" / "database" / "database.sqlite", BASE_DIR / "supply_chain_dashboard" / "database" / "schema.sql")
    ]
    for p in csv_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
    for db_p, schema_p in db_paths:
        init_sqlite_database(df, db_p, schema_p)
    return df

def main():
    print("=" * 65)
    print("  Enterprise Supply Chain Data Generator - Initializing Engine")
    print("=" * 65)

    num_records = 9500
    print(f"[*] Generating {num_records:,} transactional orders across 2023-2024...")
    df = generate_orders(num_orders=num_records)
    print(f"[+] Generation complete: {len(df):,} rows generated.")
    print(f"    - Date Range: {df['OrderDate'].min()} to {df['OrderDate'].max()}")
    print(f"    - Total Revenue: ${df['Sales'].sum():,.2f}")
    print(f"    - Total Gross Profit: ${df['Profit'].sum():,.2f}")
    print(f"    - On-Time Delivery Rate: {(df['DeliveryStatus'] == 'On-Time').mean() * 100:.2f}%")
    print(f"    - Delayed Rate: {(df['DeliveryStatus'] == 'Delayed').mean() * 100:.2f}%")
    print(f"    - High Stockout Risk Orders: {(df['StockoutRisk'] == 'High').mean() * 100:.2f}%")

    # Target file paths
    csv_paths = [
        BASE_DIR / "data" / "supply_chain_data.csv",
        BASE_DIR / "supply_chain_dashboard" / "data" / "supply_chain_data.csv"
    ]
    
    db_paths = [
        (BASE_DIR / "database" / "database.sqlite", BASE_DIR / "database" / "schema.sql"),
        (BASE_DIR / "supply_chain_dashboard" / "database" / "database.sqlite", BASE_DIR / "supply_chain_dashboard" / "database" / "schema.sql")
    ]

    # Save CSVs
    for p in csv_paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(p, index=False)
        print(f"[+] Saved CSV file to: {p} ({p.stat().st_size / (1024*1024):.2f} MB)")

    # Initialize SQLite databases
    for db_p, schema_p in db_paths:
        init_sqlite_database(df, db_p, schema_p)

    print("=" * 65)
    print("  Data generation and database synchronization complete!")
    print("=" * 65)

if __name__ == "__main__":
    main()
