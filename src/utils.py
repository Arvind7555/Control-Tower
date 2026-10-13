"""
src/utils.py
Enterprise Dashboard Utilities: Formatting, Database Connections, Plotly Themes, and Styling.
"""

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Union, Tuple
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

# Base paths resolution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if PROJECT_ROOT.name == "supply_chain_dashboard":
    WORKSPACE_ROOT = PROJECT_ROOT.parent
else:
    WORKSPACE_ROOT = PROJECT_ROOT

def get_database_path() -> Path:
    """Returns the absolute path to database.sqlite."""
    candidates = [
        PROJECT_ROOT / "database" / "database.sqlite",
        WORKSPACE_ROOT / "database" / "database.sqlite",
        WORKSPACE_ROOT / "supply_chain_dashboard" / "database" / "database.sqlite"
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / "database" / "database.sqlite"

def get_csv_path() -> Path:
    """Returns the absolute path to supply_chain_data.csv."""
    candidates = [
        PROJECT_ROOT / "data" / "supply_chain_data.csv",
        WORKSPACE_ROOT / "data" / "supply_chain_data.csv",
        WORKSPACE_ROOT / "supply_chain_dashboard" / "data" / "supply_chain_data.csv"
    ]
    for p in candidates:
        if p.exists():
            return p
    return PROJECT_ROOT / "data" / "supply_chain_data.csv"

def ensure_dataset_ready() -> Tuple[Path, Path]:
    """
    Guarantees supply_chain_data.csv and database.sqlite exist and are populated with transactions.
    If missing (e.g. on fresh deployment to Streamlit Cloud or Docker), generates 9,500 orders on the fly.
    """
    csv_path = get_csv_path()
    db_path = get_database_path()

    needs_generation = False
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        needs_generation = True
    elif not db_path.exists() or db_path.stat().st_size == 0:
        needs_generation = True
    else:
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cur = conn.cursor()
                cur.execute("SELECT count(*) FROM sqlite_master WHERE type='table' AND name='orders'")
                if cur.fetchone()[0] == 0:
                    needs_generation = True
        except Exception:
            needs_generation = True

    if needs_generation:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        db_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            from src.data_generator import generate_orders
            df = generate_orders(num_orders=9500)
        except Exception:
            try:
                from data_generator import generate_orders
                df = generate_orders(num_orders=9500)
            except Exception:
                import importlib.util
                matches = list(PROJECT_ROOT.rglob("data_generator.py")) + list(WORKSPACE_ROOT.rglob("data_generator.py"))
                if matches:
                    spec = importlib.util.spec_from_file_location("dg", str(matches[0]))
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    df = mod.generate_orders(9500)
                else:
                    raise FileNotFoundError("Could not locate data_generator.py to generate transactional data.")

        # Save CSV file
        df.to_csv(csv_path, index=False)

        # Save SQLite DB file with indexes
        with sqlite3.connect(str(db_path)) as conn:
            df.to_sql("orders", conn, if_exists="replace", index=False)
            cur = conn.cursor()
            cur.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON orders(OrderDate);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_region ON orders(Region);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_category ON orders(ProductCategory);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_delivery_status ON orders(DeliveryStatus);")
            conn.commit()

    return csv_path, db_path

def ensure_database_exists(db_path: Optional[Path] = None) -> Path:
    """Guarantees SQLite database exists and is populated with transactions table."""
    _, db_p = ensure_dataset_ready()
    return db_p

@contextmanager
def get_db_connection(db_path: Optional[Union[str, Path]] = None):
    """Yields a sqlite3 connection with Row factory enabled and auto-closes upon exit."""
    if db_path is None:
        _, db_path = ensure_dataset_ready()
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def format_currency(val: float, compact: bool = True) -> str:
    """Formats numerical value to Indian Rupee (₹) currency representation."""
    if val is None or pd.isna(val):
        return "₹0.00"
    if compact:
        if abs(val) >= 1_000_000:
            return f"₹{val / 1_000_000:.2f}M"
        if abs(val) >= 1_000:
            return f"₹{val / 1_000:.1f}K"
    return f"₹{val:,.2f}"

def format_percentage(val: float, decimals: int = 1) -> str:
    """Formats numeric value as percentage string."""
    if val is None or pd.isna(val):
        return "0.0%"
    return f"{val:.{decimals}f}%"

def format_number(val: Union[int, float], compact: bool = False) -> str:
    """Formats integer or float with thousand commas or compact suffixes."""
    if val is None or pd.isna(val):
        return "0"
    if compact:
        if abs(val) >= 1_000_000:
            return f"{val / 1_000_000:.1f}M"
        if abs(val) >= 1_000:
            return f"{val / 1_000:.1f}K"
    return f"{val:,.0f}" if isinstance(val, (int, float)) and val.is_integer() else f"{val:,.1f}"

# Executive Color Palette
EXECUTIVE_PALETTE = {
    "primary": "#2563EB",       # Royal Blue
    "secondary": "#4F46E5",     # Indigo
    "accent": "#06B6D4",        # Cyan
    "success": "#10B981",       # Emerald Green
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Crimson Red
    "neutral_dark": "#0F172A",  # Slate 900
    "neutral_light": "#F8FAFC", # Slate 50
    "card_bg": "#FFFFFF",
    "text_dark": "#0F172A",
    "text_muted": "#475569",
    "border": "#E2E8F0"
}

CATEGORY_COLORS = {
    "Electronics": "#3B82F6",
    "Furniture": "#8B5CF6",
    "Office Supplies": "#10B981",
    "Industrial": "#F97316"
}

STATUS_COLORS = {
    "On-Time": "#10B981",
    "Delayed": "#EF4444",
    "Cancelled": "#6B7280"
}

def apply_plotly_theme(fig: go.Figure, height: int = 420, title: Optional[str] = None) -> go.Figure:
    """Applies executive styling, clean margins, and readable typography to Plotly charts."""
    fig.update_layout(
        template="plotly_white",
        height=max(height, 420),
        title=dict(
            text=f"<b>{title}</b>" if title else "",
            font=dict(size=15, color="#0F172A", family="Inter, system-ui, -apple-system, sans-serif"),
            x=0.01,
            y=0.96
        ) if title else None,
        margin=dict(l=45, r=30, t=55 if title else 35, b=45),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1.0,
            font=dict(size=12, color="#334155")
        ),
        font=dict(family="Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif", size=12, color="#0F172A"),
        hoverlabel=dict(
            bgcolor="#0F172A",
            font_size=12,
            font_color="#FFFFFF",
            font_family="Inter, sans-serif"
        ),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF"
    )
    
    # Gridline and axis polish
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=11, family="Inter, sans-serif")
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#F1F5F9",
        linecolor="#CBD5E1",
        tickfont=dict(color="#334155", size=11, family="Inter, sans-serif")
    )
    return fig

def get_custom_css() -> str:
    """Returns rock-solid production CSS that preserves native BaseWeb rendering while adding executive polish."""
    return """
    <style>
        /* Base page container spacing */
        .block-container {
            padding-top: 1.8rem !important;
            padding-bottom: 2.5rem !important;
            max-width: 98% !important;
        }

        /* Top Executive Banner */
        .exec-banner {
            background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
            border-radius: 14px;
            padding: 22px 28px;
            color: #FFFFFF;
            box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.15);
            margin-bottom: 22px;
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .exec-banner h1 {
            font-size: 1.85rem !important;
            font-weight: 800 !important;
            color: #FFFFFF !important;
            margin: 0 !important;
            padding: 0 !important;
            letter-spacing: -0.02em;
        }

        .exec-banner p {
            font-size: 0.95rem !important;
            color: #94A3B8 !important;
            margin: 6px 0 0 0 !important;
            padding: 0 !important;
        }

        /* Executive KPI Metric Cards (No clipping, generous padding) */
        .metric-card {
            background: #FFFFFF;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 2px 10px rgba(15, 23, 42, 0.05);
            border: 1px solid #E2E8F0;
            display: flex;
            flex-direction: column;
            justify-content: flex-start;
            min-height: 115px;
            margin-bottom: 12px;
            position: relative;
        }

        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: #2563EB;
            border-radius: 4px 0 0 4px;
        }

        .metric-card.success::before { background: #10B981; }
        .metric-card.warning::before { background: #F59E0B; }
        .metric-card.danger::before { background: #EF4444; }
        .metric-card.purple::before { background: #8B5CF6; }

        .metric-title {
            color: #475569;
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 6px;
        }

        .metric-value {
            color: #0F172A;
            font-size: 1.70rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            line-height: 1.15;
            margin-bottom: 8px;
        }

        .metric-badge {
            display: inline-block;
            width: fit-content;
            padding: 3px 8px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 700;
        }

        .badge-positive {
            background-color: #ECFDF5;
            color: #065F46;
            border: 1px solid #A7F3D0;
        }

        .badge-negative {
            background-color: #FEF2F2;
            color: #991B1B;
            border: 1px solid #FECACA;
        }

        .badge-neutral {
            background-color: #F1F5F9;
            color: #334155;
            border: 1px solid #E2E8F0;
        }

        /* Streamlit Tabs Navigation Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #F1F5F9;
            padding: 6px;
            border-radius: 10px;
            margin-bottom: 20px;
            border: 1px solid #E2E8F0;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            padding: 8px 18px;
            font-weight: 700;
            font-size: 0.90rem;
            color: #475569;
            background-color: transparent;
            border: none;
        }

        .stTabs [aria-selected="true"] {
            background-color: #FFFFFF !important;
            color: #2563EB !important;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.08) !important;
        }

        /* Section Headings */
        h4 {
            color: #0F172A !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
            margin-top: 10px !important;
            margin-bottom: 12px !important;
        }

        /* Sidebar Info Card */
        .sidebar-card {
            background: #FFFFFF;
            border: 1px solid #CBD5E1;
            border-radius: 8px;
            padding: 12px 14px;
            font-size: 0.82rem;
            color: #0F172A;
            margin-top: 16px;
        }

        /* ML Risk Card */
        .risk-container {
            border-radius: 12px;
            padding: 20px 24px;
            margin: 12px 0 20px 0;
            border: 2px solid;
            background: #FFFFFF;
        }

        .risk-high {
            border-color: #EF4444;
            background: #FEF2F2;
        }

        .risk-medium {
            border-color: #F59E0B;
            background: #FFFBEB;
        }

        .risk-low {
            border-color: #10B981;
            background: #ECFDF5;
        }

        .risk-pill {
            display: inline-block;
            font-size: 0.95rem;
            font-weight: 800;
            padding: 6px 14px;
            border-radius: 9999px;
            text-transform: uppercase;
        }

        .pill-high { background: #EF4444; color: #FFFFFF; }
        .pill-medium { background: #F59E0B; color: #FFFFFF; }
        .pill-low { background: #10B981; color: #FFFFFF; }
    </style>
    """
