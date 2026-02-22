"""
Data Layer: Loads and processes Superstore CSV sales data.
Computes KPIs, breakdowns, and time-series for the AI context.
"""
import pandas as pd
import os

# Resolve the CSV path relative to this file's location (project root)
_DATA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CSV_PATH = os.path.join(_DATA_DIR, "superstore.csv")


def generate_data() -> pd.DataFrame:
    """
    Load the Superstore CSV dataset and return a clean DataFrame
    with Year, Quarter, and Month columns derived from Order Date.
    """
    df = pd.read_csv(_CSV_PATH, parse_dates=["Order Date"])
    df = df.sort_values("Order Date").reset_index(drop=True)

    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = "Q" + df["Order Date"].dt.quarter.astype(str)

    # Keep only the columns the rest of the app expects
    cols = [
        "Order ID", "Order Date", "Year", "Quarter", "Month",
        "Region", "Segment", "Category", "Sub-Category",
        "Sales", "Quantity", "Discount", "Profit",
    ]
    return df[cols]


# ── KPI Computation ──────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame) -> dict:
    """Compute full KPI set including YoY, regional, and category breakdowns."""
    rev = df["Sales"].sum()
    profit = df["Profit"].sum()
    margin = (profit / rev * 100) if rev else 0
    orders = df["Order ID"].nunique()
    avg_disc = df["Discount"].mean() * 100

    # YoY: use the two most recent years in the dataset
    latest_year = int(df["Year"].max())
    prev_year = latest_year - 1
    d_prev = df[df["Year"] == prev_year]
    d_latest = df[df["Year"] == latest_year]
    rev_prev, rev_latest = d_prev["Sales"].sum(), d_latest["Sales"].sum()
    prf_prev, prf_latest = d_prev["Profit"].sum(), d_latest["Profit"].sum()
    yoy_rev = (rev_latest - rev_prev) / rev_prev * 100 if rev_prev else 0
    yoy_prf = (prf_latest - prf_prev) / prf_prev * 100 if prf_prev else 0

    # Regional
    regional = (
        df.groupby("Region")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "count"))
        .reset_index()
    )
    regional["Margin%"] = (regional["Profit"] / regional["Revenue"] * 100).round(1)
    regional["AvgDiscount%"] = (
        df.groupby("Region")["Discount"].mean().values * 100
    ).round(1)
    regional = regional.sort_values("Revenue", ascending=False).reset_index(drop=True)

    # Category
    cat = (
        df.groupby("Category")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), AvgDiscount=("Discount", "mean"))
        .reset_index()
    )
    cat["Margin%"] = (cat["Profit"] / cat["Revenue"] * 100).round(1)
    cat["AvgDiscount%"] = (cat["AvgDiscount"] * 100).round(1)
    cat = cat.drop(columns="AvgDiscount")

    # Sub-category
    sub = (
        df.groupby(["Category", "Sub-Category"])
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), AvgDiscount=("Discount", "mean"))
        .reset_index()
    )
    sub["Margin%"] = (sub["Profit"] / sub["Revenue"] * 100).round(1)
    sub["AvgDiscount%"] = (sub["AvgDiscount"] * 100).round(1)
    sub = sub.drop(columns="AvgDiscount").sort_values("Revenue", ascending=False)

    # Quarterly
    quarterly = (
        df.groupby(["Year", "Quarter"])
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    quarterly["Margin%"] = (quarterly["Profit"] / quarterly["Revenue"] * 100).round(1)
    quarterly["YQ"] = quarterly["Year"].astype(str) + " " + quarterly["Quarter"]

    return {
        "total_revenue": rev,
        "total_profit": profit,
        "profit_margin": margin,
        "total_orders": orders,
        "avg_discount": avg_disc,
        "yoy_revenue_growth": yoy_rev,
        "yoy_profit_growth": yoy_prf,
        "latest_year": latest_year,
        "prev_year": prev_year,
        "revenue_latest": rev_latest,
        "revenue_prev": rev_prev,
        "profit_latest": prf_latest,
        "profit_prev": prf_prev,
        # legacy aliases so existing app.py references still resolve
        "revenue_2024": rev_latest,
        "revenue_2023": rev_prev,
        "profit_2024": prf_latest,
        "profit_2023": prf_prev,
        "regional": regional,
        "category": cat,
        "sub_category": sub,
        "quarterly": quarterly,
        "best_region": regional.iloc[0]["Region"],
        "worst_region": regional.iloc[-1]["Region"],
        "best_margin_cat": cat.loc[cat["Margin%"].idxmax(), "Category"],
        "worst_margin_cat": cat.loc[cat["Margin%"].idxmin(), "Category"],
        "high_discount_cat": cat.loc[cat["AvgDiscount%"].idxmax(), "Category"],
    }


def get_time_series(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """Monthly or quarterly revenue/profit time series."""
    tmp = df.copy()
    tmp["Period"] = tmp["Order Date"].dt.to_period(freq)
    ts = (
        tmp.groupby("Period")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "count"))
        .reset_index()
    )
    ts["Period"] = ts["Period"].astype(str)
    ts["Margin%"] = (ts["Profit"] / ts["Revenue"] * 100).round(1)
    return ts


def get_filtered(df: pd.DataFrame, year: int = None, region: str = None,
                 category: str = None) -> pd.DataFrame:
    """Return filtered slice of the dataframe."""
    mask = pd.Series([True] * len(df))
    if year:
        mask &= df["Year"] == year
    if region:
        mask &= df["Region"] == region
    if category:
        mask &= df["Category"] == category
    return df[mask]


def format_context(kpis: dict) -> str:
    """Render KPIs as a compact text block for AI prompt injection."""
    reg = kpis["regional"]
    cat = kpis["category"]
    qtr = kpis["quarterly"]

    # Recent quarters (last 8)
    recent_q = qtr.tail(8)[["YQ", "Revenue", "Profit", "Margin%"]].to_string(index=False)

    reg_str = reg[["Region", "Revenue", "Profit", "Margin%", "AvgDiscount%"]].to_string(index=False)
    cat_str = cat[["Category", "Revenue", "Profit", "Margin%", "AvgDiscount%"]].to_string(index=False)

    yr1 = kpis.get("prev_year", "prior year")
    yr2 = kpis.get("latest_year", "latest year")
    years_range = f"{min(yr1, yr2) - (yr2 - yr1)}–{yr2}"  # approximate full span

    return f"""
=== BUSINESS INTELLIGENCE SNAPSHOT ({yr1}–{yr2}) ===

OVERALL KPIs:
  Total Revenue  : ${kpis['total_revenue']:>12,.0f}
  Total Profit   : ${kpis['total_profit']:>12,.0f}
  Profit Margin  : {kpis['profit_margin']:>8.1f}%
  Total Orders   : {kpis['total_orders']:>12,}
  Avg Discount   : {kpis['avg_discount']:>8.1f}%

YEAR-OVER-YEAR ({yr1} → {yr2}):
  Revenue Growth : {kpis['yoy_revenue_growth']:>+8.1f}%  (${kpis['revenue_prev']:,.0f} → ${kpis['revenue_latest']:,.0f})
  Profit Growth  : {kpis['yoy_profit_growth']:>+8.1f}%  (${kpis['profit_prev']:,.0f} → ${kpis['profit_latest']:,.0f})

REGIONAL BREAKDOWN:
{reg_str}

CATEGORY BREAKDOWN:
{cat_str}

RECENT QUARTERLY TREND:
{recent_q}

KEY FINDINGS:
  • Best revenue region  : {kpis['best_region']}
  • Weakest region       : {kpis['worst_region']}
  • Highest margin cat.  : {kpis['best_margin_cat']}
  • Lowest margin cat.   : {kpis['worst_margin_cat']}
  • Most discounted cat. : {kpis['high_discount_cat']}
""".strip()
