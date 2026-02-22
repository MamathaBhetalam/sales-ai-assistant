"""
Metric Tree Engine: Hierarchical root-cause analysis for business metrics.

Tree structure:
  Revenue → Region → Category → Sub-Category → Discount / Quantity
  Profit  → Revenue − Costs − Discount Impact
  Margin  → Profit / Revenue

When the AI is asked "Why did X change?", this module drills down
the tree and returns a structured diagnosis with real numbers.
"""
import pandas as pd
import numpy as np
from typing import Optional


# ── Tree Definition ───────────────────────────────────────────────────────────

METRIC_TREE = {
    "revenue": {
        "formula": "Quantity × Avg Price × (1 − Discount)",
        "dimensions": ["Region", "Category", "Sub-Category", "Segment"],
        "drivers": ["quantity", "price", "discount"],
        "children": ["profit"],
    },
    "profit": {
        "formula": "Revenue × Effective Margin",
        "dimensions": ["Region", "Category", "Sub-Category"],
        "drivers": ["revenue", "discount", "margin"],
        "children": ["profit_margin"],
    },
    "profit_margin": {
        "formula": "Profit ÷ Revenue × 100",
        "dimensions": ["Region", "Category"],
        "drivers": ["profit", "revenue", "discount"],
        "children": [],
    },
    "discount": {
        "formula": "Avg Discount Rate",
        "dimensions": ["Region", "Category", "Sub-Category"],
        "drivers": [],
        "children": ["profit", "profit_margin"],
    },
}


# ── Analysis Functions ────────────────────────────────────────────────────────

def analyze_metric(metric: str, df: pd.DataFrame,
                   compare_df: Optional[pd.DataFrame] = None) -> str:
    """
    Return a textual root-cause analysis for the given metric.
    If compare_df is provided, shows period-over-period change.
    """
    m = metric.lower().strip()

    if "profit" in m and "margin" not in m:
        return _analyze_profit(df, compare_df)
    elif "revenue" in m or "sales" in m:
        return _analyze_revenue(df, compare_df)
    elif "margin" in m:
        return _analyze_margin(df, compare_df)
    elif "discount" in m:
        return _analyze_discount(df, compare_df)
    elif "growth" in m:
        return _analyze_growth(df)
    else:
        return _analyze_overview(df)


def _analyze_profit(df: pd.DataFrame, compare_df: Optional[pd.DataFrame]) -> str:
    """Drill: Profit → Region → Category → Discount impact."""
    lines = ["PROFIT ROOT-CAUSE ANALYSIS (Metric Tree Traversal)", "=" * 52]

    # Level 1 – overall
    total_profit = df["Profit"].sum()
    total_rev = df["Sales"].sum()
    margin = total_profit / total_rev * 100 if total_rev else 0
    avg_disc = df["Discount"].mean() * 100

    lines.append(f"\n[L1] Overall  →  Profit=${total_profit:,.0f}  |  Margin={margin:.1f}%  |  AvgDiscount={avg_disc:.1f}%")

    if compare_df is not None:
        prev_profit = compare_df["Profit"].sum()
        delta = total_profit - prev_profit
        pct = (delta / abs(prev_profit) * 100) if prev_profit else 0
        lines.append(f"       Change vs prior period: {delta:+,.0f} ({pct:+.1f}%)")

    # Level 2 – Region breakdown
    lines.append("\n[L2] Region Drill-Down:")
    reg = (
        df.groupby("Region")
        .agg(Profit=("Profit", "sum"), Revenue=("Sales", "sum"),
             AvgDisc=("Discount", "mean"))
        .sort_values("Profit")
        .reset_index()
    )
    reg["Margin%"] = (reg["Profit"] / reg["Revenue"] * 100).round(1)
    reg["AvgDisc%"] = (reg["AvgDisc"] * 100).round(1)

    worst_region = reg.iloc[0]["Region"]
    for _, row in reg.iterrows():
        flag = " ◄ LOWEST" if row["Region"] == worst_region else ""
        lines.append(
            f"  {row['Region']:<10}  Profit=${row['Profit']:>10,.0f}  "
            f"Margin={row['Margin%']:>5.1f}%  Disc={row['AvgDisc%']:>5.1f}%{flag}"
        )

    # Level 3 – Category in worst region
    lines.append(f"\n[L3] Category Drill-Down inside {worst_region}:")
    cat_region = (
        df[df["Region"] == worst_region]
        .groupby("Category")
        .agg(Profit=("Profit", "sum"), Revenue=("Sales", "sum"),
             AvgDisc=("Discount", "mean"))
        .reset_index()
    )
    cat_region["Margin%"] = (cat_region["Profit"] / cat_region["Revenue"] * 100).round(1)
    cat_region["AvgDisc%"] = (cat_region["AvgDisc"] * 100).round(1)

    for _, row in cat_region.iterrows():
        flag = " ◄ PROBLEM" if row["Margin%"] < 5 else ""
        lines.append(
            f"  {row['Category']:<20}  Profit=${row['Profit']:>9,.0f}  "
            f"Margin={row['Margin%']:>5.1f}%  Disc={row['AvgDisc%']:>5.1f}%{flag}"
        )

    # Level 4 – Discount impact quantification
    lines.append("\n[L4] Discount Impact Quantification:")
    high_disc = df[df["Discount"] > 0.25]
    low_disc = df[df["Discount"] <= 0.25]
    hm = (high_disc["Profit"] / high_disc["Sales"] * 100).mean() if len(high_disc) else 0
    lm = (low_disc["Profit"] / low_disc["Sales"] * 100).mean() if len(low_disc) else 0
    lines.append(f"  Avg Margin (Discount > 25%): {hm:.1f}%  ({len(high_disc):,} orders)")
    lines.append(f"  Avg Margin (Discount ≤ 25%): {lm:.1f}%  ({len(low_disc):,} orders)")
    lines.append(f"  Margin erosion from high discounting: {lm - hm:.1f} percentage points")

    # Root-cause summary
    lines.append("\n[ROOT CAUSE SUMMARY]")
    worst_cat_in_worst = cat_region.loc[cat_region["Margin%"].idxmin()]
    lines.append(
        f"  Primary driver: {worst_region} region → {worst_cat_in_worst['Category']} category "
        f"(Margin: {worst_cat_in_worst['Margin%']:.1f}%, "
        f"Avg Discount: {worst_cat_in_worst['AvgDisc%']:.1f}%)"
    )

    return "\n".join(lines)


def _analyze_revenue(df: pd.DataFrame, compare_df: Optional[pd.DataFrame]) -> str:
    """Drill: Revenue → Region × Category."""
    lines = ["REVENUE ROOT-CAUSE ANALYSIS (Metric Tree Traversal)", "=" * 52]

    total_rev = df["Sales"].sum()
    lines.append(f"\n[L1] Total Revenue: ${total_rev:,.0f}")

    if compare_df is not None:
        prev = compare_df["Sales"].sum()
        delta = total_rev - prev
        pct = (delta / prev * 100) if prev else 0
        lines.append(f"       vs prior period: {delta:+,.0f} ({pct:+.1f}%)")

    # Region
    lines.append("\n[L2] Revenue by Region:")
    reg = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
    for region, rev in reg.items():
        share = rev / total_rev * 100
        lines.append(f"  {region:<10}  ${rev:>10,.0f}  ({share:.1f}% share)")

    # Category
    lines.append("\n[L3] Revenue by Category:")
    cat = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
    for c, rev in cat.items():
        share = rev / total_rev * 100
        lines.append(f"  {c:<22}  ${rev:>9,.0f}  ({share:.1f}% share)")

    # Volume vs Price
    lines.append("\n[L4] Volume vs Pricing Drivers:")
    lines.append(f"  Total Quantity Sold : {df['Quantity'].sum():>10,}")
    lines.append(f"  Avg Revenue/Order   : ${df.groupby('Order ID')['Sales'].sum().mean():>9,.0f}")
    lines.append(f"  Avg Discount Applied: {df['Discount'].mean() * 100:>9.1f}%")

    return "\n".join(lines)


def _analyze_margin(df: pd.DataFrame, compare_df: Optional[pd.DataFrame]) -> str:
    """Drill: Margin by region × category × discount band."""
    lines = ["PROFIT MARGIN ROOT-CAUSE ANALYSIS", "=" * 52]

    overall = df["Profit"].sum() / df["Sales"].sum() * 100
    lines.append(f"\n[L1] Overall Margin: {overall:.1f}%")

    # By category (biggest spread)
    lines.append("\n[L2] Margin by Category:")
    cat = (
        df.groupby("Category")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), AvgDisc=("Discount", "mean"))
        .reset_index()
    )
    cat["Margin%"] = cat["Profit"] / cat["Revenue"] * 100
    cat["AvgDisc%"] = cat["AvgDisc"] * 100
    for _, row in cat.iterrows():
        lines.append(
            f"  {row['Category']:<22}  Margin={row['Margin%']:>5.1f}%  "
            f"Discount={row['AvgDisc%']:>5.1f}%"
        )

    # By region
    lines.append("\n[L3] Margin by Region:")
    reg = (
        df.groupby("Region")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    reg["Margin%"] = reg["Profit"] / reg["Revenue"] * 100
    for _, row in reg.sort_values("Margin%").iterrows():
        flag = " ◄ BELOW TARGET" if row["Margin%"] < 10 else ""
        lines.append(f"  {row['Region']:<10}  Margin={row['Margin%']:>5.1f}%{flag}")

    # Discount bands
    lines.append("\n[L4] Margin by Discount Band:")
    df2 = df.copy()
    df2["DiscBand"] = pd.cut(df2["Discount"], bins=[0, 0.1, 0.2, 0.3, 0.5],
                             labels=["0–10%", "10–20%", "20–30%", "30%+"])
    band = (
        df2.groupby("DiscBand", observed=True)
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"), Orders=("Order ID", "count"))
        .reset_index()
    )
    band["Margin%"] = band["Profit"] / band["Revenue"] * 100
    for _, row in band.iterrows():
        lines.append(
            f"  Discount {str(row['DiscBand']):<8}  "
            f"Margin={row['Margin%']:>5.1f}%  Orders={row['Orders']:,}"
        )

    return "\n".join(lines)


def _analyze_discount(df: pd.DataFrame, compare_df: Optional[pd.DataFrame]) -> str:
    """Analyze discount patterns and their profit impact."""
    lines = ["DISCOUNT IMPACT ANALYSIS", "=" * 52]

    avg_disc = df["Discount"].mean() * 100
    lines.append(f"\n[L1] Overall Avg Discount: {avg_disc:.1f}%")

    # By region
    lines.append("\n[L2] Avg Discount by Region:")
    reg = df.groupby("Region")["Discount"].mean().sort_values(ascending=False) * 100
    for region, d in reg.items():
        lines.append(f"  {region:<10}  {d:.1f}%")

    # By category
    lines.append("\n[L3] Avg Discount by Category:")
    cat = df.groupby("Category")["Discount"].mean().sort_values(ascending=False) * 100
    for c, d in cat.items():
        lines.append(f"  {c:<22}  {d:.1f}%")

    # High-discount orders impact
    lines.append("\n[L4] High-Discount (>30%) Order Analysis:")
    hi = df[df["Discount"] > 0.30]
    if len(hi):
        hi_margin = hi["Profit"].sum() / hi["Sales"].sum() * 100
        hi_rev_share = hi["Sales"].sum() / df["Sales"].sum() * 100
        lines.append(f"  Orders with >30% discount: {len(hi):,} ({hi_rev_share:.1f}% of revenue)")
        lines.append(f"  Their avg margin: {hi_margin:.1f}%  (vs overall {avg_disc:.1f}%)")
    else:
        lines.append("  No orders with >30% discount found.")

    return "\n".join(lines)


def _analyze_growth(df: pd.DataFrame) -> str:
    """YoY and QoQ growth analysis."""
    lines = ["GROWTH ANALYSIS", "=" * 52]

    yearly = (
        df.groupby("Year")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    yearly["RevGrowth%"] = yearly["Revenue"].pct_change() * 100
    yearly["PrfGrowth%"] = yearly["Profit"].pct_change() * 100
    yearly["Margin%"] = yearly["Profit"] / yearly["Revenue"] * 100

    lines.append("\n[L1] Year-over-Year:")
    for _, row in yearly.iterrows():
        g = f"{row['RevGrowth%']:+.1f}%" if not pd.isna(row["RevGrowth%"]) else "—"
        lines.append(
            f"  {int(row['Year'])}  Revenue=${row['Revenue']:>10,.0f}  "
            f"Growth={g:>8}  Margin={row['Margin%']:.1f}%"
        )

    # Quarterly
    lines.append("\n[L2] Quarterly Revenue Trend:")
    qtr = (
        df.groupby(["Year", "Quarter"])
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    qtr["YQ"] = qtr["Year"].astype(str) + " " + qtr["Quarter"]
    for _, row in qtr.tail(8).iterrows():
        lines.append(f"  {row['YQ']}  Revenue=${row['Revenue']:>10,.0f}")

    return "\n".join(lines)


def _analyze_overview(df: pd.DataFrame) -> str:
    """General business overview traversal."""
    lines = ["BUSINESS OVERVIEW — METRIC TREE SUMMARY", "=" * 52]

    rev = df["Sales"].sum()
    prf = df["Profit"].sum()
    margin = prf / rev * 100 if rev else 0

    lines.append(f"\nRevenue : ${rev:>12,.0f}")
    lines.append(f"Profit  : ${prf:>12,.0f}  (Margin: {margin:.1f}%)")

    best_reg = df.groupby("Region")["Sales"].sum().idxmax()
    best_cat = df.groupby("Category")["Profit"].sum().idxmax()
    lines.append(f"\nTop Region   : {best_reg}")
    lines.append(f"Top Category : {best_cat}")

    return "\n".join(lines)


# ── Question → Relevant Metric Detection ─────────────────────────────────────

def detect_metric(question: str) -> str:
    """Map user question to the most relevant metric for tree traversal."""
    q = question.lower()
    if any(w in q for w in ["profit drop", "profit fell", "profit down", "profit declin", "profit loss"]):
        return "profit"
    if any(w in q for w in ["margin", "profitability"]):
        return "profit_margin"
    if any(w in q for w in ["revenue drop", "revenue fell", "revenue down", "sales down", "sales declin"]):
        return "revenue"
    if any(w in q for w in ["discount", "promotion", "pricing"]):
        return "discount"
    if any(w in q for w in ["growth", "trend", "year over year", "yoy", "quarter"]):
        return "growth"
    if any(w in q for w in ["profit", "earn", "income"]):
        return "profit"
    if any(w in q for w in ["revenue", "sales", "turnover"]):
        return "revenue"
    return "overview"


def get_tree_analysis(question: str, df: pd.DataFrame,
                      compare_df: Optional[pd.DataFrame] = None) -> str:
    """Entry point: auto-detect metric and run full tree traversal."""
    metric = detect_metric(question)
    return analyze_metric(metric, df, compare_df)
