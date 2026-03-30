"""
Metric Tree Engine: Hierarchical analysis for the E-Commerce dataset.

Tree structure:
  Revenue → Channel → Conversion → Ratings
  Orders  → By Status → By Country → By Device
  Customers → Retention → Gender → Geography
  Quality → Product Rating → Delivery Rating → by Channel

When the AI is asked "Why did X change?", this module drills down
the tree and returns a structured diagnosis with real numbers.
"""
import pandas as pd
from typing import Optional


# ── Tree Definition ───────────────────────────────────────────────────────────

METRIC_TREE = {
    "revenue": {
        "formula": "Total Revenue = sum(Sales) for completed orders",
        "dimensions": ["TrafficSource", "Country", "DeviceCategory", "Gender"],
        "drivers": ["order volume", "average order value", "completion rate", "channel mix"],
        "children": ["orders", "channel", "geography"],
    },
    "orders": {
        "formula": "Total Orders = count(InvoiceNumber); Completion Rate = Completed / Total",
        "dimensions": ["OrderStatus", "TrafficSource", "Country"],
        "drivers": ["traffic volume", "conversion", "cancellations"],
        "children": ["channel", "geography"],
    },
    "channel": {
        "formula": "Revenue per Channel = sum(Sales) grouped by TrafficSource",
        "dimensions": ["TrafficSource"],
        "drivers": ["session quality", "channel conversion", "AOV by channel"],
        "children": ["ratings"],
    },
    "geography": {
        "formula": "Revenue by Country = sum(Sales) grouped by Country",
        "dimensions": ["Country"],
        "drivers": ["market size", "order frequency", "product fit"],
        "children": [],
    },
    "ratings": {
        "formula": "Avg Product Rating + Avg Delivery Rating for completed orders",
        "dimensions": ["TrafficSource", "Country", "DeviceCategory"],
        "drivers": ["product quality", "delivery speed", "customer expectations"],
        "children": [],
    },
}


# ── Analysis Functions ─────────────────────────────────────────────────────────

def analyze_metric(metric: str, df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    m = metric.lower().strip()

    if any(w in m for w in ["revenue", "sales", "income", "earning"]):
        return _analyze_revenue(df, kpis)
    elif any(w in m for w in ["order", "conversion", "completion", "status"]):
        return _analyze_orders(df, kpis)
    elif any(w in m for w in ["channel", "traffic", "source", "acquisition"]):
        return _analyze_channels(df, kpis)
    elif any(w in m for w in ["country", "geography", "region", "location", "market"]):
        return _analyze_geography(df, kpis)
    elif any(w in m for w in ["rating", "quality", "satisfaction", "delivery", "product"]):
        return _analyze_ratings(df, kpis)
    elif any(w in m for w in ["customer", "gender", "device", "segment"]):
        return _analyze_customers(df, kpis)
    else:
        return _analyze_overview(df, kpis)


def _analyze_revenue(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["REVENUE ROOT-CAUSE ANALYSIS (Metric Tree Traversal)", "=" * 55]

    total_rev = kpis["total_revenue"] if kpis else float(df["revenue"].sum())
    aov       = kpis["avg_order_value"] if kpis else float(df[df["is_completed"]]["Total"].mean())
    comp_rate = kpis["completion_rate"] if kpis else 0.0
    lines.append(
        f"\n[L1] Overall  →  Total Revenue = ${total_rev:,.2f} | "
        f"AOV = ${aov:.2f} | Completion Rate = {comp_rate:.1f}%"
    )

    by_ch = kpis.get("by_channel") if kpis else None
    if by_ch is not None and not by_ch.empty:
        lines.append("\n[L2] Revenue by Traffic Source:")
        for _, row in by_ch.iterrows():
            share = row["revenue"] / total_rev * 100 if total_rev else 0
            lines.append(
                f"  {str(row['TrafficSource']):<30}  "
                f"Revenue=${row['revenue']:>10,.2f}  ({share:.1f}%)  "
                f"Orders={int(row['orders']):,}  Conv={row['completion_rate']:.1f}%"
            )

    by_co = kpis.get("by_country") if kpis else None
    if by_co is not None and not by_co.empty:
        lines.append("\n[L3] Revenue by Country (top 10):")
        for _, row in by_co.head(10).iterrows():
            share = row["revenue"] / total_rev * 100 if total_rev else 0
            lines.append(
                f"  {str(row['Country']):<25}  "
                f"Revenue=${row['revenue']:>10,.2f}  ({share:.1f}%)  "
                f"Orders={int(row['orders']):,}"
            )

    return "\n".join(lines)


def _analyze_orders(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["ORDER ANALYSIS (Metric Tree Traversal)", "=" * 55]

    total = kpis["total_orders"] if kpis else len(df)
    comp  = kpis["completed_orders"] if kpis else int(df["is_completed"].sum())
    rate  = kpis["completion_rate"] if kpis else comp / total * 100
    lines.append(f"\n[L1] Total Orders: {total:,} | Completed: {comp:,} ({rate:.1f}%)")

    sb = kpis.get("status_breakdown") if kpis else None
    if sb is not None and not sb.empty:
        lines.append("\n[L2] Order Status Breakdown:")
        for _, row in sb.iterrows():
            pct = row["count"] / total * 100 if total else 0
            lines.append(f"  {str(row['status']):<20}  {row['count']:,} ({pct:.1f}%)")

    by_ch = kpis.get("by_channel") if kpis else None
    if by_ch is not None and not by_ch.empty:
        lines.append("\n[L3] Orders & Completion Rate by Channel:")
        for _, row in by_ch.sort_values("orders", ascending=False).iterrows():
            lines.append(
                f"  {str(row['TrafficSource']):<30}  "
                f"Orders={int(row['orders']):,}  Conv={row['completion_rate']:.1f}%"
            )

    return "\n".join(lines)


def _analyze_channels(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["CHANNEL ANALYSIS (Metric Tree Traversal)", "=" * 55]

    by_ch = kpis.get("by_channel") if kpis else None
    if by_ch is None or by_ch.empty:
        lines.append("\n  Channel data not available.")
        return "\n".join(lines)

    total_rev      = kpis["total_revenue"] if kpis else 1.0
    overall_rating = kpis["avg_product_rating"] if kpis else 0.0

    has_ltv = "avg_ltv" in by_ch.columns

    lines.append(f"\n[L1] Total Channels: {len(by_ch)}")
    lines.append("\n[L2] Full Channel Breakdown:")
    for _, row in by_ch.iterrows():
        rev_share   = row["revenue"] / total_rev * 100 if total_rev else 0
        ltv_str     = f"  LTV=${row['avg_ltv']:.2f}" if has_ltv and pd.notna(row.get("avg_ltv")) else ""
        rating_flag = ""
        if pd.notna(row["avg_product_rating"]):
            if row["avg_product_rating"] >= overall_rating + 0.3:
                rating_flag = " ◄ HIGH QUALITY"
            elif row["avg_product_rating"] < overall_rating - 0.3:
                rating_flag = " ◄ LOW QUALITY"
        lines.append(
            f"  {str(row['TrafficSource']):<30}  "
            f"Revenue=${row['revenue']:>10,.2f} ({rev_share:.1f}%)  "
            f"Orders={int(row['orders']):,}  Conv={row['completion_rate']:.1f}%  "
            f"Rating={row['avg_product_rating']:.2f}  Session={row['avg_session']:.1f}min"
            f"{ltv_str}{rating_flag}"
        )

    best  = by_ch.loc[by_ch["completion_rate"].idxmax()]
    worst = by_ch.loc[by_ch["completion_rate"].idxmin()]
    lines.append(f"\n[L3] Best Conversion  : {best['TrafficSource']} ({best['completion_rate']:.1f}%)")
    lines.append(f"     Worst Conversion : {worst['TrafficSource']} ({worst['completion_rate']:.1f}%)")

    if has_ltv and not by_ch["avg_ltv"].isna().all():
        best_ltv  = by_ch.loc[by_ch["avg_ltv"].idxmax()]
        worst_ltv = by_ch.loc[by_ch["avg_ltv"].idxmin()]
        lines.append(f"\n[L4] Highest LTV Channel : {best_ltv['TrafficSource']} (${best_ltv['avg_ltv']:.2f}/customer)")
        lines.append(f"     Lowest  LTV Channel : {worst_ltv['TrafficSource']} (${worst_ltv['avg_ltv']:.2f}/customer)")
        _ltv_gap = best_ltv["avg_ltv"] - worst_ltv["avg_ltv"]
        lines.append(f"     LTV Gap              : ${_ltv_gap:.2f} — shift budget toward {best_ltv['TrafficSource']}")

    return "\n".join(lines)


def _analyze_geography(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["GEOGRAPHY ANALYSIS (Metric Tree Traversal)", "=" * 55]

    by_co = kpis.get("by_country") if kpis else None
    if by_co is None or by_co.empty:
        lines.append("\n  Country data not available.")
        return "\n".join(lines)

    total_rev  = kpis["total_revenue"] if kpis else 1.0
    total_cust = kpis["unique_customers"] if kpis else 1

    lines.append(f"\n[L1] Total Countries: {len(by_co)}")
    lines.append("\n[L2] Revenue by Country (all):")
    for _, row in by_co.iterrows():
        rev_share  = row["revenue"] / total_rev * 100 if total_rev else 0
        cust_share = row["unique_customers"] / total_cust * 100 if total_cust else 0
        lines.append(
            f"  {str(row['Country']):<25}  "
            f"Revenue=${row['revenue']:>10,.2f} ({rev_share:.1f}%)  "
            f"Orders={int(row['orders']):,}  "
            f"Customers={int(row['unique_customers']):,} ({cust_share:.1f}%)"
        )

    return "\n".join(lines)


def _analyze_ratings(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["QUALITY / RATINGS ANALYSIS (Metric Tree Traversal)", "=" * 55]

    avg_pr = kpis["avg_product_rating"] if kpis else 0.0
    avg_dr = kpis["avg_delivery_rating"] if kpis else 0.0
    lines.append(
        f"\n[L1] Avg Product Rating = {avg_pr:.2f}/5  |  "
        f"Avg Delivery Rating = {avg_dr:.2f}/5"
    )

    by_ch = kpis.get("by_channel") if kpis else None
    if by_ch is not None and not by_ch.empty:
        lines.append("\n[L2] Ratings by Traffic Source:")
        for _, row in by_ch.iterrows():
            lines.append(
                f"  {str(row['TrafficSource']):<30}  "
                f"Product={row['avg_product_rating']:.2f}  "
                f"Delivery={row['avg_delivery_rating']:.2f}"
            )

    rd = kpis.get("rating_dist") if kpis else None
    if rd is not None and not rd.empty:
        total_rated = int(rd["count"].sum())
        lines.append(f"\n[L3] Product Rating Distribution (n={total_rated:,}):")
        for _, row in rd.iterrows():
            pct = row["count"] / total_rated * 100 if total_rated else 0
            bar = "█" * int(pct / 3)
            lines.append(f"  {str(row['band']):<8}  {int(row['count']):>5,}  {bar} {pct:.1f}%")

    return "\n".join(lines)


def _analyze_customers(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["CUSTOMER ANALYSIS (Metric Tree Traversal)", "=" * 55]

    unique        = kpis["unique_customers"] if kpis else int(df["CustomerID"].nunique())
    total         = kpis["total_orders"] if kpis else len(df)
    avg_orders    = total / unique if unique else 0
    retention     = kpis.get("retention_rate", 0) if kpis else 0
    repeat_custs  = kpis.get("repeat_customers", 0) if kpis else 0
    avg_freq      = kpis.get("avg_purchase_freq", avg_orders) if kpis else avg_orders
    lines.append(f"\n[L1] Unique Customers: {unique:,}  |  Avg Purchase Frequency: {avg_freq:.2f} orders")
    lines.append(f"     Repeat Customers : {repeat_custs:,}  ({retention:.1f}% retention rate)")

    by_g = kpis.get("by_gender") if kpis else None
    if by_g is not None and not by_g.empty:
        lines.append("\n[L2] Orders & Revenue by Gender:")
        total_rev = kpis["total_revenue"] if kpis else 1.0
        for _, row in by_g.iterrows():
            share = row["revenue"] / total_rev * 100 if total_rev else 0
            lines.append(
                f"  {str(row['Gender']):<12}  Orders={int(row['orders']):,}  "
                f"Revenue=${row['revenue']:>10,.2f} ({share:.1f}%)"
            )

    by_d = kpis.get("by_device") if kpis else None
    if by_d is not None and not by_d.empty:
        lines.append("\n[L3] Orders by Device Category:")
        for _, row in by_d.iterrows():
            pct = row["orders"] / total * 100 if total else 0
            lines.append(
                f"  {str(row['DeviceCategory']):<15}  Orders={int(row['orders']):,} ({pct:.1f}%)  "
                f"Revenue=${row['revenue']:>10,.2f}"
            )

    return "\n".join(lines)


def _analyze_overview(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    lines = ["E-COMMERCE OVERVIEW — METRIC TREE SUMMARY", "=" * 55]
    if kpis:
        yoy     = kpis.get("yoy_stats", {})
        rev_yoy = yoy.get("rev_yoy_pct")
        lines.append(f"\nTotal Orders       : {kpis['total_orders']:>10,}")
        lines.append(f"Completed Orders   : {kpis['completed_orders']:>10,}")
        lines.append(f"Completion Rate    : {kpis['completion_rate']:>9.1f}%")
        lines.append(f"Total Revenue      : ${kpis['total_revenue']:>12,.2f}")
        lines.append(f"Avg Order Value    : ${kpis['avg_order_value']:>11,.2f}")
        lines.append(f"Unique Customers   : {kpis['unique_customers']:>10,}")
        lines.append(f"Avg Product Rating : {kpis['avg_product_rating']:>9.2f} / 5.0")
        lines.append(f"Avg Delivery Rating: {kpis['avg_delivery_rating']:>9.2f} / 5.0")
        lines.append(f"\nTop Channel        : {kpis['top_channel']}")
        lines.append(f"Top Country        : {kpis['top_country']}")
        if rev_yoy is not None:
            lines.append(f"Revenue YoY        : {rev_yoy:+.1f}%")
    return "\n".join(lines)


# ── Question → Metric Detection ───────────────────────────────────────────────

def detect_metric(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["revenue", "sales", "income", "earning", "money"]):
        return "revenue"
    if any(w in q for w in ["order", "conversion", "completion", "status", "cancel"]):
        return "orders"
    if any(w in q for w in ["channel", "traffic", "source", "acquisition", "social", "paid", "organic"]):
        return "channel"
    if any(w in q for w in ["country", "geography", "region", "location", "market", "india", "usa", "united states"]):
        return "geography"
    if any(w in q for w in ["rating", "quality", "satisfaction", "delivery", "product rating"]):
        return "ratings"
    if any(w in q for w in ["customer", "gender", "device", "mobile", "desktop", "segment"]):
        return "customers"
    return "overview"


def get_tree_analysis(
    question: str,
    df: pd.DataFrame,
    kpis: Optional[dict] = None,
    focus_channel: Optional[str] = None,
    focus_country: Optional[str] = None,
) -> str:
    """Entry point: auto-detect metric and run full tree traversal."""
    if focus_channel:
        return _analyze_channel_deep_dive(df, focus_channel, kpis)
    if focus_country:
        return _analyze_country_deep_dive(df, focus_country, kpis)
    metric = detect_metric(question)
    return analyze_metric(metric, df, kpis)


def _analyze_channel_deep_dive(df: pd.DataFrame, channel: str,
                                kpis: Optional[dict] = None) -> str:
    lines = [f"CHANNEL DEEP-DIVE: {channel}", "=" * 60]

    ch_df = df[df["TrafficSource"].str.lower() == channel.lower()]
    if ch_df.empty:
        ch_df = df[df["TrafficSource"].str.contains(channel, case=False, na=False)]

    if ch_df.empty:
        lines.append(f"\n  No data found for channel: {channel}")
        return "\n".join(lines)

    total_rev  = kpis["total_revenue"] if kpis else float(df["revenue"].sum())
    mkt_aov    = kpis["avg_order_value"] if kpis else float(df[df["is_completed"]]["Total"].mean())
    mkt_rating = kpis["avg_product_rating"] if kpis else 0.0

    comp    = ch_df[ch_df["is_completed"]]
    ch_rev  = float(ch_df["revenue"].sum())
    ch_ord  = len(ch_df)
    ch_comp = len(comp)
    ch_rate = ch_comp / ch_ord * 100 if ch_ord else 0
    ch_aov  = float(comp["Total"].mean()) if not comp.empty else 0
    ch_pr   = float(comp[comp["ProductRating"] > 0]["ProductRating"].mean()) if not comp.empty else 0
    ch_dr   = float(comp[comp["DeliveryRating"] > 0]["DeliveryRating"].mean()) if not comp.empty else 0
    ch_sess = float(ch_df["SessionDuration"].mean())
    rev_share = ch_rev / total_rev * 100 if total_rev else 0

    lines.append(f"\n[L1] Channel Summary")
    lines.append(f"  Revenue Share     : ${ch_rev:,.2f}  ({rev_share:.1f}% of total)")
    lines.append(f"  Orders            : {ch_ord:,}  |  Completed: {ch_comp:,}  ({ch_rate:.1f}%)")
    lines.append(f"  AOV               : ${ch_aov:.2f}  (market avg ${mkt_aov:.2f}, delta {ch_aov - mkt_aov:+.2f})")
    lines.append(f"  Product Rating    : {ch_pr:.2f}  (market avg {mkt_rating:.2f})")
    lines.append(f"  Delivery Rating   : {ch_dr:.2f}")
    lines.append(f"  Avg Session       : {ch_sess:.1f} min")

    lines.append(f"\n[L2] Top Countries via {channel}:")
    ch_co = (
        ch_df.groupby("Country")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
        .head(5)
        .reset_index()
    )
    for _, row in ch_co.iterrows():
        lines.append(
            f"  {str(row['Country']):<25}  Orders={int(row['orders']):,}  "
            f"Revenue=${row['revenue']:,.2f}"
        )

    lines.append(f"\n[L3] Device Mix for {channel}:")
    ch_dev = (
        ch_df.groupby("DeviceCategory")
        .agg(orders=("InvoiceNumber", "count"))
        .reset_index()
    )
    for _, row in ch_dev.iterrows():
        pct = row["orders"] / ch_ord * 100 if ch_ord else 0
        lines.append(f"  {str(row['DeviceCategory']):<15}  {int(row['orders']):,} ({pct:.1f}%)")

    return "\n".join(lines)


def _analyze_country_deep_dive(df: pd.DataFrame, country: str,
                                kpis: Optional[dict] = None) -> str:
    lines = [f"COUNTRY DEEP-DIVE: {country}", "=" * 60]

    co_df = df[df["Country"].str.lower() == country.lower()]
    if co_df.empty:
        co_df = df[df["Country"].str.contains(country, case=False, na=False)]

    if co_df.empty:
        lines.append(f"\n  No data found for country: {country}")
        return "\n".join(lines)

    total_rev = kpis["total_revenue"] if kpis else float(df["revenue"].sum())
    comp      = co_df[co_df["is_completed"]]
    co_rev    = float(co_df["revenue"].sum())
    co_ord    = len(co_df)
    co_comp   = len(comp)
    co_rate   = co_comp / co_ord * 100 if co_ord else 0
    co_aov    = float(comp["Total"].mean()) if not comp.empty else 0
    rev_share = co_rev / total_rev * 100 if total_rev else 0

    lines.append(f"\n[L1] Country Summary")
    lines.append(f"  Revenue Share : ${co_rev:,.2f}  ({rev_share:.1f}% of total)")
    lines.append(f"  Orders        : {co_ord:,}  |  Completion: {co_rate:.1f}%")
    lines.append(f"  AOV           : ${co_aov:.2f}")

    lines.append(f"\n[L2] Channel Mix in {country}:")
    co_ch = (
        co_df.groupby("TrafficSource")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
        .reset_index()
    )
    for _, row in co_ch.iterrows():
        pct = row["orders"] / co_ord * 100 if co_ord else 0
        lines.append(
            f"  {str(row['TrafficSource']):<30}  "
            f"Orders={int(row['orders']):,} ({pct:.1f}%)  Revenue=${row['revenue']:,.2f}"
        )

    return "\n".join(lines)
