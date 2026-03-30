"""
Tool schemas and executors for the OpenAI tool-use agent.
Each tool gives the model a way to query the e-commerce DataFrame directly.
"""
import pandas as pd
from datetime import datetime
from .metric_tree import get_tree_analysis, analyze_metric

# ── Date parsing helper ───────────────────────────────────────────────────────

def _parse_date(s: str) -> pd.Timestamp | None:
    """Parse a flexible date string into a Timestamp. Returns None on failure."""
    for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%m-%d-%Y', '%d-%m-%Y',
                '%Y/%m/%d', '%B %d, %Y', '%b %d, %Y'):
        try:
            return pd.Timestamp(datetime.strptime(s.strip(), fmt))
        except ValueError:
            continue
    try:
        return pd.Timestamp(s.strip())
    except Exception:
        return None


# ── Tool Schemas (OpenAI function-calling format) ─────────────────────────────

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "filter_orders",
            "description": (
                "Filter and retrieve individual orders using any combination of filters: "
                "date, date range, channel, country, status, device, gender, product ID, "
                "customer ID, price range, or minimum rating. "
                "Use this for specific transaction lookups, date-based queries, "
                "product queries, or any multi-filter combination."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": (
                            "Exact date to filter (e.g. '2019-01-09', '1/9/2019', '09/01/2019'). "
                            "IMPORTANT: '1/09/2019' means January 9, 2019 (MM/DD/YYYY format)."
                        ),
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start of date range (inclusive). Same format as 'date'.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End of date range (inclusive). Same format as 'date'.",
                    },
                    "month": {
                        "type": "string",
                        "description": "Filter by month in YYYY-MM format (e.g. '2019-06' for June 2019).",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Filter by year (e.g. 2019).",
                    },
                    "channel": {
                        "type": "string",
                        "description": "Filter by TrafficSource (partial match, case-insensitive).",
                    },
                    "country": {
                        "type": "string",
                        "description": "Filter by Country (partial match, case-insensitive).",
                    },
                    "status": {
                        "type": "string",
                        "description": "Filter by OrderStatus: 'completed', 'in process', 'cancelled', etc.",
                    },
                    "device": {
                        "type": "string",
                        "description": "Filter by DeviceCategory: 'computer', 'mobile'.",
                    },
                    "gender": {
                        "type": "string",
                        "description": "Filter by Gender: 'male', 'female'.",
                    },
                    "product_id": {
                        "type": "string",
                        "description": "Filter by ProductID (exact match).",
                    },
                    "customer_id": {
                        "type": "string",
                        "description": "Filter by CustomerID (exact match).",
                    },
                    "min_total": {
                        "type": "number",
                        "description": "Minimum order Total (USD).",
                    },
                    "max_total": {
                        "type": "number",
                        "description": "Maximum order Total (USD).",
                    },
                    "min_product_rating": {
                        "type": "number",
                        "description": "Minimum ProductRating (1–5).",
                    },
                    "sort_by": {
                        "type": "string",
                        "enum": ["Total", "SessionDuration", "ProductRating", "InvoiceDate", "Quantity"],
                        "description": "Sort results by this field (default: InvoiceDate).",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default: 50, max: 200).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_date_summary",
            "description": (
                "Get a complete aggregated summary for a specific date or date range. "
                "Returns total orders, revenue, quantities, product breakdown, channel mix, "
                "country mix, and status breakdown. "
                "Use this for questions like 'what happened on Jan 9', 'how was last week', "
                "'summarise March 2020', 'daily orders in Q1 2019'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "Exact date (e.g. '2019-01-09', '1/9/2019').",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Start of date range (inclusive).",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "End of date range (inclusive).",
                    },
                    "month": {
                        "type": "string",
                        "description": "Month in YYYY-MM format.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Full year (e.g. 2020).",
                    },
                    "group_by": {
                        "type": "string",
                        "enum": ["product", "channel", "country", "status", "device", "day"],
                        "description": "Optional: group the summary by this dimension.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product_stats",
            "description": (
                "Get statistics for one or all products: total quantity sold, revenue, "
                "order count, completion rate, average ratings, top countries, top channels. "
                "Use for questions about specific products or 'top selling products'."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "product_id": {
                        "type": "string",
                        "description": "Specific ProductID to analyse (omit for all products).",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional start date filter.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional end date filter.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max products to return when listing all (default: 20).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_channel_stats",
            "description": (
                "Get detailed statistics for a specific traffic source / acquisition channel: "
                "revenue, orders, completion rate, AOV, ratings, session duration, top countries."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "TrafficSource name (partial match, case-insensitive).",
                    },
                },
                "required": ["channel"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cancellation_analysis",
            "description": (
                "Analyse cancelled and non-completed orders in depth: "
                "cancellation rate by channel, country, device, and gender; "
                "rating patterns for cancelled vs completed orders; "
                "session duration comparison; top cancellation segments. "
                "Use this for any question about why orders are cancelled, "
                "cancellation drivers, or how to reduce cancellations."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "group_by": {
                        "type": "string",
                        "enum": ["channel", "country", "device", "gender", "all"],
                        "description": "Dimension to break down cancellations by (default: all).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_country_stats",
            "description": (
                "Get detailed statistics for a specific country: "
                "revenue, orders, completion rate, channel mix, device mix."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "country": {
                        "type": "string",
                        "description": "Country name (partial match, case-insensitive).",
                    },
                },
                "required": ["country"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_revenue_breakdown",
            "description": (
                "Get revenue and order breakdowns by a specified dimension: "
                "channel, country, device, gender, or month."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": ["channel", "country", "device", "gender", "month"],
                        "description": "Dimension to group revenue by.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows to return (default: 10).",
                    },
                },
                "required": ["dimension"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_rating_analysis",
            "description": (
                "Get product and delivery rating analysis, optionally filtered by channel or country."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Filter by TrafficSource (optional).",
                    },
                    "country": {
                        "type": "string",
                        "description": "Filter by Country (optional).",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_overview",
            "description": (
                "Get store-wide KPIs: total revenue, orders, completion rate, AOV, "
                "top channels, top countries, and rating averages."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_revenue_trend",
            "description": "Get monthly revenue and order count trend over time, optionally filtered by date range, channel, or country.",
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": "Optionally filter trend by TrafficSource.",
                    },
                    "country": {
                        "type": "string",
                        "description": "Optionally filter trend by Country.",
                    },
                    "date_from": {
                        "type": "string",
                        "description": "Optional start date to limit the trend window.",
                    },
                    "date_to": {
                        "type": "string",
                        "description": "Optional end date to limit the trend window.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_metric_tree",
            "description": (
                "Run a full metric tree traversal for a given dimension. "
                "Use this for drill-down questions like 'break down revenue', "
                "'analyze channel performance with LTV', 'why did conversion drop', "
                "'retention and quality analysis', or 'geographic breakdown'. "
                "Returns a structured L1→L4 causal analysis with LTV, "
                "purchase frequency, and retention where relevant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "metric": {
                        "type": "string",
                        "enum": ["revenue", "orders", "channel", "geography", "ratings", "customers", "overview"],
                        "description": "Which branch of the metric tree to traverse.",
                    },
                    "focus_channel": {
                        "type": "string",
                        "description": "Optional: deep-dive into a specific TrafficSource.",
                    },
                    "focus_country": {
                        "type": "string",
                        "description": "Optional: deep-dive into a specific Country.",
                    },
                },
                "required": ["metric"],
            },
        },
    },
]


# ── Date filter helper ────────────────────────────────────────────────────────

def _apply_date_filters(df: pd.DataFrame, inputs: dict) -> pd.DataFrame:
    """Apply date/month/year filters from inputs dict to df. Returns filtered copy."""
    result = df.copy()

    date_str   = inputs.get("date")
    date_from  = inputs.get("date_from")
    date_to    = inputs.get("date_to")
    month_str  = inputs.get("month")
    year_val   = inputs.get("year")

    if date_str:
        d = _parse_date(date_str)
        if d is not None:
            result = result[result["InvoiceDate"].dt.date == d.date()]
        else:
            return result.iloc[0:0]  # return empty if date unparseable

    if date_from:
        d = _parse_date(date_from)
        if d is not None:
            result = result[result["InvoiceDate"] >= d]

    if date_to:
        d = _parse_date(date_to)
        if d is not None:
            result = result[result["InvoiceDate"] <= d + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)]

    if month_str:
        # e.g. "2019-06"
        result = result[result["year_month"] == month_str]

    if year_val:
        result = result[result["InvoiceDate"].dt.year == int(year_val)]

    return result


# ── Tool Executors ────────────────────────────────────────────────────────────

def _filter_orders(inputs: dict, df: pd.DataFrame) -> str:
    result = _apply_date_filters(df, inputs)

    channel   = inputs.get("channel")
    country   = inputs.get("country")
    status    = inputs.get("status")
    device    = inputs.get("device")
    gender    = inputs.get("gender")
    product   = inputs.get("product_id")
    customer  = inputs.get("customer_id")
    min_total = inputs.get("min_total")
    max_total = inputs.get("max_total")
    min_pr    = inputs.get("min_product_rating")
    sort_by   = inputs.get("sort_by", "InvoiceDate")
    limit     = min(int(inputs.get("limit", 50)), 200)

    if channel:
        result = result[result["TrafficSource"].str.contains(channel, case=False, na=False)]
    if country:
        result = result[result["Country"].str.contains(country, case=False, na=False)]
    if status:
        result = result[result["OrderStatus"].str.lower().str.contains(status.lower(), na=False)]
    if device:
        result = result[result["DeviceCategory"].str.lower().str.contains(device.lower(), na=False)]
    if gender:
        result = result[result["Gender"].str.lower() == gender.lower()]
    if product:
        result = result[result["ProductID"].astype(str) == str(product)]
    if customer:
        result = result[result["CustomerID"].astype(str) == str(customer)]
    if min_total is not None:
        result = result[result["Total"] >= float(min_total)]
    if max_total is not None:
        result = result[result["Total"] <= float(max_total)]
    if min_pr is not None:
        result = result[result["ProductRating"] >= min_pr]

    sort_col = sort_by if sort_by in result.columns else "InvoiceDate"
    result = result.dropna(subset=[sort_col]).sort_values(sort_col, ascending=False).head(limit)

    if result.empty:
        return "No orders matched the given filters."

    lines = [f"Found {len(result)} order(s):\n"]
    for _, row in result.iterrows():
        date_str = row["InvoiceDate"].strftime("%Y-%m-%d") if pd.notna(row["InvoiceDate"]) else "N/A"
        lines.append(
            f"• Invoice {row['InvoiceNumber']}  |  Date: {date_str}  |  Customer {row['CustomerID']}\n"
            f"  Product={row['ProductID']}  Qty={int(row['Quantity'])}  "
            f"Price=${row['Price']:.2f}  Total=${row['Total']:.2f}\n"
            f"  Country={row['Country']}  Channel={row['TrafficSource']}  "
            f"Status={row['OrderStatus']}  Device={row['DeviceCategory']}\n"
            f"  ProductRating={row['ProductRating']}  DeliveryRating={row['DeliveryRating']}"
        )
    return "\n".join(lines)


def _get_date_summary(inputs: dict, df: pd.DataFrame) -> str:
    result = _apply_date_filters(df, inputs)
    group_by = inputs.get("group_by")

    if result.empty:
        # Build a readable label for what was requested
        label = (inputs.get("date") or
                 f"{inputs.get('date_from','')}–{inputs.get('date_to','')}" or
                 inputs.get("month") or str(inputs.get("year", "")) or "that period")
        return f"No orders found for {label}."

    total_orders  = len(result)
    completed     = result[result["is_completed"]]
    total_revenue = float(result["revenue"].sum())
    total_qty     = int(result["Quantity"].sum())
    comp_rate     = len(completed) / total_orders * 100 if total_orders else 0

    # Build period label
    dates = result["InvoiceDate"].dropna()
    if len(dates):
        d_min = dates.min().strftime("%Y-%m-%d")
        d_max = dates.max().strftime("%Y-%m-%d")
        period = d_min if d_min == d_max else f"{d_min} to {d_max}"
    else:
        period = "period"

    lines = [
        f"=== Summary for {period} ===",
        f"Total Orders     : {total_orders}",
        f"Completed Orders : {len(completed)}  ({comp_rate:.1f}%)",
        f"Total Revenue    : ${total_revenue:,.2f}",
        f"Total Qty Sold   : {total_qty}",
        f"Unique Products  : {result['ProductID'].nunique()}",
        f"Unique Customers : {result['CustomerID'].nunique()}",
    ]

    if group_by == "product":
        lines.append("\nBy Product:")
        grp = (result.groupby("ProductID")
               .agg(qty=("Quantity","sum"), revenue=("revenue","sum"),
                    orders=("InvoiceNumber","count"))
               .sort_values("qty", ascending=False).reset_index())
        for _, row in grp.iterrows():
            lines.append(f"  Product {row['ProductID']:<8}  Qty={int(row['qty']):>5}  "
                         f"Revenue=${row['revenue']:>10,.2f}  Orders={int(row['orders'])}")

    elif group_by == "channel":
        lines.append("\nBy Channel:")
        grp = (result.groupby("TrafficSource")
               .agg(orders=("InvoiceNumber","count"), revenue=("revenue","sum"),
                    completed=("is_completed","sum"))
               .reset_index())
        for _, row in grp.iterrows():
            conv = row["completed"] / row["orders"] * 100 if row["orders"] else 0
            lines.append(f"  {str(row['TrafficSource']):<30}  Orders={int(row['orders'])}  "
                         f"Revenue=${row['revenue']:>10,.2f}  Conv={conv:.1f}%")

    elif group_by == "country":
        lines.append("\nBy Country:")
        grp = (result.groupby("Country")
               .agg(orders=("InvoiceNumber","count"), revenue=("revenue","sum"))
               .sort_values("revenue", ascending=False).reset_index())
        for _, row in grp.iterrows():
            lines.append(f"  {str(row['Country']):<25}  Orders={int(row['orders'])}  "
                         f"Revenue=${row['revenue']:>10,.2f}")

    elif group_by == "status":
        lines.append("\nBy Status:")
        grp = result["OrderStatus"].value_counts().reset_index()
        grp.columns = ["status", "count"]
        for _, row in grp.iterrows():
            pct = row["count"] / total_orders * 100
            lines.append(f"  {str(row['status']):<20}  {row['count']}  ({pct:.1f}%)")

    elif group_by == "day":
        lines.append("\nDaily Breakdown:")
        grp = (result.groupby("year_month" if inputs.get("month") or inputs.get("year") else
                              result["InvoiceDate"].dt.date)
               .agg(orders=("InvoiceNumber","count"), revenue=("revenue","sum"))
               .reset_index())
        for _, row in grp.iterrows():
            lines.append(f"  {str(row.iloc[0])}  Orders={int(row['orders'])}  "
                         f"Revenue=${row['revenue']:>10,.2f}")

    else:
        # Default: show product breakdown + status breakdown
        lines.append("\nBy Product:")
        grp = (result.groupby("ProductID")
               .agg(qty=("Quantity","sum"), revenue=("revenue","sum"),
                    orders=("InvoiceNumber","count"))
               .sort_values("qty", ascending=False).reset_index())
        for _, row in grp.iterrows():
            lines.append(f"  Product {row['ProductID']:<8}  Qty={int(row['qty']):>5}  "
                         f"Revenue=${row['revenue']:>10,.2f}  Orders={int(row['orders'])}")

        lines.append("\nBy Status:")
        for status_val, cnt in result["OrderStatus"].value_counts().items():
            pct = cnt / total_orders * 100
            lines.append(f"  {str(status_val):<20}  {cnt}  ({pct:.1f}%)")

        lines.append("\nBy Channel:")
        for ch, cnt in result["TrafficSource"].value_counts().items():
            lines.append(f"  {str(ch):<30}  {cnt} orders")

        lines.append("\nBy Country:")
        for co, cnt in result["Country"].value_counts().head(10).items():
            lines.append(f"  {str(co):<25}  {cnt} orders")

    return "\n".join(lines)


def _get_product_stats(inputs: dict, df: pd.DataFrame) -> str:
    result = _apply_date_filters(df, inputs)
    product_id = inputs.get("product_id")
    limit      = int(inputs.get("limit", 20))

    if product_id:
        result = result[result["ProductID"].astype(str) == str(product_id)]
        if result.empty:
            return f"No data found for ProductID '{product_id}'."

        comp      = result[result["is_completed"]]
        total_rev = float(result["revenue"].sum())
        total_qty = int(result["Quantity"].sum())
        orders    = len(result)
        comp_rate = len(comp) / orders * 100 if orders else 0
        pr = comp[comp["ProductRating"] > 0]["ProductRating"]
        dr = comp[comp["DeliveryRating"] > 0]["DeliveryRating"]

        lines = [
            f"=== Product {product_id} ===",
            f"Total Orders    : {orders}",
            f"Completed       : {len(comp)}  ({comp_rate:.1f}%)",
            f"Total Qty Sold  : {total_qty}",
            f"Total Revenue   : ${total_rev:,.2f}",
            f"Avg Unit Price  : ${float(result['Price'].mean()):.2f}",
            f"Avg Product Rtg : {pr.mean():.2f}/5  (n={len(pr)})" if not pr.empty else "Avg Product Rtg : N/A",
            f"Avg Delivery Rtg: {dr.mean():.2f}/5  (n={len(dr)})" if not dr.empty else "Avg Delivery Rtg: N/A",
        ]
        lines.append("\nTop Channels:")
        for ch, cnt in result["TrafficSource"].value_counts().head(5).items():
            lines.append(f"  {str(ch):<30}  {cnt} orders")
        lines.append("\nTop Countries:")
        for co, cnt in result["Country"].value_counts().head(5).items():
            lines.append(f"  {str(co):<25}  {cnt} orders")
        return "\n".join(lines)

    else:
        # All products ranked
        grp = (result.groupby("ProductID")
               .agg(qty=("Quantity","sum"), revenue=("revenue","sum"),
                    orders=("InvoiceNumber","count"),
                    completed=("is_completed","sum"))
               .reset_index()
               .sort_values("revenue", ascending=False)
               .head(limit))
        grp["completion_rate"] = (grp["completed"] / grp["orders"] * 100).round(1)

        lines = [f"Top {len(grp)} Products by Revenue:\n",
                 f"  {'ProductID':<12}  {'Qty':>8}  {'Revenue':>12}  {'Orders':>8}  {'Conv%':>7}"]
        lines.append("  " + "-" * 55)
        for _, row in grp.iterrows():
            lines.append(
                f"  {str(row['ProductID']):<12}  {int(row['qty']):>8}  "
                f"${row['revenue']:>11,.2f}  {int(row['orders']):>8}  "
                f"{row['completion_rate']:>6.1f}%"
            )
        return "\n".join(lines)


def _get_cancellation_analysis(inputs: dict, df: pd.DataFrame, kpis: dict) -> str:
    group_by  = inputs.get("group_by", "all")
    cancelled = df[df["OrderStatus"].str.lower().str.contains("cancel", na=False)]
    completed = df[df["is_completed"]]
    total     = len(df)
    n_cancel  = len(cancelled)
    cancel_rate = n_cancel / total * 100 if total else 0

    lines = [
        "=== CANCELLATION ANALYSIS ===",
        f"Total Orders     : {total:,}",
        f"Cancelled Orders : {n_cancel:,}  ({cancel_rate:.1f}% cancellation rate)",
        f"Completed Orders : {len(completed):,}  ({kpis['completion_rate']:.1f}% completion rate)",
    ]

    def _add_dim(label: str, col: str):
        lines.append(f"\nCancellation Rate by {label}:")
        by_dim = (
            df.groupby(col)
            .agg(total=("InvoiceNumber", "count"),
                 cancelled=("OrderStatus",
                             lambda x: x.str.lower().str.contains("cancel", na=False).sum()))
            .reset_index()
        )
        by_dim["cancel_rate"] = (by_dim["cancelled"] / by_dim["total"] * 100).round(1)
        by_dim = by_dim.sort_values("cancel_rate", ascending=False)
        for _, row in by_dim.iterrows():
            lines.append(
                f"  {str(row[col]):<30}  Cancelled={int(row['cancelled']):,}  "
                f"of {int(row['total']):,}  ({row['cancel_rate']:.1f}%)"
            )

    if group_by in ("channel", "all"):
        _add_dim("Channel", "TrafficSource")
    if group_by in ("country", "all"):
        _add_dim("Country", "Country")
    if group_by in ("device", "all"):
        _add_dim("Device", "DeviceCategory")
    if group_by in ("gender", "all"):
        _add_dim("Gender", "Gender")

    # Session duration comparison: cancelled vs completed
    if not cancelled.empty and not completed.empty:
        lines.append("\nSession Duration Comparison:")
        lines.append(f"  Cancelled orders avg session : {cancelled['SessionDuration'].mean():.1f} min")
        lines.append(f"  Completed orders avg session : {completed['SessionDuration'].mean():.1f} min")
        diff = cancelled["SessionDuration"].mean() - completed["SessionDuration"].mean()
        lines.append(f"  Difference                   : {diff:+.1f} min")

    # Rating pattern for cancelled orders (where rated)
    rated_cancel = cancelled[cancelled["ProductRating"] > 0]
    if not rated_cancel.empty:
        lines.append("\nRatings on Cancelled Orders (rated before cancel):")
        lines.append(f"  Avg Product Rating  : {rated_cancel['ProductRating'].mean():.2f}/5")
        lines.append(f"  Avg Delivery Rating : {rated_cancel['DeliveryRating'].mean():.2f}/5")
        lines.append(f"  Avg for Completed   : {kpis['avg_product_rating']:.2f}/5 (product)  "
                     f"{kpis['avg_delivery_rating']:.2f}/5 (delivery)")

    # Top cancellation segments
    lines.append("\nHighest Cancellation Segments (Channel × Country):")
    seg = (
        df.groupby(["TrafficSource", "Country"])
        .agg(total=("InvoiceNumber", "count"),
             cancelled=("OrderStatus",
                        lambda x: x.str.lower().str.contains("cancel", na=False).sum()))
        .reset_index()
    )
    seg["cancel_rate"] = (seg["cancelled"] / seg["total"] * 100).round(1)
    seg = seg[seg["total"] >= 5].sort_values("cancel_rate", ascending=False).head(5)
    for _, row in seg.iterrows():
        lines.append(
            f"  {str(row['TrafficSource']):<25} × {str(row['Country']):<20}  "
            f"{row['cancel_rate']:.1f}% ({int(row['cancelled'])}/{int(row['total'])} orders)"
        )

    return "\n".join(lines)


def _get_channel_stats(inputs: dict, df: pd.DataFrame, kpis: dict) -> str:
    channel = inputs["channel"]
    ch_df = df[df["TrafficSource"].str.contains(channel, case=False, na=False)]
    if ch_df.empty:
        return f"No data found for channel '{channel}'."

    comp      = ch_df[ch_df["is_completed"]]
    total_rev = kpis["total_revenue"] or 1.0
    ch_rev    = float(ch_df["revenue"].sum())
    ch_ord    = len(ch_df)
    ch_comp   = len(comp)
    ch_rate   = ch_comp / ch_ord * 100 if ch_ord else 0
    ch_aov    = float(comp["Total"].mean()) if not comp.empty else 0
    ch_pr     = float(comp[comp["ProductRating"] > 0]["ProductRating"].mean()) if not comp.empty else 0
    ch_dr     = float(comp[comp["DeliveryRating"] > 0]["DeliveryRating"].mean()) if not comp.empty else 0
    ch_sess   = float(ch_df["SessionDuration"].mean())
    rev_share = ch_rev / total_rev * 100 if total_rev else 0

    # LTV for this channel
    ch_ltv_row = kpis.get("by_channel", pd.DataFrame())
    ltv_str = ""
    if not ch_ltv_row.empty:
        mask = ch_ltv_row["TrafficSource"].str.contains(channel, case=False, na=False)
        if mask.any():
            ltv = ch_ltv_row.loc[mask, "avg_ltv"].iloc[0]
            if pd.notna(ltv):
                ltv_str = f"\nAvg Customer LTV: ${ltv:.2f}"

    lines = [
        f"=== Channel: {ch_df['TrafficSource'].iloc[0]} ===",
        f"Revenue     : ${ch_rev:,.2f}  ({rev_share:.1f}% of total)",
        f"Orders      : {ch_ord:,}  |  Completed: {ch_comp:,}  ({ch_rate:.1f}%)",
        f"AOV         : ${ch_aov:.2f}",
        f"Prod Rating : {ch_pr:.2f}/5",
        f"Deliv Rating: {ch_dr:.2f}/5",
        f"Avg Session : {ch_sess:.1f} min{ltv_str}",
        "\nTop Countries for this channel:",
    ]
    top_co = (
        ch_df.groupby("Country")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False).head(5).reset_index()
    )
    for _, row in top_co.iterrows():
        lines.append(f"  • {row['Country']}: {int(row['orders'])} orders  ${row['revenue']:,.2f}")

    return "\n".join(lines)


def _get_country_stats(inputs: dict, df: pd.DataFrame, kpis: dict) -> str:
    country = inputs["country"]
    co_df = df[df["Country"].str.contains(country, case=False, na=False)]
    if co_df.empty:
        return f"No data found for country '{country}'."

    comp      = co_df[co_df["is_completed"]]
    total_rev = kpis["total_revenue"] or 1.0
    co_rev    = float(co_df["revenue"].sum())
    co_ord    = len(co_df)
    co_comp   = len(comp)
    co_rate   = co_comp / co_ord * 100 if co_ord else 0
    co_aov    = float(comp["Total"].mean()) if not comp.empty else 0
    rev_share = co_rev / total_rev * 100 if total_rev else 0

    lines = [
        f"=== Country: {co_df['Country'].iloc[0]} ===",
        f"Revenue     : ${co_rev:,.2f}  ({rev_share:.1f}% of total)",
        f"Orders      : {co_ord:,}  |  Completion: {co_rate:.1f}%",
        f"AOV         : ${co_aov:.2f}",
        f"Customers   : {co_df['CustomerID'].nunique():,}",
        "\nChannel Mix:",
    ]
    ch_mix = (
        co_df.groupby("TrafficSource")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False).reset_index()
    )
    for _, row in ch_mix.iterrows():
        pct = row["orders"] / co_ord * 100 if co_ord else 0
        lines.append(
            f"  • {row['TrafficSource']}: {int(row['orders'])} orders ({pct:.1f}%)  "
            f"${row['revenue']:,.2f}"
        )

    lines.append("\nDevice Mix:")
    dev_mix = co_df.groupby("DeviceCategory").agg(orders=("InvoiceNumber","count")).reset_index()
    for _, row in dev_mix.iterrows():
        pct = row["orders"] / co_ord * 100 if co_ord else 0
        lines.append(f"  • {row['DeviceCategory']}: {int(row['orders'])} ({pct:.1f}%)")

    return "\n".join(lines)


def _get_revenue_breakdown(inputs: dict, df: pd.DataFrame, kpis: dict) -> str:
    dim   = inputs["dimension"]
    limit = int(inputs.get("limit", 10))

    dim_map = {
        "channel": "TrafficSource",
        "country": "Country",
        "device":  "DeviceCategory",
        "gender":  "Gender",
        "month":   "year_month",
    }
    col = dim_map.get(dim)
    if not col or col not in df.columns:
        return f"Unknown dimension '{dim}'."

    total_rev = kpis["total_revenue"] or 1.0
    breakdown = (
        df.groupby(col)
        .agg(
            orders=("InvoiceNumber", "count"),
            revenue=("revenue", "sum"),
            completed=("is_completed", "sum"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .head(limit)
    )
    breakdown["completion_rate"] = (breakdown["completed"] / breakdown["orders"] * 100).round(1)

    lines = [f"Revenue Breakdown by {dim.title()}:\n"]
    for _, row in breakdown.iterrows():
        share = row["revenue"] / total_rev * 100
        lines.append(
            f"  {str(row[col]):<30}  Revenue=${row['revenue']:>10,.2f} ({share:.1f}%)  "
            f"Orders={int(row['orders']):,}  Conv={row['completion_rate']:.1f}%"
        )
    return "\n".join(lines)


def _get_rating_analysis(inputs: dict, df: pd.DataFrame) -> str:
    channel = inputs.get("channel")
    country = inputs.get("country")

    result = df[df["is_completed"]].copy()
    label  = "All Completed Orders"

    if channel:
        result = result[result["TrafficSource"].str.contains(channel, case=False, na=False)]
        label  = f"Channel: {channel}"
    if country:
        result = result[result["Country"].str.contains(country, case=False, na=False)]
        label  = f"Country: {country}"

    pr = result[result["ProductRating"] > 0]["ProductRating"]
    dr = result[result["DeliveryRating"] > 0]["DeliveryRating"]

    if pr.empty:
        return f"No rating data found for {label}."

    lines = [
        f"=== Rating Analysis: {label} ===",
        f"Avg Product Rating  : {pr.mean():.2f}/5  (n={len(pr):,})",
        f"Avg Delivery Rating : {dr.mean():.2f}/5  (n={len(dr):,})",
        "",
        "Product Rating Distribution:",
    ]
    for stars in sorted(pr.unique()):
        cnt = int((pr == stars).sum())
        pct = cnt / len(pr) * 100
        bar = "█" * int(pct / 4)
        lines.append(f"  {int(stars)}★  {cnt:>4,}  {bar} {pct:.1f}%")

    lines.append("\nDelivery Rating Distribution:")
    for stars in sorted(dr.unique()):
        cnt = int((dr == stars).sum())
        pct = cnt / len(dr) * 100
        bar = "█" * int(pct / 4)
        lines.append(f"  {int(stars)}★  {cnt:>4,}  {bar} {pct:.1f}%")

    return "\n".join(lines)


def _get_revenue_trend(inputs: dict, df: pd.DataFrame) -> str:
    result = _apply_date_filters(df, inputs)
    channel = inputs.get("channel")
    country = inputs.get("country")
    label   = "All"

    if channel:
        result = result[result["TrafficSource"].str.contains(channel, case=False, na=False)]
        label  = channel
    if country:
        result = result[result["Country"].str.contains(country, case=False, na=False)]
        label  = country

    trend = (
        result.groupby("year_month")
        .agg(
            orders=("InvoiceNumber", "count"),
            revenue=("revenue", "sum"),
            completed=("is_completed", "sum"),
        )
        .reset_index()
        .sort_values("year_month")
    )
    trend["completion_rate"] = (trend["completed"] / trend["orders"] * 100).round(1)

    lines = [f"Monthly Revenue Trend ({label}):\n"]
    for _, row in trend.iterrows():
        lines.append(
            f"  {row['year_month']}  Revenue=${row['revenue']:>10,.2f}  "
            f"Orders={int(row['orders']):,}  Conv={row['completion_rate']:.1f}%"
        )
    return "\n".join(lines)


def _get_market_overview(kpis: dict) -> str:
    by_ch = kpis.get("by_channel", pd.DataFrame())
    by_co = kpis.get("by_country", pd.DataFrame())
    yoy   = kpis.get("yoy_stats", {})

    lines = [
        "=== E-Commerce Market Overview ===",
        f"Total Orders       : {kpis['total_orders']:,}",
        f"Completed Orders   : {kpis['completed_orders']:,}  ({kpis['completion_rate']:.1f}%)",
        f"Total Revenue      : ${kpis['total_revenue']:,.2f}",
        f"Avg Order Value    : ${kpis['avg_order_value']:.2f}",
        f"Unique Customers   : {kpis['unique_customers']:,}",
        f"Avg Product Rating : {kpis['avg_product_rating']:.2f}/5",
        f"Avg Delivery Rating: {kpis['avg_delivery_rating']:.2f}/5",
        f"Top Channel        : {kpis['top_channel']}",
        f"Top Country        : {kpis['top_country']}",
    ]
    if yoy.get("rev_yoy_pct") is not None:
        lines.append(f"Revenue YoY        : {yoy['rev_yoy_pct']:+.1f}%  ({yoy['prior_year']} → {yoy['curr_year']})")

    if not by_ch.empty:
        lines.append("\nChannel Performance:")
        for _, row in by_ch.iterrows():
            ltv_str = f"  LTV=${row['avg_ltv']:.2f}" if "avg_ltv" in row and pd.notna(row.get("avg_ltv")) else ""
            lines.append(
                f"  • {row['TrafficSource']}: ${row['revenue']:,.2f}  "
                f"{int(row['orders'])} orders  {row['completion_rate']:.1f}% conv{ltv_str}"
            )

    if not by_co.empty:
        lines.append("\nTop 5 Countries by Revenue:")
        for _, row in by_co.head(5).iterrows():
            lines.append(
                f"  • {row['Country']}: ${row['revenue']:,.2f}  {int(row['orders'])} orders"
            )

    return "\n".join(lines)


# ── Public Dispatcher ──────────────────────────────────────────────────────────

def _get_metric_tree(inputs: dict, df: pd.DataFrame, kpis: dict) -> str:
    metric        = inputs.get("metric", "overview")
    focus_channel = inputs.get("focus_channel")
    focus_country = inputs.get("focus_country")
    if focus_channel or focus_country:
        return get_tree_analysis("", df, kpis,
                                 focus_channel=focus_channel,
                                 focus_country=focus_country)
    return analyze_metric(metric, df, kpis)


def execute_tool(name: str, inputs: dict, df: pd.DataFrame,
                 tables: dict, kpis: dict) -> str:
    try:
        if name == "filter_orders":
            return _filter_orders(inputs, df)
        if name == "get_cancellation_analysis":
            return _get_cancellation_analysis(inputs, df, kpis)
        if name == "get_date_summary":
            return _get_date_summary(inputs, df)
        if name == "get_product_stats":
            return _get_product_stats(inputs, df)
        if name == "get_channel_stats":
            return _get_channel_stats(inputs, df, kpis)
        if name == "get_country_stats":
            return _get_country_stats(inputs, df, kpis)
        if name == "get_revenue_breakdown":
            return _get_revenue_breakdown(inputs, df, kpis)
        if name == "get_rating_analysis":
            return _get_rating_analysis(inputs, df)
        if name == "get_market_overview":
            return _get_market_overview(kpis)
        if name == "get_revenue_trend":
            return _get_revenue_trend(inputs, df)
        if name == "get_metric_tree":
            return _get_metric_tree(inputs, df, kpis)
        return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error in '{name}': {str(e)}"
