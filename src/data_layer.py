"""
Data Layer: Loads and processes the E-Commerce dataset.

File expected in <project_root>/data/:
  E-Commerce.csv — one row per transaction with columns:
    CustomerID, Gender, InvoiceDate, InvoiceNumber, ProductID,
    Quantity, Price, Total, OrderStatus, Country, TrafficSource,
    SessionDuration, DeviceCategory, Device, OS,
    DeliveryRating, ProductRating, Sales

Hot-reload: call get_data_fingerprint() on every Streamlit run and pass it as an
argument to load_data(). Streamlit's @st.cache_data caches on all arguments, so
a changed mtime automatically invalidates the cache.
"""
import os
import glob
import hashlib
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


# ── Hot-reload helper ─────────────────────────────────────────────────────────

def get_data_fingerprint() -> str:
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        h.update(f"{path}:{os.path.getmtime(path)}".encode())
    return h.hexdigest()


# ── Raw loader ────────────────────────────────────────────────────────────────

def generate_data(_fingerprint: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Load and enrich the E-Commerce CSV.

    Returns:
        df    : enriched transactions DataFrame
        tables: empty dict (kept for API compatibility with ai_engine)
    """
    path = os.path.join(DATA_DIR, "E-Commerce.csv")
    df = pd.read_csv(path)

    # ── 1. Parse dates ─────────────────────────────────────────────────────
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], dayfirst=False, errors="coerce")
    df["year_month"] = df["InvoiceDate"].dt.to_period("M").astype(str)
    df["year"] = df["InvoiceDate"].dt.year

    # ── 2. Clean numeric columns ───────────────────────────────────────────
    for col in ["Quantity", "Price", "Total", "Sales", "SessionDuration",
                "DeliveryRating", "ProductRating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 3. Completion flag ────────────────────────────────────────────────
    df["is_completed"] = df["OrderStatus"].str.strip().str.lower() == "completed"

    # ── 4. Normalise TrafficSource casing / typos ─────────────────────────
    df["TrafficSource"] = df["TrafficSource"].str.strip()

    # ── 5. Revenue = Sales (only credited for completed orders in source) ─
    df["revenue"] = df["Sales"].fillna(0)

    return df, {}


# ── KPI Computation ───────────────────────────────────────────────────────────

def compute_kpis(df: pd.DataFrame, tables: dict = {}) -> dict:  # noqa: ARG001
    """Compute the full KPI set from the enriched transactions DataFrame."""

    completed = df[df["is_completed"]]

    total_orders     = len(df)
    completed_orders = len(completed)
    completion_rate  = completed_orders / total_orders * 100 if total_orders else 0
    total_revenue    = float(df["revenue"].sum())
    unique_customers = int(df["CustomerID"].nunique())
    avg_order_value  = float(completed["Total"].mean()) if not completed.empty else 0.0

    # Ratings only on completed orders with a real rating
    rated = completed[completed["ProductRating"] > 0]
    avg_product_rating  = float(rated["ProductRating"].mean()) if not rated.empty else 0.0
    rated_d = completed[completed["DeliveryRating"] > 0]
    avg_delivery_rating = float(rated_d["DeliveryRating"].mean()) if not rated_d.empty else 0.0

    # Retention and purchase frequency (completed orders only)
    cust_orders = (
        completed.groupby("CustomerID")["InvoiceNumber"].count()
        .reset_index()
        .rename(columns={"InvoiceNumber": "order_count"})
    )
    customers_who_bought = int(completed["CustomerID"].nunique())
    repeat_customers     = int((cust_orders["order_count"] > 1).sum())
    # Denominator = customers who actually completed ≥1 order, not all visitors
    retention_rate    = repeat_customers / customers_who_bought * 100 if customers_who_bought else 0.0
    avg_purchase_freq = float(cust_orders["order_count"].mean()) if not cust_orders.empty else 1.0

    # ── By-channel aggregation ─────────────────────────────────────────────
    by_channel = (
        df.groupby("TrafficSource")
        .agg(
            orders=("InvoiceNumber", "count"),
            revenue=("revenue", "sum"),
            completed=("is_completed", "sum"),
            avg_product_rating=("ProductRating", lambda x: x[x > 0].mean()),
            avg_delivery_rating=("DeliveryRating", lambda x: x[x > 0].mean()),
            avg_session=("SessionDuration", "mean"),
        )
        .reset_index()
    )
    by_channel["completion_rate"] = (by_channel["completed"] / by_channel["orders"] * 100).round(1)
    by_channel["avg_product_rating"] = by_channel["avg_product_rating"].round(2)
    by_channel["avg_delivery_rating"] = by_channel["avg_delivery_rating"].round(2)
    by_channel["avg_session"] = by_channel["avg_session"].round(2)
    by_channel = by_channel.sort_values("revenue", ascending=False).reset_index(drop=True)

    # LTV per channel: each customer attributed to their PRIMARY channel (most completed
    # orders), then avg of total lifetime revenue across those customers per channel.
    # This avoids double-counting customers who bought across multiple channels.
    _cust_primary_ch = (
        completed.groupby(["CustomerID", "TrafficSource"])
        .size()
        .reset_index(name="ch_orders")
        .sort_values(["CustomerID", "ch_orders"], ascending=[True, False])
        .drop_duplicates(subset=["CustomerID"], keep="first")[["CustomerID", "TrafficSource"]]
    )
    _cust_total_ltv = (
        completed.groupby("CustomerID")["revenue"].sum()
        .reset_index()
        .rename(columns={"revenue": "customer_ltv"})
    )
    ltv_by_ch = (
        _cust_primary_ch.merge(_cust_total_ltv, on="CustomerID")
        .groupby("TrafficSource")["customer_ltv"].mean()
        .reset_index()
        .rename(columns={"customer_ltv": "avg_ltv"})
    )
    ltv_by_ch["avg_ltv"] = ltv_by_ch["avg_ltv"].round(2)
    by_channel = by_channel.merge(ltv_by_ch, on="TrafficSource", how="left")

    top_channel = by_channel.iloc[0]["TrafficSource"] if not by_channel.empty else "N/A"

    # ── By-country aggregation ─────────────────────────────────────────────
    by_country = (
        df.groupby("Country")
        .agg(
            orders=("InvoiceNumber", "count"),
            revenue=("revenue", "sum"),
            unique_customers=("CustomerID", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    top_country = by_country.iloc[0]["Country"] if not by_country.empty else "N/A"

    # ── By-device aggregation ──────────────────────────────────────────────
    by_device = (
        df.groupby("DeviceCategory")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .reset_index()
        .sort_values("orders", ascending=False)
        .reset_index(drop=True)
    )

    # ── Order status breakdown ─────────────────────────────────────────────
    status_breakdown = df["OrderStatus"].value_counts().reset_index()
    status_breakdown.columns = ["status", "count"]

    # ── Revenue trend (monthly) ────────────────────────────────────────────
    revenue_trend = (
        df.groupby("year_month")
        .agg(
            order_count=("InvoiceNumber", "count"),
            revenue=("revenue", "sum"),
            avg_rating=("ProductRating", lambda x: x[x > 0].mean()),
        )
        .reset_index()
        .sort_values("year_month")
    )
    revenue_trend["avg_rating"] = revenue_trend["avg_rating"].round(2)

    # ── Gender split ───────────────────────────────────────────────────────
    by_gender = (
        df.groupby("Gender")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .reset_index()
    )

    # ── Rating distribution ────────────────────────────────────────────────
    bins   = [0, 1, 2, 3, 4, 4.5, 5.01]
    labels = ["0–1", "1–2", "2–3", "3–4", "4–4.5", "4.5–5"]
    pr = rated["ProductRating"].dropna()
    if len(pr):
        rating_dist = (
            pd.cut(pr, bins=bins, labels=labels, right=False)
            .value_counts()
            .reindex(labels, fill_value=0)
            .reset_index()
        )
        rating_dist.columns = ["band", "count"]
    else:
        rating_dist = pd.DataFrame({"band": labels, "count": [0] * len(labels)})

    # ── YoY stats ─────────────────────────────────────────────────────────
    yoy_stats = _compute_yoy(revenue_trend)

    kpis: dict = {
        "total_orders":         total_orders,
        "completed_orders":     completed_orders,
        "completion_rate":      round(completion_rate, 1),
        "total_revenue":        round(total_revenue, 2),
        "unique_customers":     unique_customers,
        "avg_order_value":      round(avg_order_value, 2),
        "avg_product_rating":   round(avg_product_rating, 2),
        "avg_delivery_rating":  round(avg_delivery_rating, 2),
        "retention_rate":       round(retention_rate, 1),
        "repeat_customers":     repeat_customers,
        "avg_purchase_freq":    round(avg_purchase_freq, 2),
        "top_channel":          top_channel,
        "top_country":          top_country,
        "by_channel":           by_channel,
        "by_country":           by_country,
        "by_device":            by_device,
        "by_gender":            by_gender,
        "status_breakdown":     status_breakdown,
        "revenue_trend":        revenue_trend,
        "rating_dist":          rating_dist,
        "yoy_stats":            yoy_stats,
    }
    return kpis


# ── YoY helper ────────────────────────────────────────────────────────────────

def _compute_yoy(revenue_trend: pd.DataFrame) -> dict:
    if revenue_trend.empty:
        return {}
    trend = revenue_trend.copy()
    trend["year"] = trend["year_month"].str[:4]
    years = sorted(trend["year"].unique())
    if len(years) < 2:
        return {}
    curr_year  = years[-1]
    prior_year = years[-2]
    curr  = trend[trend["year"] == curr_year]
    prior = trend[trend["year"] == prior_year]

    curr_rev  = float(curr["revenue"].sum())
    prior_rev = float(prior["revenue"].sum())
    rev_yoy   = (curr_rev - prior_rev) / prior_rev * 100 if prior_rev else None

    curr_orders  = int(curr["order_count"].sum())
    prior_orders = int(prior["order_count"].sum())
    ord_yoy      = (curr_orders - prior_orders) / prior_orders * 100 if prior_orders else None

    return {
        "rev_yoy_pct":   round(rev_yoy, 1)   if rev_yoy is not None   else None,
        "order_yoy_pct": round(ord_yoy, 1)   if ord_yoy is not None   else None,
        "curr_year":     curr_year,
        "prior_year":    prior_year,
    }
