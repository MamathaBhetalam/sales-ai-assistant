"""
Data Layer: Loads and processes the Shopify App Store multi-CSV dataset.

Files expected in  <project_root>/data/:
  apps.csv                 — one row per app (rating, reviews_count, developer …)
  categories.csv           — category master list
  apps_categories.csv      — many-to-many join: app_id ↔ category_id
  pricing_plans.csv        — pricing plans per app
  pricing_plan_features.csv— features per pricing plan
  key_benefits.csv         — key benefit bullets per app
  reviews.csv              — individual user reviews (1 M+ rows; loaded for trends only)

Hot-reload: call get_data_fingerprint() on every Streamlit run and pass it as an
argument to load_data().  Streamlit's @st.cache_data caches on all arguments, so
a changed mtime/fingerprint automatically invalidates the cache.
"""
import os
import glob
import hashlib
import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_PROJECT_ROOT, "data")


# ── Hot-reload helper ─────────────────────────────────────────────────────────

def get_data_fingerprint() -> str:
    """
    Return a hex digest that changes whenever any CSV in DATA_DIR is modified.
    Pass this as an argument to @st.cache_data-decorated loaders so Streamlit
    automatically re-runs them when the user updates a file.
    """
    h = hashlib.md5()
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.csv"))):
        h.update(f"{path}:{os.path.getmtime(path)}".encode())
    return h.hexdigest()


# ── Raw loaders ───────────────────────────────────────────────────────────────

def _load_raw() -> dict[str, pd.DataFrame]:
    """Load all CSVs from DATA_DIR; return a dict keyed by table name."""
    tables = {}
    for fname in ["apps", "categories", "apps_categories",
                  "pricing_plans", "pricing_plan_features", "key_benefits"]:
        path = os.path.join(DATA_DIR, f"{fname}.csv")
        if os.path.exists(path):
            tables[fname] = pd.read_csv(path)
        else:
            tables[fname] = pd.DataFrame()

    # reviews.csv is large (1 M+ rows) — load only columns needed for trends
    reviews_path = os.path.join(DATA_DIR, "reviews.csv")
    if os.path.exists(reviews_path):
        tables["reviews"] = pd.read_csv(
            reviews_path,
            usecols=["app_id", "rating", "posted_at", "helpful_count"],
            low_memory=True,
        )
    else:
        tables["reviews"] = pd.DataFrame()

    return tables


# ── Main data builder ─────────────────────────────────────────────────────────

def generate_data(_fingerprint: str = "") -> tuple[pd.DataFrame, dict]:
    """
    Build the master apps DataFrame and compute all KPIs.

    The _fingerprint argument is unused internally but forces @st.cache_data
    to recompute when files change (caller passes get_data_fingerprint()).

    Returns:
        df    : apps-level DataFrame (one row per app, enriched with category/pricing)
        tables: raw sub-tables dict for chart-level access
    """
    tables = _load_raw()
    apps = tables["apps"].copy()
    categories = tables["categories"]
    apps_cats = tables["apps_categories"]
    pricing = tables["pricing_plans"]

    # --- 1. Derive primary category + full category list per app ---------------
    if not apps_cats.empty and not categories.empty:
        cat_lookup = categories.set_index("id")["title"].to_dict()

        # Primary category = first assignment for each app
        primary = (
            apps_cats.groupby("app_id")["category_id"]
            .first()
            .map(cat_lookup)
            .reset_index()
            .rename(columns={"category_id": "primary_category"})
        )
        apps = apps.merge(primary, left_on="id", right_on="app_id", how="left").drop(
            columns=["app_id"], errors="ignore"
        )

        # All categories as a list string
        all_cats = (
            apps_cats.assign(cat_title=apps_cats["category_id"].map(cat_lookup))
            .groupby("app_id")["cat_title"]
            .apply(lambda x: ", ".join(x.dropna()))
            .reset_index()
            .rename(columns={"cat_title": "categories"})
        )
        apps = apps.merge(all_cats, left_on="id", right_on="app_id", how="left").drop(
            columns=["app_id"], errors="ignore"
        )
    else:
        apps["primary_category"] = "Unknown"
        apps["categories"] = ""

    # --- 2. Pricing: flag free-plan apps --------------------------------------
    if not pricing.empty:
        has_free = (
            pricing[pricing["price"].str.lower().str.contains("free", na=False)]["app_id"]
            .unique()
        )
        apps["has_free_plan"] = apps["id"].isin(has_free)
        apps["pricing_type"] = apps["has_free_plan"].map(
            {True: "Free / Freemium", False: "Paid only"}
        )
    else:
        apps["has_free_plan"] = False
        apps["pricing_type"] = "Unknown"

    # --- 3. Temporal: parse lastmod -------------------------------------------
    if "lastmod" in apps.columns:
        apps["lastmod"] = pd.to_datetime(apps["lastmod"], errors="coerce")
        apps["year_updated"] = apps["lastmod"].dt.year
    else:
        apps["year_updated"] = np.nan

    # --- 4. Clean numeric columns ---------------------------------------------
    apps["rating"] = pd.to_numeric(apps["rating"], errors="coerce")
    apps["reviews_count"] = pd.to_numeric(apps["reviews_count"], errors="coerce").fillna(0).astype(int)

    return apps, tables


# ── KPI Computation ───────────────────────────────────────────────────────────

def compute_kpis(apps: pd.DataFrame, tables: dict) -> dict:
    """Compute the full KPI set from the enriched apps DataFrame."""
    categories = tables.get("categories", pd.DataFrame())
    apps_cats = tables.get("apps_categories", pd.DataFrame())
    pricing = tables.get("pricing_plans", pd.DataFrame())
    reviews = tables.get("reviews", pd.DataFrame())

    total_apps = len(apps)
    avg_rating = apps["rating"].mean()
    total_reviews = int(apps["reviews_count"].sum())
    total_categories = len(categories) if not categories.empty else 0
    pct_free = apps["has_free_plan"].mean() * 100 if total_apps else 0

    # --- By-category aggregation (for breakdowns) ---
    if not apps_cats.empty and not categories.empty:
        cat_lookup = categories.set_index("id")["title"].to_dict()
        by_cat = (
            apps_cats
            .assign(category=apps_cats["category_id"].map(cat_lookup))
            .merge(
                apps[["id", "rating", "reviews_count", "has_free_plan"]],
                left_on="app_id", right_on="id", how="left"
            )
            .groupby("category")
            .agg(
                app_count=("app_id", "nunique"),
                avg_rating=("rating", "mean"),
                total_reviews=("reviews_count", "sum"),
                pct_free=("has_free_plan", "mean"),
            )
            .reset_index()
        )
        by_cat["avg_rating"] = by_cat["avg_rating"].round(2)
        by_cat["pct_free"] = (by_cat["pct_free"] * 100).round(1)
        by_cat["total_reviews"] = by_cat["total_reviews"].astype(int)
        by_cat = by_cat.sort_values("app_count", ascending=False).reset_index(drop=True)
    else:
        by_cat = pd.DataFrame(columns=["category", "app_count", "avg_rating",
                                        "total_reviews", "pct_free"])

    top_category = by_cat.iloc[0]["category"] if not by_cat.empty else "N/A"
    highest_rated_cat = (
        by_cat.loc[by_cat["avg_rating"].idxmax(), "category"]
        if not by_cat.empty else "N/A"
    )

    # --- Developer aggregation ---
    dev = (
        apps.groupby("developer")
        .agg(
            app_count=("id", "count"),
            avg_rating=("rating", "mean"),
            total_reviews=("reviews_count", "sum"),
        )
        .reset_index()
        .sort_values("app_count", ascending=False)
        .reset_index(drop=True)
    )
    top_developer = dev.iloc[0]["developer"] if not dev.empty else "N/A"

    # --- Rating distribution ---
    bins = [0, 1, 2, 3, 4, 4.5, 5.01]
    labels = ["0–1", "1–2", "2–3", "3–4", "4–4.5", "4.5–5"]
    apps_rated = apps.dropna(subset=["rating"])
    if len(apps_rated):
        rating_dist = (
            apps_rated.assign(
                band=pd.cut(apps_rated["rating"], bins=bins, labels=labels, right=False)
            )
            .groupby("band", observed=True)
            .size()
            .reset_index(name="count")
        )
    else:
        rating_dist = pd.DataFrame({"band": labels, "count": [0] * len(labels)})

    # --- Pricing breakdown ---
    pricing_breakdown = (
        apps.groupby("pricing_type")
        .agg(app_count=("id", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )

    # --- Review trend (from reviews.csv if available) ---
    if not reviews.empty and "posted_at" in reviews.columns:
        reviews["posted_dt"] = pd.to_datetime(reviews["posted_at"], errors="coerce")
        reviews["year_month"] = reviews["posted_dt"].dt.to_period("M")
        review_trend = (
            reviews.groupby("year_month")
            .agg(review_count=("rating", "count"), avg_rating=("rating", "mean"))
            .reset_index()
        )
        review_trend["year_month"] = review_trend["year_month"].astype(str)
        review_trend = review_trend.sort_values("year_month")
    else:
        review_trend = pd.DataFrame(columns=["year_month", "review_count", "avg_rating"])

    # Build the kpis dict first, then attach opportunity signals which need
    # by_category to be present — both functions live in this module so no
    # circular import; signals are computed once here and reused everywhere.
    kpis: dict = {
        "total_apps": total_apps,
        "avg_rating": round(avg_rating, 2) if not np.isnan(avg_rating) else 0,
        "total_reviews": total_reviews,
        "total_categories": total_categories,
        "pct_free": round(pct_free, 1),
        "top_category": top_category,
        "highest_rated_cat": highest_rated_cat,
        "top_developer": top_developer,
        "by_category": by_cat,
        "by_developer": dev,
        "rating_dist": rating_dist,
        "pricing_breakdown": pricing_breakdown,
        "review_trend": review_trend,
    }
    kpis["opportunity_signals"] = get_opportunity_signals(kpis)
    return kpis


# ── Opportunity Signal Detection ─────────────────────────────────────────────

def get_opportunity_signals(kpis: dict) -> dict:
    """
    Identify under-served and over-saturated categories using data-driven
    percentile thresholds — no hardcoded numbers.

    Under-served  = low app count (below p25) AND high avg rating (above p75)
                    → quality niche with room for new entrants
    Over-saturated = high app count (above p75) AND low avg rating (below p25)
                    → crowded market with dissatisfied users (hard to stand out)
    """
    by_cat = kpis["by_category"].dropna(subset=["avg_rating"]).copy()
    if by_cat.empty:
        return {"under_served": pd.DataFrame(), "over_saturated": pd.DataFrame(),
                "thresholds": {}}

    # Noise filter: drop categories with negligible review volume.
    # Geographic entries / data artifacts typically have 0–few reviews while
    # real product categories accumulate significant engagement.
    # Threshold = 10th percentile of total_reviews across all categories (data-driven).
    min_reviews = max(1.0, float(np.percentile(by_cat["total_reviews"], 10)))
    by_cat = by_cat[by_cat["total_reviews"] >= min_reviews]

    if by_cat.empty:
        return {"under_served": pd.DataFrame(), "over_saturated": pd.DataFrame(),
                "thresholds": {}}

    p25_count  = float(np.percentile(by_cat["app_count"], 25))
    p75_count  = float(np.percentile(by_cat["app_count"], 75))
    p25_rating = float(np.percentile(by_cat["avg_rating"], 25))
    p75_rating = float(np.percentile(by_cat["avg_rating"], 75))

    under_served = (
        by_cat[
            (by_cat["app_count"] <= p25_count) &
            (by_cat["avg_rating"] >= p75_rating)
        ]
        .sort_values("avg_rating", ascending=False)
        .head(5)
    )
    over_saturated = (
        by_cat[
            (by_cat["app_count"] >= p75_count) &
            (by_cat["avg_rating"] <= p25_rating)
        ]
        .sort_values("app_count", ascending=False)
        .head(5)
    )

    return {
        "under_served": under_served,
        "over_saturated": over_saturated,
        "thresholds": {
            "p25_count": p25_count,
            "p75_count": p75_count,
            "p25_rating": p25_rating,
            "p75_rating": p75_rating,
            "min_reviews": min_reviews,
        },
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _category_signal(app_count: int, avg_rating: float, thresholds: dict) -> str:
    """
    Return a human-readable market signal for a category.
    Extracted from the f-string so it's testable and readable.
    All comparisons use data-driven percentile thresholds, never hardcoded values.
    """
    if app_count <= thresholds.get("p25_count", 0) and avg_rating >= thresholds.get("p75_rating", 0):
        return "OPPORTUNITY (quality niche)"
    if app_count >= thresholds.get("p75_count", 0) and avg_rating <= thresholds.get("p25_rating", 0):
        return "SATURATED (crowded+low quality)"
    return "COMPETITIVE (normal market)"


# ── AI Context Formatter ──────────────────────────────────────────────────────

def format_context(kpis: dict, focus_category: str | None = None) -> str:
    """
    Render KPIs as a compact text block for AI prompt injection.

    When focus_category is provided (detected from user's question), prepend a
    detailed category deep-dive section so the AI can answer category-specific
    questions with real numbers.
    """
    by_cat = kpis["by_category"]
    by_dev = kpis["by_developer"]

    top_cats = by_cat.head(10)[["category", "app_count", "avg_rating",
                                 "total_reviews", "pct_free"]].to_string(index=False)
    top_devs = by_dev.head(10)[["developer", "app_count", "avg_rating",
                                  "total_reviews"]].to_string(index=False)
    pricing = kpis["pricing_breakdown"].to_string(index=False)

    # ── Opportunity signals — read from pre-computed value in kpis (no recompute) ─
    signals = kpis.get("opportunity_signals") or get_opportunity_signals(kpis)
    th = signals["thresholds"]
    opp_lines = []
    if not signals["under_served"].empty:
        for _, row in signals["under_served"].iterrows():
            opp_lines.append(
                f"  OPPORTUNITY  {row['category']:<30}  "
                f"Apps={row['app_count']:,}  Rating={row['avg_rating']:.2f}"
            )
    if not signals["over_saturated"].empty:
        for _, row in signals["over_saturated"].iterrows():
            opp_lines.append(
                f"  SATURATED    {row['category']:<30}  "
                f"Apps={row['app_count']:,}  Rating={row['avg_rating']:.2f}"
            )
    opp_block = "\n".join(opp_lines) if opp_lines else "  (insufficient data)"

    base = f"""
=== SHOPIFY APP STORE — INTELLIGENCE SNAPSHOT ===

OVERALL KPIs:
  Total Apps       : {kpis['total_apps']:>10,}
  Avg Rating       : {kpis['avg_rating']:>10.2f} / 5.0
  Total Reviews    : {kpis['total_reviews']:>10,}
  Total Categories : {kpis['total_categories']:>10,}
  % Free / Freemium: {kpis['pct_free']:>9.1f}%

TOP 10 CATEGORIES (by app count):
{top_cats}

TOP 10 DEVELOPERS (by app count):
{top_devs}

PRICING BREAKDOWN:
{pricing}

MARKET OPPORTUNITY SIGNALS
(Thresholds derived from data: low_count≤{th.get('p25_count',0):.0f}, high_rating≥{th.get('p75_rating',0):.2f},
 high_count≥{th.get('p75_count',0):.0f}, low_rating≤{th.get('p25_rating',0):.2f})
{opp_block}

KEY FINDINGS:
  • Largest category  : {kpis['top_category']}
  • Highest-rated cat.: {kpis['highest_rated_cat']}
  • Most active dev.  : {kpis['top_developer']}
  • Free/Freemium apps: {kpis['pct_free']:.1f}% of total
""".strip()

    # ── Optional: category deep-dive block ────────────────────────────────────
    if focus_category:
        row = by_cat[by_cat["category"] == focus_category]
        if not row.empty:
            r = row.iloc[0]
            vs_rating = r["avg_rating"] - kpis["avg_rating"]
            vs_count  = r["app_count"] - by_cat["app_count"].mean()
            deep = f"""

--- CATEGORY FOCUS: {focus_category} ---
  App Count    : {r['app_count']:,}  ({vs_count:+.0f} vs store avg {by_cat['app_count'].mean():.0f})
  Avg Rating   : {r['avg_rating']:.2f}  ({vs_rating:+.2f} vs store avg {kpis['avg_rating']:.2f})
  Total Reviews: {r['total_reviews']:,}
  % Free/Fremm : {r['pct_free']:.1f}%
  Signal       : {_category_signal(int(r['app_count']), float(r['avg_rating']), th)}
"""
            base += deep

    return base
