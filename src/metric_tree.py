"""
Metric Tree Engine: Hierarchical analysis for the Shopify App Store dataset.

Tree structure:
  Apps → Category → Rating bands → Pricing
  Reviews → Over time → By category
  Rating → By developer → By category
  Pricing → Free vs Paid → By category

When the AI is asked "Why did X change?", this module drills down
the tree and returns a structured diagnosis with real numbers.
"""
import pandas as pd
import numpy as np
from typing import Optional


# ── Tree Definition ───────────────────────────────────────────────────────────

METRIC_TREE = {
    "apps": {
        "formula": "Total published apps in the store",
        "dimensions": ["Category", "Developer", "Pricing Type"],
        "drivers": ["new listings", "category growth", "developer activity"],
        "children": ["rating", "reviews", "pricing"],
    },
    "rating": {
        "formula": "Avg Rating = sum(star ratings) ÷ total ratings given",
        "dimensions": ["Category", "Developer", "Pricing Type"],
        "drivers": ["review quality", "app reliability", "developer responsiveness"],
        "children": [],
    },
    "reviews": {
        "formula": "Total user reviews submitted",
        "dimensions": ["Category", "Time period"],
        "drivers": ["app installs", "user engagement", "developer replies"],
        "children": ["rating"],
    },
    "pricing": {
        "formula": "Distribution of Free / Freemium vs Paid apps",
        "dimensions": ["Category", "Price tier"],
        "drivers": ["monetization strategy", "market competition"],
        "children": [],
    },
}


# ── Analysis Functions ─────────────────────────────────────────────────────────

def analyze_metric(metric: str, df: pd.DataFrame,
                   kpis: Optional[dict] = None) -> str:
    """
    Return a textual root-cause analysis for the given metric.
    kpis: pre-computed KPI dict — avoids redundant groupby on every call.
    """
    m = metric.lower().strip()

    if "rating" in m or "score" in m or "star" in m:
        return _analyze_rating(df, kpis)
    elif "review" in m or "feedback" in m or "comment" in m:
        return _analyze_reviews(df, kpis)
    elif "pric" in m or "free" in m or "paid" in m or "monetiz" in m:
        return _analyze_pricing(df, kpis)
    elif "developer" in m or "partner" in m or "vendor" in m:
        return _analyze_developers(df, kpis)
    elif "category" in m or "segment" in m or "market" in m:
        return _analyze_categories(df, kpis)
    else:
        return _analyze_overview(df, kpis)


def _analyze_rating(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    """Drill: Rating → by category → by pricing type → top/bottom apps."""
    lines = ["RATING ROOT-CAUSE ANALYSIS (Metric Tree Traversal)", "=" * 55]

    # L1 – Overall (use pre-computed kpis to avoid recomputing mean)
    overall = kpis["avg_rating"] if kpis else float(df["rating"].mean())
    n_rated  = int(df["rating"].notna().sum())
    lines.append(f"\n[L1] Overall  →  Avg Rating = {overall:.2f}/5.0  ({n_rated:,} rated apps)")

    # L2 – By category (read from pre-computed kpis["by_category"] — no groupby)
    by_cat = kpis.get("by_category") if kpis else None
    if by_cat is not None and not by_cat.empty:
        lines.append("\n[L2] Rating by Category (top 10 by avg rating):")
        top10 = (
            by_cat.dropna(subset=["avg_rating"])
            .sort_values("avg_rating", ascending=False)
            .head(10)
        )
        for _, row in top10.iterrows():
            flag = (
                " ◄ TOP"       if row["avg_rating"] >= overall + 0.3 else
                " ◄ BELOW AVG" if row["avg_rating"] <  overall - 0.3 else ""
            )
            lines.append(
                f"  {str(row['category']):<35}  "
                f"Rating={row['avg_rating']:.2f}  Apps={row['app_count']:,}{flag}"
            )

    # L3 – By pricing type (cheap: only 2–3 groups, and avg_reviews needs df)
    lines.append("\n[L3] Rating by Pricing Type:")
    if "pricing_type" in df.columns:
        pt = (
            df.dropna(subset=["rating"])
            .groupby("pricing_type")
            .agg(avg_rating=("rating", "mean"), app_count=("id", "count"))
            .reset_index()
        )
        for _, row in pt.iterrows():
            lines.append(
                f"  {str(row['pricing_type']):<25}  "
                f"Rating={row['avg_rating']:.2f}  Apps={row['app_count']:,}"
            )

    # L4 – Top apps (threshold = 60th percentile of review counts, not hardcoded 100)
    review_threshold = max(1, int(df["reviews_count"].quantile(0.6)))
    lines.append(f"\n[L4] Highest-Rated Apps (≥{review_threshold:,} reviews):")
    top = (
        df[df["reviews_count"] >= review_threshold]
        .sort_values("rating", ascending=False)
        .head(5)[["title", "developer", "rating", "reviews_count"]]
    )
    for _, row in top.iterrows():
        lines.append(
            f"  {str(row['title'])[:40]:<42}  Rating={row['rating']:.1f}  "
            f"Reviews={row['reviews_count']:,}"
        )

    lines.append("\n[ROOT CAUSE SUMMARY]")
    lines.append(
        f"  Overall avg rating: {overall:.2f}/5.0 across {n_rated:,} apps. "
        f"Apps with {review_threshold:,}+ reviews skew higher due to selection bias "
        f"(only popular apps accumulate that many reviews)."
    )
    return "\n".join(lines)


def _analyze_reviews(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    """Drill: Reviews → by category → by developer → developer responsiveness."""
    lines = ["REVIEW ANALYSIS (Metric Tree Traversal)", "=" * 55]

    # L1 – totals (use pre-computed kpis)
    total_reviews = kpis["total_reviews"] if kpis else int(df["reviews_count"].sum())
    avg_reviews   = float(df["reviews_count"].mean())
    lines.append(f"\n[L1] Total Reviews: {total_reviews:,}  (avg per app: {avg_reviews:.0f})")

    # L2 – By category (from pre-computed kpis — no groupby)
    by_cat = kpis.get("by_category") if kpis else None
    if by_cat is not None and not by_cat.empty:
        lines.append("\n[L2] Review Volume by Category (top 10):")
        top10 = by_cat.sort_values("total_reviews", ascending=False).head(10)
        for _, row in top10.iterrows():
            share = int(row["total_reviews"]) / max(total_reviews, 1) * 100
            lines.append(
                f"  {str(row['category']):<35}  "
                f"Reviews={int(row['total_reviews']):>8,}  ({share:.1f}% share)"
            )

    # L3 – Most reviewed apps (individual app data — needs df)
    lines.append("\n[L3] Top 5 Most-Reviewed Apps:")
    top = df.sort_values("reviews_count", ascending=False).head(5)
    for _, row in top.iterrows():
        lines.append(
            f"  {str(row['title'])[:40]:<42}  Reviews={row['reviews_count']:,}  "
            f"Rating={row['rating'] if pd.notna(row['rating']) else 'N/A'}"
        )

    lines.append("\n[ROOT CAUSE SUMMARY]")
    lines.append(
        "  Top-10% of apps by reviews account for the majority of review volume. "
        "High review count strongly correlates with app visibility and category leadership."
    )
    return "\n".join(lines)


def _analyze_pricing(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    """Drill: Pricing → Free vs Paid → by category → rating impact."""
    lines = ["PRICING ANALYSIS (Metric Tree Traversal)", "=" * 55]

    if "has_free_plan" in df.columns:
        # L1 – totals (from kpis)
        pct_free = kpis["pct_free"] if kpis else df["has_free_plan"].mean() * 100
        total    = kpis["total_apps"] if kpis else len(df)
        n_free   = round(total * pct_free / 100)
        n_paid   = total - n_free
        lines.append(
            f"\n[L1] Total Apps: {total:,}  |  "
            f"Free/Freemium: {n_free:,} ({pct_free:.1f}%)  |  Paid: {n_paid:,}"
        )

        # L2 – Rating by pricing type (cheap: 2–3 groups, and avg_reviews needs df)
        lines.append("\n[L2] Rating by Pricing Model:")
        pt = (
            df.dropna(subset=["rating"])
            .groupby("pricing_type")
            .agg(avg_rating=("rating", "mean"), app_count=("id", "count"),
                 avg_reviews=("reviews_count", "mean"))
            .reset_index()
        )
        for _, row in pt.iterrows():
            lines.append(
                f"  {str(row['pricing_type']):<25}  "
                f"Rating={row['avg_rating']:.2f}  Apps={row['app_count']:,}  "
                f"Avg Reviews={row['avg_reviews']:.0f}"
            )

        # L3 – Free % by top category (from pre-computed kpis — no groupby)
        by_cat = kpis.get("by_category") if kpis else None
        if by_cat is not None and not by_cat.empty:
            lines.append("\n[L3] Free % by Top Category:")
            for _, row in by_cat.head(10).iterrows():
                lines.append(
                    f"  {str(row['category']):<35}  "
                    f"Free={row['pct_free']:.0f}%  Apps={row['app_count']:,}"
                )
    else:
        lines.append("\n  Pricing data not available.")

    lines.append("\n[ROOT CAUSE SUMMARY]")
    lines.append(
        "  Free/Freemium dominates the Shopify App Store. Paid-only apps "
        "typically serve niche professional workflows and may command higher ratings "
        "due to lower install volume filtering out casual users."
    )
    return "\n".join(lines)


def _analyze_developers(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    """Drill: Developer → by app count → by rating → by reviews."""
    lines = ["DEVELOPER ANALYSIS (Metric Tree Traversal)", "=" * 55]

    by_dev = kpis.get("by_developer") if kpis else None

    # L1 – Total developers (from kpis — no nunique scan)
    total_devs = len(by_dev) if by_dev is not None else df["developer"].nunique()
    lines.append(f"\n[L1] Total Unique Developers: {total_devs:,}")

    # L2 – Top developers by app count (from pre-computed kpis — no groupby)
    lines.append("\n[L2] Top 10 Developers by App Count:")
    if by_dev is not None and not by_dev.empty:
        for _, row in by_dev.head(10).iterrows():
            lines.append(
                f"  {str(row['developer'])[:40]:<42}  Apps={row['app_count']:>4}  "
                f"Rating={row['avg_rating']:.2f}  Reviews={row['total_reviews']:,}"
            )

    # L3 – Top by rating (threshold = median app count, not hardcoded 3)
    lines.append("\n[L3] Top 10 Developers by Avg Rating (≥ median app count):")
    dev_rated = by_dev.copy() if by_dev is not None else (
        df.groupby("developer")
        .agg(app_count=("id", "count"), avg_rating=("rating", "mean"))
        .reset_index()
    )
    # Data-driven minimum: median app count per developer
    min_apps = max(2, int(dev_rated["app_count"].median()))
    dev_rated = (
        dev_rated[dev_rated["app_count"] >= min_apps]
        .sort_values("avg_rating", ascending=False)
        .head(10)
    )
    for _, row in dev_rated.iterrows():
        lines.append(
            f"  {str(row['developer'])[:40]:<42}  Apps={row['app_count']:>4}  "
            f"Rating={row['avg_rating']:.2f}"
        )

    return "\n".join(lines)


def _analyze_categories(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    """Drill: Category → app count → avg rating → pricing mix."""
    lines = ["CATEGORY ANALYSIS (Metric Tree Traversal)", "=" * 55]

    by_cat = kpis.get("by_category") if kpis else None

    # L1 – total categories (from kpis)
    total_cats = (
        kpis["total_categories"] if kpis
        else (df["primary_category"].nunique() if "primary_category" in df.columns else 0)
    )
    lines.append(f"\n[L1] Total Categories: {total_cats:,}")

    # L2 – Top 15 by app count (from pre-computed kpis — no groupby)
    lines.append("\n[L2] Top 15 Categories by App Count:")
    if by_cat is not None and not by_cat.empty:
        for _, row in by_cat.head(15).iterrows():
            lines.append(
                f"  {str(row['category']):<35}  Apps={row['app_count']:>5,}  "
                f"Rating={row['avg_rating']:.2f}  Free={row['pct_free']:.0f}%"
            )
    elif "primary_category" in df.columns:
        cat = (
            df.groupby("primary_category")
            .agg(app_count=("id", "count"), avg_rating=("rating", "mean"),
                 pct_free=("has_free_plan", "mean"))
            .reset_index()
            .sort_values("app_count", ascending=False)
            .head(15)
        )
        for _, row in cat.iterrows():
            lines.append(
                f"  {str(row['primary_category']):<35}  Apps={row['app_count']:>5,}  "
                f"Rating={row['avg_rating']:.2f}  Free={row['pct_free']*100:.0f}%"
            )

    # L3 – Highest-rated categories (threshold = p25 app count, not hardcoded 20)
    if by_cat is not None and not by_cat.empty:
        min_apps = max(5, int(by_cat["app_count"].quantile(0.25)))
        lines.append(f"\n[L3] Highest-Rated Categories (≥{min_apps:,} apps):")
        cat_large = (
            by_cat[by_cat["app_count"] >= min_apps]
            .sort_values("avg_rating", ascending=False)
            .head(5)
        )
        for _, row in cat_large.iterrows():
            lines.append(
                f"  {str(row['category']):<35}  Avg Rating={row['avg_rating']:.2f}"
            )

    return "\n".join(lines)


def _analyze_overview(df: pd.DataFrame, kpis: Optional[dict] = None) -> str:
    """General app store overview traversal (fully from kpis when available)."""
    lines = ["APP STORE OVERVIEW — METRIC TREE SUMMARY", "=" * 55]

    if kpis:
        # Zero groupbys — all stats from pre-computed kpis
        total        = kpis["total_apps"]
        avg_rating   = kpis["avg_rating"]
        total_reviews = kpis["total_reviews"]
        pct_free     = kpis["pct_free"]
        total_devs   = len(kpis.get("by_developer", []))
        top_cat      = kpis["top_category"]
        best_rated   = kpis["highest_rated_cat"]
        top_dev      = kpis["top_developer"]
    else:
        total        = len(df)
        avg_rating   = float(df["rating"].mean())
        total_reviews = int(df["reviews_count"].sum())
        pct_free     = df["has_free_plan"].mean() * 100 if "has_free_plan" in df.columns else 0
        total_devs   = df["developer"].nunique()
        top_cat      = (
            df.groupby("primary_category")["id"].count().idxmax()
            if "primary_category" in df.columns else "N/A"
        )
        best_rated   = (
            df.groupby("primary_category")["rating"].mean().idxmax()
            if "primary_category" in df.columns else "N/A"
        )
        top_dev      = df.groupby("developer")["id"].count().idxmax()

    lines.append(f"\nTotal Apps       : {total:>10,}")
    lines.append(f"Avg Rating       : {avg_rating:>10.2f} / 5.0")
    lines.append(f"Total Reviews    : {total_reviews:>10,}")
    lines.append(f"Free/Freemium    : {pct_free:>9.1f}%")
    lines.append(f"Unique Developers: {total_devs:>10,}")
    lines.append(f"\nLargest Category : {top_cat}")
    lines.append(f"Best Rated Cat.  : {best_rated}")
    lines.append(f"Most Active Dev. : {top_dev}")

    return "\n".join(lines)


# ── Question → Relevant Metric Detection ─────────────────────────────────────

def detect_metric(question: str) -> str:
    """Map user question to the most relevant metric for tree traversal."""
    q = question.lower()

    if any(w in q for w in ["rating", "score", "stars", "rated", "quality"]):
        return "rating"
    if any(w in q for w in ["review", "feedback", "comment", "opinion", "sentiment"]):
        return "reviews"
    if any(w in q for w in ["price", "pricing", "free", "paid", "freemium", "plan", "cost", "monetiz"]):
        return "pricing"
    if any(w in q for w in ["developer", "partner", "vendor", "publisher", "maker"]):
        return "developer"
    if any(w in q for w in ["category", "categories", "segment", "market", "niche", "vertical"]):
        return "category"
    if any(w in q for w in ["app", "apps", "tool", "plugin", "overview", "total", "how many"]):
        return "apps"
    return "overview"


def _analyze_category_deep_dive(df: pd.DataFrame, category_name: str,
                                 kpis: Optional[dict] = None) -> str:
    """
    Full drill-down for a specific named category.
    All benchmarks are computed from the data — no hardcoded thresholds.
    """
    lines = [f"CATEGORY DEEP-DIVE: {category_name}", "=" * 60]

    # Proper column existence check — avoids the df.get() anti-pattern
    # (df.get(col, pd.Series(dtype=str)) creates a length-0 Series which
    # raises a ValueError when used as a boolean mask on a non-empty DataFrame)
    if "primary_category" in df.columns:
        cat_df = df[df["primary_category"] == category_name]
    else:
        cat_df = pd.DataFrame()

    if cat_df.empty and "categories" in df.columns:
        cat_df = df[df["categories"].str.contains(category_name, case=False, na=False)]

    if cat_df.empty:
        lines.append(f"\n  No apps found for category: {category_name}")
        return "\n".join(lines)

    # L1: Category vs market
    cat_apps    = len(cat_df)
    cat_rating  = float(cat_df["rating"].mean())
    cat_reviews = int(cat_df["reviews_count"].sum())
    cat_free    = cat_df["has_free_plan"].mean() * 100 if "has_free_plan" in cat_df.columns else 0

    mkt_apps   = kpis["total_apps"] if kpis else len(df)
    mkt_rating = kpis["avg_rating"]  if kpis else float(df["rating"].mean())

    lines.append("\n[L1] Category Summary")
    lines.append(f"  Apps in category : {cat_apps:,}  ({cat_apps / max(mkt_apps, 1) * 100:.1f}% of store)")
    lines.append(f"  Avg Rating       : {cat_rating:.2f}  (store avg {mkt_rating:.2f}, delta {cat_rating - mkt_rating:+.2f})")
    lines.append(f"  Total Reviews    : {cat_reviews:,}")
    lines.append(f"  % Free / Freemium: {cat_free:.1f}%")

    # L2: Top apps in this category (by review count — proxy for popularity)
    lines.append(f"\n[L2] Top Apps in {category_name} (by review volume):")
    top = (
        cat_df.dropna(subset=["reviews_count", "rating"])
        .sort_values("reviews_count", ascending=False)
        .head(5)[["title", "developer", "rating", "reviews_count", "pricing_type"]]
    )
    for _, row in top.iterrows():
        lines.append(
            f"  {str(row['title'])[:38]:<40}  "
            f"Rating={row['rating']:.1f}  Reviews={row['reviews_count']:,}  "
            f"({row['pricing_type']})"
        )

    # L3: Rating distribution within category (data-driven bands)
    lines.append(f"\n[L3] Rating Distribution in {category_name}:")
    rated = cat_df.dropna(subset=["rating"])
    if not rated.empty:
        p25  = float(rated["rating"].quantile(0.25))
        med  = float(rated["rating"].median())
        p75  = float(rated["rating"].quantile(0.75))
        low  = int((rated["rating"] < p25).sum())
        mid  = int(((rated["rating"] >= p25) & (rated["rating"] < p75)).sum())
        high = int((rated["rating"] >= p75).sum())
        lines.append(f"  Below p25 ({p25:.1f})  : {low:,} apps")
        lines.append(f"  p25–p75 ({p25:.1f}–{p75:.1f}): {mid:,} apps")
        lines.append(f"  Above p75 ({p75:.1f})  : {high:,} apps")
        lines.append(f"  Median rating       : {med:.2f}")

    # L4: Competitive signal — reuse pre-computed percentile thresholds from kpis
    lines.append("\n[L4] Competitive Signal (data-driven):")
    if kpis and "opportunity_signals" in kpis:
        # Reuse thresholds already computed in get_opportunity_signals()
        th         = kpis["opportunity_signals"]["thresholds"]
        count_p25  = th.get("p25_count",  0.0)
        count_p75  = th.get("p75_count",  0.0)
        rating_p75 = th.get("p75_rating", 0.0)
        rating_p25 = th.get("p25_rating", 0.0)
    elif kpis and "by_category" in kpis:
        counts     = kpis["by_category"]["app_count"]
        ratings    = kpis["by_category"]["avg_rating"].dropna()
        count_p25  = float(counts.quantile(0.25))
        count_p75  = float(counts.quantile(0.75))
        rating_p75 = float(ratings.quantile(0.75))
        rating_p25 = float(ratings.quantile(0.25))
    else:
        # Fallback: derive from df (one groupby, only when kpis not provided)
        if "primary_category" in df.columns:
            sizes     = df.groupby("primary_category").size()
            count_p25 = float(sizes.quantile(0.25))
            count_p75 = float(sizes.quantile(0.75))
        else:
            count_p25 = count_p75 = 0.0
        rating_p75 = float(df["rating"].quantile(0.75))
        rating_p25 = float(df["rating"].quantile(0.25))

    if cat_apps <= count_p25 and cat_rating >= rating_p75:
        signal = "OPPORTUNITY — few apps, high satisfaction. Strong entry potential."
    elif cat_apps >= count_p75 and cat_rating <= rating_p25:
        signal = "SATURATED + LOW QUALITY — crowded with poor ratings. Hard to differentiate."
    elif cat_apps >= count_p75:
        signal = "COMPETITIVE — large category. Need strong differentiation to stand out."
    elif cat_rating >= rating_p75:
        signal = "QUALITY NICHE — high user satisfaction. Consider entering."
    else:
        signal = "NORMAL MARKET — average competition and quality."

    lines.append(f"  {signal}")

    return "\n".join(lines)


def get_tree_analysis(
    question: str,
    df: pd.DataFrame,
    kpis: Optional[dict] = None,
    focus_category: Optional[str] = None,
) -> str:
    """
    Entry point: auto-detect metric and run full tree traversal.

    kpis: pre-computed KPI dict — passing this avoids re-running groupby
          aggregations on every AI call (the data doesn't change between calls).
    focus_category: when set, runs a category deep-dive instead of metric routing.
    """
    if focus_category:
        return _analyze_category_deep_dive(df, focus_category, kpis)
    metric = detect_metric(question)
    return analyze_metric(metric, df, kpis)
