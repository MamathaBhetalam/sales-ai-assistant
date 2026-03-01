"""
Visualizations: Auto-generated Plotly charts for the Shopify App Store dataset.

auto_chart()        — keyword-routing: picks the best chart for a given question
overview_chart()    — default landing view (top categories bar chart)
metric_tree_chart() — sidebar sunburst: Categories → Free/Paid split (color=rating)
"""
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.data_layer import get_opportunity_signals
from src.entity_extractor import extract_entities as _extract_entities

# ── Palette ───────────────────────────────────────────────────────────────────
_BLUE   = "#2563eb"
_GREEN  = "#16a34a"
_AMBER  = "#d97706"
_RED    = "#dc2626"
_PURPLE = "#7c3aed"
_SLATE  = "#64748b"
_BG     = "#ffffff"
_GRID   = "#e2e8f0"


# ── Shared style helpers ──────────────────────────────────────────────────────

def _base_layout(**kwargs) -> dict:
    return dict(
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family="Inter, sans-serif", color="#1a2744", size=12),
        margin=dict(l=50, r=30, t=50, b=50),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID),
        **kwargs,
    )


def _axis_style() -> dict:
    return dict(
        gridcolor=_GRID,
        zerolinecolor=_GRID,
        tickfont=dict(size=11, color=_SLATE),
        title_font=dict(size=12, color=_SLATE),
    )


# ── Individual chart builders ─────────────────────────────────────────────────

def _top_categories_chart(df: pd.DataFrame, kpis: dict, n: int = 15) -> go.Figure:
    """Horizontal bar: top N categories by app count."""
    by_cat = kpis["by_category"].head(n).sort_values("app_count")
    hover = [
        f"<b>{c}</b><br>Apps: {a:,}<br>Avg Rating: {r:.2f}"
        for c, a, r in zip(by_cat["category"], by_cat["app_count"], by_cat["avg_rating"])
    ]
    fig = go.Figure(go.Bar(
        x=by_cat["app_count"],
        y=by_cat["category"],
        orientation="h",
        marker_color=_BLUE,
        text=by_cat["app_count"],
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Top Categories by App Count",
        xaxis=dict(title="Number of Apps", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=420,
        **_base_layout(),
    )
    return fig


def _rating_distribution_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """Bar chart: rating band distribution."""
    dist = kpis["rating_dist"]
    colors = [_RED, _AMBER, _AMBER, _GREEN, _BLUE, _BLUE]
    fig = go.Figure(go.Bar(
        x=dist["band"].astype(str),
        y=dist["count"],
        marker_color=colors[:len(dist)],
        text=dist["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertemplate="Rating %{x}<br>Apps: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="App Rating Distribution",
        xaxis=dict(title="Rating Band", **_axis_style()),
        yaxis=dict(title="Number of Apps", **_axis_style()),
        height=380,
        **_base_layout(),
    )
    return fig


def _pricing_breakdown_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """Pie chart: Free/Freemium vs Paid-only."""
    pb = kpis["pricing_breakdown"]
    if pb.empty:
        return _top_categories_chart(df, kpis)
    fig = go.Figure(go.Pie(
        labels=pb["pricing_type"],
        values=pb["app_count"],
        hole=0.45,
        marker=dict(colors=[_BLUE, _GREEN, _AMBER]),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Apps: %{value:,}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Pricing Model Breakdown",
        height=380,
        **_base_layout(),
    )
    return fig


def _top_developers_chart(df: pd.DataFrame, kpis: dict, n: int = 15) -> go.Figure:
    """Horizontal bar: top N developers by app count."""
    devs = kpis["by_developer"].head(n).sort_values("app_count")
    hover = [
        f"<b>{d}</b><br>Apps: {a}<br>Avg Rating: {r:.2f}"
        for d, a, r in zip(devs["developer"], devs["app_count"], devs["avg_rating"])
    ]
    fig = go.Figure(go.Bar(
        x=devs["app_count"],
        y=devs["developer"],
        orientation="h",
        marker_color=_PURPLE,
        text=devs["app_count"],
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Top Developers by App Count",
        xaxis=dict(title="Number of Apps", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=420,
        **_base_layout(),
    )
    return fig


def _category_rating_chart(df: pd.DataFrame, kpis: dict, n: int = 20) -> go.Figure:
    """Scatter: category avg rating vs app count (bubble size = total reviews)."""
    by_cat = kpis["by_category"].head(n).dropna(subset=["avg_rating"])
    max_rev = by_cat["total_reviews"].max() or 1
    sizes = (by_cat["total_reviews"] / max_rev * 40 + 8).clip(8, 40)
    fig = go.Figure(go.Scatter(
        x=by_cat["app_count"],
        y=by_cat["avg_rating"],
        mode="markers+text",
        text=by_cat["category"].apply(lambda s: s[:20] if isinstance(s, str) else ""),
        textposition="top center",
        marker=dict(
            size=sizes,
            color=by_cat["avg_rating"],
            colorscale=[[0, _RED], [0.5, _AMBER], [1, _GREEN]],
            showscale=True,
            colorbar=dict(title="Avg Rating"),
        ),
        hovertemplate="<b>%{text}</b><br>Apps: %{x:,}<br>Rating: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Category: App Count vs Avg Rating (bubble = review volume)",
        xaxis=dict(title="Number of Apps", **_axis_style()),
        yaxis=dict(title="Avg Rating", **_axis_style()),
        height=420,
        **_base_layout(),
    )
    return fig


def _review_trend_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """Line chart: monthly review volume over time."""
    trend = kpis.get("review_trend", pd.DataFrame())
    if trend.empty or len(trend) < 2:
        return _top_categories_chart(df, kpis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["year_month"],
        y=trend["review_count"],
        name="Reviews",
        mode="lines+markers",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="Period: %{x}<br>Reviews: %{y:,}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trend["year_month"],
        y=trend["avg_rating"],
        name="Avg Rating",
        mode="lines",
        line=dict(color=_GREEN, width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Period: %{x}<br>Avg Rating: %{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        title="Review Volume & Avg Rating Over Time",
        xaxis=dict(title="Month", **_axis_style()),
        yaxis=dict(title="Review Count", **_axis_style()),
        yaxis2=dict(
            title="Avg Rating", overlaying="y", side="right",
            range=[0, 5.5], **_axis_style()
        ),
        height=380,
        **_base_layout(),
    )
    fig.update_layout(legend=dict(x=0.01, y=0.99))
    return fig


def _apps_rating_scatter(df: pd.DataFrame, kpis: dict, n: int = 500) -> go.Figure:
    """Scatter: individual apps — reviews_count vs rating, colored by pricing."""
    sample = df.dropna(subset=["rating", "reviews_count"]).head(n)
    color_map = {"Free / Freemium": _BLUE, "Paid only": _AMBER, "Unknown": _SLATE}

    fig = go.Figure()
    for ptype, group in sample.groupby("pricing_type"):
        fig.add_trace(go.Scatter(
            x=group["reviews_count"],
            y=group["rating"],
            mode="markers",
            name=ptype,
            marker=dict(color=color_map.get(str(ptype), _SLATE), size=6, opacity=0.65),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Dev: %{customdata[1]}<br>"
                "Reviews: %{x:,}<br>Rating: %{y:.2f}<extra></extra>"
            ),
            customdata=group[["title", "developer"]].values,
        ))
    fig.update_layout(
        title="Apps: Rating vs Review Count",
        xaxis=dict(title="Review Count", **_axis_style()),
        yaxis=dict(title="Rating", **_axis_style()),
        height=420,
        **_base_layout(),
    )
    return fig


def _free_vs_paid_by_category(df: pd.DataFrame, kpis: dict, n: int = 12) -> go.Figure:
    """Stacked bar: free vs paid app count per top category."""
    by_cat = kpis["by_category"].head(n)
    free_counts = (by_cat["app_count"] * by_cat["pct_free"] / 100).round().astype(int)
    paid_counts = by_cat["app_count"] - free_counts
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Free / Freemium",
        x=by_cat["category"],
        y=free_counts,
        marker_color=_BLUE,
    ))
    fig.add_trace(go.Bar(
        name="Paid only",
        x=by_cat["category"],
        y=paid_counts,
        marker_color=_AMBER,
    ))
    fig.update_layout(
        barmode="stack",
        title="Free vs Paid Apps by Category",
        xaxis=dict(title="", tickangle=-35, **_axis_style()),
        yaxis=dict(title="Number of Apps", **_axis_style()),
        height=400,
        **_base_layout(),
    )
    return fig


# ── Category deep-dive chart ─────────────────────────────────────────────────

def _category_detail_chart(df: pd.DataFrame, kpis: dict, category_name: str) -> go.Figure:
    """
    Top apps in a specific category by review count, bars colored by rating.
    Fully dynamic — title and data built from category_name at runtime.
    """
    # Filter apps to this category — proper column check avoids the df.get()
    # anti-pattern (a length-0 pd.DataFrame() fallback has a different Pyright
    # inferred type than a filtered slice of df, causing overload errors downstream).
    # df.iloc[0:0] gives an empty DataFrame with the same schema as df.
    if "primary_category" in df.columns:
        cat_df = df[df["primary_category"] == category_name]
    else:
        cat_df = df.iloc[0:0]  # empty, same columns/dtypes as df

    if cat_df.empty and "categories" in df.columns:
        cat_df = df[df["categories"].str.contains(category_name, case=False, na=False)]

    if cat_df.empty:
        return _top_categories_chart(df, kpis)

    _mask = cat_df["reviews_count"].notna() & cat_df["rating"].notna()
    top   = cat_df[_mask].sort_values("reviews_count", ascending=False).head(15)

    # Color each bar by its rating relative to category median (data-driven)
    median_rating = top["rating"].median()
    bar_colors = [
        _GREEN if r >= median_rating + 0.3 else
        _RED   if r < median_rating - 0.3 else
        _AMBER
        for r in top["rating"]
    ]

    hover = [
        f"<b>{t}</b><br>Dev: {d}<br>Reviews: {rv:,}<br>Rating: {r:.1f}<br>{pt}"
        for t, d, rv, r, pt in zip(
            top["title"], top["developer"],
            top["reviews_count"], top["rating"],
            top.get("pricing_type", [""] * len(top))
        )
    ]

    fig = go.Figure(go.Bar(
        x=top["reviews_count"],
        y=top["title"].apply(lambda s: s[:35] if isinstance(s, str) else s),
        orientation="h",
        marker_color=bar_colors,
        text=top["rating"].apply(lambda r: f"★{r:.1f}"),
        textposition="inside",
        hovertext=hover,
        hoverinfo="text",
    ))

    store_avg  = kpis["avg_rating"]
    cat_row    = kpis["by_category"][kpis["by_category"]["category"] == category_name]
    cat_avg    = cat_row.iloc[0]["avg_rating"] if not cat_row.empty else store_avg
    n_apps     = cat_row.iloc[0]["app_count"]  if not cat_row.empty else len(cat_df)

    fig.update_layout(
        title=f"Top Apps in '{category_name}'  ({n_apps:,} apps · avg rating {cat_avg:.2f} vs store {store_avg:.2f})",
        xaxis=dict(title="Review Count", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=450,
        **_base_layout(),
    )
    return fig


# ── Opportunity / saturation map ─────────────────────────────────────────────

def _opportunities_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """
    Scatter of all categories: x=app_count, y=avg_rating.
    Quadrant boundaries derived from data percentiles — no hardcoded values.
    Annotates opportunity and saturated zones.
    """
    # Read pre-computed signals from kpis (computed once in compute_kpis).
    # Falls back to recomputing only if kpis was built without it.
    by_cat  = kpis["by_category"].dropna(subset=["avg_rating"])
    signals = kpis.get("opportunity_signals") or get_opportunity_signals(kpis)
    th      = signals["thresholds"]

    p25_count  = th.get("p25_count", by_cat["app_count"].quantile(0.25))
    p75_count  = th.get("p75_count", by_cat["app_count"].quantile(0.75))
    p25_rating = th.get("p25_rating", by_cat["avg_rating"].quantile(0.25))
    p75_rating = th.get("p75_rating", by_cat["avg_rating"].quantile(0.75))

    under_set = set(signals["under_served"]["category"].tolist())
    over_set  = set(signals["over_saturated"]["category"].tolist())

    def _quadrant_color(row):
        if row["category"] in under_set:
            return _GREEN
        if row["category"] in over_set:
            return _RED
        return _BLUE

    colors = by_cat.apply(_quadrant_color, axis=1).tolist()
    labels = by_cat["category"].apply(lambda s: s[:18] if isinstance(s, str) else "")

    fig = go.Figure(go.Scatter(
        x=by_cat["app_count"],
        y=by_cat["avg_rating"],
        mode="markers+text",
        text=labels,
        textposition="top center",
        textfont=dict(size=8),
        marker=dict(
            color=colors,
            size=by_cat["total_reviews"].apply(
                lambda v: max(6, min(30, v / by_cat["total_reviews"].max() * 30))
            ),
            opacity=0.75,
        ),
        hovertemplate=(
            "<b>%{text}</b><br>"
            "Apps: %{x:,}<br>"
            "Avg Rating: %{y:.2f}<br>"
            "<extra></extra>"
        ),
    ))

    # Quadrant reference lines (data-driven)
    fig.add_vline(x=p25_count,  line=dict(color=_SLATE, dash="dot", width=1))
    fig.add_vline(x=p75_count,  line=dict(color=_SLATE, dash="dot", width=1))
    fig.add_hline(y=p25_rating, line=dict(color=_SLATE, dash="dot", width=1))
    fig.add_hline(y=p75_rating, line=dict(color=_SLATE, dash="dot", width=1))

    # Zone annotations
    x_max = float(by_cat["app_count"].max()) * 0.95
    fig.add_annotation(x=p25_count * 0.5, y=p75_rating + 0.05,
                       text="Opportunity Zone", showarrow=False,
                       font=dict(color=_GREEN, size=10, family="Inter"))
    fig.add_annotation(x=x_max, y=p25_rating - 0.05,
                       text="Saturated Zone", showarrow=False,
                       font=dict(color=_RED, size=10, family="Inter"))

    fig.update_layout(
        title=(
            f"Market Map: App Count vs Avg Rating  "
            f"(thresholds: count p25={p25_count:.0f}/p75={p75_count:.0f}, "
            f"rating p25={p25_rating:.2f}/p75={p75_rating:.2f})"
        ),
        xaxis=dict(title="Number of Apps in Category", **_axis_style()),
        yaxis=dict(title="Avg Rating", **_axis_style()),
        height=480,
        **_base_layout(),
    )
    return fig


# ── Side-by-side category comparison ─────────────────────────────────────────

def _compare_categories_chart(
    df: pd.DataFrame, kpis: dict, cat1: str, cat2: str
) -> go.Figure:
    """
    2×2 subplot comparing two categories across four key metrics:
    App Count, Avg Rating, Total Reviews, % Free.
    Each subplot has its own y-axis scale so no metric is squashed.
    """
    by_cat = kpis["by_category"]
    rows: dict = {}
    for cat in [cat1, cat2]:
        row = by_cat[by_cat["category"] == cat]
        if not row.empty:
            rows[cat] = row.iloc[0]

    if len(rows) < 2:
        return _top_categories_chart(df, kpis)

    r1, r2 = rows[cat1], rows[cat2]
    labels = [cat1[:25], cat2[:25]]
    colors = [_BLUE, _GREEN]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("App Count", "Avg Rating (/5)", "Total Reviews", "% Free Apps"),
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    def _bar(vals: list, texts: list) -> go.Bar:
        return go.Bar(
            x=labels, y=vals,
            marker_color=colors,
            text=texts, textposition="outside",
            showlegend=False,
        )

    fig.add_trace(_bar(
        [int(r1["app_count"]),   int(r2["app_count"])],
        [f"{r1['app_count']:,}", f"{r2['app_count']:,}"],
    ), row=1, col=1)

    fig.add_trace(_bar(
        [float(r1["avg_rating"]),     float(r2["avg_rating"])],
        [f"{r1['avg_rating']:.2f}",   f"{r2['avg_rating']:.2f}"],
    ), row=1, col=2)

    fig.add_trace(_bar(
        [int(r1["total_reviews"]),   int(r2["total_reviews"])],
        [f"{r1['total_reviews']:,}", f"{r2['total_reviews']:,}"],
    ), row=2, col=1)

    fig.add_trace(_bar(
        [float(r1["pct_free"]),      float(r2["pct_free"])],
        [f"{r1['pct_free']:.1f}%",   f"{r2['pct_free']:.1f}%"],
    ), row=2, col=2)

    # Pin sensible y-axis ranges
    fig.update_yaxes(range=[0, 5.5],  row=1, col=2)   # rating
    fig.update_yaxes(range=[0, 105],  row=2, col=2)   # % free

    # Apply axis style to all subplots
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(**_axis_style(), row=r, col=c)
            fig.update_yaxes(**_axis_style(), row=r, col=c)

    fig.update_layout(
        title=f"Category Comparison: {cat1}  vs  {cat2}",
        height=480,
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(family="Inter, sans-serif", color="#1a2744", size=12),
        margin=dict(l=50, r=30, t=80, b=50),
        showlegend=False,
    )
    return fig


# ── Overview (default landing) chart ─────────────────────────────────────────

def overview_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """Default chart shown before the user asks anything."""
    return _top_categories_chart(df, kpis)


# ── Keyword-routing auto-chart ────────────────────────────────────────────────

def auto_chart(
    question: str,
    df: pd.DataFrame,
    kpis: dict,
    entities: dict | None = None,
) -> go.Figure:
    """
    Map a natural-language question to the most relevant chart.

    entities: pre-computed result from entity_extractor.extract_entities().
    If provided, uses it directly; otherwise falls back to keyword routing.
    """
    q = question.lower()
    ents = entities if entities is not None else _extract_entities(question, kpis)

    # ── 1a. Two categories detected → side-by-side comparison chart ─────────
    if ents.get("category") and ents.get("category2"):
        return _compare_categories_chart(df, kpis, ents["category"], ents["category2"])

    # ── 1b. Single category detected → category deep-dive chart ─────────────
    if ents.get("category"):
        return _category_detail_chart(df, kpis, ents["category"])

    # ── 2. Opportunity / saturation questions ────────────────────────────────
    if any(w in q for w in ["opportunit", "underserved", "where should", "gap",
                             "saturated", "market map", "which category to",
                             "where to build", "where to enter"]):
        return _opportunities_chart(df, kpis)

    # ── 3. Standard keyword routing (unchanged) ──────────────────────────────
    if any(w in q for w in ["trend", "time", "month", "year", "history",
                             "grow", "over time", "timeline"]):
        return _review_trend_chart(df, kpis)

    if any(w in q for w in ["developer", "partner", "vendor", "publisher",
                             "maker", "built by"]):
        return _top_developers_chart(df, kpis)

    if any(w in q for w in ["scatter", "correlation", "compare apps",
                             "vs review", "versus"]):
        return _apps_rating_scatter(df, kpis)

    if any(w in q for w in ["rating", "score", "stars", "rated", "distribution"]):
        if any(w in q for w in ["category", "categories", "type", "segment"]):
            return _category_rating_chart(df, kpis)
        return _rating_distribution_chart(df, kpis)

    if any(w in q for w in ["price", "pricing", "free", "paid", "freemium",
                             "plan", "cost", "subscription", "monetiz"]):
        if any(w in q for w in ["category", "categories", "segment"]):
            return _free_vs_paid_by_category(df, kpis)
        return _pricing_breakdown_chart(df, kpis)

    if any(w in q for w in ["review", "feedback", "comment",
                             "user opinion", "sentiment"]):
        return _review_trend_chart(df, kpis)

    if any(w in q for w in ["category", "categories", "segment", "top",
                             "popular", "market", "niche"]):
        return _top_categories_chart(df, kpis)

    # default
    return _top_categories_chart(df, kpis)


# ── Sidebar metric tree (sunburst) ────────────────────────────────────────────

def metric_tree_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """
    Sunburst: root = total apps, level-1 = top categories,
    level-2 = Free / Paid split.  Color intensity = avg rating.
    """
    by_cat = kpis["by_category"].head(10)

    ids, labels, parents, values, colors = [], [], [], [], []

    ids.append("root")
    labels.append(f"All Apps")
    parents.append("")
    values.append(kpis["total_apps"])
    colors.append(kpis["avg_rating"])

    for _, row in by_cat.iterrows():
        cat = str(row["category"])
        cat_id = f"cat_{cat}"
        ids.append(cat_id)
        labels.append(f"{cat[:15]}")
        parents.append("root")
        values.append(int(row["app_count"]))
        colors.append(float(row["avg_rating"]))

        free_n = max(1, round(row["app_count"] * row["pct_free"] / 100))
        paid_n = max(0, int(row["app_count"]) - free_n)

        ids.append(f"{cat_id}_free")
        labels.append("Free")
        parents.append(cat_id)
        values.append(free_n)
        colors.append(min(5.0, float(row["avg_rating"]) + 0.05))

        if paid_n:
            ids.append(f"{cat_id}_paid")
            labels.append("Paid")
            parents.append(cat_id)
            values.append(paid_n)
            colors.append(max(0.0, float(row["avg_rating"]) - 0.05))

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            colorscale=[[0, "#dc2626"], [0.5, "#d97706"], [1, "#16a34a"]],
            cmin=0,
            cmax=5,
            showscale=False,
        ),
        branchvalues="total",
        textfont=dict(size=9),
        hovertemplate="<b>%{label}</b><br>Apps: %{value:,}<extra></extra>",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        height=260,
        font=dict(size=9),
    )
    return fig
