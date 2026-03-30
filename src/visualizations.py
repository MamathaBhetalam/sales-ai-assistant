"""
Visualizations: Auto-generated Plotly charts for the E-Commerce dataset.

auto_chart()        — keyword-routing: picks the best chart for a given question
overview_chart()    — default landing view (revenue by channel bar chart)
metric_tree_chart() — sidebar node-edge KPI hierarchy (click to expand)
"""
import re
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

def _revenue_by_channel(df: pd.DataFrame, kpis: dict) -> go.Figure:
    by_ch = kpis["by_channel"].sort_values("revenue", ascending=True)
    total = kpis["total_revenue"] or 1
    hover = [
        f"<b>{ch}</b><br>Revenue: ${rev:,.2f} ({rev/total*100:.1f}%)<br>"
        f"Orders: {int(ord_)}<br>Conv: {conv:.1f}%"
        for ch, rev, ord_, conv in zip(
            by_ch["TrafficSource"], by_ch["revenue"],
            by_ch["orders"], by_ch["completion_rate"]
        )
    ]
    fig = go.Figure(go.Bar(
        x=by_ch["revenue"],
        y=by_ch["TrafficSource"],
        orientation="h",
        marker_color=_BLUE,
        text=by_ch["revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Revenue by Traffic Source",
        xaxis=dict(title="Revenue (USD)", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=380,
        **_base_layout(),
    )
    return fig


def _revenue_by_country(df: pd.DataFrame, kpis: dict, n: int = 12) -> go.Figure:
    by_co = kpis["by_country"].head(n).sort_values("revenue", ascending=True)
    total = kpis["total_revenue"] or 1
    hover = [
        f"<b>{co}</b><br>Revenue: ${rev:,.2f} ({rev/total*100:.1f}%)<br>Orders: {int(ord_)}"
        for co, rev, ord_ in zip(by_co["Country"], by_co["revenue"], by_co["orders"])
    ]
    fig = go.Figure(go.Bar(
        x=by_co["revenue"],
        y=by_co["Country"],
        orientation="h",
        marker_color=_PURPLE,
        text=by_co["revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Revenue by Country (Top {n})",
        xaxis=dict(title="Revenue (USD)", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=420,
        **_base_layout(),
    )
    return fig


def _conversion_rate_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    by_ch = kpis["by_channel"].sort_values("completion_rate", ascending=True)
    avg_conv = kpis["completion_rate"]
    colors = [
        _GREEN if r >= avg_conv + 3 else
        _RED   if r < avg_conv - 3 else
        _AMBER
        for r in by_ch["completion_rate"]
    ]
    fig = go.Figure(go.Bar(
        x=by_ch["completion_rate"],
        y=by_ch["TrafficSource"],
        orientation="h",
        marker_color=colors,
        text=by_ch["completion_rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Conversion: %{x:.1f}%<extra></extra>",
    ))
    fig.add_vline(x=avg_conv, line=dict(color=_SLATE, dash="dot", width=1.5))
    fig.add_annotation(
        x=avg_conv, y=1.02, yref="paper",
        text=f"Avg {avg_conv:.1f}%",
        showarrow=False, font=dict(color=_SLATE, size=10),
    )
    fig.update_layout(
        title="Order Completion Rate by Channel  (green = above avg)",
        xaxis=dict(title="Completion Rate (%)", range=[0, 110], **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=350,
        **_base_layout(),
    )
    return fig


def _aov_by_channel(df: pd.DataFrame, kpis: dict) -> go.Figure:
    by_ch = kpis["by_channel"].copy()
    mkt_aov = kpis["avg_order_value"]
    by_ch_sorted = by_ch.sort_values("revenue", ascending=False)
    # Compute AOV per channel from raw data
    completed = df[df["is_completed"]]
    ch_aov = (
        completed.groupby("TrafficSource")["Total"].mean()
        .reset_index()
        .rename(columns={"Total": "aov"})
    )
    merged = by_ch_sorted.merge(ch_aov, on="TrafficSource", how="left")
    merged = merged.sort_values("aov", ascending=True)
    colors = [
        _GREEN if v >= mkt_aov + 2 else
        _RED   if v < mkt_aov - 2 else
        _AMBER
        for v in merged["aov"]
    ]
    fig = go.Figure(go.Bar(
        x=merged["aov"],
        y=merged["TrafficSource"],
        orientation="h",
        marker_color=colors,
        text=merged["aov"].apply(lambda v: f"${v:.2f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>AOV: $%{x:.2f}<extra></extra>",
    ))
    fig.add_vline(x=mkt_aov, line=dict(color=_SLATE, dash="dot", width=1.5))
    fig.update_layout(
        title=f"Average Order Value by Channel  (market avg ${mkt_aov:.2f})",
        xaxis=dict(title="AOV (USD)", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=350,
        **_base_layout(),
    )
    return fig


def _rating_distribution_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    dist = kpis["rating_dist"]
    colors = [_RED, _AMBER, _AMBER, _GREEN, _BLUE, _BLUE]
    fig = go.Figure(go.Bar(
        x=dist["band"].astype(str),
        y=dist["count"],
        marker_color=colors[:len(dist)],
        text=dist["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertemplate="Rating %{x}<br>Orders: %{y:,}<extra></extra>",
    ))
    fig.update_layout(
        title="Product Rating Distribution",
        xaxis=dict(title="Rating Band", **_axis_style()),
        yaxis=dict(title="Number of Orders", **_axis_style()),
        height=360,
        **_base_layout(),
    )
    return fig


def _ratings_by_channel(df: pd.DataFrame, kpis: dict) -> go.Figure:
    by_ch = kpis["by_channel"].dropna(subset=["avg_product_rating"]).copy()
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Product Rating",
        x=by_ch["TrafficSource"],
        y=by_ch["avg_product_rating"],
        marker_color=_BLUE,
        text=by_ch["avg_product_rating"].apply(lambda v: f"{v:.2f}"),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Delivery Rating",
        x=by_ch["TrafficSource"],
        y=by_ch["avg_delivery_rating"],
        marker_color=_GREEN,
        text=by_ch["avg_delivery_rating"].apply(lambda v: f"{v:.2f}"),
        textposition="outside",
    ))
    fig.update_layout(
        barmode="group",
        title="Product vs Delivery Rating by Channel",
        xaxis=dict(title="", **_axis_style()),
        yaxis=dict(title="Avg Rating (/5)", range=[0, 5.5], **_axis_style()),
        height=380,
        **_base_layout(),
    )
    return fig


def _revenue_trend_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    trend = kpis.get("revenue_trend", pd.DataFrame())
    if trend.empty or len(trend) < 2:
        return _revenue_by_channel(df, kpis)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=trend["year_month"],
        y=trend["revenue"],
        name="Revenue",
        mode="lines+markers",
        line=dict(color=_BLUE, width=2),
        fill="tozeroy",
        fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="Period: %{x}<br>Revenue: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=trend["year_month"],
        y=trend["order_count"],
        name="Orders",
        mode="lines",
        line=dict(color=_GREEN, width=2, dash="dot"),
        yaxis="y2",
        hovertemplate="Period: %{x}<br>Orders: %{y:,}<extra></extra>",
    ))
    _layout = _base_layout()
    _layout["legend"] = dict(x=0.01, y=0.99, bgcolor="rgba(0,0,0,0)", bordercolor=_GRID)
    fig.update_layout(
        title="Monthly Revenue & Order Volume",
        xaxis=dict(title="Month", **_axis_style()),
        yaxis=dict(title="Revenue (USD)", **_axis_style()),
        yaxis2=dict(title="Order Count", overlaying="y", side="right", **_axis_style()),
        height=380,
        **_layout,
    )
    return fig


def _device_breakdown_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    by_dev = kpis["by_device"]
    fig = go.Figure(go.Pie(
        labels=by_dev["DeviceCategory"],
        values=by_dev["revenue"],
        hole=0.45,
        marker=dict(colors=[_BLUE, _GREEN, _AMBER]),
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.2f}<br>Share: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        title="Revenue by Device Category",
        height=360,
        **_base_layout(),
    )
    return fig


def _gender_breakdown_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    by_g = kpis["by_gender"]
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Orders by Gender", "Revenue by Gender"))
    colors = [_BLUE, _PURPLE]
    fig.add_trace(go.Bar(
        x=by_g["Gender"], y=by_g["orders"],
        marker_color=colors,
        text=by_g["orders"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=by_g["Gender"], y=by_g["revenue"],
        marker_color=colors,
        text=by_g["revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        showlegend=False,
    ), row=1, col=2)
    for c in (1, 2):
        fig.update_xaxes(**_axis_style(), row=1, col=c)
        fig.update_yaxes(**_axis_style(), row=1, col=c)
    fig.update_layout(
        title="Orders & Revenue by Gender",
        height=360,
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="Inter, sans-serif", color="#1a2744", size=12),
        margin=dict(l=50, r=30, t=80, b=50),
    )
    return fig


def _order_status_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    sb = kpis["status_breakdown"]
    total = kpis["total_orders"] or 1
    colors_map = {"Completed": _GREEN, "In Process": _AMBER,
                  "Cancelled": _RED, "Returned": _PURPLE}
    bar_colors = [colors_map.get(s, _BLUE) for s in sb["status"]]
    hover = [
        f"<b>{s}</b><br>Count: {c:,}<br>Share: {c/total*100:.1f}%"
        for s, c in zip(sb["status"], sb["count"])
    ]
    fig = go.Figure(go.Bar(
        x=sb["status"],
        y=sb["count"],
        marker_color=bar_colors,
        text=sb["count"].apply(lambda v: f"{v:,}"),
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))
    fig.update_layout(
        title="Order Status Breakdown",
        xaxis=dict(title="", **_axis_style()),
        yaxis=dict(title="Number of Orders", **_axis_style()),
        height=360,
        **_base_layout(),
    )
    return fig


def _cancellation_analysis_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    """Cancellation rate and count by channel — used for 'why cancelled' questions."""
    cancelled = df[df["OrderStatus"].str.lower().str.contains("cancel", na=False)]
    total_by_ch = df.groupby("TrafficSource")["InvoiceNumber"].count().reset_index()
    total_by_ch.columns = ["TrafficSource", "total"]
    cancel_by_ch = cancelled.groupby("TrafficSource")["InvoiceNumber"].count().reset_index()
    cancel_by_ch.columns = ["TrafficSource", "cancelled"]
    merged = total_by_ch.merge(cancel_by_ch, on="TrafficSource", how="left").fillna(0)
    merged["cancel_rate"] = (merged["cancelled"] / merged["total"] * 100).round(1)
    merged = merged.sort_values("cancel_rate", ascending=True)

    avg_rate = float(cancelled.__len__() / len(df) * 100) if len(df) else 0
    colors = [
        _RED   if r > avg_rate + 2 else
        _AMBER if r > avg_rate - 2 else
        _GREEN
        for r in merged["cancel_rate"]
    ]

    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Cancellation Rate by Channel (%)", "Cancelled Orders Count"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        x=merged["cancel_rate"], y=merged["TrafficSource"],
        orientation="h", marker_color=colors,
        text=merged["cancel_rate"].apply(lambda v: f"{v:.1f}%"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Cancel Rate: %{x:.1f}%<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=merged["cancelled"], y=merged["TrafficSource"],
        orientation="h", marker_color=_SLATE,
        text=merged["cancelled"].apply(lambda v: f"{int(v)}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Cancelled: %{x:,}<extra></extra>",
        showlegend=False,
    ), row=1, col=2)

    fig.add_vline(x=avg_rate, row=1, col=1,
                  line=dict(color=_RED, dash="dot", width=1.5))
    _layout = _base_layout()
    fig.update_layout(
        title=f"Cancellation Analysis by Channel  (avg rate: {avg_rate:.1f}%)",
        height=380,
        **_layout,
    )
    fig.update_xaxes(**_axis_style())
    fig.update_yaxes(**_axis_style())
    return fig


def _channel_deep_dive_chart(df: pd.DataFrame, kpis: dict, channel: str) -> go.Figure:
    ch_df = df[df["TrafficSource"].str.contains(channel, case=False, na=False)]
    if ch_df.empty:
        return _revenue_by_channel(df, kpis)

    by_co = (
        ch_df.groupby("Country")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=False)
        .head(10)
        .sort_values("revenue", ascending=True)
        .reset_index()
    )
    hover = [
        f"<b>{co}</b><br>Revenue: ${rev:,.2f}<br>Orders: {int(ord_)}"
        for co, rev, ord_ in zip(by_co["Country"], by_co["revenue"], by_co["orders"])
    ]
    fig = go.Figure(go.Bar(
        x=by_co["revenue"],
        y=by_co["Country"],
        orientation="h",
        marker_color=_BLUE,
        text=by_co["revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertext=hover,
        hoverinfo="text",
    ))
    ch_rev  = float(ch_df["revenue"].sum())
    ch_ord  = len(ch_df)
    comp    = ch_df[ch_df["is_completed"]]
    ch_rate = len(comp) / ch_ord * 100 if ch_ord else 0
    fig.update_layout(
        title=f"{channel}  —  ${ch_rev:,.0f} revenue · {ch_ord:,} orders · {ch_rate:.1f}% conv",
        xaxis=dict(title="Revenue (USD)", **_axis_style()),
        yaxis=dict(title="", **_axis_style()),
        height=420,
        **_base_layout(),
    )
    return fig


def _country_deep_dive_chart(df: pd.DataFrame, kpis: dict, country: str) -> go.Figure:
    co_df = df[df["Country"].str.contains(country, case=False, na=False)]
    if co_df.empty:
        return _revenue_by_country(df, kpis)

    ch_mix = (
        co_df.groupby("TrafficSource")
        .agg(orders=("InvoiceNumber", "count"), revenue=("revenue", "sum"))
        .sort_values("revenue", ascending=True)
        .reset_index()
    )
    fig = go.Figure(go.Bar(
        x=ch_mix["revenue"],
        y=ch_mix["TrafficSource"],
        orientation="h",
        marker_color=_PURPLE,
        text=ch_mix["revenue"].apply(lambda v: f"${v:,.0f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Revenue: $%{x:,.2f}<extra></extra>",
    ))
    co_rev = float(co_df["revenue"].sum())
    co_ord = len(co_df)
    fig.update_layout(
        title=f"{country}  —  ${co_rev:,.0f} revenue · {co_ord:,} orders",
        xaxis=dict(title="Revenue (USD)", **_axis_style()),
        yaxis=dict(title="Channel", **_axis_style()),
        height=360,
        **_base_layout(),
    )
    return fig


def _compare_channels_chart(df: pd.DataFrame, kpis: dict, ch1: str, ch2: str) -> go.Figure:
    by_ch = kpis["by_channel"]
    rows = {}
    for ch in [ch1, ch2]:
        r = by_ch[by_ch["TrafficSource"].str.lower() == ch.lower()]
        if not r.empty:
            rows[ch] = r.iloc[0]

    if len(rows) < 2:
        return _revenue_by_channel(df, kpis)

    r1, r2 = rows[ch1], rows[ch2]
    labels = [ch1[:20], ch2[:20]]
    colors = [_BLUE, _GREEN]

    # Compute per-channel AOV
    completed = df[df["is_completed"]]
    aov = completed.groupby("TrafficSource")["Total"].mean()
    aov1 = float(aov.get(ch1, 0))
    aov2 = float(aov.get(ch2, 0))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Revenue (USD)", "Conversion Rate (%)", "Avg Order Value ($)", "Avg Product Rating"),
        vertical_spacing=0.18, horizontal_spacing=0.12,
    )

    def _bar(vals, texts):
        return go.Bar(x=labels, y=vals, marker_color=colors,
                      text=texts, textposition="outside", showlegend=False)

    fig.add_trace(_bar([r1["revenue"], r2["revenue"]],
                       [f"${r1['revenue']:,.0f}", f"${r2['revenue']:,.0f}"]), row=1, col=1)
    fig.add_trace(_bar([r1["completion_rate"], r2["completion_rate"]],
                       [f"{r1['completion_rate']:.1f}%", f"{r2['completion_rate']:.1f}%"]), row=1, col=2)
    fig.add_trace(_bar([aov1, aov2],
                       [f"${aov1:.2f}", f"${aov2:.2f}"]), row=2, col=1)
    fig.add_trace(_bar([r1["avg_product_rating"], r2["avg_product_rating"]],
                       [f"{r1['avg_product_rating']:.2f}", f"{r2['avg_product_rating']:.2f}"]), row=2, col=2)

    fig.update_yaxes(range=[0, 5.5], row=2, col=2)
    fig.update_yaxes(range=[0, 110], row=1, col=2)
    for r in (1, 2):
        for c in (1, 2):
            fig.update_xaxes(**_axis_style(), row=r, col=c)
            fig.update_yaxes(**_axis_style(), row=r, col=c)

    fig.update_layout(
        title=f"Channel Comparison: {ch1}  vs  {ch2}",
        height=480,
        paper_bgcolor=_BG, plot_bgcolor=_BG,
        font=dict(family="Inter, sans-serif", color="#1a2744", size=12),
        margin=dict(l=50, r=30, t=80, b=50),
        showlegend=False,
    )
    return fig


# ── Default overview chart ─────────────────────────────────────────────────────

def overview_chart(df: pd.DataFrame, kpis: dict) -> go.Figure:
    return _revenue_by_channel(df, kpis)


# ── Keyword-routing auto-chart ────────────────────────────────────────────────

_THEORY_PREFIXES = (
    "what is ", "what are the definition", "how does ", "how do ",
    "explain ", "why is ", "why are ", "why does ", "can you explain",
    "what does it mean", "tell me what", "what does",
    "hello", "hi ", "hey ", "thanks", "thank you",
)


def auto_chart(
    question: str,
    df: pd.DataFrame,
    kpis: dict,
    entities: dict | None = None,
) -> "go.Figure | None":
    q    = question.lower().strip()
    ents = entities if entities is not None else _extract_entities(question, kpis)

    # ── 0. Theory / greeting ─────────────────────────────────────────────────
    if any(q.startswith(p) for p in _THEORY_PREFIXES):
        return None

    # ── 1. Two channels → comparison ─────────────────────────────────────────
    if ents.get("channel") and ents.get("channel2"):
        return _compare_channels_chart(df, kpis, ents["channel"], ents["channel2"])

    # ── 2. Single channel + intent routing ───────────────────────────────────
    if ents.get("channel"):
        ch = ents["channel"]
        if any(w in q for w in ["conversion", "completion", "conv rate"]):
            return _conversion_rate_chart(df, kpis)
        if any(w in q for w in ["aov", "order value", "average order"]):
            return _aov_by_channel(df, kpis)
        if any(w in q for w in ["rating", "quality", "satisfaction"]):
            return _ratings_by_channel(df, kpis)
        if any(w in q for w in ["trend", "over time", "monthly", "month"]):
            return _revenue_trend_chart(df, kpis)
        return _channel_deep_dive_chart(df, kpis, ch)

    # ── 3. Country ────────────────────────────────────────────────────────────
    if ents.get("country"):
        return _country_deep_dive_chart(df, kpis, ents["country"])

    # ── 4. Keyword routing ────────────────────────────────────────────────────

    if any(w in q for w in ["trend", "over time", "monthly", "month", "grow", "decline", "yoy"]):
        return _revenue_trend_chart(df, kpis)

    if any(w in q for w in ["conversion", "completion rate", "completion", "conv rate", "order status"]):
        if any(w in q for w in ["channel", "source", "traffic"]):
            return _conversion_rate_chart(df, kpis)
        return _order_status_chart(df, kpis)

    # Cancellation analysis: questions about WHY/WHERE orders are cancelled
    # → show cancellation rate by channel (more actionable than simple status count)
    if any(w in q for w in ["cancel", "cancelled", "cancellation", "cancellations"]):
        if any(w in q for w in ["why", "reason", "cause", "analysis", "analyse", "analyze",
                                 "driver", "factor", "pattern", "channel", "source"]):
            return _cancellation_analysis_chart(df, kpis)
        return _order_status_chart(df, kpis)

    if any(w in q for w in ["status", "in process", "breakdown"]):
        return _order_status_chart(df, kpis)

    if any(w in q for w in ["aov", "order value", "average order value"]):
        return _aov_by_channel(df, kpis)

    if any(w in q for w in ["rating", "quality", "satisfaction", "delivery rating", "product rating"]):
        if any(w in q for w in ["channel", "source", "traffic"]):
            return _ratings_by_channel(df, kpis)
        return _rating_distribution_chart(df, kpis)

    if any(w in q for w in ["channel", "traffic source", "source", "acquisition"]):
        if any(w in q for w in ["revenue", "sales", "money"]):
            return _revenue_by_channel(df, kpis)
        if any(w in q for w in ["conversion", "completion"]):
            return _conversion_rate_chart(df, kpis)
        return _revenue_by_channel(df, kpis)

    if any(w in q for w in ["country", "countries", "geography", "region", "market"]):
        return _revenue_by_country(df, kpis)

    if any(w in q for w in ["device", "mobile", "computer", "desktop", "laptop"]):
        return _device_breakdown_chart(df, kpis)

    if any(w in q for w in ["gender", "male", "female", "men", "women"]):
        return _gender_breakdown_chart(df, kpis)

    if any(w in q for w in ["revenue", "sales", "income", "earning"]):
        if any(w in q for w in ["country", "countries", "geography"]):
            return _revenue_by_country(df, kpis)
        if any(w in q for w in ["channel", "source"]):
            return _revenue_by_channel(df, kpis)
        return _revenue_trend_chart(df, kpis)

    if any(w in q for w in ["overview", "summary", "overall", "health", "kpi"]):
        return _revenue_by_channel(df, kpis)

    return None


# ── Sidebar metric tree ────────────────────────────────────────────────────────

def metric_tree_chart(df: pd.DataFrame, kpis: dict, expanded: str | None = None) -> go.Figure:
    """
    Interactive tree: root → 4 pillar nodes.
    Pass expanded='rev'|'ord'|'qual'|'geo' to show leaf metrics.
    Pillar customdata IDs are used by app.py to detect clicks via on_select.
    """
    yoy     = kpis.get("yoy_stats", {})
    rev_yoy = yoy.get("rev_yoy_pct")
    rev_yoy_str = f"{rev_yoy:+.1f}%" if rev_yoy is not None else "N/A"
    pct_completed = kpis["completion_rate"]

    def _color(score: float) -> str:
        if score >= 4.0:   return "#16a34a"
        if score >= 3.0:   return "#d97706"
        return "#dc2626"

    ROOT_X, ROOT_Y = 0.50, 0.88
    PILLAR_Y       = 0.54
    LEAF_Y         = 0.10

    pillars = [
        ("rev",  "Revenue\nGrowth",   "Revenue Growth",   0.125,
         4.0 if (rev_yoy or 0) >= 0 else 2.5),
        ("ord",  "Order\nQuality",    "Order Quality",    0.375,
         4.0 if pct_completed >= 70 else (3.0 if pct_completed >= 50 else 2.0)),
        ("qual", "Customer\nQuality", "Customer Quality", 0.625,
         kpis["avg_product_rating"]),
        ("geo",  "Geo\nReach",        "Geographic Reach", 0.875, 3.8),
    ]

    # Build channel leaves: top 3 channels, clickable, customdata = "ch:<name>"
    _by_ch = kpis.get("by_channel", pd.DataFrame())
    _rev_xpos = [0.03, 0.125, 0.22]
    _rev_leaves: list[tuple] = []
    for _i, (_, _row) in enumerate(_by_ch.head(3).iterrows()):
        _ch = _row["TrafficSource"]
        _ltv = _row.get("avg_ltv", None)
        _ltv_str = f" | LTV:${_ltv:.0f}" if _ltv and pd.notna(_ltv) else ""
        _rev_leaves.append((
            f"{_ch[:9]}\n${_row['revenue']/1000:.0f}k",
            f"{_ch} | Rev:${_row['revenue']:,.0f} | Conv:{_row['completion_rate']:.1f}%{_ltv_str}",
            _rev_xpos[_i],
            4.0 if _row["completion_rate"] >= 60 else 3.0,
            f"ch:{_ch}",
        ))

    # Build country leaves: top 3 countries, clickable, customdata = "co:<name>"
    _by_co = kpis.get("by_country", pd.DataFrame())
    _geo_xpos = [0.78, 0.875, 0.97]
    _geo_leaves: list[tuple] = []
    for _i, (_, _row) in enumerate(_by_co.head(3).iterrows()):
        _co = _row["Country"]
        _geo_leaves.append((
            f"{_co[:10]}\n${_row['revenue']/1000:.0f}k",
            f"{_co} | Rev:${_row['revenue']:,.0f} | Orders:{int(_row['orders']):,}",
            _geo_xpos[_i],
            4.0,
            f"co:{_co}",
        ))

    _freq = kpis.get("avg_purchase_freq", 1.0)
    _ret  = kpis.get("retention_rate", 0)
    leaf_data: dict[str, list[tuple]] = {
        "rev": _rev_leaves,
        "ord": [
            (f"{kpis['completion_rate']:.0f}%\nconv",
             f"Completion Rate: {kpis['completion_rate']:.1f}%", 0.29,
             4.0 if pct_completed >= 70 else 2.5, None),
            (f"{kpis['total_orders']:,}\norders",
             f"Total Orders: {kpis['total_orders']:,}", 0.375, 4.0, None),
            (f"{_freq:.2f}x\nfreq",
             f"Avg Purchase Frequency: {_freq:.2f} orders/customer", 0.46,
             4.0 if _freq >= 2.0 else (3.0 if _freq >= 1.5 else 2.5), None),
        ],
        "qual": [
            (f"{kpis['avg_product_rating']:.2f}★\nproduct",
             f"Avg Product Rating: {kpis['avg_product_rating']:.2f}/5", 0.54,
             kpis["avg_product_rating"], None),
            (f"{kpis['avg_delivery_rating']:.2f}★\ndelivery",
             f"Avg Delivery Rating: {kpis['avg_delivery_rating']:.2f}/5", 0.625,
             kpis["avg_delivery_rating"], None),
            (f"{_ret:.0f}%\nrepeat",
             f"Repeat Purchase Rate: {_ret:.1f}%  ({kpis.get('repeat_customers', 0):,} repeat buyers)",
             0.71,
             4.0 if _ret >= 30 else (3.0 if _ret >= 15 else 2.0), None),
        ],
        "geo": _geo_leaves,
    }

    fig = go.Figure()

    # Root → pillar edges
    ex, ey = [], []
    for pid, _, _, px, _ in pillars:
        ex += [ROOT_X, px, None]
        ey += [ROOT_Y, PILLAR_Y, None]
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(color="#9ca3af", width=1.5),
                             hoverinfo="skip", showlegend=False))

    # Pillar → leaf edges
    if expanded and expanded in leaf_data:
        pillar_x = next(p[3] for p in pillars if p[0] == expanded)
        lex, ley = [], []
        for leaf in leaf_data[expanded]:
            lex += [pillar_x, leaf[2], None]
            ley += [PILLAR_Y, LEAF_Y, None]
        fig.add_trace(go.Scatter(x=lex, y=ley, mode="lines",
                                 line=dict(color="#6b7280", width=1.2, dash="dot"),
                                 hoverinfo="skip", showlegend=False))

        leaves = leaf_data[expanded]
        _has_click = any(leaf[4] is not None for leaf in leaves)
        fig.add_trace(go.Scatter(
            x=[leaf[2] for leaf in leaves], y=[LEAF_Y] * len(leaves),
            mode="markers+text",
            marker=dict(size=44, color=[_color(leaf[3]) for leaf in leaves],
                        line=dict(color=["#facc15" if leaf[4] else "white" for leaf in leaves],
                                  width=[2.5 if leaf[4] else 1.5 for leaf in leaves])),
            text=[leaf[0] for leaf in leaves],
            textfont=dict(size=7, color="white"),
            textposition="middle center",
            customdata=[leaf[4] for leaf in leaves],   # click ID: "ch:X", "co:X", or None
            hovertext=[leaf[1] for leaf in leaves],
            hovertemplate="<b>%{hovertext}</b>"
                          + (" — click to drill down" if _has_click else "")
                          + "<extra></extra>",
            showlegend=False,
        ))

    # Pillar nodes
    p_border_colors = ["#facc15" if p[0] == expanded else "white" for p in pillars]
    p_border_widths = [3 if p[0] == expanded else 2 for p in pillars]
    fig.add_trace(go.Scatter(
        x=[p[3] for p in pillars], y=[PILLAR_Y] * 4,
        mode="markers+text",
        marker=dict(size=58, color=[_color(p[4]) for p in pillars],
                    line=dict(color=p_border_colors, width=p_border_widths)),
        text=[p[1] for p in pillars],
        textfont=dict(size=8, color="white"),
        textposition="middle center",
        customdata=[p[0] for p in pillars],   # IDs: "rev","ord","qual","geo"
        hovertemplate="%{text}  — click to expand<extra></extra>",
        showlegend=False,
    ))

    # Root node
    fig.add_trace(go.Scatter(
        x=[ROOT_X], y=[ROOT_Y],
        mode="markers+text",
        marker=dict(size=68, color=["#1e40af"], line=dict(color="white", width=2)),
        text=["E-Commerce"],
        textfont=dict(size=9, color="white"),
        textposition="middle center",
        customdata=["root"],
        hovertemplate=f"Revenue: ${kpis['total_revenue']:,.0f}<extra></extra>",
        showlegend=False,
    ))

    hint = "Click root to collapse ↑" if expanded else "Click a pillar to expand ↓"
    fig.update_layout(
        margin=dict(l=2, r=2, t=20, b=2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        xaxis=dict(visible=False, range=[-0.06, 1.06]),
        yaxis=dict(visible=False, range=[0.0, 1.0]),
        hovermode="closest",
        font=dict(size=9),
        title=dict(text=hint, font=dict(size=9, color="#6b7280"), x=0.5, y=0.995, xanchor="center"),
    )
    return fig
