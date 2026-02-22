"""
Visualizations: Smart auto-selecting Plotly charts.
The app calls `auto_chart(question, df, kpis)` and gets back the right chart.
"""
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Color Palette ─────────────────────────────────────────────────────────────
COLORS = {
    "primary": "#2563eb",   # professional blue
    "accent":  "#f97316",   # warm orange
    "success": "#16a34a",   # clean green
    "warning": "#d97706",   # amber
    "bg":      "#f0f4fb",   # light blue-gray page background
    "card":    "#ffffff",   # white card surface
    "text":    "#1a2744",   # dark navy text
    "grid":    "#d1ddf0",   # soft blue-gray grid lines
}

REGION_COLORS = {"West": "#00b4d8", "East": "#2ecc71", "South": "#e94560", "Central": "#f5a623"}
CAT_COLORS = {"Technology": "#00b4d8", "Furniture": "#f5a623", "Office Supplies": "#2ecc71"}

LAYOUT_BASE = dict(
    paper_bgcolor=COLORS["card"],
    plot_bgcolor=COLORS["card"],
    font=dict(color=COLORS["text"], family="Inter, sans-serif", size=12),
    margin=dict(t=50, b=40, l=40, r=20),
    xaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
    yaxis=dict(gridcolor=COLORS["grid"], showgrid=True),
)


# ── Chart Type Detection ──────────────────────────────────────────────────────

def detect_chart_type(question: str) -> str:
    q = question.lower()
    if any(w in q for w in ["region", "geographic", "east", "west", "south", "central", "location", "where"]):
        return "regional"
    if any(w in q for w in ["trend", "time", "month", "quarter", "year", "growth", "forecast", "over time"]):
        return "timeseries"
    if any(w in q for w in ["category", "furniture", "technology", "office", "product", "segment"]):
        return "category"
    if any(w in q for w in ["discount", "price", "promotion"]):
        return "discount"
    if any(w in q for w in ["profit drop", "why", "root cause", "cause", "reason", "explain"]):
        return "waterfall"
    if any(w in q for w in ["margin", "profitability"]):
        return "margin"
    if any(w in q for w in ["compare", "vs", "versus", "difference"]):
        return "comparison"
    return "overview"


def auto_chart(question: str, df: pd.DataFrame, kpis: dict):
    """Return the best Plotly figure for the given question."""
    chart_type = detect_chart_type(question)
    dispatch = {
        "regional": regional_chart,
        "timeseries": timeseries_chart,
        "category": category_chart,
        "discount": discount_chart,
        "waterfall": waterfall_chart,
        "margin": margin_chart,
        "comparison": comparison_chart,
        "overview": overview_chart,
    }
    fn = dispatch.get(chart_type, overview_chart)
    return fn(df, kpis)


# ── Individual Chart Builders ─────────────────────────────────────────────────

def overview_chart(df: pd.DataFrame, kpis: dict):
    """4-panel executive dashboard."""
    ts = _monthly_ts(df)
    reg = kpis["regional"]
    cat = kpis["category"]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Revenue & Profit Trend", "Revenue by Region",
                        "Category Revenue Mix", "Quarterly Margin %"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "bar"}]],
        vertical_spacing=0.18,
        horizontal_spacing=0.12,
    )

    # 1. Revenue + Profit line
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["Revenue"], name="Revenue",
                             line=dict(color=COLORS["primary"], width=2),
                             fill="tozeroy", fillcolor="rgba(0,180,216,0.08)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["Profit"], name="Profit",
                             line=dict(color=COLORS["success"], width=2)), row=1, col=1)

    # 2. Regional bar
    fig.add_trace(go.Bar(x=reg["Region"], y=reg["Revenue"],
                         marker_color=[REGION_COLORS.get(r, COLORS["primary"]) for r in reg["Region"]],
                         name="Regional Revenue", showlegend=False), row=1, col=2)

    # 3. Category pie
    fig.add_trace(go.Pie(labels=cat["Category"], values=cat["Revenue"],
                         hole=0.55, name="",
                         marker_colors=[CAT_COLORS.get(c, "#888") for c in cat["Category"]],
                         textinfo="label+percent"), row=2, col=1)

    # 4. Quarterly margin bar
    qtr = kpis["quarterly"].tail(8)
    fig.add_trace(go.Bar(x=qtr["YQ"], y=qtr["Margin%"],
                         marker_color=[
                             COLORS["success"] if m >= 12 else COLORS["warning"] if m >= 8 else COLORS["accent"]
                             for m in qtr["Margin%"]
                         ],
                         name="Margin %", showlegend=False), row=2, col=2)

    fig.update_layout(
        title_text="Business Overview Dashboard",
        title_font_size=16,
        height=520,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        legend=dict(orientation="h", x=0, y=1.12, font_color=COLORS["text"]),
    )
    _apply_axis_style(fig)
    return fig


def regional_chart(df: pd.DataFrame, kpis: dict):
    """Side-by-side revenue + profit by region with margin annotation."""
    reg = kpis["regional"].sort_values("Revenue")

    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Revenue & Profit by Region", "Profit Margin % by Region"),
                        horizontal_spacing=0.14)

    fig.add_trace(go.Bar(y=reg["Region"], x=reg["Revenue"], name="Revenue",
                         orientation="h",
                         marker_color=COLORS["primary"]), row=1, col=1)
    fig.add_trace(go.Bar(y=reg["Region"], x=reg["Profit"], name="Profit",
                         orientation="h",
                         marker_color=COLORS["success"]), row=1, col=1)
    fig.add_trace(go.Bar(y=reg["Region"], x=reg["Margin%"],
                         orientation="h", name="Margin %",
                         marker_color=[
                             COLORS["success"] if m >= 12 else COLORS["warning"] if m >= 8 else COLORS["accent"]
                             for m in reg["Margin%"]
                         ]), row=1, col=2)

    fig.update_layout(
        barmode="group",
        title_text="Regional Performance Analysis",
        height=380,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
    )
    _apply_axis_style(fig)
    return fig


def timeseries_chart(df: pd.DataFrame, kpis: dict):
    """Monthly revenue + profit trend with YoY comparison bands."""
    ts = _monthly_ts(df)

    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=("Monthly Revenue & Profit", "Rolling Profit Margin %"),
                        vertical_spacing=0.16, row_heights=[0.65, 0.35])

    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["Revenue"],
                             name="Revenue", mode="lines",
                             line=dict(color=COLORS["primary"], width=2.5),
                             fill="tozeroy", fillcolor="rgba(0,180,216,0.07)"), row=1, col=1)
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["Profit"],
                             name="Profit", mode="lines",
                             line=dict(color=COLORS["success"], width=2)), row=1, col=1)

    # Rolling margin
    ts["RollingMargin"] = ts["Margin%"].rolling(3, min_periods=1).mean()
    fig.add_trace(go.Scatter(x=ts["Period"], y=ts["RollingMargin"],
                             name="3-Mo Avg Margin%", mode="lines",
                             line=dict(color=COLORS["warning"], width=2, dash="dot")), row=2, col=1)

    # Reference line at 10% margin
    fig.add_hline(y=10, line_dash="dash", line_color=COLORS["grid"],
                  annotation_text="10% target", row=2, col=1)

    fig.update_layout(
        title_text="Revenue & Profit Time Series",
        height=480,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        legend=dict(orientation="h", x=0, y=1.06, font_color=COLORS["text"]),
    )
    _apply_axis_style(fig)
    return fig


def category_chart(df: pd.DataFrame, kpis: dict):
    """Treemap + bar combo for category / sub-category breakdown."""
    sub = kpis["sub_category"].copy()
    sub["text"] = sub.apply(
        lambda r: f"${r['Revenue']:,.0f}<br>Margin: {r['Margin%']:.1f}%", axis=1
    )

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.55, 0.45],
        subplot_titles=("Revenue by Category & Sub-Category", "Margin % by Category"),
        specs=[[{"type": "treemap"}, {"type": "bar"}]],
        horizontal_spacing=0.06,
    )

    fig.add_trace(go.Treemap(
        labels=sub["Sub-Category"],
        parents=sub["Category"],
        values=sub["Revenue"],
        customdata=sub[["Margin%", "AvgDiscount%"]],
        hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<br>Margin: %{customdata[0]:.1f}%<br>Avg Discount: %{customdata[1]:.1f}%<extra></extra>",
        texttemplate="%{label}<br>%{percentRoot:.1%}",
        marker_colorscale=[[0, "#e94560"], [0.5, "#f5a623"], [1, "#2ecc71"]],
        marker_cmid=0,
        textfont_size=11,
    ), row=1, col=1)

    cat = kpis["category"].sort_values("Margin%")
    fig.add_trace(go.Bar(
        y=cat["Category"],
        x=cat["Margin%"],
        orientation="h",
        marker_color=[CAT_COLORS.get(c, "#888") for c in cat["Category"]],
        text=[f"{m:.1f}%" for m in cat["Margin%"]],
        textposition="outside",
        name="Margin %",
    ), row=1, col=2)

    fig.update_layout(
        title_text="Category & Sub-Category Analysis",
        height=400,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        showlegend=False,
    )
    _apply_axis_style(fig)
    return fig


def discount_chart(df: pd.DataFrame, kpis: dict):
    """Scatter: Discount % vs Profit Margin, colored by category."""
    sample = df.sample(min(1000, len(df)), random_state=42).copy()
    sample["Margin%"] = sample["Profit"] / sample["Sales"] * 100
    sample["Discount%"] = sample["Discount"] * 100

    fig = px.scatter(
        sample, x="Discount%", y="Margin%",
        color="Category",
        color_discrete_map=CAT_COLORS,
        size="Sales", size_max=14,
        hover_data=["Region", "Sub-Category"],
        opacity=0.65,
        trendline="ols",
        labels={"Discount%": "Discount %", "Margin%": "Profit Margin %"},
    )

    fig.add_hline(y=0, line_color=COLORS["accent"], line_dash="dash",
                  annotation_text="Break-even")

    fig.update_layout(
        title_text="Discount vs Profit Margin (Each Dot = 1 Order)",
        height=420,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
    )
    _apply_axis_style(fig)
    return fig


def waterfall_chart(df: pd.DataFrame, kpis: dict):
    """Revenue → Gross Profit waterfall showing discount drain."""
    rev = df["Sales"].sum()
    prf = df["Profit"].sum()
    disc_loss = df["Sales"].sum() * df["Discount"].mean()
    cost = rev - disc_loss - prf

    labels = ["Gross Revenue", "Discount Impact", "Operating Costs", "Net Profit"]
    values = [rev, -disc_loss, -cost, prf]
    measures = ["absolute", "relative", "relative", "total"]
    colors = [COLORS["primary"], COLORS["accent"], COLORS["warning"], COLORS["success"]]

    fig = go.Figure(go.Waterfall(
        name="P&L Bridge",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"${abs(v):,.0f}" for v in values],
        textposition="outside",
        connector=dict(line=dict(color=COLORS["grid"])),
        increasing=dict(marker_color=COLORS["success"]),
        decreasing=dict(marker_color=COLORS["accent"]),
        totals=dict(marker_color=COLORS["primary"]),
    ))

    fig.update_layout(
        title_text="P&L Waterfall — Revenue to Profit Bridge",
        height=400,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        showlegend=False,
    )
    _apply_axis_style(fig)
    return fig


def margin_chart(df: pd.DataFrame, kpis: dict):
    """Margin heatmap: Region × Category."""
    pivot = (
        df.groupby(["Region", "Category"])
        .apply(lambda g: g["Profit"].sum() / g["Sales"].sum() * 100)
        .reset_index(name="Margin%")
    )
    matrix = pivot.pivot(index="Region", columns="Category", values="Margin%")

    fig = go.Figure(go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=[[0, "#e94560"], [0.4, "#f5a623"], [1, "#2ecc71"]],
        text=[[f"{v:.1f}%" for v in row] for row in matrix.values],
        texttemplate="%{text}",
        hovertemplate="<b>%{y} × %{x}</b><br>Margin: %{z:.1f}%<extra></extra>",
        zmin=0, zmax=25,
    ))

    fig.update_layout(
        title_text="Profit Margin Heatmap: Region × Category",
        height=340,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
    )
    return fig


def comparison_chart(df: pd.DataFrame, kpis: dict):
    """YoY comparison: 2022 vs 2023 vs 2024 per quarter."""
    qtr = kpis["quarterly"]
    years = sorted(qtr["Year"].unique())[-3:]  # last 3 years

    fig = go.Figure()
    year_colors = {years[-3]: COLORS["grid"], years[-2]: COLORS["warning"], years[-1]: COLORS["primary"]}

    for yr in years:
        sub = qtr[qtr["Year"] == yr]
        fig.add_trace(go.Bar(
            x=sub["Quarter"], y=sub["Revenue"],
            name=str(yr),
            marker_color=year_colors.get(yr, COLORS["primary"]),
        ))

    fig.update_layout(
        barmode="group",
        title_text="Year-over-Year Revenue Comparison by Quarter",
        height=380,
        **{k: v for k, v in LAYOUT_BASE.items() if k not in ("xaxis", "yaxis")},
        legend=dict(orientation="h", x=0, y=1.08, font_color=COLORS["text"]),
    )
    _apply_axis_style(fig)
    return fig


# ── Metric Tree: Profit Driver Tree ──────────────────────────────────────────

def metric_tree_chart(df: pd.DataFrame, kpis: dict):
    """
    KPI Driver Tree — shows how data columns connect to business outcomes.

    4-level causal decomposition:
      L0  NET PROFIT         ← Profit column
      L1  REVENUE            ← Sales column
          MARGIN %           ← Profit ÷ Sales
      L2  VOLUME (Qty col)   PRICING (Sales÷Orders)   DISCOUNT (Discount col)
          CATEGORY (Cat col) DISC BAND (Discount col)
      L3  Tech | Furn | OfficeSup   (Category column breakdown)

    Node color = margin health:  Red(<5%) / Amber(5-12%) / Green(>12%)
    Hover each node for source column, formula, and full detail.
    """

    # ── Data computations ─────────────────────────────────────────────────────
    latest_yr  = kpis["latest_year"]
    prev_yr    = kpis["prev_year"]
    cat_df     = kpis["category"].copy()
    sub_df     = kpis["sub_category"].copy()

    qty        = df["Quantity"].sum()
    avg_ord    = df.groupby("Order ID")["Sales"].sum().mean()
    disc_drain = (df["Sales"] * df["Discount"]).sum()
    avg_disc   = df["Discount"].mean() * 100

    m_latest      = kpis["profit_latest"] / kpis["revenue_latest"] * 100 if kpis["revenue_latest"] else 0
    m_prev        = kpis["profit_prev"]   / kpis["revenue_prev"]   * 100 if kpis["revenue_prev"]   else 0
    margin_pp_yoy = m_latest - m_prev

    hi = df[df["Discount"] >  0.20]
    lo = df[df["Discount"] <= 0.20]
    hi_margin   = hi["Profit"].sum() / hi["Sales"].sum() * 100 if len(hi) else 0
    lo_margin   = lo["Profit"].sum() / lo["Sales"].sum() * 100 if len(lo) else 0
    hi_pct      = len(hi) / len(df) * 100

    cat_l   = df[df["Year"] == latest_yr].groupby("Category")["Sales"].sum()
    cat_p   = df[df["Year"] == prev_yr  ].groupby("Category")["Sales"].sum()
    cat_yoy = {
        c: (cat_l[c] - cat_p.get(c, cat_l[c])) / cat_p.get(c, cat_l[c]) * 100
        if cat_p.get(c, 0) > 0 else 0
        for c in cat_l.index
    }

    # ── Color palette ─────────────────────────────────────────────────────────
    CHART_BG    = "#f0f4fb"
    BLUE_FILL   = "#1e40af"   # structural blue (Revenue root)
    BLUE_BDR    = "#3b82f6"
    ROOT_FILL   = "#1e3a8a"   # darker blue for root
    ROOT_BDR    = "#60a5fa"
    PURPLE_FILL = "#5b21b6"   # dimension nodes (Category, Disc Band)
    PURPLE_BDR  = "#8b5cf6"
    RED_FILL    = "#991b1b"   # discount drain (bad actor)
    RED_BDR     = "#f87171"
    def mcolor(m):
        if m < 5:    return "#dc2626", "#fca5a5"
        elif m < 12: return "#d97706", "#fcd34d"
        return             "#16a34a", "#86efac"

    def trend(v, suffix="%"):
        return f"{'▲' if v >= 0 else '▼'} {abs(v):.1f}{suffix}"

    # ── Node positions (x: 0–100, y: 0–100) ───────────────────────────────────
    # L0
    ROOT_X, ROOT_Y = 50, 92
    # L1
    REV_X,  REV_Y  = 27, 73
    MAR_X,  MAR_Y  = 73, 73
    # L2 under Revenue
    QTY_X,  QTY_Y  =  9, 50
    ORD_X,  ORD_Y  = 27, 50
    DSC_X,  DSC_Y  = 45, 50
    # L2 under Margin
    CAT_X,  CAT_Y  = 63, 50
    BND_X,  BND_Y  = 84, 50
    # L3 under Category
    TCH_X,  TCH_Y  = 52, 26
    FRN_X,  FRN_Y  = 67, 26
    OSP_X,  OSP_Y  = 83, 26

    # Box half-widths / half-heights per level
    R_HW,  R_HH  = 30,  5.5
    L1_HW, L1_HH = 20,  5.5
    L2_HW, L2_HH = 11,  8.5
    L3_HW, L3_HH =  9,  8.5

    shapes, annots, hdata = [], [], []

    # ── Drawing helpers ───────────────────────────────────────────────────────
    def edge(x0, y0, x1, y1, color, w=1.5):
        cy = (y0 + y1) / 2
        shapes.append(dict(
            type="path",
            path=f"M {x0},{y0} C {x0},{cy} {x1},{cy} {x1},{y1}",
            line=dict(color=color, width=w),
            fillcolor="rgba(0,0,0,0)",
            layer="below",
        ))

    def box(cx, cy, hw, hh, fill, border):
        shapes.append(dict(
            type="rect",
            x0=cx - hw, y0=cy - hh, x1=cx + hw, y1=cy + hh,
            fillcolor=fill,
            line=dict(color=border, width=1.8),
            layer="above",
        ))

    def lbl(cx, cy, txt, sz=9):
        annots.append(dict(
            x=cx, y=cy, text=txt,
            showarrow=False,
            font=dict(size=sz, color="#ffffff", family="Inter, sans-serif"),
            xanchor="center", yanchor="middle",
        ))

    def elabel(x, y, txt, color):
        annots.append(dict(
            x=x, y=y, text=f"<i>{txt}</i>",
            showarrow=False,
            font=dict(size=6.5, color=color, family="Inter, sans-serif"),
            xanchor="center", yanchor="middle",
            bgcolor="rgba(240,244,251,0.85)",
        ))

    # ── Edges ─────────────────────────────────────────────────────────────────
    # L0 → L1
    edge(ROOT_X, ROOT_Y - R_HH,  REV_X, REV_Y + L1_HH, "#3b82f6", 2.2)
    edge(ROOT_X, ROOT_Y - R_HH,  MAR_X, MAR_Y + L1_HH, "#3b82f6", 2.2)
    elabel((ROOT_X + REV_X) / 2 - 4, (ROOT_Y + REV_Y) / 2, "Σ Sales col", "#2563eb")
    elabel((ROOT_X + MAR_X) / 2 + 4, (ROOT_Y + MAR_Y) / 2, "÷ Revenue", "#7c3aed")

    # L1 Revenue → L2
    edge(REV_X, REV_Y - L1_HH, QTY_X, QTY_Y + L2_HH, "rgba(37,99,235,0.5)", 1.3)
    edge(REV_X, REV_Y - L1_HH, ORD_X, ORD_Y + L2_HH, "rgba(37,99,235,0.5)", 1.3)
    edge(REV_X, REV_Y - L1_HH, DSC_X, DSC_Y + L2_HH, "rgba(220,38,38,0.55)", 1.5)
    elabel((REV_X + DSC_X) / 2 + 3, (REV_Y + DSC_Y) / 2 + 1, "reduces ↓", "#dc2626")

    # L1 Margin → L2
    edge(MAR_X, MAR_Y - L1_HH, CAT_X, CAT_Y + L2_HH, "rgba(91,33,182,0.5)", 1.3)
    edge(MAR_X, MAR_Y - L1_HH, BND_X, BND_Y + L2_HH, "rgba(220,38,38,0.5)", 1.3)
    elabel((MAR_X + BND_X) / 2 + 3, (MAR_Y + BND_Y) / 2, "erodes ↓", "#dc2626")

    # Category → L3
    edge(CAT_X, CAT_Y - L2_HH, TCH_X, TCH_Y + L3_HH, "rgba(91,33,182,0.4)", 1.2)
    edge(CAT_X, CAT_Y - L2_HH, FRN_X, FRN_Y + L3_HH, "rgba(91,33,182,0.4)", 1.2)
    edge(CAT_X, CAT_Y - L2_HH, OSP_X, OSP_Y + L3_HH, "rgba(91,33,182,0.4)", 1.2)

    # ── L0: Net Profit ────────────────────────────────────────────────────────
    box(ROOT_X, ROOT_Y, R_HW, R_HH, ROOT_FILL, ROOT_BDR)
    lbl(ROOT_X, ROOT_Y,
        f"<b>NET PROFIT</b>  ${kpis['total_profit']/1e3:.0f}K  ·  "
        f"{kpis['profit_margin']:.1f}% margin  ·  "
        f"{trend(kpis['yoy_profit_growth'])} YoY  "
        f"<span style='opacity:0.7'>← Profit col</span>",
        sz=8.5)
    hdata.append((ROOT_X, ROOT_Y,
        f"<b>NET PROFIT</b><br>"
        f"Source column: <b>Profit</b><br>"
        f"Formula: Σ(Profit) = Revenue × Margin%<br>"
        f"Value: ${kpis['total_profit']:,.0f}<br>"
        f"Margin: {kpis['profit_margin']:.1f}%<br>"
        f"YoY: {kpis['yoy_profit_growth']:+.1f}%"))

    # ── L1: Revenue ───────────────────────────────────────────────────────────
    box(REV_X, REV_Y, L1_HW, L1_HH, BLUE_FILL, BLUE_BDR)
    lbl(REV_X, REV_Y,
        f"<b>REVENUE</b>  ${kpis['total_revenue']/1e6:.2f}M<br>"
        f"{trend(kpis['yoy_revenue_growth'])} YoY  "
        f"<span style='opacity:0.75'>← Sales col</span>",
        sz=8.5)
    hdata.append((REV_X, REV_Y,
        f"<b>REVENUE</b><br>"
        f"Source column: <b>Sales</b><br>"
        f"Formula: Σ(Sales) = Quantity × Price × (1−Discount)<br>"
        f"Value: ${kpis['total_revenue']:,.0f}<br>"
        f"YoY: {kpis['yoy_revenue_growth']:+.1f}%"))

    # ── L1: Margin % ──────────────────────────────────────────────────────────
    mfill, mbdr = mcolor(kpis["profit_margin"])
    box(MAR_X, MAR_Y, L1_HW, L1_HH, mfill, mbdr)
    lbl(MAR_X, MAR_Y,
        f"<b>MARGIN %</b>  {kpis['profit_margin']:.1f}%<br>"
        f"{trend(margin_pp_yoy, 'pp')} YoY  "
        f"<span style='opacity:0.75'>← Profit÷Sales</span>",
        sz=8.5)
    hdata.append((MAR_X, MAR_Y,
        f"<b>PROFIT MARGIN %</b><br>"
        f"Source columns: <b>Profit, Sales</b><br>"
        f"Formula: Σ(Profit) ÷ Σ(Sales) × 100<br>"
        f"Value: {kpis['profit_margin']:.1f}%<br>"
        f"YoY change: {margin_pp_yoy:+.1f}pp<br>"
        f"Discount drain on revenue: ${disc_drain:,.0f}"))

    # ── L2: Volume (Quantity column) ──────────────────────────────────────────
    box(QTY_X, QTY_Y, L2_HW, L2_HH, BLUE_FILL, BLUE_BDR)
    lbl(QTY_X, QTY_Y,
        f"<b>VOLUME</b><br>"
        f"col: Quantity<br>"
        f"{qty:,} units<br>"
        f"{kpis['total_orders']:,} orders",
        sz=7.5)
    hdata.append((QTY_X, QTY_Y,
        f"<b>VOLUME DRIVER</b><br>"
        f"Source column: <b>Quantity</b><br>"
        f"Total units sold: {qty:,}<br>"
        f"Total orders: {kpis['total_orders']:,}<br>"
        f"Drives revenue via: Qty × Avg Price"))

    # ── L2: Pricing (Sales ÷ Order ID) ────────────────────────────────────────
    box(ORD_X, ORD_Y, L2_HW, L2_HH, BLUE_FILL, BLUE_BDR)
    lbl(ORD_X, ORD_Y,
        f"<b>PRICING</b><br>"
        f"col: Sales÷Orders<br>"
        f"${avg_ord:.0f} avg<br>"
        f"per order",
        sz=7.5)
    hdata.append((ORD_X, ORD_Y,
        f"<b>AVERAGE ORDER VALUE</b><br>"
        f"Derived from: Sales ÷ Order ID<br>"
        f"Avg order value: ${avg_ord:,.0f}<br>"
        f"Total orders: {kpis['total_orders']:,}<br>"
        f"Higher avg order = higher revenue without more orders"))

    # ── L2: Discount (reduces Revenue) ────────────────────────────────────────
    box(DSC_X, DSC_Y, L2_HW, L2_HH, RED_FILL, RED_BDR)
    lbl(DSC_X, DSC_Y,
        f"<b>DISCOUNT ↓</b><br>"
        f"col: Discount<br>"
        f"${disc_drain/1e3:.0f}K drained<br>"
        f"avg {avg_disc:.1f}%",
        sz=7.5)
    hdata.append((DSC_X, DSC_Y,
        f"<b>DISCOUNT IMPACT</b><br>"
        f"Source column: <b>Discount</b><br>"
        f"Revenue drained by discounts: ${disc_drain:,.0f}<br>"
        f"Average discount rate: {avg_disc:.1f}%<br>"
        f"⚠ This column hurts both Revenue AND Margin%"))

    # ── L2: Category Mix (drives Margin%) ─────────────────────────────────────
    box(CAT_X, CAT_Y, L2_HW, L2_HH, PURPLE_FILL, PURPLE_BDR)
    lbl(CAT_X, CAT_Y,
        f"<b>CATEGORY</b><br>"
        f"col: Category<br>"
        f"3 product groups<br>"
        f"margin varies",
        sz=7.5)
    hdata.append((CAT_X, CAT_Y,
        f"<b>CATEGORY MIX</b><br>"
        f"Source column: <b>Category</b><br>"
        f"Drives margin variation across product lines:<br>"
        + "<br>".join(
            f"  {r['Category']}: {r['Margin%']:.1f}% margin  (${r['Revenue']/1e3:.0f}K rev)"
            for _, r in cat_df.iterrows()
        )))

    # ── L2: Discount Band (erodes Margin%) ────────────────────────────────────
    box(BND_X, BND_Y, L2_HW, L2_HH, RED_FILL, RED_BDR)
    lbl(BND_X, BND_Y,
        f"<b>DISC BAND ↓</b><br>"
        f"col: Discount<br>"
        f">20%: {hi_margin:.1f}%m<br>"
        f"≤20%: {lo_margin:.1f}%m",
        sz=7.5)
    hdata.append((BND_X, BND_Y,
        f"<b>DISCOUNT BAND IMPACT ON MARGIN</b><br>"
        f"Source column: <b>Discount</b><br>"
        f"High discount (>20%): {len(hi):,} rows → margin {hi_margin:.1f}%<br>"
        f"Low discount  (≤20%): {len(lo):,} rows → margin {lo_margin:.1f}%<br>"
        f"Margin gap: {lo_margin - hi_margin:.1f} percentage points<br>"
        f"⚠ {hi_pct:.0f}% of orders carry high discount"))

    # ── L3: Category breakdown ────────────────────────────────────────────────
    for cname, cx, cy in [
        ("Technology",      TCH_X, TCH_Y),
        ("Furniture",       FRN_X, FRN_Y),
        ("Office Supplies", OSP_X, OSP_Y),
    ]:
        row   = cat_df[cat_df["Category"] == cname].iloc[0]
        fill, bdr = mcolor(row["Margin%"])
        short = {"Technology": "TECH", "Furniture": "FURN", "Office Supplies": "OFF.SUP"}[cname]
        subs  = sub_df[sub_df["Category"] == cname].sort_values("Revenue", ascending=False)
        sub_lines = "<br>".join(
            f"  ▸ {r['Sub-Category']}: {r['Margin%']:.1f}% marg"
            for _, r in subs.iterrows()
        )

        box(cx, cy, L3_HW, L3_HH, fill, bdr)
        lbl(cx, cy,
            f"<b>{short}</b><br>"
            f"${row['Revenue']/1e3:.0f}K<br>"
            f"Marg {row['Margin%']:.1f}%<br>"
            f"Disc {row['AvgDiscount%']:.1f}%",
            sz=7)
        hdata.append((cx, cy,
            f"<b>{cname}</b><br>"
            f"Source column: <b>Category</b><br>"
            f"Revenue: ${row['Revenue']:,.0f}<br>"
            f"Profit:  ${row['Profit']:,.0f}<br>"
            f"Margin:  {row['Margin%']:.1f}%<br>"
            f"Avg Discount: {row['AvgDiscount%']:.1f}%<br>"
            f"YoY Revenue: {cat_yoy.get(cname, 0):+.1f}%<br>"
            f"<br><b>Sub-Categories:</b><br>{sub_lines}"))

    # ── Legend ────────────────────────────────────────────────────────────────
    annots += [
        dict(x=2,  y=12, text="<b>Column → KPI map:</b>",
             showarrow=False, font=dict(size=6.5, color="#475569"), xanchor="left"),
        dict(x=2,  y=10, text="Sales → Revenue",
             showarrow=False, font=dict(size=6.5, color="#2563eb"), xanchor="left"),
        dict(x=24, y=10, text="Profit → Margin%",
             showarrow=False, font=dict(size=6.5, color="#16a34a"), xanchor="left"),
        dict(x=50, y=10, text="Discount → erodes both ↓",
             showarrow=False, font=dict(size=6.5, color="#dc2626"), xanchor="left"),
        dict(x=2,  y=8,  text="Quantity → Volume  |  Category → mix  |  Order ID → avg order",
             showarrow=False, font=dict(size=6.5, color="#7c3aed"), xanchor="left"),
        dict(x=2,  y=6,  text=f"▪ Red = margin <5%   ▪ Amber = 5–12%   ▪ Green = >12%",
             showarrow=False, font=dict(size=6.5, color="#64748b"), xanchor="left"),
    ]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[d[0] for d in hdata],
        y=[d[1] for d in hdata],
        mode="markers",
        marker=dict(size=40, opacity=0),
        hovertext=[d[2] for d in hdata],
        hoverinfo="text",
        showlegend=False,
    ))
    fig.update_layout(
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        shapes=shapes,
        annotations=annots,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2, 102]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 100]),
        margin=dict(t=10, b=10, l=6, r=6),
        height=520,
        hovermode="closest",
    )
    return fig


# ── Helpers ───────────────────────────────────────────────────────────────────

def _monthly_ts(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["Period"] = tmp["Order Date"].dt.to_period("M")
    ts = (
        tmp.groupby("Period")
        .agg(Revenue=("Sales", "sum"), Profit=("Profit", "sum"))
        .reset_index()
    )
    ts["Period"] = ts["Period"].astype(str)
    ts["Margin%"] = (ts["Profit"] / ts["Revenue"] * 100).round(1)
    return ts


def _apply_axis_style(fig):
    """Apply dark-theme axis styling to all axes in figure."""
    for ax in ["xaxis", "yaxis", "xaxis2", "yaxis2", "xaxis3", "yaxis3", "xaxis4", "yaxis4"]:
        try:
            fig.update_layout(**{ax: dict(gridcolor=COLORS["grid"], color=COLORS["text"])})
        except Exception:
            pass
