"""
Business AI Copilot — Main Streamlit Application
An intelligent sales analytics assistant with role-based AI responses,
metric tree reasoning, and auto-updating visualizations.
"""
import os
import streamlit as st
from dotenv import load_dotenv

from src.data_layer import generate_data, compute_kpis
from src.ai_engine import SalesAIEngine
from src.visualizations import auto_chart, overview_chart, metric_tree_chart
from src.role_system import ROLE_CONFIG, get_example_questions
from src.email_service import compose_email, send_email, is_email_request, html_for_preview
import streamlit.components.v1 as components

load_dotenv()

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Business AI Copilot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Base & font ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* ── Light background ── */
  .stApp { background-color: #f0f4fb; color: #1a2744; }
  .main .block-container { padding: 1.5rem 2rem 3rem; }
  section[data-testid="stSidebar"] { background-color: #dde6f5; border-right: 1px solid #b8cceb; }

  /* ── Header banner ── */
  .app-header {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 60%, #1e40af 100%);
    border-radius: 12px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    border: 1px solid #1d4ed8;
  }
  .app-header h1 { margin: 0; font-size: 1.75rem; font-weight: 700; color: #ffffff; }
  .app-header p  { margin: 0.3rem 0 0; font-size: 0.9rem; color: #bfdbfe; }

  /* ── KPI metric cards ── */
  div[data-testid="metric-container"] {
    background: #ffffff;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border: 1px solid #d1ddf0;
    border-left: 3px solid #2563eb;
  }
  div[data-testid="metric-container"] label { color: #64748b !important; font-size: 0.78rem; }
  div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-size: 1.55rem !important; font-weight: 600 !important; color: #1a2744 !important;
  }

  /* ── Chat ── */
  .stChatMessage { background: #ffffff !important; border-radius: 10px !important;
                   border: 1px solid #d1ddf0 !important; margin-bottom: 0.6rem; }
  .stChatMessage p { color: #1a2744 !important; line-height: 1.65; }
  div[data-testid="stChatInput"] textarea {
    background: #ffffff !important; color: #1a2744 !important;
    border: 1px solid #b8cceb !important; border-radius: 8px !important;
  }

  /* ── Role badge ── */
  .role-badge {
    display: inline-block;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.8rem;
  }

  /* ── Sidebar sections ── */
  .sidebar-section {
    background: #ffffff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.8rem;
    border: 1px solid #d1ddf0;
  }
  .sidebar-section h4 { margin: 0 0 0.5rem; font-size: 0.82rem;
                         color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }

  /* ── Metric tree display ── */
  .metric-tree {
    font-family: 'Courier New', monospace;
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.7;
    background: #f0f4fb;
    padding: 0.6rem;
    border-radius: 6px;
  }

  /* ── Divider ── */
  hr { border-color: #d1ddf0 !important; }

  /* ── Selectbox ── */
  div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #b8cceb !important;
    color: #1a2744 !important;
  }

  /* ── Button ── */
  .stButton > button {
    background: #ffffff; color: #1a2744;
    border: 1px solid #b8cceb; border-radius: 8px;
    font-size: 0.82rem; padding: 0.4rem 1rem;
  }
  .stButton > button:hover { background: #dde6f5; border-color: #2563eb; }

  /* ── Expander ── */
  details { background: #ffffff !important; border: 1px solid #d1ddf0 !important;
             border-radius: 8px !important; }
  summary { color: #64748b !important; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ────────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "messages": [],
        "current_chart": None,
        "role": "CEO",
        "engine": None,
        "df": None,
        "kpis": None,
        "az_api_key": os.environ.get("AZURE_OPENAI_API_KEY", ""),
        "az_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
        "az_deployment": os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4o"),
        "email_step": None,
        "email_address": "",
        "email_preview": None,
        "email_result": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Data Bootstrap ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading Superstore dataset…")
def load_data():
    df = generate_data()
    kpis = compute_kpis(df)
    return df, kpis


if st.session_state.df is None:
    st.session_state.df, st.session_state.kpis = load_data()

df = st.session_state.df
kpis = st.session_state.kpis


# ── AI Engine Init ────────────────────────────────────────────────────────────

def get_engine(api_key: str, endpoint: str, deployment: str) -> SalesAIEngine:
    creds = (api_key, endpoint, deployment)
    if (st.session_state.engine is None or
            getattr(st.session_state.engine, "_creds_used", None) != creds):
        engine = SalesAIEngine(api_key=api_key, endpoint=endpoint, deployment=deployment)
        engine._creds_used = creds
        st.session_state.engine = engine
    return st.session_state.engine


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🤖 Business AI Copilot")
    st.markdown("---")

    # Role selector
    st.markdown("**Analyst Mode**")
    role = st.selectbox(
        "Select your role",
        options=list(ROLE_CONFIG.keys()),
        index=list(ROLE_CONFIG.keys()).index(st.session_state.role),
        format_func=lambda r: f"{ROLE_CONFIG[r]['emoji']}  {ROLE_CONFIG[r]['label']}",
        label_visibility="collapsed",
    )
    if role != st.session_state.role:
        st.session_state.role = role
        if st.session_state.engine:
            st.session_state.engine.reset_history()

    cfg = ROLE_CONFIG[role]
    st.markdown(
        f'<div class="role-badge" style="background:{cfg["color"]}22;color:{cfg["color"]};'
        f'border:1px solid {cfg["color"]}44">{cfg["emoji"]} {cfg["description"]}</div>',
        unsafe_allow_html=True,
    )

    engine = get_engine(
        st.session_state.az_api_key,
        st.session_state.az_endpoint,
        st.session_state.az_deployment,
    )

    st.markdown("---")

    # Quick KPI snapshot
    _latest_yr = kpis.get("latest_year", "Latest")
    st.markdown("**Quick KPIs**")
    st.markdown(f"""
<div class="sidebar-section">
<h4>{_latest_yr} Snapshot</h4>
<p style="margin:0;font-size:0.82rem;color:#1a2744">
  Revenue  <strong style="float:right;color:#2563eb">${kpis['revenue_latest']:,.0f}</strong><br><br>
  Profit   <strong style="float:right;color:#16a34a">${kpis['profit_latest']:,.0f}</strong><br><br>
  Margin   <strong style="float:right;color:#d97706">{kpis['profit_latest']/kpis['revenue_latest']*100:.1f}%</strong><br><br>
  YoY Rev  <strong style="float:right;color:{'#16a34a' if kpis['yoy_revenue_growth']>=0 else '#dc2626'}">{kpis['yoy_revenue_growth']:+.1f}%</strong>
</p>
</div>
""", unsafe_allow_html=True)

    # Metric tree — interactive sunburst
    st.markdown("**Metric Tree** <small style='color:#8b949e;font-size:0.72rem'>Revenue → Category → Sub-Category · Color = Margin%</small>", unsafe_allow_html=True)
    st.plotly_chart(
        metric_tree_chart(df, kpis),
        use_container_width=True,
        config={"displayModeBar": False},
        key="sidebar_metric_tree",
    )

    st.markdown("---")

    # Clear conversation
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.current_chart = None
        if st.session_state.engine:
            st.session_state.engine.reset_history()
        st.rerun()


# ── Main Layout ───────────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="app-header">
  <h1>🤖 Business AI Copilot</h1>
  <p>Intelligent sales analytics · Role-aware insights · Metric tree reasoning &nbsp;·&nbsp;
     <strong style="color:{cfg['color']}">{cfg['emoji']} {cfg['label']}</strong></p>
</div>
""", unsafe_allow_html=True)

# ── KPI Cards Row ─────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)

with k1:
    st.metric(
        "Total Revenue",
        f"${kpis['total_revenue']/1e6:.1f}M",
        f"{kpis['yoy_revenue_growth']:+.1f}% YoY",
    )
with k2:
    st.metric(
        "Total Profit",
        f"${kpis['total_profit']/1e6:.1f}M",
        f"{kpis['yoy_profit_growth']:+.1f}% YoY",
    )
with k3:
    st.metric(
        "Profit Margin",
        f"{kpis['profit_margin']:.1f}%",
        f"{kpis['profit_latest']/kpis['revenue_latest']*100 - kpis['profit_prev']/kpis['revenue_prev']*100:+.1f}pp YoY",
    )
with k4:
    st.metric(
        "Total Orders",
        f"{kpis['total_orders']:,}",
        f"Best: {kpis['best_region']}",
    )
with k5:
    st.metric(
        "Avg Discount",
        f"{kpis['avg_discount']:.1f}%",
        f"High in {kpis['high_discount_cat']}",
        delta_color="inverse",
    )

st.markdown("---")

# ── Chart Area ────────────────────────────────────────────────────────────────
chart_placeholder = st.empty()

# Show default overview or last auto-generated chart
if st.session_state.current_chart is not None:
    chart_placeholder.plotly_chart(
        st.session_state.current_chart,
        use_container_width=True,
        config={"displayModeBar": False},
    )
else:
    default_fig = overview_chart(df, kpis)
    chart_placeholder.plotly_chart(
        default_fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )

st.markdown("---")

# ── Chat Interface ────────────────────────────────────────────────────────────
st.markdown(f"#### {cfg['emoji']} AI Assistant &nbsp;<small style='color:#8b949e;font-size:0.8rem'>({cfg['label']})</small>", unsafe_allow_html=True)

# Render conversation history
for _msg_idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "assistant" else "👤"):
        st.markdown(msg["content"])
        if msg.get("chart") is not None:
            st.plotly_chart(
                msg["chart"],
                use_container_width=True,
                config={"displayModeBar": False},
                key=f"hist_chart_{_msg_idx}",
            )

# Example question pills (only when history is empty)
if not st.session_state.messages:
    examples = get_example_questions(role)
    st.markdown("<small style='color:#8b949e'>Try asking:</small>", unsafe_allow_html=True)
    cols = st.columns(len(examples))
    for i, (col, q) in enumerate(zip(cols, examples)):
        with col:
            if st.button(q, key=f"eg_{i}", use_container_width=True):
                st.session_state["_pending_question"] = q
                st.rerun()

# Handle example button press
pending = st.session_state.pop("_pending_question", None)

# Chat input
user_input = st.chat_input(
    placeholder=f"{cfg['greeting']} What would you like to explore?",
)

question = user_input or pending

if question:
    # Display user message immediately
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="👤"):
        st.markdown(question)

    # Generate chart for this question (stored in session state; top chart updates on next render)
    new_chart = auto_chart(question, df, kpis)
    st.session_state.current_chart = new_chart

    # Get AI response
    with st.chat_message("assistant", avatar="🤖"):
        _is_email = is_email_request(question)
        if _is_email:
            response = (
                "Sure! I'll prepare a summary of our conversation for you. "
                "Please enter your email address below ↓"
            )
            st.session_state.email_step = "input"
        else:
            with st.spinner("Analyzing…"):
                response = engine.ask(
                    question=question,
                    role=role,
                    kpis=kpis,
                    df=df,
                )
        st.markdown(response)
        if not _is_email:
            # Inline chart in the message too
            _inline_key = f"hist_chart_{len(st.session_state.messages)}"
            st.plotly_chart(new_chart, use_container_width=True, config={"displayModeBar": False}, key=_inline_key)

    # Store in history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "chart": new_chart,
    })

# ── Email Flow ────────────────────────────────────────────────────────────────

_email_step = st.session_state.get("email_step")

if _email_step == "input":
    st.markdown("---")
    st.markdown("#### ✉️ Send Analysis Summary by Email")
    with st.form("email_address_form"):
        _email_input = st.text_input(
            "Your email address",
            placeholder="you@example.com",
            value=st.session_state.get("email_address", ""),
        )
        _col1, _col2 = st.columns([3, 1])
        with _col1:
            _preview_btn = st.form_submit_button("Preview Email →", use_container_width=True)
        with _col2:
            _cancel_btn = st.form_submit_button("Cancel", use_container_width=True)

    if _preview_btn:
        if not _email_input or "@" not in _email_input or "." not in _email_input.split("@")[-1]:
            st.error("Please enter a valid email address.")
        else:
            # Collect unique charts — deduplicate by title, keep last 3 distinct
            _seen_titles, _charts = set(), []
            for _m in st.session_state.messages:
                _fig = _m.get("chart")
                if _fig is not None:
                    _title = (_fig.layout.title.text or "") if _fig.layout.title else ""
                    if _title not in _seen_titles:
                        _seen_titles.add(_title)
                        _charts.append(_fig)
            _charts = _charts[-3:]
            _subject, _html_body, _inline_images = compose_email(
                st.session_state.messages, role, kpis, figures=_charts
            )
            st.session_state.email_address  = _email_input
            st.session_state.email_preview  = (_subject, _html_body, _inline_images)
            st.session_state.email_step     = "preview"
            st.rerun()
    elif _cancel_btn:
        st.session_state.email_step = None
        st.rerun()

elif _email_step == "preview":
    st.markdown("---")
    st.markdown("#### ✉️ Confirm & Send")
    _subject, _html_body, _inline_images = st.session_state.email_preview
    _n_turns = sum(1 for m in st.session_state.messages if m["role"] == "user"
                   and not is_email_request(m["content"]))

    st.markdown(f"**To:** `{st.session_state.email_address}`")
    st.markdown(f"**Subject:** {_subject}")
    st.markdown(f"**Includes:** {_n_turns} question(s) · KPI snapshot · {len(_inline_images)} chart(s) · {role} mode context")

    with st.expander("Preview email content", expanded=True):
        # Swap cid: → data: URIs so charts render in the browser preview
        _preview_html = html_for_preview(_html_body, _inline_images)
        components.html(_preview_html, height=500, scrolling=True)

    _col1, _col2 = st.columns(2)
    with _col1:
        if st.button("✉️ Confirm & Send", use_container_width=True, type="primary"):
            _success, _msg = send_email(
                st.session_state.email_address, _subject, _html_body, _inline_images
            )
            st.session_state.email_result = (_success, _msg)
            st.session_state.email_step = "sent"
            st.rerun()
    with _col2:
        if st.button("← Edit Email Address", use_container_width=True):
            st.session_state.email_step = "input"
            st.rerun()

elif _email_step == "sent":
    st.markdown("---")
    _success, _msg = st.session_state.get("email_result", (False, "Unknown error"))
    if _success:
        st.success(f"✅ {_msg}")
    else:
        st.error(f"❌ {_msg}")
        st.info(
            "Gmail tip: Go to myaccount.google.com → Security → App Passwords "
            "and use that password as SMTP_PASSWORD in your .env file."
        )
    if st.button("Done"):
        st.session_state.email_step = None
        st.session_state.email_result = None
        st.rerun()

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:2rem 0 0.5rem;color:#94a3b8;font-size:0.75rem'>
  Business AI Copilot &nbsp;·&nbsp; Powered by GPT-4o &amp; Streamlit &nbsp;·&nbsp;
  Metric Tree Reasoning &nbsp;·&nbsp; Superstore Dataset (2015–2018)
</div>
""", unsafe_allow_html=True)
