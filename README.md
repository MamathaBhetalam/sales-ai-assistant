# Business AI Copilot

An intelligent enterprise-grade sales analytics assistant.
Built with **Claude AI**, **Streamlit**, and **Plotly**.

## Features

| Capability | Details |
|---|---|
| **Role-Based AI** | CEO / Manager / Analyst — different depth and tone per role |
| **Metric Tree Reasoning** | Root-cause drills: Revenue → Region → Category → Discount |
| **Auto Visualizations** | Charts auto-select based on your question keywords |
| **5,000-row Dataset** | Synthetic Superstore data (2021–2024) with built-in story patterns |
| **KPI Dashboard** | Live metric cards with YoY deltas |
| **Conversation Memory** | Multi-turn chat with full business context in every message |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your API key (or paste it in the sidebar)
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=sk-ant-...

# 3. Run
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

## Project Structure

```
sales-ai-assistant/
├── app.py                  # Streamlit UI + orchestration
├── src/
│   ├── data_layer.py       # Data generation & KPI computation
│   ├── metric_tree.py      # Root-cause drill-down engine
│   ├── ai_engine.py        # Claude API integration
│   ├── visualizations.py   # Plotly chart library
│   └── role_system.py      # CEO / Manager / Analyst prompts
└── requirements.txt
```

## Example Questions to Try

**CEO Mode**
- "How is the business doing?"
- "What are our biggest risks?"
- "Give me a 30-second briefing."

**Manager Mode**
- "Which region is underperforming and why?"
- "Where should I focus my team this quarter?"
- "Compare this year vs last year by category."

**Analyst Mode**
- "Decompose the profit margin by region and category."
- "What is the statistical impact of discounting on profit?"
- "Identify anomalies in the 2023 Q3 data."
- "Walk me through the metric tree for revenue."

## Architecture

```
User Question
     │
     ├─► Metric Tree Engine  ──► Root-cause analysis (real numbers)
     │
     ├─► KPI Context Builder ──► Business snapshot injection
     │
     ├─► Role System         ──► CEO / Manager / Analyst system prompt
     │
     └─► Claude API          ──► Grounded, role-aware response
               │
               └─► Auto Chart ──► Relevant Plotly visualization
```
