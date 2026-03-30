"""
AI Engine: Orchestrates OpenAI GPT-4o calls with tool use for e-commerce data querying.

Supports both standard OpenAI (api.openai.com) and Azure OpenAI.
The engine receives a data schema description and callable tool functions.
For each question it decides which tools to call, executes them against the
real DataFrames, and produces an answer grounded in actual data.
"""
import json
import os
from typing import Optional

from openai import OpenAI, AzureOpenAI

from .tools import TOOL_SCHEMAS, execute_tool
from .role_system import get_system_prompt

MODEL           = "gpt-4o"
MAX_TOOL_ROUNDS = 8

_ANALYTICAL_FRAMEWORK = """
ANALYTICAL FRAMEWORK — follow this process internally for every question:

STEP 1 — UNDERSTAND THE QUESTION
  • Identify intent: informational / comparison / trend / root-cause / prediction / recommendation
  • Map user terms to dataset fields (even if wording differs — e.g. "sales" → revenue, "source" → TrafficSource)
  • If vague, assume the most logical business interpretation and state it

STEP 2 — CALL THE RIGHT TOOL(S)
  • Always retrieve real data via tools before answering
  • Choose tools based on the TOOL SELECTION GUIDE below
  • For multi-dimensional questions, call multiple tools in sequence

STEP 3 — BUILD THE METRIC TREE
  Revenue          = Quantity × Price  (= Total field per line item; credited only when Completed → Sales field)
  Completion Rate  = Completed orders ÷ Total orders  ⚠️ NOT a true conversion rate (see DATASET LIMITATIONS)
  Profit           = Revenue − Cost  ⚠️ NOT computable — no cost data in dataset
  LTV              = AOV × Purchase Frequency × Customer Lifespan
  Lifespan         = 1 ÷ Churn Rate
  Churn            = 1 − Retention Rate
  Use this tree to explain WHY a metric is high or low; call out limitations where components are missing

STEP 4 — ANALYZE
  • Calculate: sums, averages, ratios, growth rates
  • Concentration: only claim 80/20 if you have actually run a breakdown and the data supports it — never assert it generically
  • Compare: channel vs channel, country vs country, current vs baseline
  • Identify: anomalies, outliers, patterns, correlations
  • Flag: retention risk, channel/geo over-dependence, funnel inefficiencies (see INSIGHT DEPTH rule)

STEP 5 — STRUCTURE THE RESPONSE
  Adapt depth to the role and question complexity:

  For factual/lookup questions:
    → Direct Answer with exact numbers from the tool

  For analytical/strategic questions, use this structure:
    **Direct Answer**      — one-sentence bottom line with the key number
    **Key Insight**        — what is happening (pattern, trend, anomaly)
    **Root Cause**         — WHY it is happening (use metric tree reasoning)
    **Recommendation**     — what to do next (specific, actionable)
    **⚠️ Risks & Watch-Outs** — risks, concentration, funnel gaps (always required; see INSIGHT DEPTH rule)

STEP 6 — QUALITY CHECK
  • Never return raw numbers without context
  • Always provide insight and reasoning, not just data
  • If data is missing or tool returns empty → say so clearly, do NOT fabricate
"""

_DATA_SCHEMA = """
DATABASE SCHEMA (E-Commerce Transactions):

TABLE STRUCTURE:
  Each row is a TRANSACTION LINE ITEM — one product within an order.
  A single InvoiceNumber can have multiple rows (multiple products).
  Do NOT treat one row as one order when counting orders — group by InvoiceNumber.

COLUMNS:
  CustomerID      — unique customer identifier
  Gender          — Male / Female
  InvoiceDate     — date of the transaction (datetime; source format M/D/YYYY)
                    DATE PARSING RULE: '1/09/2019' = January 9 2019 (month first)
  InvoiceNumber   — unique order ID (one order can span multiple rows)
  ProductID       — numeric product identifier (no product names in dataset)
  Quantity        — units ordered for this line item
  Price           — unit price per item (USD)
  Total           — line-item subtotal = Price × Quantity (pre-status; always positive)
  OrderStatus     — Completed / In Process / Cancelled / etc.
  Country         — customer country
  TrafficSource   — acquisition channel (Social Media, Paid Advertisement, Organic Search, etc.)
  SessionDuration — time spent on site (minutes) before purchase
  DeviceCategory  — Computer / Mobile
  Device          — Laptop / Desktop / Tablet / Smartphone / etc.
  OS              — Windows / iOS / Android / etc.
  DeliveryRating  — customer rating of delivery (1–5; 0 = not yet rated)
  ProductRating   — customer rating of product (1–5; 0 = not yet rated)
  Sales           — REVENUE FIELD: = Total when OrderStatus is Completed, else 0
  revenue         — identical to Sales (derived alias; use either)
  is_completed    — boolean: True when OrderStatus == 'Completed'
  year_month      — YYYY-MM string derived from InvoiceDate

FIELD RELATIONSHIPS (critical for correct analysis):
  Revenue (credited)  = Sales field  (only Completed rows have non-zero Sales)
  Line-item subtotal  = Total field   (all rows, regardless of status)
  Completion rate     = Completed InvoiceNumbers ÷ Total InvoiceNumbers
                        ⚠️ This is ORDER COMPLETION RATE, NOT true conversion rate
                           (true conversion = website visitors → purchase; traffic data absent)
  AOV (completed)     = sum(Sales) ÷ count(distinct completed InvoiceNumbers)

DATASET LIMITATIONS (always surface when relevant):
  ❌ No website traffic volume → true conversion rate (visitors → orders) cannot be computed
  ❌ No cost / COGS data       → gross profit and margin cannot be computed
  ❌ No customer lifespan field → Customer Lifespan must be estimated via retention rate
  ❌ No product names          → product analysis is by ProductID only
  ⚠️  Short observation window → retention rate may be underestimated; flag this when reporting LTV
"""


class SalesAIEngine:
    """
    OpenAI-powered AI engine with tool use for e-commerce data querying.

    Automatically uses AzureOpenAI when an endpoint is provided, otherwise
    falls back to standard OpenAI (api.openai.com).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
    ):
        self._api_key    = api_key    or os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
        self._endpoint   = endpoint   or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", MODEL)
        self.client: Optional[OpenAI] = None
        self.history: list[dict] = []
        self._build_client()

    def _build_client(self):
        if not self._api_key:
            return
        if self._endpoint:
            self.client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version="2024-12-01-preview",
            )
        else:
            self.client = OpenAI(api_key=self._api_key)

    def is_ready(self) -> bool:
        return self.client is not None

    def reset_history(self):
        self.history = []

    def ask(
        self,
        question: str,
        role: str,
        kpis: dict,
        df,
        tables: dict | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """
        Send a question to GPT-4o with tool-use capability.

        The model calls the appropriate tools to retrieve real data, then
        synthesises a grounded answer. Only clean text is stored in history
        so follow-up questions work correctly without leaking raw tool blocks.
        """
        if not self.is_ready():
            return (
                "⚠️ OpenAI not configured. Please add OPENAI_API_KEY (or Azure keys) "
                "to the .env file to activate the AI engine.\n\n"
                "You can still explore the data using the charts and KPI cards above."
            )

        tables = tables or {}

        system_content = f"""{get_system_prompt(role)}

---
{_ANALYTICAL_FRAMEWORK}
---
{_DATA_SCHEMA}
---
CURRENT SNAPSHOT (top-level KPIs — use tools for detailed queries):
  Total Orders        : {kpis['total_orders']:,}
  Completed Orders    : {kpis['completed_orders']:,}  ({kpis['completion_rate']:.1f}%)
  Total Revenue       : ${kpis['total_revenue']:,.2f}
  Avg Order Value     : ${kpis['avg_order_value']:.2f}
  Unique Customers    : {kpis['unique_customers']:,}
  Avg Purchase Freq   : {kpis.get('avg_purchase_freq', 'N/A')} orders/customer
  Retention Rate      : {kpis.get('retention_rate', 'N/A')}%  ({kpis.get('repeat_customers', 'N/A')} repeat customers)
  Avg Product Rating  : {kpis['avg_product_rating']:.2f} / 5.0
  Avg Delivery Rating : {kpis['avg_delivery_rating']:.2f} / 5.0
  Top Channel         : {kpis['top_channel']}
  Top Country         : {kpis['top_country']}
---
CRITICAL RULES (non-negotiable):

▸ STRICT DATA GROUNDING:
1. NEVER invent invoice numbers, product IDs, quantities, countries, channels, or any data.
2. ALWAYS call a tool to retrieve real data before answering ANY factual question.
3. If a tool returns "No orders found" or empty results → say exactly that. Do NOT make up data.
4. If you are unsure which tool to use, call filter_orders with the available filters.
5. For date questions: always pass the date to the tool — do not compute or guess from memory.
6. If a question cannot be answered with available tools, say:
   "This specific query isn't supported by the available tools."
   Never substitute a plausible-sounding fabricated answer.
7. Before using any value in your answer, verify it came from a tool result in this conversation.
   If the data was not returned by a tool → label it as "approximation" or "inference", never state it as fact.
8. DO NOT introduce external assumptions not present in the dataset.
   FORBIDDEN examples: inflation rates, market competition, industry benchmarks, seasonal norms,
   macroeconomic factors, platform algorithm changes, consumer sentiment shifts.
   If context outside the dataset would be needed, say:
   "This analysis is limited to what the dataset contains — external factors (e.g. competition,
   inflation) are not represented in the data and cannot be assumed."

▸ METRIC CONSISTENCY:
9. Use consistent denominators throughout a single answer:
   - Revenue analysis → use COMPLETED orders only (Sales field)
   - Funnel / cancellation analysis → use ALL orders as the base
   - NEVER mix completed and total orders in the same ratio without explicitly stating which is which
10. When reporting percentages, always state what the denominator is.
    Good: "Completion rate = 68% (completed orders ÷ total orders)"
    Bad:  "Completion rate = 68%"
11. NEVER call the completion rate a "conversion rate".
    - Completion rate  = completed orders ÷ total orders  (what this dataset can measure)
    - True conversion  = website visitors → purchases      (NOT measurable — no traffic data)
    Always use the term "order completion rate" or "completion rate (order success rate)".

▸ LTV CALCULATION:
12. The correct LTV formula is:
      LTV = AOV × Purchase Frequency × Customer Lifespan
    Where:
      AOV               = total completed revenue ÷ completed orders
      Purchase Frequency = completed orders ÷ unique customers  (orders per customer)
      Customer Lifespan  = 1 ÷ Churn Rate
      Churn Rate         = 1 − Retention Rate
      Retention Rate     = repeat customers ÷ unique customers

    Full derived form:
      LTV = AOV × Purchase Frequency × (1 / (1 − Retention Rate))

    - For channel-level LTV: use primary-channel attribution — assign each customer to their
      most-used channel, compute their lifetime revenue, then average per channel (avoids double-counting).
    - LIFESPAN CAVEAT: if the dataset covers a short time window, retention rate will be
      underestimated and lifespan will be inflated. Always state the date range of the data
      used when reporting LTV.
    - If Retention Rate is not computable from the data (e.g. insufficient repeat purchase
      history), state explicitly:
      "LTV calculation method not defined in dataset — Customer Lifespan cannot be determined
      without reliable retention data."
13. Always state the exact formula and all input values used. Never present LTV without
    explaining the basis. Never infer or estimate LTV components from outside the dataset.

▸ INSIGHT DEPTH (mandatory for every analytical response):
14. After your main analysis, always scan for and surface the following if present in the data:
    RISKS:
    - Low retention (repeat customer rate < 30%) → flag as churn risk
    - High order drop-off / cancellation rate (> 20%) → flag as funnel leakage
    - Low delivery or product ratings (avg < 3.5 / 5) → flag as satisfaction risk
    OVER-DEPENDENCE:
    - If a single channel drives > 50% of revenue → flag as channel concentration risk
    - If a single country drives > 50% of revenue → flag as geographic concentration risk
    HIDDEN FUNNEL INEFFICIENCIES:
    - High session-to-completion gap (high SessionDuration or session count, low completion rate)
    - Disproportionate cancellation rates in specific channels or countries
    - Large revenue gap between top and bottom performers (products, channels, countries)
    Format these findings under a **⚠️ Risks & Watch-Outs** section at the end of your response.
    If none of the above thresholds are triggered, explicitly state: "No critical risks identified
    in current data." — do NOT silently omit the section.

RESPONSE STYLE:
- Simple factual questions → show exact data from the tool, clearly formatted.
- Analytical/strategic questions → follow the 4-part structure from the analytical framework.

TOOL SELECTION GUIDE:
- Specific date / date range query    → get_date_summary(date=...) or filter_orders(date=...)
- Multi-filter lookup (date+country+channel etc.) → filter_orders(date=..., country=..., channel=...)
- Product query                       → get_product_stats(product_id=...) or filter_orders(product_id=...)
- Customer query                      → filter_orders(customer_id=...)
- Deep dive on a channel              → get_channel_stats(channel=...)
- Deep dive on a country              → get_country_stats(country=...)
- Revenue/orders split by dimension   → get_revenue_breakdown(dimension=...)
- Product or delivery ratings         → get_rating_analysis(...)
- Monthly revenue trend               → get_revenue_trend(...)
- Store-wide KPIs                     → get_market_overview()
- Cancellation analysis / why cancelled → get_cancellation_analysis(group_by="all")
- Metric tree drill-down (causal)     → get_metric_tree(metric=...)
  Use for: "break down revenue", "analyze channels with LTV", "why completion rate dropped",
  "retention analysis", "geographic breakdown", "purchase frequency drivers"
"""

        # Append question to history; trim to last 6 turns (12 messages)
        self.history.append({"role": "user", "content": question})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        # in-flight messages — may include tool blocks; history stays text-only
        messages = [{"role": "system", "content": system_content}] + list(self.history)

        try:
            response = None

            for _round in range(MAX_TOOL_ROUNDS):
                # Round 0: force at least one tool call so the model never answers
                # from parametric memory. Subsequent rounds: auto (so it can stop).
                _tool_choice = "required" if _round == 0 else "auto"
                response = self.client.chat.completions.create(
                    model=self._deployment,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice=_tool_choice,
                    max_tokens=max_tokens,
                )

                choice = response.choices[0]

                if choice.finish_reason == "stop":
                    answer = choice.message.content or ""
                    self.history.append({"role": "assistant", "content": answer})
                    return answer

                if choice.finish_reason == "tool_calls":
                    msg = choice.message

                    # Append the assistant message with tool_calls to in-flight messages
                    messages.append({
                        "role": "assistant",
                        "content": msg.content,  # may be None
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                            for tc in (msg.tool_calls or [])
                        ],
                    })

                    # Execute each tool call and append results
                    for tc in (msg.tool_calls or []):
                        try:
                            inputs = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            inputs = {}

                        result = execute_tool(
                            name=tc.function.name,
                            inputs=inputs,
                            df=df,
                            tables=tables,
                            kpis=kpis,
                        )
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        })

                    continue  # next round

                # Unexpected finish reason
                break

            # Fell through MAX_TOOL_ROUNDS — return whatever text is available
            answer = ""
            if response is not None:
                answer = response.choices[0].message.content or ""
            if not answer:
                answer = (
                    "I reached the maximum number of tool calls without a final answer. "
                    "Please try a more specific question."
                )
            self.history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            # Remove the question we added to keep history consistent
            if self.history and self.history[-1]["role"] == "user":
                self.history.pop()
            err = str(e).lower()
            if "authentication" in err or "401" in err or "api key" in err:
                return "❌ Invalid API key. Please check your key in the .env file."
            if "rate limit" in err or "429" in err:
                return "⏳ Rate limit reached. Please wait a moment and try again."
            if "deployment" in err or "not found" in err or "404" in err:
                return f"❌ Deployment '{self._deployment}' not found. Check your deployment name."
            return f"❌ API error: {str(e)}"
