"""
AI Engine: Orchestrates Azure OpenAI GPT-4o calls with role-aware prompting,
KPI context injection, and metric tree reasoning.
"""
import os
from typing import Optional
from openai import AzureOpenAI

from .role_system import get_system_prompt
from .data_layer import format_context
from .metric_tree import get_tree_analysis
from .entity_extractor import extract_entities


class SalesAIEngine:
    """
    Main AI orchestration class.
    Injects business context into every GPT-4o call so answers are
    grounded in real data rather than generic advice.
    """

    MODEL = "gpt-4o"  # Azure deployment name

    def __init__(
        self,
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
    ):
        self._api_key = api_key or os.environ.get("AZURE_OPENAI_API_KEY", "")
        self._endpoint = endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT", "")
        self._deployment = deployment or os.environ.get("AZURE_OPENAI_DEPLOYMENT", self.MODEL)
        self.client = None
        self.history: list[dict] = []
        self._build_client()

    def _build_client(self):
        if self._api_key and self._endpoint:
            self.client = AzureOpenAI(
                api_key=self._api_key,
                azure_endpoint=self._endpoint,
                api_version="2024-12-01-preview",
            )

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
        max_tokens: int = 600,
    ) -> str:
        """
        Send a question to GPT-4o with full business context.

        1. Runs metric tree analysis on the question
        2. Formats KPI snapshot
        3. Builds role-appropriate system prompt
        4. Maintains conversation history
        5. Returns the model's response
        """
        if not self.is_ready():
            return (
                "⚠️ Azure OpenAI not configured. Please add your API Key and Endpoint "
                "in the sidebar to activate the AI engine.\n\n"
                "You can still explore the data using the charts and KPI cards above."
            )

        # --- Extract named entities (category / developer) from question ---
        entities      = extract_entities(question, kpis)
        focus_cat     = entities.get("category")

        # --- Build context blocks (category-aware) ---
        kpi_context   = format_context(kpis, focus_category=focus_cat)
        tree_analysis = get_tree_analysis(question, df, kpis=kpis, focus_category=focus_cat)

        # Tell the AI which category was detected (if any) so it frames its answer
        focus_note = (
            f"\nDETECTED FOCUS: User is asking specifically about the '{focus_cat}' category. "
            f"Prioritise the CATEGORY FOCUS section in the data above.\n"
        ) if focus_cat else ""

        system_content = f"""{get_system_prompt(role)}
{focus_note}
---
LIVE BUSINESS DATA (always use these real numbers in your answer):

{kpi_context}

---
METRIC TREE ANALYSIS FOR THIS QUERY:
(Use this structured drill-down to ground your reasoning)

{tree_analysis}
---
IMPORTANT RULES:
- Always reference specific numbers from the data above
- Never fabricate metrics — if data is absent, say so
- Match your depth and tone to your assigned role
- The user is asking about a REAL business, not a hypothetical
- When a specific category is identified, lead with that category's data
"""

        # Maintain rolling history (last 6 turns to stay within context)
        self.history.append({"role": "user", "content": question})
        if len(self.history) > 12:
            self.history = self.history[-12:]

        messages = [{"role": "system", "content": system_content}] + self.history

        try:
            response = self.client.chat.completions.create(
                model=self._deployment,
                messages=messages,
                max_tokens=max_tokens,
            )
            answer = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": answer})
            return answer

        except Exception as e:
            err = str(e).lower()
            if "authentication" in err or "401" in err or "api key" in err:
                return "❌ Invalid API key. Please check your Azure OpenAI key in the sidebar."
            if "rate limit" in err or "429" in err:
                return "⏳ Rate limit reached. Please wait a moment and try again."
            if "deployment" in err or "not found" in err or "404" in err:
                return f"❌ Deployment '{self._deployment}' not found. Check your deployment name."
            return f"❌ API error: {str(e)}"
