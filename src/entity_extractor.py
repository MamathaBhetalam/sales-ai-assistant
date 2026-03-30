"""
Entity Extractor: Identify channel (TrafficSource) and country names in free-text questions.
All matching is done against the live dataset — no hardcoded lists.
"""

_COMPARE_SIGNALS = ("vs ", " vs", "versus", "compare", "against", "compared to")

_GENERIC_PHRASES = (
    "all channels", "all countries", "overall", "overview", "summary",
    "total orders", "total revenue", "how many orders", "trend", "trend over time",
)


def extract_channel(question: str, kpis: dict) -> str | None:
    if "by_channel" not in kpis or kpis["by_channel"].empty:
        return None
    q_lower  = question.lower()
    channels = kpis["by_channel"]["TrafficSource"].tolist()
    best_match, best_len = None, 0
    for ch in channels:
        ch_lower = ch.lower()
        if ch_lower in q_lower and len(ch_lower) > best_len:
            best_match, best_len = ch, len(ch_lower)
    return best_match


def extract_country(question: str, kpis: dict) -> str | None:
    if "by_country" not in kpis or kpis["by_country"].empty:
        return None
    q_lower   = question.lower()
    countries = kpis["by_country"]["Country"].tolist()
    best_match, best_len = None, 0
    for co in countries:
        co_lower = co.lower()
        if co_lower in q_lower and len(co_lower) > best_len:
            best_match, best_len = co, len(co_lower)
    return best_match


def extract_second_channel(question: str, kpis: dict, first_ch: str) -> str | None:
    if not first_ch or "by_channel" not in kpis or kpis["by_channel"].empty:
        return None
    q_lower  = question.lower()
    channels = kpis["by_channel"]["TrafficSource"].tolist()
    best_match, best_len = None, 0
    for ch in channels:
        if ch == first_ch:
            continue
        ch_lower = ch.lower()
        if ch_lower in q_lower and len(ch_lower) > best_len:
            best_match, best_len = ch, len(ch_lower)
    return best_match


def extract_entities(question: str, kpis: dict) -> dict:
    """
    Returns:
        {
            "channel":  str | None  — TrafficSource found in question
            "channel2": str | None  — second channel when comparing
            "country":  str | None  — Country name found in question
        }
    """
    if len(question.strip()) < 5:
        return {"channel": None, "channel2": None, "country": None}

    q_lower = question.lower()
    if any(p in q_lower for p in _GENERIC_PHRASES):
        return {"channel": None, "channel2": None, "country": None}

    channel  = extract_channel(question, kpis)
    country  = extract_country(question, kpis)

    is_comparison = any(sig in q_lower for sig in _COMPARE_SIGNALS)
    channel2 = extract_second_channel(question, kpis, channel) if (is_comparison and channel) else None

    return {"channel": channel, "channel2": channel2, "country": country}
