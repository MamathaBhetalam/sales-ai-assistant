"""
Entity Extractor: Identify category and developer names in free-text questions.

All matching is done against the live dataset — no hardcoded lists.
Strategy: exact substring matching only (the full category/developer name must
appear in the question). This is intentionally strict to prevent false positives
like matching "Developer tools" from a question about developers-in-general.
"""

# Words/phrases that signal the user wants to compare two things side by side.
_COMPARE_SIGNALS = ("vs ", " vs", "versus", "compare", "against", "compared to")

# Store-wide questions where category extraction would produce false positives.
# Defined at module level so the tuple is allocated once, not on every call.
_GENERIC_PHRASES = (
    "all categories", "all apps", "whole store", "top categories",
    "best categories", "compare categories", "overall", "overview",
    "summary", "trend over time", "how many apps", "total apps",
)


def extract_category(question: str, kpis: dict) -> str | None:
    """
    Return the category name that appears (as a substring) in the question,
    or None if no category name is found.

    Matches against kpis["by_category"]["category"] — the live list from data.
    Longest match wins (prevents "Pop" matching before "Pop-ups").
    """
    if "by_category" not in kpis or kpis["by_category"].empty:
        return None

    q_lower = question.lower()
    cats = kpis["by_category"]["category"].tolist()

    best_match, best_len = None, 0
    for cat in cats:
        cat_lower = cat.lower()
        if cat_lower in q_lower and len(cat_lower) > best_len:
            best_match, best_len = cat, len(cat_lower)

    return best_match


def extract_developer(question: str, kpis: dict) -> str | None:
    """
    Return the developer name that appears (as a substring) in the question,
    or None if none found.

    Exact substring only — developer names are too varied for fuzzy matching.
    """
    if "by_developer" not in kpis or kpis["by_developer"].empty:
        return None

    q_lower = question.lower()
    for dev in kpis["by_developer"]["developer"].tolist():
        if dev.lower() in q_lower:
            return dev

    return None


def extract_second_category(question: str, kpis: dict, first_cat: str) -> str | None:
    """
    When comparison intent is detected, find a second category in the question
    that is different from first_cat.  Uses the same longest-substring rule.
    Returns None if no second category is found.
    """
    if not first_cat or "by_category" not in kpis or kpis["by_category"].empty:
        return None

    q_lower = question.lower()
    cats = kpis["by_category"]["category"].tolist()

    best_match, best_len = None, 0
    for cat in cats:
        if cat == first_cat:
            continue
        cat_lower = cat.lower()
        if cat_lower in q_lower and len(cat_lower) > best_len:
            best_match, best_len = cat, len(cat_lower)

    return best_match


def extract_entities(question: str, kpis: dict) -> dict:
    """
    Extract named entities from a question.

    Returns:
        {
            "category":  str | None  — primary category name found in question
            "category2": str | None  — second category when comparison detected
            "developer": str | None  — developer name found in question
        }
    """
    if len(question.strip()) < 5:
        return {"category": None, "category2": None, "developer": None}

    q_lower = question.lower()
    # Questions about the whole store — skip category extraction to avoid
    # accidentally matching a category name that happens to be a common word
    if any(p in q_lower for p in _GENERIC_PHRASES):
        return {"category": None, "category2": None, "developer": None}

    category  = extract_category(question, kpis)
    developer = extract_developer(question, kpis)

    # Detect comparison intent — try to extract a second category
    is_comparison = any(sig in q_lower for sig in _COMPARE_SIGNALS)
    category2 = extract_second_category(question, kpis, category) if (is_comparison and category) else None

    return {"category": category, "category2": category2, "developer": developer}
