import re

AI_KEYWORDS = [
    "artificial intelligence",
    "machine learning",
    "generative ai",
    "large language model",
    "deep learning",
    "neural network",
    "ai",
    "llm",
    "gpu",
    "gpus",
]

# \b word boundaries keep short terms like "ai"/"llm"/"gpu" from matching inside
# unrelated words (e.g. "said", "maintain").
_AI_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(keyword) for keyword in AI_KEYWORDS) + r")\b", re.IGNORECASE
)

BUSINESS_SUMMARY_MAX_CHARS = 800


def find_ai_keywords(text: str) -> list[str]:
    if not text:
        return []
    matches = {match.group(0).lower() for match in _AI_PATTERN.finditer(text)}
    return sorted(matches)


def summarize_ai_exposure(info: dict, news: list[dict]) -> dict:
    business_summary = info.get("longBusinessSummary") or ""
    summary_matches = find_ai_keywords(business_summary)

    news_items = []
    ai_related_count = 0
    for item in news:
        text = f"{item.get('title', '')} {item.get('summary', '')}"
        is_ai_related = bool(find_ai_keywords(text))
        if is_ai_related:
            ai_related_count += 1
        news_items.append({**item, "is_ai_related": is_ai_related})

    return {
        "business_summary": business_summary,
        "ai_mentioned_in_summary": bool(summary_matches),
        "ai_keyword_matches": summary_matches,
        "news_items": news_items,
        "ai_related_news_count": ai_related_count,
        "total_news_count": len(news_items),
    }


def format_ai_exposure_for_prompt(ai: dict) -> str:
    summary = ai["business_summary"]
    if summary:
        if len(summary) > BUSINESS_SUMMARY_MAX_CHARS:
            summary = summary[:BUSINESS_SUMMARY_MAX_CHARS].rsplit(" ", 1)[0] + "..."
        summary_line = f"Business description: {summary}"
    else:
        summary_line = "Business description: N/A"

    keywords_line = (
        "AI-related terms found in business description: "
        + (", ".join(ai["ai_keyword_matches"]) if ai["ai_keyword_matches"] else "none detected")
    )

    lines = [summary_line, keywords_line]

    if ai["news_items"]:
        lines.append(
            f"Recent headlines ({ai['ai_related_news_count']}/{ai['total_news_count']} AI-related):"
        )
        for item in ai["news_items"]:
            tag = "[AI-related] " if item["is_ai_related"] else ""
            lines.append(f"- {tag}{item['title']} ({item.get('publisher', 'unknown source')})")
    else:
        lines.append("Recent headlines: none available")

    return "\n".join(lines)
