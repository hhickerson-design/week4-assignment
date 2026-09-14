from src.ai_exposure import find_ai_keywords, summarize_ai_exposure


def test_find_ai_keywords_matches_phrases_and_short_acronyms():
    text = "The company builds GPUs for generative AI and large language model training."
    matches = find_ai_keywords(text)
    assert "gpus" in matches
    assert "generative ai" in matches
    assert "large language model" in matches


def test_find_ai_keywords_avoids_false_positives_inside_words():
    text = "The board said it would maintain its chair and email policies."
    assert find_ai_keywords(text) == []


def test_find_ai_keywords_case_insensitive_and_hyphenated():
    assert "ai" in find_ai_keywords("An AI-powered platform.")
    assert "machine learning" in find_ai_keywords("Uses Machine Learning models.")


def test_summarize_ai_exposure_detects_summary_mentions():
    info = {"longBusinessSummary": "This company provides artificial intelligence infrastructure."}
    result = summarize_ai_exposure(info, news=[])
    assert result["ai_mentioned_in_summary"] is True
    assert "artificial intelligence" in result["ai_keyword_matches"]
    assert result["total_news_count"] == 0


def test_summarize_ai_exposure_no_mentions():
    info = {"longBusinessSummary": "This company sells furniture and home goods."}
    result = summarize_ai_exposure(info, news=[])
    assert result["ai_mentioned_in_summary"] is False
    assert result["ai_keyword_matches"] == []


def test_summarize_ai_exposure_flags_ai_related_news_items():
    info = {"longBusinessSummary": "A retailer."}
    news = [
        {"title": "Company launches new AI-powered assistant", "summary": ""},
        {"title": "Company reports quarterly earnings", "summary": "Revenue grew 5%."},
    ]
    result = summarize_ai_exposure(info, news)
    assert result["ai_related_news_count"] == 1
    assert result["total_news_count"] == 2
    assert result["news_items"][0]["is_ai_related"] is True
    assert result["news_items"][1]["is_ai_related"] is False
