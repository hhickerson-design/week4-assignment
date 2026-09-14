# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## About the user and how to work with them

- The user is a working professional, not a computer science student. They are comfortable with everyday computer use but are not deeply technical.
- Their background is in finance, with prior experience as a logistics officer in the U.S. Marine Corps. Analogies to finance or logistics concepts (budgets, supply chains, risk, resource allocation) may help explain technical ideas.
- Outside of work, they are a homeowner, a parent of a 5-year-old and a 7-year-old, a coach for 2nd-grade basketball and flag football, and an avid fitness enthusiast. Analogies to coaching, training/practice routines, or household management may also help explain technical ideas.
- The user is enrolled in an M.S. program studying applied AI, and this repository is the deliverable for an assignment in the Foundations of AI course.
- When helping build this assignment, explain step by step what you are doing and why — do not just make changes silently.
- Introduce technical terms in plain language before using them.

## Status

A Streamlit-based stock analysis chatbot combining fundamental, technical, valuation,
earnings-quality, and AI-exposure analysis with the Anthropic Claude API. Designed to
weigh a company's earnings and sales growth from its core business as the central signal
of business health, to distinguish that core-operations growth from growth driven by
non-operating items, and to proactively assess (and probe, when unclear) the company's
ties to AI. Also supports mid-conversation multi-company comparison and DCF (discounted
cash flow) valuation via Claude API tool use — see the "Tool use" entry below. See
[README.md](README.md) for full setup/usage instructions.

## Commands

```bash
pip install -r requirements.txt   # install dependencies
streamlit run app.py              # run the app (http://localhost:8501)
pytest -v                         # run all tests
pytest tests/test_technical.py::test_rsi_all_gains_is_100 -v   # run a single test
```

## Architecture

- `app.py` — Streamlit entrypoint. Owns the UI (sidebar ticker input/metrics, main chat
  panel) and orchestrates calls into `src/`. Holds conversation state in
  `st.session_state`.
- `src/config.py` — loads `.env`, exposes the Anthropic API key and shared constants
  (model name, cache TTL, CAPM assumptions, etc.).
- `src/data.py` — fetches live price history and company info from Yahoo Finance via
  `yfinance`; framework-agnostic (no Streamlit import) so it's independently testable.
  Raises `TickerDataError` on bad ticker/network failure.
- `src/technical.py` — hand-rolled technical indicators (SMA, EMA, RSI, MACD, Bollinger
  Bands) computed with pandas, plus `summarize_technical()` which turns a price-history
  DataFrame into a flat dict of latest values and plain-language signals.
- `src/fundamentals.py` — extracts and formats fundamental metrics (P/E, market cap,
  margins, revenue/earnings growth, etc.) from the raw `yfinance` info dict. Handles
  yfinance's inconsistent scaling (e.g. `dividendYield` is already a percent, unlike
  fraction-based fields like `profitMargins`).
- `src/valuation.py` — estimates cost of equity via CAPM from beta, and approximates the
  stock's own historical P/E range from price history against current trailing EPS
  (a simplification, clearly labeled as such, since point-in-time historical EPS isn't
  available from this data source).
- `src/earnings_quality.py` — compares YoY growth of Operating Income (core business) vs.
  Net Income (bottom line, includes non-operating items) from the annual income statement
  to flag whether earnings growth is coming from core operations or elsewhere (investment
  gains, other income, tax effects). Degrades gracefully to "unknown" when a ticker lacks
  enough income statement history.
- `src/ai_exposure.py` — scans the `yfinance` business description and recent headlines
  (`fetch_recent_news` in `src/data.py`) for AI-related terms via word-boundary regex
  matching (so short terms like "AI"/"LLM"/"GPU" don't false-positive inside unrelated
  words). Feeds the raw text + match flags to Claude rather than trying to classify AI
  relevance itself — categorization is left to the LLM's judgment in the system prompt.
- `src/dcf.py` — deterministic DCF calculator: projects free cash flow at a given growth
  rate, discounts it (default rate = CAPM cost of equity from `valuation.py`, overridable),
  adds a Gordon-growth terminal value, and derives enterprise → equity value → implied
  share price. All arithmetic happens here in Python, not in the LLM's own reasoning —
  `format_dcf_for_prompt()` returns a deliberately compact summary (assumptions + implied
  price only, no year-by-year table, per user preference).
- `src/claude_chat.py` — builds the Claude system prompt from the current
  fundamentals/technicals/valuation/earnings-quality/AI-exposure snapshot, defines `TOOLS`
  (`get_stock_analysis`, `run_dcf_scenario`), and implements the tool-use loop in
  `get_response()`: calls the API with `tools=TOOLS`, and whenever `stop_reason ==
  "tool_use"`, runs each requested tool through a caller-supplied `tool_executor`
  callback, appends the `tool_result`, and calls again (capped at
  `MAX_TOOL_ITERATIONS`) until the model returns text. Stays Streamlit-agnostic — the
  actual tool implementations (using `st.cache_data`-wrapped fetchers) live in `app.py`'s
  `execute_tool()`. Wraps SDK exceptions into `ClaudeChatError`.
- `tests/test_technical.py`, `tests/test_valuation.py`, `tests/test_earnings_quality.py`,
  `tests/test_ai_exposure.py`, `tests/test_dcf.py` — unit tests for the indicator,
  valuation, earnings-quality, AI-keyword-matching, and DCF math against known-answer
  fixtures; no network or API key required. `tests/test_claude_chat.py` verifies the
  tool-use loop's shape (executes the right tool, appends `tool_result` correctly, stops
  once no more tools are requested) against a small hand-written fake client — no live
  API calls.

Data flow: user enters a ticker → `data.py` fetches (cached 5 min) → `technical.py` /
`fundamentals.py` / `valuation.py` / `earnings_quality.py` / `ai_exposure.py` summarize →
sidebar displays the summary → each chat turn rebuilds the system prompt from the current
data snapshot and calls Claude via `claude_chat.py`, which may call back into `app.py`'s
`execute_tool()` (one or more times) to analyze another ticker or run a DCF scenario
before producing its final answer.

**Known Streamlit gotcha**: `st.markdown()` treats a pair of `$` as LaTeX math
delimiters, which garbles any chat message containing two-plus dollar amounts (very
common in this app). `app.py`'s `render_chat_text()` escapes `$` before rendering — use
it (not raw `st.markdown()`) for any new chat-message display.
