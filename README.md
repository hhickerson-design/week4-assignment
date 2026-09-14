# Stock Analysis Chatbot

A chatbot for exploring public equities that combines **fundamental analysis** (company
financials like P/E ratio, market cap, margins, revenue/earnings growth), **technical
analysis** (price-chart indicators like moving averages, RSI, MACD, Bollinger Bands), and
**valuation analysis** (current multiples vs. the stock's own historical range and vs. an
estimated cost of equity), an **earnings quality** check (whether earnings growth is
coming from the core business vs. non-operating items like investment gains or tax
effects), and an **AI exposure** check (whether and how the company is tied to AI — as a
builder/seller of AI tech, an AI infrastructure supplier, an internal adopter, or facing
AI-driven disruption risk), powered by the Anthropic Claude API and live market data from
Yahoo Finance. The chatbot is instructed to weigh a company's earnings and sales growth
from its core business as the central signal of business health, and to probe AI ties
using the company's own business description and recent headlines rather than assuming
them.

The chatbot can also, mid-conversation, **pull the same analysis for a second company**
(e.g. "compare this to its closest competitor") and **run a real DCF (discounted cash
flow) valuation** under a specified growth assumption ("what price does 12% growth
imply?") — both via Claude API tool use, so the numbers are computed exactly in Python
rather than estimated by the model.

For substantive questions, responses are formatted as a structured "Investment Screener
Report" — a titled header, then one section per relevant pillar (🏛️ Fundamentals,
🔍 Earnings Quality, 💰 Valuation, 📈 Technicals, 🤖 AI Exposure), each with a metric
table and a bolded **Assessment** takeaway.

> **Educational tool only. Not financial advice.** Data may be delayed, incomplete, or
> inaccurate. Always verify independently before making any investment decision.

## Prerequisites

- **Python 3.10+**. On Windows, running `python` may launch a "Install from Microsoft
  Store" prompt instead of a real interpreter — if so, install Python from
  [python.org](https://www.python.org/downloads/) (check "Add python.exe to PATH" during
  setup), or disable the Store shortcut under
  *Settings > Apps > Advanced app settings > App execution aliases*.
- An **Anthropic API key** — create one at [console.anthropic.com](https://console.anthropic.com/).

## Setup

```bash
# from the project root
python -m venv .venv
.venv\Scripts\activate        # on Windows
pip install -r requirements.txt

copy .env.example .env        # on Windows (use `cp` on macOS/Linux)
# then edit .env and set ANTHROPIC_API_KEY=your-actual-key
```

## Run

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).

## How to use

1. Enter a stock ticker (e.g. `AAPL`, `MSFT`, `TSLA`) in the sidebar and click **Load / Refresh**.
2. Review the fundamental and technical metrics that appear in the sidebar.
3. Ask questions in the chat box — e.g. "How does this company's valuation compare to its
   sector?" or "What does the current RSI suggest?" Claude answers using the data shown in
   the sidebar plus its general financial knowledge.
4. Ask it to compare the loaded ticker to any other company by name (e.g. "compare this
   to Medtronic (MDT)") — it will fetch and analyze that company too.
5. Ask a DCF/valuation-model question (e.g. "what implied share price does 12% growth
   imply over 5 years?") — it will run a real discounted cash flow calculation, not
   estimate the number itself.
6. Click **Clear conversation** to reset the chat while keeping the loaded ticker.

## Project structure

```
app.py                  Streamlit entrypoint — UI and orchestration
src/
  config.py              Environment/config loading, API key check, shared constants
  data.py                 Live data fetching from Yahoo Finance (yfinance)
  technical.py            Hand-rolled technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
  fundamentals.py         Fundamental metric extraction and formatting
  valuation.py             Historical P/E range and CAPM-based cost of equity estimate
  earnings_quality.py      Core operating income growth vs. bottom-line net income growth
  ai_exposure.py           Detects AI-related terms in business description and headlines
  dcf.py                    Discounted cash flow calculator (projection, terminal value, implied price)
  claude_chat.py           Claude API wrapper, tool-use loop, and prompt construction
tests/
  test_technical.py       Unit tests for the technical indicator math
  test_valuation.py        Unit tests for the valuation calculations
  test_earnings_quality.py Unit tests for the earnings quality calculations
  test_ai_exposure.py      Unit tests for the AI keyword detection
  test_dcf.py               Unit tests for the DCF math and edge cases
  test_claude_chat.py       Unit tests for the tool-use loop (no live API calls)
```

## Multi-company comparison and DCF (tool use)

The chatbot has two tools it can call mid-conversation via the Anthropic API's function
calling:

- **`get_stock_analysis`** — fetches the same 5-pillar analysis (fundamentals,
  technicals, valuation, earnings quality, AI exposure) for any company the user
  mentions besides the one loaded in the sidebar, so it can produce real side-by-side
  comparisons.
- **`run_dcf_scenario`** — runs an actual DCF calculation (`src/dcf.py`) for a given
  ticker and FCF growth rate, returning the exact resulting implied share price. The
  discount rate defaults to a CAPM-estimated cost of equity but can be overridden.

Both tools return plain-text results that Claude incorporates into its answer; the
underlying arithmetic always happens in Python, not in the model's own reasoning.

## Running tests

```bash
pytest -v
```

Tests cover the technical indicator calculations against known-answer fixtures and do not
require network access or an API key.

## Single-file build (for BoddleBox)

`boddlebox_app.py` is a combined, self-contained version of `app.py` + all of `src/`, for
platforms (like BoddleBox) that need one pasteable Python file instead of a project
folder. It behaves identically to the modular app. `app.py`/`src/` remain the source of
truth — if you change those, regenerate `boddlebox_app.py` to match.

## Known limitations

- Single active ticker at a time — loading a new ticker resets the conversation.
- Yahoo Finance data (via `yfinance`) is free but can be delayed, occasionally incomplete
  for a given ticker, or rate-limited if refreshed too frequently.
- Not a real-time trading tool and not a substitute for professional financial advice.
