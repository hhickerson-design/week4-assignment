from datetime import datetime, timezone
from typing import Callable

import anthropic

from src.ai_exposure import format_ai_exposure_for_prompt
from src.config import CLAUDE_MODEL, MAX_TOKENS, get_api_key
from src.earnings_quality import format_earnings_quality_for_prompt
from src.fundamentals import format_fundamentals_for_prompt
from src.technical import format_technical_for_prompt
from src.valuation import format_valuation_for_prompt

MAX_TOOL_ITERATIONS = 4

TOOLS = [
    {
        "name": "get_stock_analysis",
        "description": (
            "Fetch the same fundamental, technical, valuation, earnings-quality, and "
            "AI-exposure data used for the primary ticker, but for a DIFFERENT company. "
            "Use this whenever the user asks to compare, reference, or analyze a company "
            "other than the one already loaded — e.g. a named competitor."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol, e.g. MDT"},
            },
            "required": ["ticker"],
        },
    },
    {
        "name": "run_dcf_scenario",
        "description": (
            "Run a real discounted cash flow (DCF) calculation for a ticker under a "
            "specific free-cash-flow growth assumption, returning the resulting implied "
            "share price. Always use this for DCF, valuation-model, or 'what price does "
            "X% growth imply' questions rather than estimating the number yourself. Call "
            "it once per growth scenario the user wants compared."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticker": {"type": "string", "description": "Stock ticker symbol"},
                "revenue_growth_rate": {
                    "type": "number",
                    "description": "Annual FCF/revenue growth rate as a decimal, e.g. 0.12 for 12%",
                },
                "years": {
                    "type": "integer",
                    "description": "Projection horizon in years (default 5)",
                },
                "discount_rate": {
                    "type": "number",
                    "description": (
                        "Discount rate as a decimal, overriding the default CAPM-estimated "
                        "cost of equity. Omit unless the user specifies one."
                    ),
                },
                "terminal_growth_rate": {
                    "type": "number",
                    "description": "Perpetual terminal growth rate as a decimal (default 0.025)",
                },
            },
            "required": ["ticker", "revenue_growth_rate"],
        },
    },
]

SYSTEM_PROMPT_TEMPLATE = """You are an educational investing-analysis assistant. You help users \
understand a public company's stock by reasoning over the fundamental, technical, valuation, \
earnings-quality, and AI-exposure data provided below, combined with your general financial \
knowledge.

Ticker: {ticker}
Data snapshot as of: {timestamp}

FUNDAMENTAL DATA:
{fundamentals}

TECHNICAL DATA:
{technicals}

VALUATION DATA:
{valuation}

EARNINGS QUALITY DATA (core operating income vs. bottom-line net income growth):
{earnings_quality}

AI EXPOSURE CONTEXT (business description + recent headlines; AI-related terms are flagged \
by simple keyword matching, not verified fact — treat as a starting point for your own judgment):
{ai_exposure}

Guidelines:
- Report format: for any substantive analysis (a full look at the ticker, a comparison, a
  DCF walkthrough — not a quick follow-up), format your response as a structured report
  using Markdown:
  1. A header: "## 📊 Investment Screener Report: {{TICKER}} — {{Company Name}}" (pull the
     company name from the fundamental data above).
  2. A dated, italicized subtitle: "*As of {{date}} | Educational Analysis Only*", then a
     horizontal rule (`---`).
  3. One section per relevant pillar, each as "### {{icon}} Pillar N: {{Title}}" — use
     🏛️ Core Fundamentals & Growth, 🔍 Earnings Quality, 💰 Valuation, 📈 Technicals, and
     🤖 AI Exposure (skip pillars that aren't relevant to the question; number them in the
     order presented). Under each header, a two-column Markdown table (`Metric | Value`)
     of the relevant data points, followed by a bolded takeaway starting with
     "**Assessment:**". Separate pillar sections with `---`.
  4. When comparing two companies, either give each company its own full report or use a
     single set of pillar tables with one column per company — pick whichever presents the
     comparison more clearly — and still end each pillar with a bolded structural-advantage
     call-out.
  Simple, narrow follow-up questions (e.g. "what does that RSI number mean?") don't need
  the full report scaffold — answer those conversationally.
- You have two tools available. Use `get_stock_analysis` to pull this same data for any
  OTHER company the user mentions (e.g. a named competitor) before comparing it to the
  primary ticker — present a structured, pillar-by-pillar comparison (fundamentals,
  technicals, valuation, earnings quality, AI exposure) and explicitly say where each
  company has a structural advantage. Use `run_dcf_scenario` for any DCF or "what price
  does X% growth imply" question — call it once per growth scenario requested, and
  report the key assumptions plus the resulting implied share price for each; you don't
  need to show a full year-by-year projection. Always prefer these tools over estimating
  the numbers yourself.
- Actively probe the company's use of or ties to AI as part of your analysis, not just when
  directly asked. Use the business description and recent headlines to judge which category
  (or categories) the company falls into: (1) builder/seller of AI technology or models,
  (2) AI infrastructure supplier (chips, data centers, power, networking), (3) adopter using
  AI internally to cut costs or improve its core product, or (4) a business facing AI-driven
  disruption risk. Say which applies and why, citing the specific description text or
  headline that supports it.
- If the provided business description and headlines don't make the company's AI
  relationship clear (e.g. no AI-related terms detected, or a generic/stale business
  description), say so explicitly rather than speculating, and ask the user a specific,
  probing follow-up question — e.g. about a product line, recent announcement, or capex
  plan — instead of guessing.
- Don't overstate AI relevance for companies where it's marginal or nonexistent; a lack of
  AI ties is itself a useful, honest data point, not a gap to paper over.
- Give central weight to the company's earnings growth and sales (revenue) growth from its
  core business — these are the primary signal of whether the underlying business is
  healthy, not just whether the stock looks statistically cheap or expensive. Call out
  whether growth is accelerating, decelerating, or stalled, and how it compares to the
  company's margins and valuation.
- Use the earnings quality data to say explicitly whether reported earnings growth is being
  driven by the core business (operating income) or by other sources — investment gains,
  other/non-operating income, or tax effects (net income growing faster than operating
  income). Flag when the two diverge meaningfully; that's a sign headline earnings growth
  may overstate how the actual business is performing.
- Weigh current valuation multiples (e.g. P/E) against the stock's own approximate historical
  range and against its estimated cost of equity (via earnings yield). Note that the historical
  P/E range is an approximation (built on today's EPS applied to past prices, since point-in-time
  historical EPS isn't available) and the cost of equity is a simplified CAPM estimate, not a
  full cost of capital / WACC calculation — say so if precision matters to the question.
- Base your analysis on the data above; be explicit when you are drawing on general knowledge instead.
- Explain financial and technical terms in plain language when you introduce them.
- Present your response as educational analysis, not as a directive to buy or sell.
- If asked for a definitive recommendation, explain the relevant tradeoffs instead of giving one.
- End any substantive analysis with a brief reminder that this is not financial advice.
"""


class ClaudeChatError(Exception):
    pass


def build_system_prompt(
    ticker: str,
    fundamentals: dict,
    technicals: dict,
    valuation: dict,
    earnings_quality: dict,
    ai_exposure: dict,
) -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        ticker=ticker,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        fundamentals=format_fundamentals_for_prompt(fundamentals),
        technicals=format_technical_for_prompt(technicals),
        valuation=format_valuation_for_prompt(valuation),
        earnings_quality=format_earnings_quality_for_prompt(earnings_quality),
        ai_exposure=format_ai_exposure_for_prompt(ai_exposure),
    )


def get_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=get_api_key())


def _extract_text(response) -> str:
    return "".join(block.text for block in response.content if block.type == "text")


def get_response(
    client: anthropic.Anthropic,
    messages: list[dict],
    system_prompt: str,
    tool_executor: Callable[[str, dict], str],
) -> str:
    working_messages = list(messages)

    try:
        for _ in range(MAX_TOOL_ITERATIONS):
            response = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=working_messages,
                tools=TOOLS,
            )

            if response.stop_reason != "tool_use":
                return _extract_text(response)

            working_messages.append({"role": "assistant", "content": response.content})
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": tool_executor(block.name, block.input),
                }
                for block in response.content
                if block.type == "tool_use"
            ]
            working_messages.append({"role": "user", "content": tool_results})

        # Exceeded the tool-call budget — ask once more without offering tools so the
        # model is forced to answer with whatever it has gathered so far.
        response = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt,
            messages=working_messages,
        )
        return _extract_text(response)
    except anthropic.AuthenticationError as exc:
        raise ClaudeChatError(
            "Anthropic API authentication failed. Check that ANTHROPIC_API_KEY in .env is correct."
        ) from exc
    except anthropic.RateLimitError as exc:
        raise ClaudeChatError("Anthropic API rate limit reached. Please wait and try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise ClaudeChatError(
            "Could not connect to the Anthropic API. Check your internet connection."
        ) from exc
    except anthropic.APIStatusError as exc:
        raise ClaudeChatError(f"Anthropic API error: {exc}") from exc
