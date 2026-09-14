# Single-file combined build of this project (app.py + src/*.py) for pasting directly
# into the BoddleBox chatbot builder. The modular version in app.py / src/ is the
# canonical source — regenerate this file if those change.
import os
import re
from datetime import datetime, timezone
from typing import Callable

import anthropic
import pandas as pd
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-5"
MAX_TOKENS = 2500
PRICE_HISTORY_PERIOD = "5y"
CACHE_TTL_SECONDS = 300

RISK_FREE_RATE = 0.045
EQUITY_RISK_PREMIUM = 0.05


class MissingAPIKeyError(Exception):
    pass


def load_env() -> None:
    load_dotenv()


def get_api_key() -> str:
    load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


class TickerDataError(Exception):
    pass


def fetch_price_history(ticker: str) -> pd.DataFrame:
    try:
        history = yf.Ticker(ticker).history(period=PRICE_HISTORY_PERIOD)
    except Exception as exc:
        raise TickerDataError(f"Could not fetch price history for '{ticker}': {exc}") from exc

    if history.empty:
        raise TickerDataError(f"No price history found for '{ticker}'. Check the ticker symbol.")

    return history


def fetch_company_info(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
    except Exception as exc:
        raise TickerDataError(f"Could not fetch company info for '{ticker}': {exc}") from exc

    if not info or not (info.get("longName") or info.get("shortName")):
        raise TickerDataError(f"No company info found for '{ticker}'. Check the ticker symbol.")

    return info


def fetch_income_statement(ticker: str) -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).income_stmt
    except Exception:
        return pd.DataFrame()


def fetch_cash_flow_statement(ticker: str) -> pd.DataFrame:
    try:
        return yf.Ticker(ticker).cashflow
    except Exception:
        return pd.DataFrame()


def fetch_recent_news(ticker: str, limit: int = 8) -> list[dict]:
    try:
        raw_items = yf.Ticker(ticker).news or []
    except Exception:
        return []

    normalized = []
    for item in raw_items[:limit]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = content.get("title")
        if not title:
            continue
        provider = content.get("provider") or {}
        normalized.append(
            {
                "title": title,
                "publisher": provider.get("displayName") or content.get("publisher") or "unknown source",
                "published": content.get("pubDate") or content.get("providerPublishTime") or "",
                "summary": content.get("summary") or content.get("description") or "",
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# technical
# ---------------------------------------------------------------------------


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: int = 2
) -> tuple[pd.Series, pd.Series, pd.Series]:
    middle = sma(series, window)
    std = series.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _latest(series: pd.Series):
    valid = series.dropna()
    if valid.empty:
        return None
    return float(valid.iloc[-1])


def _rsi_signal(latest_rsi):
    if latest_rsi is None:
        return "unknown"
    if latest_rsi >= 70:
        return "overbought"
    if latest_rsi <= 30:
        return "oversold"
    return "neutral"


def _macd_signal(macd_line: pd.Series, signal_line: pd.Series) -> str:
    diff = (macd_line - signal_line).dropna()
    if len(diff) < 2:
        return "neutral"
    prev, curr = diff.iloc[-2], diff.iloc[-1]
    if prev <= 0 and curr > 0:
        return "bullish crossover"
    if prev >= 0 and curr < 0:
        return "bearish crossover"
    if curr > 0:
        return "bullish"
    if curr < 0:
        return "bearish"
    return "neutral"


def _price_vs_sma(latest_close, latest_sma):
    if latest_close is None or latest_sma is None:
        return "unknown"
    return "above" if latest_close > latest_sma else "below"


def summarize_technical(df: pd.DataFrame) -> dict:
    close = df["Close"]

    sma20, sma50, sma200 = sma(close, 20), sma(close, 50), sma(close, 200)
    rsi14 = rsi(close, 14)
    macd_line, signal_line, histogram = macd(close)
    bb_upper, bb_middle, bb_lower = bollinger_bands(close)

    latest_close = _latest(close)
    latest_rsi = _latest(rsi14)
    latest_sma50 = _latest(sma50)

    return {
        "latest_close": latest_close,
        "sma20": _latest(sma20),
        "sma50": latest_sma50,
        "sma200": _latest(sma200),
        "rsi14": latest_rsi,
        "rsi_signal": _rsi_signal(latest_rsi),
        "macd_line": _latest(macd_line),
        "macd_signal_line": _latest(signal_line),
        "macd_histogram": _latest(histogram),
        "macd_signal": _macd_signal(macd_line, signal_line),
        "bb_upper": _latest(bb_upper),
        "bb_middle": _latest(bb_middle),
        "bb_lower": _latest(bb_lower),
        "price_vs_sma50": _price_vs_sma(latest_close, latest_sma50),
    }


def format_technical_for_prompt(tech: dict) -> str:
    def fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else str(v)

    lines = [
        f"Latest Close: {fmt(tech['latest_close'])}",
        f"SMA20: {fmt(tech['sma20'])} | SMA50: {fmt(tech['sma50'])} | SMA200: {fmt(tech['sma200'])}",
        f"Price vs SMA50: {tech['price_vs_sma50']}",
        f"RSI(14): {fmt(tech['rsi14'])} ({tech['rsi_signal']})",
        f"MACD: {fmt(tech['macd_line'])} | Signal: {fmt(tech['macd_signal_line'])} "
        f"| Histogram: {fmt(tech['macd_histogram'])} ({tech['macd_signal']})",
        f"Bollinger Bands: upper {fmt(tech['bb_upper'])} | middle {fmt(tech['bb_middle'])} "
        f"| lower {fmt(tech['bb_lower'])}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# fundamentals
# ---------------------------------------------------------------------------


def _fmt_currency(v) -> str:
    return f"${v:,.2f}"


def _fmt_large_currency(v) -> str:
    if v >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if v >= 1e6:
        return f"${v / 1e6:,.2f}M"
    return f"${v:,.0f}"


def _fmt_percent(v) -> str:
    return f"{v * 100:.2f}%"


def _fmt_percent_already_scaled(v) -> str:
    return f"{v:.2f}%"


def _fmt_ratio(v) -> str:
    return f"{v:.2f}"


def _fmt_raw(v) -> str:
    return str(v)


FUNDAMENTAL_FIELDS = [
    ("Company", ["longName", "shortName"], _fmt_raw),
    ("Sector", ["sector"], _fmt_raw),
    ("Industry", ["industry"], _fmt_raw),
    ("Current Price", ["currentPrice", "regularMarketPrice"], _fmt_currency),
    ("Market Cap", ["marketCap"], _fmt_large_currency),
    ("Revenue Growth (YoY)", ["revenueGrowth"], _fmt_percent),
    ("Earnings Growth (YoY)", ["earningsGrowth"], _fmt_percent),
    ("Operating Margin", ["operatingMargins"], _fmt_percent),
    ("Trailing P/E", ["trailingPE"], _fmt_ratio),
    ("Forward P/E", ["forwardPE"], _fmt_ratio),
    ("EPS (TTM)", ["trailingEps"], _fmt_currency),
    ("Price/Book", ["priceToBook"], _fmt_ratio),
    ("Profit Margin", ["profitMargins"], _fmt_percent),
    ("Debt/Equity", ["debtToEquity"], _fmt_ratio),
    ("Dividend Yield", ["dividendYield"], _fmt_percent_already_scaled),
    ("Revenue (TTM)", ["totalRevenue"], _fmt_large_currency),
    ("Beta", ["beta"], _fmt_ratio),
    ("52-Week High", ["fiftyTwoWeekHigh"], _fmt_currency),
    ("52-Week Low", ["fiftyTwoWeekLow"], _fmt_currency),
]


def extract_fundamentals(info: dict) -> dict:
    result = {}
    for label, keys, formatter in FUNDAMENTAL_FIELDS:
        value = None
        for key in keys:
            value = info.get(key)
            if value is not None:
                break

        if value is None:
            result[label] = "N/A"
            continue

        try:
            result[label] = formatter(value)
        except (TypeError, ValueError):
            result[label] = str(value)

    return result


def format_fundamentals_for_prompt(fund: dict) -> str:
    return "\n".join(f"{label}: {value}" for label, value in fund.items())


# ---------------------------------------------------------------------------
# valuation
# ---------------------------------------------------------------------------


def estimate_cost_of_equity(beta: float) -> float:
    return RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM


def historical_pe_stats(price_history: pd.DataFrame, trailing_eps: float) -> dict:
    if trailing_eps is None or trailing_eps <= 0 or price_history.empty:
        return {"pe_5y_min": None, "pe_5y_median": None, "pe_5y_max": None}

    implied_pe = price_history["Close"] / trailing_eps
    return {
        "pe_5y_min": float(implied_pe.min()),
        "pe_5y_median": float(implied_pe.median()),
        "pe_5y_max": float(implied_pe.max()),
    }


def _pe_vs_history(current_pe, median_pe) -> str:
    if current_pe is None or median_pe is None:
        return "unknown"
    if current_pe > median_pe * 1.1:
        return "above its own historical range"
    if current_pe < median_pe * 0.9:
        return "below its own historical range"
    return "in line with its own historical range"


def _earnings_yield_vs_cost_of_equity(earnings_yield, cost_of_equity) -> str:
    if earnings_yield is None or cost_of_equity is None:
        return "unknown"
    if earnings_yield > cost_of_equity:
        return "earnings yield exceeds estimated cost of equity"
    return "earnings yield is below estimated cost of equity"


def summarize_valuation(info: dict, price_history: pd.DataFrame) -> dict:
    trailing_pe = info.get("trailingPE")
    trailing_eps = info.get("trailingEps")
    beta = info.get("beta")

    cost_of_equity = estimate_cost_of_equity(beta) if beta is not None else None
    earnings_yield = (1 / trailing_pe) if trailing_pe else None
    pe_history = historical_pe_stats(price_history, trailing_eps)

    return {
        "trailing_pe": trailing_pe,
        "earnings_yield": earnings_yield,
        "cost_of_equity": cost_of_equity,
        "earnings_yield_signal": _earnings_yield_vs_cost_of_equity(earnings_yield, cost_of_equity),
        "pe_5y_min": pe_history["pe_5y_min"],
        "pe_5y_median": pe_history["pe_5y_median"],
        "pe_5y_max": pe_history["pe_5y_max"],
        "pe_vs_history_signal": _pe_vs_history(trailing_pe, pe_history["pe_5y_median"]),
    }


def format_valuation_for_prompt(val: dict) -> str:
    def fmt_pct(v):
        return f"{v * 100:.2f}%" if v is not None else "N/A"

    def fmt_ratio(v):
        return f"{v:.2f}" if v is not None else "N/A"

    lines = [
        f"Trailing P/E: {fmt_ratio(val['trailing_pe'])}",
        f"Approx. 5-year P/E range (current-EPS basis): "
        f"{fmt_ratio(val['pe_5y_min'])} - {fmt_ratio(val['pe_5y_median'])} (median) - {fmt_ratio(val['pe_5y_max'])}",
        f"Current P/E vs own history: {val['pe_vs_history_signal']}",
        f"Earnings Yield (1/PE): {fmt_pct(val['earnings_yield'])}",
        f"Estimated Cost of Equity (CAPM, approx.): {fmt_pct(val['cost_of_equity'])}",
        f"Earnings yield vs cost of equity: {val['earnings_yield_signal']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# earnings_quality
# ---------------------------------------------------------------------------

DIVERGENCE_THRESHOLD = 0.05


def _income_stmt_get(income_stmt: pd.DataFrame, row: str, col) -> float | None:
    if row not in income_stmt.index:
        return None
    value = income_stmt.loc[row, col]
    return None if pd.isna(value) else float(value)


def _pct_growth(latest, prior):
    if latest is None or prior is None or prior == 0:
        return None
    return (latest - prior) / abs(prior)


def _earnings_quality_signal(operating_income_growth, net_income_growth) -> str:
    if operating_income_growth is None or net_income_growth is None:
        return "unknown (insufficient historical income statement data)"

    gap = net_income_growth - operating_income_growth
    if gap > DIVERGENCE_THRESHOLD:
        return (
            "net income growth outpaces operating income growth — suggests some earnings "
            "growth is coming from non-operating items (investment gains, other income, "
            "or tax effects) rather than the core business alone"
        )
    if gap < -DIVERGENCE_THRESHOLD:
        return (
            "operating income growth outpaces net income growth — core operations are "
            "growing faster than the bottom line, likely due to taxes, interest expense, "
            "or other non-operating charges"
        )
    return (
        "net income growth is closely aligned with operating income growth — earnings "
        "growth appears driven primarily by core operations"
    )


def summarize_earnings_quality(income_stmt: pd.DataFrame) -> dict:
    if income_stmt is None or income_stmt.empty or income_stmt.shape[1] < 2:
        return {
            "operating_income_growth": None,
            "net_income_growth": None,
            "other_income_pct_of_pretax": None,
            "earnings_quality_signal": "unknown (insufficient historical income statement data)",
            "latest_fiscal_year": None,
            "prior_fiscal_year": None,
        }

    columns = sorted(income_stmt.columns, reverse=True)
    latest_col, prior_col = columns[0], columns[1]

    operating_income_latest = _income_stmt_get(income_stmt, "Operating Income", latest_col)
    operating_income_prior = _income_stmt_get(income_stmt, "Operating Income", prior_col)
    net_income_latest = _income_stmt_get(income_stmt, "Net Income", latest_col)
    net_income_prior = _income_stmt_get(income_stmt, "Net Income", prior_col)
    other_income_latest = _income_stmt_get(income_stmt, "Other Income Expense", latest_col)
    pretax_income_latest = _income_stmt_get(income_stmt, "Pretax Income", latest_col)

    operating_income_growth = _pct_growth(operating_income_latest, operating_income_prior)
    net_income_growth = _pct_growth(net_income_latest, net_income_prior)

    other_income_pct_of_pretax = None
    if other_income_latest is not None and pretax_income_latest not in (None, 0):
        other_income_pct_of_pretax = other_income_latest / abs(pretax_income_latest)

    return {
        "operating_income_growth": operating_income_growth,
        "net_income_growth": net_income_growth,
        "other_income_pct_of_pretax": other_income_pct_of_pretax,
        "earnings_quality_signal": _earnings_quality_signal(
            operating_income_growth, net_income_growth
        ),
        "latest_fiscal_year": latest_col,
        "prior_fiscal_year": prior_col,
    }


def format_earnings_quality_for_prompt(eq: dict) -> str:
    def fmt_pct(v):
        return f"{v * 100:.2f}%" if v is not None else "N/A"

    def fmt_year(v):
        return v.strftime("%Y") if v is not None else "N/A"

    lines = [
        f"Fiscal years compared: {fmt_year(eq['prior_fiscal_year'])} -> {fmt_year(eq['latest_fiscal_year'])}",
        f"Operating Income growth (core business): {fmt_pct(eq['operating_income_growth'])}",
        f"Net Income growth (bottom line, incl. non-operating items): {fmt_pct(eq['net_income_growth'])}",
        f"Other/non-operating income as % of pretax income (latest year): {fmt_pct(eq['other_income_pct_of_pretax'])}",
        f"Earnings quality signal: {eq['earnings_quality_signal']}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# ai_exposure
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# dcf
# ---------------------------------------------------------------------------


class DCFError(Exception):
    pass


def compute_dcf(
    base_fcf: float,
    shares_outstanding: float,
    net_debt: float,
    growth_rate: float,
    discount_rate: float,
    years: int = 5,
    terminal_growth_rate: float = 0.025,
) -> dict:
    if base_fcf <= 0:
        raise DCFError("Base free cash flow must be positive to run a DCF.")
    if not shares_outstanding or shares_outstanding <= 0:
        raise DCFError("Shares outstanding must be a positive number to run a DCF.")
    if discount_rate <= terminal_growth_rate:
        raise DCFError(
            f"Discount rate ({discount_rate:.2%}) must exceed the terminal growth rate "
            f"({terminal_growth_rate:.2%}) for the terminal value formula to be valid."
        )

    yearly = []
    for year in range(1, years + 1):
        projected_fcf = base_fcf * (1 + growth_rate) ** year
        discounted_fcf = projected_fcf / (1 + discount_rate) ** year
        yearly.append({"year": year, "projected_fcf": projected_fcf, "discounted_fcf": discounted_fcf})

    terminal_value = (
        yearly[-1]["projected_fcf"] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    )
    discounted_terminal_value = terminal_value / (1 + discount_rate) ** years

    enterprise_value = sum(y["discounted_fcf"] for y in yearly) + discounted_terminal_value
    equity_value = enterprise_value - net_debt
    implied_share_price = equity_value / shares_outstanding

    return {
        "yearly": yearly,
        "terminal_value": terminal_value,
        "discounted_terminal_value": discounted_terminal_value,
        "enterprise_value": enterprise_value,
        "net_debt": net_debt,
        "equity_value": equity_value,
        "shares_outstanding": shares_outstanding,
        "implied_share_price": implied_share_price,
        "assumptions": {
            "base_fcf": base_fcf,
            "growth_rate": growth_rate,
            "discount_rate": discount_rate,
            "years": years,
            "terminal_growth_rate": terminal_growth_rate,
        },
    }


def get_dcf_inputs(info: dict, cash_flow: pd.DataFrame) -> dict:
    base_fcf = None
    if cash_flow is not None and not cash_flow.empty and "Free Cash Flow" in cash_flow.index:
        latest_col = sorted(cash_flow.columns, reverse=True)[0]
        value = cash_flow.loc["Free Cash Flow", latest_col]
        if pd.notna(value):
            base_fcf = float(value)
    if base_fcf is None:
        base_fcf = info.get("freeCashflow")

    total_debt = info.get("totalDebt") or 0
    total_cash = info.get("totalCash") or 0
    beta = info.get("beta")

    return {
        "base_fcf": base_fcf,
        "net_debt": total_debt - total_cash,
        "shares_outstanding": info.get("sharesOutstanding"),
        "default_discount_rate": estimate_cost_of_equity(beta) if beta is not None else None,
    }


def run_dcf_scenario(
    info: dict,
    cash_flow: pd.DataFrame,
    growth_rate: float,
    years: int = 5,
    discount_rate: float | None = None,
    terminal_growth_rate: float = 0.025,
) -> dict:
    inputs = get_dcf_inputs(info, cash_flow)

    if inputs["base_fcf"] is None:
        raise DCFError("No free cash flow data available for this ticker; cannot run a DCF.")
    if inputs["shares_outstanding"] is None:
        raise DCFError("Shares outstanding data is unavailable for this ticker; cannot run a DCF.")

    resolved_discount_rate = discount_rate if discount_rate is not None else inputs["default_discount_rate"]
    if resolved_discount_rate is None:
        raise DCFError(
            "No discount rate could be determined (beta is missing) and none was provided."
        )

    result = compute_dcf(
        base_fcf=inputs["base_fcf"],
        shares_outstanding=inputs["shares_outstanding"],
        net_debt=inputs["net_debt"],
        growth_rate=growth_rate,
        discount_rate=resolved_discount_rate,
        years=years,
        terminal_growth_rate=terminal_growth_rate,
    )
    result["discount_rate_source"] = (
        "user-specified" if discount_rate is not None else "CAPM cost of equity (approx.)"
    )
    return result


def format_dcf_for_prompt(result: dict) -> str:
    a = result["assumptions"]
    lines = [
        f"Assumptions: {a['growth_rate']:.2%} annual FCF growth, {a['discount_rate']:.2%} discount rate "
        f"({result['discount_rate_source']}), {a['years']}-year horizon, "
        f"{a['terminal_growth_rate']:.2%} terminal growth",
        f"Base Free Cash Flow: ${a['base_fcf']:,.0f}",
        f"Enterprise Value: ${result['enterprise_value']:,.0f} "
        f"(net debt: ${result['net_debt']:,.0f}, equity value: ${result['equity_value']:,.0f})",
        f"Implied Share Price: ${result['implied_share_price']:,.2f}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# claude_chat
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# app (Streamlit UI)
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Stock Analysis Chatbot", page_icon="\U0001F4C8")

load_env()
try:
    get_api_key()
except MissingAPIKeyError as exc:
    st.error(str(exc))
    st.stop()

st.title("Stock Analysis Chatbot")
st.warning(
    "Educational tool only. This is not financial advice. "
    "Data may be delayed, incomplete, or inaccurate — verify independently before making decisions."
)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_price_history(ticker: str):
    return fetch_price_history(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_company_info(ticker: str):
    return fetch_company_info(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_income_statement(ticker: str):
    return fetch_income_statement(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_recent_news(ticker: str):
    return fetch_recent_news(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_cash_flow_statement(ticker: str):
    return fetch_cash_flow_statement(ticker)


def execute_tool(name: str, tool_input: dict) -> str:
    ticker = tool_input.get("ticker", "").strip().upper()

    if name == "get_stock_analysis":
        try:
            history = load_price_history(ticker)
            info = load_company_info(ticker)
            fundamentals = extract_fundamentals(info)
            technicals = summarize_technical(history)
            valuation = summarize_valuation(info, history)
            earnings_quality = summarize_earnings_quality(load_income_statement(ticker))
            ai_exposure = summarize_ai_exposure(info, load_recent_news(ticker))
        except TickerDataError as exc:
            return f"Error: {exc}"

        return (
            f"Ticker: {ticker}\n\n"
            f"FUNDAMENTAL DATA:\n{format_fundamentals_for_prompt(fundamentals)}\n\n"
            f"TECHNICAL DATA:\n{format_technical_for_prompt(technicals)}\n\n"
            f"VALUATION DATA:\n{format_valuation_for_prompt(valuation)}\n\n"
            f"EARNINGS QUALITY DATA:\n{format_earnings_quality_for_prompt(earnings_quality)}\n\n"
            f"AI EXPOSURE DATA:\n{format_ai_exposure_for_prompt(ai_exposure)}"
        )

    if name == "run_dcf_scenario":
        try:
            info = load_company_info(ticker)
            cash_flow = load_cash_flow_statement(ticker)
            result = run_dcf_scenario(
                info,
                cash_flow,
                growth_rate=tool_input["revenue_growth_rate"],
                years=tool_input.get("years", 5),
                discount_rate=tool_input.get("discount_rate"),
                terminal_growth_rate=tool_input.get("terminal_growth_rate", 0.025),
            )
        except (TickerDataError, DCFError) as exc:
            return f"Error: {exc}"

        return f"Ticker: {ticker}\n{format_dcf_for_prompt(result)}"

    return f"Error: unknown tool '{name}'"


if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
    st.session_state.current_fundamentals = None
    st.session_state.current_technicals = None
    st.session_state.current_valuation = None
    st.session_state.current_earnings_quality = None
    st.session_state.current_ai_exposure = None

with st.sidebar:
    st.header("Ticker")
    ticker_input = st.text_input("Symbol (e.g. AAPL)", value="").strip().upper()
    load_clicked = st.button("Load / Refresh")

    if load_clicked and ticker_input:
        try:
            history = load_price_history(ticker_input)
            info = load_company_info(ticker_input)
            st.session_state.current_ticker = ticker_input
            st.session_state.current_fundamentals = extract_fundamentals(info)
            st.session_state.current_technicals = summarize_technical(history)
            st.session_state.current_valuation = summarize_valuation(info, history)
            income_stmt = load_income_statement(ticker_input)
            st.session_state.current_earnings_quality = summarize_earnings_quality(income_stmt)
            news = load_recent_news(ticker_input)
            st.session_state.current_ai_exposure = summarize_ai_exposure(info, news)
            st.session_state.messages = []
        except TickerDataError as exc:
            st.error(str(exc))

    if st.session_state.current_ticker:
        st.subheader(f"Data: {st.session_state.current_ticker}")

        st.caption("Fundamentals")
        for label, value in st.session_state.current_fundamentals.items():
            st.metric(label, value)

        st.caption("Technicals")
        tech = st.session_state.current_technicals
        st.metric("RSI(14)", f"{tech['rsi14']:.1f}" if tech["rsi14"] is not None else "N/A", tech["rsi_signal"])
        st.metric("MACD signal", tech["macd_signal"])
        st.metric("Price vs SMA50", tech["price_vs_sma50"])

        st.caption("Valuation")
        val = st.session_state.current_valuation
        pe_display = f"{val['trailing_pe']:.2f}" if val["trailing_pe"] is not None else "N/A"
        st.metric("Trailing P/E", pe_display, val["pe_vs_history_signal"])
        coe_display = f"{val['cost_of_equity'] * 100:.2f}%" if val["cost_of_equity"] is not None else "N/A"
        st.metric("Cost of Equity (CAPM, approx.)", coe_display, val["earnings_yield_signal"])

        st.caption("Earnings Quality (core vs. other)")
        eq = st.session_state.current_earnings_quality
        op_display = (
            f"{eq['operating_income_growth'] * 100:.2f}%"
            if eq["operating_income_growth"] is not None
            else "N/A"
        )
        net_display = (
            f"{eq['net_income_growth'] * 100:.2f}%" if eq["net_income_growth"] is not None else "N/A"
        )
        st.metric("Operating Income Growth", op_display)
        st.metric("Net Income Growth", net_display)
        st.caption(eq["earnings_quality_signal"])

        st.caption("AI Exposure")
        ai = st.session_state.current_ai_exposure
        st.metric("AI mentioned in business description", "Yes" if ai["ai_mentioned_in_summary"] else "No")
        st.metric("AI-related headlines", f"{ai['ai_related_news_count']}/{ai['total_news_count']}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()


def render_chat_text(text: str) -> None:
    # st.markdown treats a pair of "$" as LaTeX math delimiters, which garbles any
    # message containing two or more dollar amounts (common in this app).
    st.markdown(text.replace("$", "\\$"))


if not st.session_state.current_ticker:
    st.info("Enter a ticker symbol in the sidebar and click **Load / Refresh** to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_chat_text(message["content"])

    user_input = st.chat_input(f"Ask about {st.session_state.current_ticker}...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            render_chat_text(user_input)

        with st.chat_message("assistant"):
            try:
                system_prompt = build_system_prompt(
                    st.session_state.current_ticker,
                    st.session_state.current_fundamentals,
                    st.session_state.current_technicals,
                    st.session_state.current_valuation,
                    st.session_state.current_earnings_quality,
                    st.session_state.current_ai_exposure,
                )
                client = get_client()
                with st.spinner("Analyzing..."):
                    reply = get_response(
                        client, st.session_state.messages, system_prompt, execute_tool
                    )
                render_chat_text(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except ClaudeChatError as exc:
                st.error(str(exc))
