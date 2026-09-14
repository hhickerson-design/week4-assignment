import pandas as pd
import yfinance as yf

from src.config import PRICE_HISTORY_PERIOD


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
    """Annual income statement (most recent fiscal years as columns). May be unavailable
    or incomplete for some tickers (e.g. certain foreign listings) — callers should treat
    an empty result as "earnings-quality breakdown unavailable" rather than a fatal error.
    """
    try:
        return yf.Ticker(ticker).income_stmt
    except Exception:
        return pd.DataFrame()


def fetch_cash_flow_statement(ticker: str) -> pd.DataFrame:
    """Annual cash flow statement (most recent fiscal years as columns). Best-effort,
    like fetch_income_statement — returns an empty DataFrame rather than raising.
    """
    try:
        return yf.Ticker(ticker).cashflow
    except Exception:
        return pd.DataFrame()


def fetch_recent_news(ticker: str, limit: int = 8) -> list[dict]:
    """Recent headlines for the ticker, normalized to {title, publisher, published, summary}.
    Best-effort: returns an empty list rather than raising if news is unavailable, since
    this is supplementary context, not core ticker data.
    """
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
