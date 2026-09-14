import pandas as pd

from src.config import EQUITY_RISK_PREMIUM, RISK_FREE_RATE


def estimate_cost_of_equity(beta: float) -> float:
    """CAPM estimate: cost of equity = risk-free rate + beta * equity risk premium.

    A simplified proxy for cost of capital, not a full WACC (which would also
    require cost of debt, tax rate, and capital structure).
    """
    return RISK_FREE_RATE + beta * EQUITY_RISK_PREMIUM


def historical_pe_stats(price_history: pd.DataFrame, trailing_eps: float) -> dict:
    """Approximate historical P/E range by dividing historical close prices by the
    *current* trailing EPS. This holds EPS constant across history (real point-in-time
    EPS at each past date isn't available from this data source), so it approximates
    how the stock's own price has moved relative to today's earnings level, not a
    precise historical P/E series.
    """
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
