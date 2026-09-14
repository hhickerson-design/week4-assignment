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
    # yfinance returns dividendYield already as a percent (e.g. 0.33 == 0.33%),
    # unlike fields such as profitMargins which are fractions (e.g. 0.276 == 27.6%).
    return f"{v:.2f}%"


def _fmt_ratio(v) -> str:
    return f"{v:.2f}"


def _fmt_raw(v) -> str:
    return str(v)


# (display label, candidate yfinance info keys in priority order, formatter)
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
