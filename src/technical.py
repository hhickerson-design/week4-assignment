import pandas as pd


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
