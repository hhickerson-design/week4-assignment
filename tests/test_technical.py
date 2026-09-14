import numpy as np
import pandas as pd
import pytest

from src.technical import bollinger_bands, ema, macd, rsi, sma


def test_sma_known_values():
    series = pd.Series([1, 2, 3, 4, 5])
    result = sma(series, window=3)
    expected = pd.Series([np.nan, np.nan, 2.0, 3.0, 4.0])
    pd.testing.assert_series_equal(result, expected)


def test_ema_matches_pandas_ewm():
    series = pd.Series([10, 11, 12, 11, 13, 14, 15, 13, 12, 16], dtype=float)
    result = ema(series, span=5)
    expected = series.ewm(span=5, adjust=False).mean()
    pd.testing.assert_series_equal(result, expected)


def test_rsi_all_gains_is_100():
    series = pd.Series(range(1, 31), dtype=float)  # strictly increasing
    result = rsi(series, period=14)
    last_value = result.dropna().iloc[-1]
    assert last_value == pytest.approx(100.0)


def test_rsi_all_losses_is_0():
    series = pd.Series(range(30, 0, -1), dtype=float)  # strictly decreasing
    result = rsi(series, period=14)
    last_value = result.dropna().iloc[-1]
    assert last_value == pytest.approx(0.0)


def test_macd_output_shapes_and_histogram_relationship():
    series = pd.Series(np.linspace(100, 150, 60))
    macd_line, signal_line, histogram = macd(series)

    assert len(macd_line) == len(series)
    assert len(signal_line) == len(series)
    assert len(histogram) == len(series)

    expected_histogram = macd_line - signal_line
    pd.testing.assert_series_equal(histogram, expected_histogram)


def test_bollinger_bands_ordering():
    series = pd.Series(np.random.default_rng(42).normal(loc=100, scale=5, size=60))
    upper, middle, lower = bollinger_bands(series, window=20)

    valid = upper.notna() & middle.notna() & lower.notna()
    assert valid.any()
    assert (upper[valid] >= middle[valid]).all()
    assert (middle[valid] >= lower[valid]).all()
