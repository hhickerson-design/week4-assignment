import pandas as pd
import pytest

from src.valuation import estimate_cost_of_equity, historical_pe_stats, summarize_valuation


def test_estimate_cost_of_equity_capm_formula():
    # cost of equity = risk_free_rate + beta * equity_risk_premium
    # with this project's default constants (0.045, 0.05), beta=1.0 -> 0.045 + 0.05 = 0.095
    result = estimate_cost_of_equity(beta=1.0)
    assert result == pytest.approx(0.095)


def test_estimate_cost_of_equity_scales_with_beta():
    low_beta = estimate_cost_of_equity(beta=0.5)
    high_beta = estimate_cost_of_equity(beta=2.0)
    assert high_beta > low_beta


def test_historical_pe_stats_known_values():
    prices = pd.DataFrame({"Close": [100.0, 150.0, 200.0]})
    result = historical_pe_stats(prices, trailing_eps=10.0)
    assert result["pe_5y_min"] == pytest.approx(10.0)
    assert result["pe_5y_median"] == pytest.approx(15.0)
    assert result["pe_5y_max"] == pytest.approx(20.0)


def test_historical_pe_stats_handles_missing_eps():
    prices = pd.DataFrame({"Close": [100.0, 150.0]})
    result = historical_pe_stats(prices, trailing_eps=None)
    assert result == {"pe_5y_min": None, "pe_5y_median": None, "pe_5y_max": None}


def test_historical_pe_stats_handles_non_positive_eps():
    prices = pd.DataFrame({"Close": [100.0, 150.0]})
    result = historical_pe_stats(prices, trailing_eps=-2.0)
    assert result == {"pe_5y_min": None, "pe_5y_median": None, "pe_5y_max": None}


def test_summarize_valuation_flags_pe_above_history():
    prices = pd.DataFrame({"Close": [100.0] * 10})  # implied historical P/E stays at 10 (EPS=10)
    info = {"trailingPE": 20.0, "trailingEps": 10.0, "beta": 1.0}
    result = summarize_valuation(info, prices)
    assert result["pe_vs_history_signal"] == "above its own historical range"


def test_summarize_valuation_earnings_yield_vs_cost_of_equity():
    prices = pd.DataFrame({"Close": [100.0] * 10})
    # trailingPE=8 -> earnings yield 12.5%, well above ~9.5% CAPM cost of equity at beta=1.0
    info = {"trailingPE": 8.0, "trailingEps": 12.5, "beta": 1.0}
    result = summarize_valuation(info, prices)
    assert result["earnings_yield_signal"] == "earnings yield exceeds estimated cost of equity"
