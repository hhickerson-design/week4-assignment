import pandas as pd
import pytest

from src.earnings_quality import summarize_earnings_quality


def _income_stmt(latest: dict, prior: dict, latest_year="2025-09-30", prior_year="2024-09-30"):
    return pd.DataFrame(
        {pd.Timestamp(latest_year): latest, pd.Timestamp(prior_year): prior}
    )


def test_flags_net_income_outpacing_operating_income():
    # Operating income flat, net income way up -> growth driven by non-operating items.
    latest = {"Operating Income": 100.0, "Net Income": 150.0, "Other Income Expense": 40.0, "Pretax Income": 160.0}
    prior = {"Operating Income": 100.0, "Net Income": 80.0, "Other Income Expense": 0.0, "Pretax Income": 90.0}
    result = summarize_earnings_quality(_income_stmt(latest, prior))

    assert result["operating_income_growth"] == pytest.approx(0.0)
    assert result["net_income_growth"] == pytest.approx(0.875)
    assert "non-operating" in result["earnings_quality_signal"]


def test_flags_operating_income_outpacing_net_income():
    latest = {"Operating Income": 200.0, "Net Income": 90.0, "Other Income Expense": -10.0, "Pretax Income": 100.0}
    prior = {"Operating Income": 100.0, "Net Income": 85.0, "Other Income Expense": 0.0, "Pretax Income": 90.0}
    result = summarize_earnings_quality(_income_stmt(latest, prior))

    assert "core operations are" in result["earnings_quality_signal"]


def test_flags_growth_aligned_with_core_operations():
    latest = {"Operating Income": 110.0, "Net Income": 88.0, "Other Income Expense": 2.0, "Pretax Income": 95.0}
    prior = {"Operating Income": 100.0, "Net Income": 80.0, "Other Income Expense": 1.0, "Pretax Income": 85.0}
    result = summarize_earnings_quality(_income_stmt(latest, prior))

    assert result["earnings_quality_signal"] == (
        "net income growth is closely aligned with operating income growth — earnings "
        "growth appears driven primarily by core operations"
    )


def test_handles_missing_or_insufficient_data():
    assert summarize_earnings_quality(pd.DataFrame())["earnings_quality_signal"].startswith("unknown")
    assert summarize_earnings_quality(None)["earnings_quality_signal"].startswith("unknown")

    single_year = pd.DataFrame(
        {pd.Timestamp("2025-09-30"): {"Operating Income": 100.0, "Net Income": 80.0}}
    )
    assert summarize_earnings_quality(single_year)["earnings_quality_signal"].startswith("unknown")
