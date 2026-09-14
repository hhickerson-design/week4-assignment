import pandas as pd
import pytest

from src.dcf import DCFError, compute_dcf, get_dcf_inputs, run_dcf_scenario


def test_compute_dcf_known_values_zero_growth():
    # With 0% FCF growth, a 1-year horizon, 10% discount rate, and 5% terminal growth,
    # this collapses to a growing-perpetuity valuation: PV = FCF*(1+g)/(r-g) discounted
    # one year, which combines with the single discounted FCF to a clean 2000.
    result = compute_dcf(
        base_fcf=100.0,
        shares_outstanding=100.0,
        net_debt=0.0,
        growth_rate=0.0,
        discount_rate=0.10,
        years=1,
        terminal_growth_rate=0.05,
    )
    assert result["enterprise_value"] == pytest.approx(2000.0)
    assert result["equity_value"] == pytest.approx(2000.0)
    assert result["implied_share_price"] == pytest.approx(20.0)


def test_compute_dcf_net_debt_reduces_implied_price():
    base = dict(
        base_fcf=100.0, shares_outstanding=100.0, growth_rate=0.0,
        discount_rate=0.10, years=1, terminal_growth_rate=0.05,
    )
    no_debt = compute_dcf(net_debt=0.0, **base)
    with_debt = compute_dcf(net_debt=500.0, **base)
    assert with_debt["implied_share_price"] == pytest.approx(no_debt["implied_share_price"] - 5.0)


def test_compute_dcf_higher_growth_implies_higher_price():
    base = dict(
        base_fcf=100.0, shares_outstanding=100.0, net_debt=0.0,
        discount_rate=0.10, years=5, terminal_growth_rate=0.025,
    )
    low_growth = compute_dcf(growth_rate=0.08, **base)
    high_growth = compute_dcf(growth_rate=0.12, **base)
    assert high_growth["implied_share_price"] > low_growth["implied_share_price"]


def test_compute_dcf_rejects_non_positive_base_fcf():
    with pytest.raises(DCFError):
        compute_dcf(
            base_fcf=0.0, shares_outstanding=100.0, net_debt=0.0,
            growth_rate=0.1, discount_rate=0.1, terminal_growth_rate=0.02,
        )


def test_compute_dcf_rejects_missing_shares_outstanding():
    with pytest.raises(DCFError):
        compute_dcf(
            base_fcf=100.0, shares_outstanding=0, net_debt=0.0,
            growth_rate=0.1, discount_rate=0.1, terminal_growth_rate=0.02,
        )


def test_compute_dcf_rejects_discount_rate_at_or_below_terminal_growth():
    with pytest.raises(DCFError):
        compute_dcf(
            base_fcf=100.0, shares_outstanding=100.0, net_debt=0.0,
            growth_rate=0.1, discount_rate=0.03, terminal_growth_rate=0.03,
        )


def test_get_dcf_inputs_prefers_cash_flow_statement_free_cash_flow_row():
    cash_flow = pd.DataFrame({pd.Timestamp("2025-09-30"): {"Free Cash Flow": 500.0}})
    info = {"freeCashflow": 999.0, "totalDebt": 200.0, "totalCash": 50.0, "beta": 1.0}
    inputs = get_dcf_inputs(info, cash_flow)
    assert inputs["base_fcf"] == pytest.approx(500.0)
    assert inputs["net_debt"] == pytest.approx(150.0)


def test_get_dcf_inputs_falls_back_to_info_free_cashflow():
    info = {"freeCashflow": 999.0, "totalDebt": 0, "totalCash": 0, "beta": 1.0}
    inputs = get_dcf_inputs(info, pd.DataFrame())
    assert inputs["base_fcf"] == pytest.approx(999.0)


def test_run_dcf_scenario_uses_capm_default_discount_rate():
    info = {"freeCashflow": 1000.0, "totalDebt": 0, "totalCash": 0, "sharesOutstanding": 100.0, "beta": 1.0}
    result = run_dcf_scenario(info, pd.DataFrame(), growth_rate=0.05, years=3)
    assert result["discount_rate_source"] == "CAPM cost of equity (approx.)"


def test_run_dcf_scenario_raises_when_fcf_unavailable():
    info = {"freeCashflow": None, "totalDebt": 0, "totalCash": 0, "sharesOutstanding": 100.0, "beta": 1.0}
    with pytest.raises(DCFError):
        run_dcf_scenario(info, pd.DataFrame(), growth_rate=0.05)
