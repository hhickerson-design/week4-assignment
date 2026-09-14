import pandas as pd

from src.valuation import estimate_cost_of_equity


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
