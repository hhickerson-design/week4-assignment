import pandas as pd

# Growth-rate gap (net income growth minus operating income growth) beyond which we call
# out a meaningful divergence between core-operations growth and bottom-line growth.
DIVERGENCE_THRESHOLD = 0.05


def _get(income_stmt: pd.DataFrame, row: str, col) -> float | None:
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

    operating_income_latest = _get(income_stmt, "Operating Income", latest_col)
    operating_income_prior = _get(income_stmt, "Operating Income", prior_col)
    net_income_latest = _get(income_stmt, "Net Income", latest_col)
    net_income_prior = _get(income_stmt, "Net Income", prior_col)
    other_income_latest = _get(income_stmt, "Other Income Expense", latest_col)
    pretax_income_latest = _get(income_stmt, "Pretax Income", latest_col)

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
