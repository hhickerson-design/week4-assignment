import streamlit as st

from src import config
from src.claude_chat import ClaudeChatError, build_system_prompt, get_client, get_response
from src.config import CACHE_TTL_SECONDS, MissingAPIKeyError
from src.ai_exposure import format_ai_exposure_for_prompt, summarize_ai_exposure
from src.data import (
    TickerDataError,
    fetch_cash_flow_statement,
    fetch_company_info,
    fetch_income_statement,
    fetch_price_history,
    fetch_recent_news,
)
from src.dcf import DCFError, format_dcf_for_prompt, run_dcf_scenario
from src.earnings_quality import format_earnings_quality_for_prompt, summarize_earnings_quality
from src.fundamentals import extract_fundamentals, format_fundamentals_for_prompt
from src.technical import format_technical_for_prompt, summarize_technical
from src.valuation import format_valuation_for_prompt, summarize_valuation

st.set_page_config(page_title="Stock Analysis Chatbot", page_icon="\U0001F4C8")

config.load_env()
try:
    config.get_api_key()
except MissingAPIKeyError as exc:
    st.error(str(exc))
    st.stop()

st.title("Stock Analysis Chatbot")
st.warning(
    "Educational tool only. This is not financial advice. "
    "Data may be delayed, incomplete, or inaccurate — verify independently before making decisions."
)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_price_history(ticker: str):
    return fetch_price_history(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_company_info(ticker: str):
    return fetch_company_info(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_income_statement(ticker: str):
    return fetch_income_statement(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_recent_news(ticker: str):
    return fetch_recent_news(ticker)


@st.cache_data(ttl=CACHE_TTL_SECONDS)
def load_cash_flow_statement(ticker: str):
    return fetch_cash_flow_statement(ticker)


def execute_tool(name: str, tool_input: dict) -> str:
    ticker = tool_input.get("ticker", "").strip().upper()

    if name == "get_stock_analysis":
        try:
            history = load_price_history(ticker)
            info = load_company_info(ticker)
            fundamentals = extract_fundamentals(info)
            technicals = summarize_technical(history)
            valuation = summarize_valuation(info, history)
            earnings_quality = summarize_earnings_quality(load_income_statement(ticker))
            ai_exposure = summarize_ai_exposure(info, load_recent_news(ticker))
        except TickerDataError as exc:
            return f"Error: {exc}"

        return (
            f"Ticker: {ticker}\n\n"
            f"FUNDAMENTAL DATA:\n{format_fundamentals_for_prompt(fundamentals)}\n\n"
            f"TECHNICAL DATA:\n{format_technical_for_prompt(technicals)}\n\n"
            f"VALUATION DATA:\n{format_valuation_for_prompt(valuation)}\n\n"
            f"EARNINGS QUALITY DATA:\n{format_earnings_quality_for_prompt(earnings_quality)}\n\n"
            f"AI EXPOSURE DATA:\n{format_ai_exposure_for_prompt(ai_exposure)}"
        )

    if name == "run_dcf_scenario":
        try:
            info = load_company_info(ticker)
            cash_flow = load_cash_flow_statement(ticker)
            result = run_dcf_scenario(
                info,
                cash_flow,
                growth_rate=tool_input["revenue_growth_rate"],
                years=tool_input.get("years", 5),
                discount_rate=tool_input.get("discount_rate"),
                terminal_growth_rate=tool_input.get("terminal_growth_rate", 0.025),
            )
        except (TickerDataError, DCFError) as exc:
            return f"Error: {exc}"

        return f"Ticker: {ticker}\n{format_dcf_for_prompt(result)}"

    return f"Error: unknown tool '{name}'"


if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_ticker" not in st.session_state:
    st.session_state.current_ticker = None
    st.session_state.current_fundamentals = None
    st.session_state.current_technicals = None
    st.session_state.current_valuation = None
    st.session_state.current_earnings_quality = None
    st.session_state.current_ai_exposure = None

with st.sidebar:
    st.header("Ticker")
    ticker_input = st.text_input("Symbol (e.g. AAPL)", value="").strip().upper()
    load_clicked = st.button("Load / Refresh")

    if load_clicked and ticker_input:
        try:
            history = load_price_history(ticker_input)
            info = load_company_info(ticker_input)
            st.session_state.current_ticker = ticker_input
            st.session_state.current_fundamentals = extract_fundamentals(info)
            st.session_state.current_technicals = summarize_technical(history)
            st.session_state.current_valuation = summarize_valuation(info, history)
            income_stmt = load_income_statement(ticker_input)
            st.session_state.current_earnings_quality = summarize_earnings_quality(income_stmt)
            news = load_recent_news(ticker_input)
            st.session_state.current_ai_exposure = summarize_ai_exposure(info, news)
            st.session_state.messages = []
        except TickerDataError as exc:
            st.error(str(exc))

    if st.session_state.current_ticker:
        st.subheader(f"Data: {st.session_state.current_ticker}")

        st.caption("Fundamentals")
        for label, value in st.session_state.current_fundamentals.items():
            st.metric(label, value)

        st.caption("Technicals")
        tech = st.session_state.current_technicals
        st.metric("RSI(14)", f"{tech['rsi14']:.1f}" if tech["rsi14"] is not None else "N/A", tech["rsi_signal"])
        st.metric("MACD signal", tech["macd_signal"])
        st.metric("Price vs SMA50", tech["price_vs_sma50"])

        st.caption("Valuation")
        val = st.session_state.current_valuation
        pe_display = f"{val['trailing_pe']:.2f}" if val["trailing_pe"] is not None else "N/A"
        st.metric("Trailing P/E", pe_display, val["pe_vs_history_signal"])
        coe_display = f"{val['cost_of_equity'] * 100:.2f}%" if val["cost_of_equity"] is not None else "N/A"
        st.metric("Cost of Equity (CAPM, approx.)", coe_display, val["earnings_yield_signal"])

        st.caption("Earnings Quality (core vs. other)")
        eq = st.session_state.current_earnings_quality
        op_display = (
            f"{eq['operating_income_growth'] * 100:.2f}%"
            if eq["operating_income_growth"] is not None
            else "N/A"
        )
        net_display = (
            f"{eq['net_income_growth'] * 100:.2f}%" if eq["net_income_growth"] is not None else "N/A"
        )
        st.metric("Operating Income Growth", op_display)
        st.metric("Net Income Growth", net_display)
        st.caption(eq["earnings_quality_signal"])

        st.caption("AI Exposure")
        ai = st.session_state.current_ai_exposure
        st.metric("AI mentioned in business description", "Yes" if ai["ai_mentioned_in_summary"] else "No")
        st.metric("AI-related headlines", f"{ai['ai_related_news_count']}/{ai['total_news_count']}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

def render_chat_text(text: str) -> None:
    # st.markdown treats a pair of "$" as LaTeX math delimiters, which garbles any
    # message containing two or more dollar amounts (common in this app).
    st.markdown(text.replace("$", "\\$"))


if not st.session_state.current_ticker:
    st.info("Enter a ticker symbol in the sidebar and click **Load / Refresh** to get started.")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            render_chat_text(message["content"])

    user_input = st.chat_input(f"Ask about {st.session_state.current_ticker}...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            render_chat_text(user_input)

        with st.chat_message("assistant"):
            try:
                system_prompt = build_system_prompt(
                    st.session_state.current_ticker,
                    st.session_state.current_fundamentals,
                    st.session_state.current_technicals,
                    st.session_state.current_valuation,
                    st.session_state.current_earnings_quality,
                    st.session_state.current_ai_exposure,
                )
                client = get_client()
                with st.spinner("Analyzing..."):
                    reply = get_response(
                        client, st.session_state.messages, system_prompt, execute_tool
                    )
                render_chat_text(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except ClaudeChatError as exc:
                st.error(str(exc))
