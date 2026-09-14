import os

from dotenv import load_dotenv

CLAUDE_MODEL = "claude-sonnet-5"
MAX_TOKENS = 2500
PRICE_HISTORY_PERIOD = "5y"
CACHE_TTL_SECONDS = 300

# CAPM assumptions used to estimate a rough cost of equity (see src/valuation.py).
# These are simplified, editable defaults, not live market data.
RISK_FREE_RATE = 0.045
EQUITY_RISK_PREMIUM = 0.05


class MissingAPIKeyError(Exception):
    pass


def load_env() -> None:
    load_dotenv()


def get_api_key() -> str:
    load_env()
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return api_key
