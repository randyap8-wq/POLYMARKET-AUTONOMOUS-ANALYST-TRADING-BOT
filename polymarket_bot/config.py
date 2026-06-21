from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
REPORT_PATH = BASE_DIR / "report.json"
TRADES_LOG_PATH = LOGS_DIR / "trades.jsonl"

DATA_DIR = BASE_DIR / "data"
PAPER_BETS_PATH = DATA_DIR / "paper_bets.jsonl"
RESOLVED_PATH = DATA_DIR / "resolved.jsonl"
PERFORMANCE_PATH = DATA_DIR / "performance.json"

DATA_DIR.mkdir(exist_ok=True)

load_dotenv(BASE_DIR / ".env")
LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
LOGGER = logging.getLogger("polymarket_bot")

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"
DATA_BASE = "https://data-api.polymarket.com"
POLYGON_CHAIN_ID = 137

MARKET_FILTERS = {
    "min_volume": 10_000,
    "max_days_to_close": 60,
    "max_outcomes": 4,
    "limit": 50,
}

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_REASONER_MODEL = "deepseek-reasoner"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"

# DeepSeek token pricing in USD per 1M tokens. Defaults track published
# deepseek-chat / deepseek-reasoner standard rates and can be overridden via env
# as the provider adjusts pricing.
DEEPSEEK_PRICING = {
    DEEPSEEK_MODEL: {
        "input": float(os.getenv("DEEPSEEK_CHAT_INPUT_PRICE", "0.27")),
        "output": float(os.getenv("DEEPSEEK_CHAT_OUTPUT_PRICE", "1.10")),
    },
    DEEPSEEK_REASONER_MODEL: {
        "input": float(os.getenv("DEEPSEEK_REASONER_INPUT_PRICE", "0.55")),
        "output": float(os.getenv("DEEPSEEK_REASONER_OUTPUT_PRICE", "2.19")),
    },
}

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

# News search window (Tavily `days`). The window is widened dynamically for
# slower-moving markets (those that close further in the future) so the model
# sees enough context, while fast-moving markets stay focused on fresh news.
NEWS_WINDOW_MIN_DAYS = int(os.getenv("NEWS_WINDOW_MIN_DAYS", "3"))
NEWS_WINDOW_MAX_DAYS = int(os.getenv("NEWS_WINDOW_MAX_DAYS", "30"))

# Pull a second, negation-focused Tavily query per market so the model also
# weighs evidence against the favoured outcome (reduces overconfidence).
COUNTER_EVIDENCE_ENABLED = os.getenv("COUNTER_EVIDENCE", "true").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

# Avoid recommending bets at extreme prices: edges there are noisy and the
# risk/reward is poor (a 0.02 -> 0.04 "edge" is mostly microstructure noise).
MIN_PRICE = float(os.getenv("MIN_PRICE", "0.05"))
MAX_PRICE = float(os.getenv("MAX_PRICE", "0.95"))

# Comma-separated market categories to skip entirely (e.g. "sports,entertainment").
# Useful once --validate shows a category consistently loses money.
DISABLED_CATEGORIES = {
    item.strip().lower()
    for item in os.getenv("DISABLED_CATEGORIES", "").split(",")
    if item.strip()
}

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
POLYGON_PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
POLYGON_WALLET_ADDRESS = os.getenv("POLYGON_WALLET_ADDRESS", "")
MAX_BET_USDC = float(os.getenv("MAX_BET_USDC", "10"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.07"))
MIN_CONFIDENCE = os.getenv("MIN_CONFIDENCE", "medium").strip().lower()
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"}
