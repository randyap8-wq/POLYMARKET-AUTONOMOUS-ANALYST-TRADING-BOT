from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
LOGS_DIR = BASE_DIR / "logs"
REPORT_PATH = BASE_DIR / "report.json"
TRADES_LOG_PATH = LOGS_DIR / "trades.jsonl"

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

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
POLYGON_PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
POLYGON_WALLET_ADDRESS = os.getenv("POLYGON_WALLET_ADDRESS", "")
MAX_BET_USDC = float(os.getenv("MAX_BET_USDC", "10"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.07"))
MIN_CONFIDENCE = os.getenv("MIN_CONFIDENCE", "medium").strip().lower()
DRY_RUN = os.getenv("DRY_RUN", "true").strip().lower() in {"1", "true", "yes", "on"}
