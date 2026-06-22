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


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


MARKET_FILTERS = {
    "min_volume": 10_000,
    "max_days_to_close": 60,
    "max_outcomes": 4,
    "limit": 50,
}

# ---------------------------------------------------------------------------
# AI scoring — Gemini 2.0 Flash with built-in Google Search grounding.
# Replaces the previous DeepSeek + Tavily stack (two paid services) with a
# single call on Gemini's free tier (1,500 requests/day, 15 requests/min).
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_MODEL_PRO = os.getenv("GEMINI_MODEL_PRO", "gemini-2.5-pro")

# Built-in Google Search grounding lets Gemini fetch its own news, removing the
# separate Tavily dependency. Disable to save grounding quota when running the
# quant-only path.
GEMINI_USE_SEARCH = _as_bool(os.getenv("GEMINI_USE_SEARCH", "true"), True)

# Optional second opinion from the Pro model on high-edge, high-confidence picks.
# Costs an extra call; disable to stay strictly on the Flash free tier.
GEMINI_PRO_RECHECK = _as_bool(os.getenv("GEMINI_PRO_RECHECK", "true"), True)

# Free tier is 15 requests/min. The scorer sleeps 60/RPM seconds between calls.
# Raise (or set 0) when using a paid Vertex key with a higher limit.
GEMINI_RPM = int(os.getenv("GEMINI_RPM", "15"))

# Cost tracking. Gemini Flash free tier = $0. If you move to a paid Vertex key,
# set USD-per-1M-token prices here (or via env) to keep cost reporting honest.
GEMINI_PRICING = {
    GEMINI_MODEL: {
        "input": float(os.getenv("GEMINI_FLASH_INPUT_PRICE", "0.0")),
        "output": float(os.getenv("GEMINI_FLASH_OUTPUT_PRICE", "0.0")),
    },
    GEMINI_MODEL_PRO: {
        "input": float(os.getenv("GEMINI_PRO_INPUT_PRICE", "0.0")),
        "output": float(os.getenv("GEMINI_PRO_OUTPUT_PRICE", "0.0")),
    },
}

CONFIDENCE_RANK = {"low": 0, "medium": 1, "high": 2}

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

# ---------------------------------------------------------------------------
# Quant engine — structured signals from CLOB price history and the live order
# book. These are computed from numeric market data (no LLM) and fused with the
# AI signal in fusion.py.
# ---------------------------------------------------------------------------
QUANT_ENABLED = _as_bool(os.getenv("QUANT_ENABLED", "true"), True)

# When True, the quant tradeability gate (liquidity, spread, extreme price) runs
# BEFORE the AI call so we never spend an LLM request on a market we cannot trade
# — this both improves quality and cuts AI cost.
QUANT_PREFILTER = _as_bool(os.getenv("QUANT_PREFILTER", "true"), True)

# Fusion weights for the final edge. They are normalised, so only the ratio
# matters. Default leans on the AI for fundamental/news fair value while letting
# the quant microstructure signal confirm or veto.
AI_WEIGHT = float(os.getenv("AI_WEIGHT", "0.6"))
QUANT_WEIGHT = float(os.getenv("QUANT_WEIGHT", "0.4"))

# Tradeability gates (order-book microstructure).
MIN_BOOK_LIQUIDITY_USDC = float(os.getenv("MIN_BOOK_LIQUIDITY_USDC", "200"))
MAX_SPREAD = float(os.getenv("MAX_SPREAD", "0.06"))

# Price-history sampling for the quant features (CLOB /prices-history).
QUANT_HISTORY_INTERVAL = os.getenv("QUANT_HISTORY_INTERVAL", "1w")
QUANT_HISTORY_FIDELITY = int(os.getenv("QUANT_HISTORY_FIDELITY", "60"))

# Momentum/mean-reversion lookbacks expressed in samples of the history series.
QUANT_MOMENTUM_LOOKBACK = int(os.getenv("QUANT_MOMENTUM_LOOKBACK", "24"))
QUANT_MA_SHORT = int(os.getenv("QUANT_MA_SHORT", "6"))
QUANT_MA_LONG = int(os.getenv("QUANT_MA_LONG", "24"))

# How strongly a quant/AI disagreement shrinks the blended edge (0..1).
DISAGREEMENT_PENALTY = float(os.getenv("DISAGREEMENT_PENALTY", "0.5"))

# ---------------------------------------------------------------------------
# Systematic risk controls — portfolio-level sizing and circuit breakers.
# ---------------------------------------------------------------------------
MAX_BET_USDC = float(os.getenv("MAX_BET_USDC", "10"))
MIN_EDGE = float(os.getenv("MIN_EDGE", "0.07"))
MIN_CONFIDENCE = os.getenv("MIN_CONFIDENCE", "medium").strip().lower()
DRY_RUN = _as_bool(os.getenv("DRY_RUN", "true"), True)

# Half-Kelly by default; volatility scaling shrinks this further on choppy books.
KELLY_FRACTION = float(os.getenv("KELLY_FRACTION", "0.5"))
VOL_SIZING = _as_bool(os.getenv("VOL_SIZING", "true"), True)

# Portfolio caps applied across the opportunities in a single scan.
MAX_PORTFOLIO_EXPOSURE_USDC = float(os.getenv("MAX_PORTFOLIO_EXPOSURE_USDC", "50"))
MAX_CATEGORY_EXPOSURE_USDC = float(os.getenv("MAX_CATEGORY_EXPOSURE_USDC", "20"))
MAX_CONCURRENT_POSITIONS = int(os.getenv("MAX_CONCURRENT_POSITIONS", "5"))

# Drawdown circuit breaker: if realised paper/live P&L over the recent window is
# below this (negative) threshold, sizing is throttled. 0 disables the guard.
MAX_DRAWDOWN_USDC = float(os.getenv("MAX_DRAWDOWN_USDC", "30"))

POLYGON_PRIVATE_KEY = os.getenv("POLYGON_PRIVATE_KEY", "")
POLYGON_WALLET_ADDRESS = os.getenv("POLYGON_WALLET_ADDRESS", "")
