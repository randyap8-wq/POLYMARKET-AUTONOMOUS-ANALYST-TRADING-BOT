"""Per-market analysis pipeline fusing the quant and AI signals.

``analyze_market`` is the orchestration the scan loop calls for each market:

1. **Quant prefilter** (binary markets) — fetch the order book once and skip the
   market entirely if it is untradeable, *before* spending an AI request. This is
   both a quality and a cost control.
2. **AI scoring** — Gemini with Google Search grounding picks a candidate outcome
   and fair value.
3. **Fusion** — the quant signal for the recommended outcome confirms, sizes, or
   vetoes the AI pick (see ``fusion.fuse_signals``).

Quant data is fetched lazily and fails soft: if the CLOB is unreachable the
pipeline degrades to AI-only rather than dropping the market.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from .config import QUANT_ENABLED, QUANT_PREFILTER
    from .fusion import fuse_signals
    from .market_data import fetch_order_book, fetch_price_history
    from .quant import compute_quant_signal
    from .scorer import score_market
except ImportError:  # pragma: no cover
    from config import QUANT_ENABLED, QUANT_PREFILTER
    from fusion import fuse_signals
    from market_data import fetch_order_book, fetch_price_history
    from quant import compute_quant_signal
    from scorer import score_market

LOGGER = logging.getLogger("pipeline")


def _quant_for_index(market: dict, index: int, book_cache: dict[int, dict]) -> dict[str, Any]:
    """Build the quant signal for one outcome token (lazy, fail-soft)."""
    token_ids = market.get("token_ids") or []
    prices = market.get("prices") or []
    snapshot_price = prices[index] if 0 <= index < len(prices) else 0.0

    if index >= len(token_ids) or not token_ids[index]:
        return compute_quant_signal(snapshot_price, [], {"bids": [], "asks": []})

    token_id = token_ids[index]
    history = fetch_price_history(token_id)
    book = book_cache.get(index)
    if book is None:
        book = fetch_order_book(token_id)
        book_cache[index] = book
    return compute_quant_signal(snapshot_price, history, book)


def analyze_market(market: dict) -> dict | None:
    """Run quant prefilter -> AI -> fusion for a single market.

    Returns a fused score dict, or ``None`` when the market is skipped (quant
    prefilter veto or AI failure).
    """
    if not QUANT_ENABLED:
        return score_market(market)

    book_cache: dict[int, dict] = {}
    outcomes = market.get("outcomes") or []

    # Cheap prefilter for binary markets: skip untradeable books before the AI
    # call. Multi-outcome books are not symmetric, so we defer their quant check
    # until after the AI picks an outcome.
    if QUANT_PREFILTER and len(outcomes) == 2:
        rep = _quant_for_index(market, 0, book_cache)
        if rep.get("available") and not rep.get("tradeable"):
            LOGGER.debug("quant prefilter skip '%s': %s", str(market.get("question", ""))[:50], rep.get("reason"))
            return None

    ai = score_market(market)
    if ai is None:
        return None

    index = ai.get("recommended_outcome_index")
    if index is None:
        return fuse_signals(ai, None)

    quant_signal = _quant_for_index(market, index, book_cache)
    return fuse_signals(ai, quant_signal)
