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
    from .config import AI_WEIGHT, QUANT_ENABLED, QUANT_PREFILTER, QUANT_WEIGHT
    from .fusion import fuse_signals
    from .market_data import fetch_market_stats, fetch_order_book, fetch_price_history
    from .quant import compute_quant_signal
    from .scorer import score_market
except ImportError:  # pragma: no cover
    from config import AI_WEIGHT, QUANT_ENABLED, QUANT_PREFILTER, QUANT_WEIGHT
    from fusion import fuse_signals
    from market_data import fetch_market_stats, fetch_order_book, fetch_price_history
    from quant import compute_quant_signal
    from scorer import score_market

LOGGER = logging.getLogger("pipeline")


def _optional_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _stats_for_market(market: dict, stats_cache: dict[str, dict[str, float]]) -> dict[str, float]:
    provided = {}
    for key in ("volume_24h", "open_interest"):
        value = _optional_float(market.get(key))
        if value is not None:
            provided[key] = value
    if provided:
        return provided

    condition_id = str(market.get("condition_id") or "")
    if not condition_id:
        return {}
    if condition_id not in stats_cache:
        stats_cache[condition_id] = fetch_market_stats(condition_id)
    return stats_cache[condition_id]


def _quant_for_index(
    market: dict,
    index: int,
    book_cache: dict[int, dict],
    stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build the quant signal for one outcome token (lazy, fail-soft)."""
    token_ids = market.get("token_ids") or []
    prices = market.get("prices") or []
    snapshot_price = prices[index] if 0 <= index < len(prices) else 0.0

    if index >= len(token_ids) or not token_ids[index]:
        return compute_quant_signal(snapshot_price, [], {"bids": [], "asks": []}, stats)

    token_id = token_ids[index]
    history = fetch_price_history(token_id)
    book = book_cache.get(index)
    if book is None:
        book = fetch_order_book(token_id)
        book_cache[index] = book
    return compute_quant_signal(snapshot_price, history, book, stats)


def _book_signal(
    market: dict,
    index: int,
    book_cache: dict[int, dict],
    stats: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Build an order-book-only quant signal for prefiltering."""
    token_ids = market.get("token_ids") or []
    prices = market.get("prices") or []
    snapshot_price = prices[index] if 0 <= index < len(prices) else 0.0

    if index >= len(token_ids) or not token_ids[index]:
        return compute_quant_signal(snapshot_price, [], {"bids": [], "asks": []}, stats)

    token_id = token_ids[index]
    book = book_cache.get(index)
    if book is None:
        book = fetch_order_book(token_id)
        book_cache[index] = book
    return compute_quant_signal(snapshot_price, [], book, stats)


def _probability_for_outcome(ai: dict[str, Any], index: int, n_outcomes: int) -> float:
    for key in ("outcome_probabilities", "probabilities"):
        raw = ai.get(key)
        if isinstance(raw, list) and 0 <= index < len(raw):
            try:
                return max(0.0, min(1.0, float(raw[index])))
            except (TypeError, ValueError):
                pass
        if isinstance(raw, dict):
            outcome = ai.get("outcomes", [])[index] if 0 <= index < len(ai.get("outcomes", [])) else None
            value = raw.get(str(index), raw.get(outcome))
            try:
                return max(0.0, min(1.0, float(value)))
            except (TypeError, ValueError):
                pass

    if index == ai.get("recommended_outcome_index"):
        try:
            return max(0.0, min(1.0, float(ai.get("probability", ai.get("fair_value_estimate")))))
        except (TypeError, ValueError):
            return 0.5
    if n_outcomes == 2:
        try:
            return max(0.0, min(1.0, 1.0 - float(ai.get("probability", ai.get("fair_value_estimate")))))
        except (TypeError, ValueError):
            return 0.5
    return 0.5


def _maybe_switch_multi_outcome(
    market: dict,
    ai: dict[str, Any],
    quant_by_index: dict[int, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    outcomes = market.get("outcomes") or []
    prices = market.get("prices") or []
    current_index = ai.get("recommended_outcome_index")
    if len(outcomes) <= 2 or current_index is None:
        return ai, quant_by_index.get(current_index) if current_index is not None else {}

    n_outcomes = len(outcomes)
    weight_sum = (AI_WEIGHT + QUANT_WEIGHT) or 1.0
    candidates: list[dict[str, Any]] = []
    for index, outcome in enumerate(outcomes):
        quant = quant_by_index.get(index) or {}
        if quant.get("available") and not quant.get("tradeable"):
            continue
        price = float(prices[index]) if index < len(prices) else float(quant.get("price") or 0.0)
        ai_probability = _probability_for_outcome({**ai, "outcomes": outcomes}, index, n_outcomes)
        quant_probability = quant.get("probability")
        if quant_probability is None:
            quant_probability = max(0.0, min(1.0, price + float(quant.get("quant_edge") or 0.0)))
        quant_probability = max(0.0, min(1.0, float(quant_probability or 0.0)))
        blended_probability = (AI_WEIGHT * ai_probability + QUANT_WEIGHT * quant_probability) / weight_sum
        edge = blended_probability - price if price > 0 else blended_probability - 0.5
        candidates.append(
            {
                "index": index,
                "outcome": outcome,
                "price": price,
                "ai_probability": ai_probability,
                "probability": blended_probability,
                "edge": edge,
                "quant": quant,
            }
        )

    if not candidates:
        return ai, quant_by_index.get(current_index, {})

    current = next((item for item in candidates if item["index"] == current_index), None)
    best = max(candidates, key=lambda item: item["edge"])
    current_edge = current["edge"] if current else float(ai.get("edge") or 0.0)
    if best["index"] != current_index and best["edge"] > current_edge + 0.02:
        LOGGER.info(
            "multi-outcome override for '%s': %s -> %s (edge %.3f > %.3f)",
            str(market.get("question", ""))[:50],
            ai.get("recommended_outcome"),
            best["outcome"],
            best["edge"],
            current_edge,
        )
        ai = {
            **ai,
            "recommended_outcome": best["outcome"],
            "recommended_outcome_index": best["index"],
            "outcome_index": best["index"],
            "current_price": round(best["price"], 4),
            "fair_value_estimate": round(best["ai_probability"], 4),
            "probability": round(best["ai_probability"], 4),
            "edge": round(best["ai_probability"] - best["price"], 4),
            "multi_outcome_override": True,
            "original_recommended_outcome": ai.get("recommended_outcome"),
        }
        return ai, best["quant"]

    return ai, quant_by_index.get(current_index, {})


def analyze_market(market: dict) -> dict | None:
    """Run quant prefilter -> AI -> fusion for a single market.

    Returns a fused score dict, or ``None`` when the market is skipped (quant
    prefilter veto or AI failure).
    """
    if not QUANT_ENABLED:
        return score_market(market)

    book_cache: dict[int, dict] = {}
    stats_cache: dict[str, dict[str, float]] = {}
    outcomes = market.get("outcomes") or []
    stats = _stats_for_market(market, stats_cache)

    # Cheap prefilter for binary markets: skip only when the order book is present
    # and untradeable. Multi-outcome books are not symmetric, so we defer their
    # quant check until after the AI picks an outcome.
    if QUANT_PREFILTER and len(outcomes) == 2:
        rep = _book_signal(market, 0, book_cache, stats)
        if rep.get("available") and not rep.get("tradeable"):
            LOGGER.debug("quant prefilter skip '%s': %s", str(market.get("question", ""))[:50], rep.get("reason"))
            return None

    ai = score_market(market)
    if ai is None:
        return None

    index = ai.get("recommended_outcome_index")
    if index is None:
        return fuse_signals(ai, None)

    quant_by_index: dict[int, dict[str, Any]] = {}
    if len(outcomes) > 2:
        for outcome_index in range(len(outcomes)):
            quant_by_index[outcome_index] = _quant_for_index(market, outcome_index, book_cache, stats)
        ai, quant_signal = _maybe_switch_multi_outcome(market, ai, quant_by_index)
    else:
        quant_signal = _quant_for_index(market, index, book_cache, stats)
    return fuse_signals(ai, quant_signal, market_data={"prices": market.get("prices") or []}, ai_weight=AI_WEIGHT, quant_weight=QUANT_WEIGHT)
