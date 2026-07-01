from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests

try:
    from .config import GAMMA_BASE, BASE_DIR
    from .utils import retry_with_backoff
except ImportError:  # pragma: no cover
    from config import GAMMA_BASE, BASE_DIR
    from utils import retry_with_backoff

LOGGER = logging.getLogger("paper_trader")

DATA_DIR = BASE_DIR / "data"
PAPER_BETS_PATH = DATA_DIR / "paper_bets.jsonl"
RESOLVED_PATH = DATA_DIR / "resolved.jsonl"

DATA_DIR.mkdir(exist_ok=True)


@retry_with_backoff(max_retries=5, base_delay=1)
def _fetch_gamma_market_by_condition(condition_id: str) -> list[dict]:
    """Fetch a Gamma market by condition id with retry/backoff."""
    resp = requests.get(
        f"{GAMMA_BASE}/markets",
        params={"condition_id": condition_id},
        timeout=20,
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else [data]


def record_paper_bet(market: dict, score: dict, news: list[dict]) -> dict | None:
    """
    Append a hypothetical bet to paper_bets.jsonl only if no pending bet for
    the same market (condition_id) already exists. Returns the entry or None
    if skipped.
    check_resolutions may later rewrite the file in place to update statuses.
    """
    condition_id = market.get("condition_id", "")
    if condition_id:
        existing = _load_paper_bets()
        already_pending = any(
            b.get("condition_id") == condition_id and b.get("status") == "pending"
            for b in existing
        )
        if already_pending:
            LOGGER.debug("skipping duplicate paper bet for %s", condition_id)
            return None

    current_price = float(score.get("current_price") or 0.01)
    usdc = 10.0  # fixed paper stake
    tokens = round(usdc / max(current_price, 0.01), 2)

    entry = {
        "bet_id": str(uuid.uuid4()),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "question": market.get("question", ""),
        "url": market.get("url", ""),
        "condition_id": condition_id,
        "recommended_outcome": score.get("recommended_outcome"),
        "recommended_outcome_index": score.get("recommended_outcome_index"),
        "category": market.get("category", score.get("category", "other")),
        "current_price": current_price,
        "fair_value_estimate": float(score.get("fair_value_estimate") or 0.0),
        "edge": float(score.get("edge") or 0.0),
        "confidence": score.get("confidence", "low"),
        "confidence_level": score.get("confidence_level", "low"),
        "ai_confidence": score.get("confidence", 0.0),
        "reasoning": score.get("reasoning", ""),
        "top_news": [
            {"title": n.get("title", ""), "url": n.get("url", ""), "date": n.get("published_date", "")}
            for n in news[:3]
        ],
        "hypothetical_usdc": usdc,
        "hypothetical_tokens": tokens,
        "end_date": market.get("end_date", ""),
        "status": "pending",
        "resolved_outcome": None,
        "resolved_at": None,
        "pnl_usdc": None,
        "correct": None,
    }

    with PAPER_BETS_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    LOGGER.info("paper bet recorded: %s → %s (edge %.2f)", entry["question"][:50], entry["recommended_outcome"], entry["edge"])
    return entry


def _load_paper_bets() -> list[dict]:
    if not PAPER_BETS_PATH.exists():
        return []
    bets = []
    with PAPER_BETS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    bets.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return bets


def _rewrite_paper_bets(bets: list[dict]) -> None:
    # Rewrite statuses in place so pending views do not need to re-derive state.
    with PAPER_BETS_PATH.open("w", encoding="utf-8") as fh:
        for bet in bets:
            fh.write(json.dumps(bet) + "\n")


def _append_resolved(entry: dict) -> None:
    with RESOLVED_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def check_resolutions() -> int:
    """
    Check all pending paper bets against Polymarket API.
    Updates paper_bets.jsonl and appends to resolved.jsonl.
    Returns the number of newly resolved bets.
    """
    bets = _load_paper_bets()
    pending = [b for b in bets if b.get("status") == "pending"]
    if not pending:
        return 0

    resolved_count = 0
    bet_map = {b["bet_id"]: b for b in bets}

    for bet in pending:
        condition_id = bet.get("condition_id")
        if not condition_id:
            continue

        try:
            markets = _fetch_gamma_market_by_condition(condition_id)
            if not markets:
                continue
            mkt = markets[0]

            is_closed = mkt.get("closed", False)
            resolution_time = mkt.get("resolutionTime") or mkt.get("resolution_time")
            if not is_closed or not resolution_time:
                continue

            raw_prices = mkt.get("outcomePrices") or "[]"
            prices = [float(p) for p in json.loads(raw_prices)]
            outcomes = json.loads(mkt.get("outcomes") or "[]")

            if not prices or not outcomes or max(prices) < 0.99:
                # Not fully resolved yet (winner should be at 1.0)
                continue

            winning_index = prices.index(max(prices))
            winning_outcome = outcomes[winning_index]

            correct = (winning_outcome.lower() == str(bet.get("recommended_outcome") or "").lower())
            entry_price = max(float(bet.get("current_price") or 0.01), 0.01)
            if correct:
                pnl = round(bet["hypothetical_usdc"] * (1.0 / entry_price - 1.0), 4)
            else:
                pnl = -bet["hypothetical_usdc"]

            updated = {
                **bet,
                "status": "resolved",
                "resolved_outcome": winning_outcome,
                "resolved_at": resolution_time,
                "pnl_usdc": pnl,
                "correct": correct,
            }

            bet_map[bet["bet_id"]] = updated
            _append_resolved(updated)
            resolved_count += 1

            symbol = "✓" if correct else "✗"
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            LOGGER.info(
                "RESOLVED %s '%s' → %s (predicted %s) %s %s",
                symbol,
                bet["question"][:50],
                winning_outcome,
                bet.get("recommended_outcome"),
                symbol,
                pnl_str,
            )

        except Exception as exc:
            LOGGER.warning("resolution check failed for %s: %s", condition_id, exc)

    if resolved_count > 0:
        _rewrite_paper_bets(list(bet_map.values()))

    return resolved_count


def load_resolved() -> list[dict]:
    if not RESOLVED_PATH.exists():
        return []
    results = []
    with RESOLVED_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def load_pending() -> list[dict]:
    return [b for b in _load_paper_bets() if b.get("status") == "pending"]
