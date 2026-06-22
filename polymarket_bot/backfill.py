from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone, timedelta

import requests

try:
    from .config import GAMMA_BASE, BASE_DIR
    from .categories import categorize
    from .scorer import score_market, reset_token_usage
except ImportError:  # pragma: no cover
    from config import GAMMA_BASE, BASE_DIR
    from categories import categorize
    from scorer import score_market, reset_token_usage

LOGGER = logging.getLogger("backfill")

DATA_DIR = BASE_DIR / "data"
RESOLVED_PATH = DATA_DIR / "resolved.jsonl"
DATA_DIR.mkdir(exist_ok=True)


def _fetch_recently_closed(days_back: int = 14, limit: int = 50) -> list[dict]:
    """Fetch markets that closed in the last `days_back` days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        resp = requests.get(
            f"{GAMMA_BASE}/markets",
            params={
                "closed": "true",
                "limit": limit,
                "order": "volume",
                "ascending": "false",
            },
            timeout=30,
        )
        resp.raise_for_status()
        all_markets = resp.json()
    except Exception as exc:
        LOGGER.error("failed to fetch closed markets: %s", exc)
        return []

    # Only keep markets that actually resolved (have a winner) and closed recently
    result = []
    for m in all_markets:
        try:
            resolution_time = m.get("resolutionTime") or m.get("resolution_time")
            if not resolution_time:
                continue
            res_dt = datetime.fromisoformat(str(resolution_time).replace("Z", "+00:00"))
            if res_dt.tzinfo is None:
                res_dt = res_dt.replace(tzinfo=timezone.utc)
            if res_dt < cutoff:
                continue  # too old

            outcomes = json.loads(m.get("outcomes") or "[]")
            prices = [float(p) for p in json.loads(m.get("outcomePrices") or "[]")]
            if not prices or max(prices) < 0.99:
                continue  # not cleanly resolved

            winning_index = prices.index(max(prices))
            result.append({
                "id": m["id"],
                "condition_id": m["conditionId"],
                "question": m["question"],
                "outcomes": outcomes,
                # Feed every outcome a flat 0.5 implied probability instead of the
                # true pre-resolution book price, so the scorer always "sees" a
                # 50/50 market here. We don't have point-in-time prices for these
                # closed markets, and backfill is already acknowledged as
                # look-ahead biased (Gemini fetches news as of *now* — see
                # run_backfill's docstring), so this placeholder is deliberate, not
                # a bug. Use --backtest for a bias-free, point-in-time price read.
                "prices": [0.5] * len(outcomes),
                "volume": float(m.get("volume", 0) or 0),
                "end_date": m.get("endDate", ""),
                "slug": m.get("slug", ""),
                "url": f"https://polymarket.com/event/{m.get('slug', '')}",
                "category": categorize(m.get("question", "")),
                "resolved_outcome": outcomes[winning_index],
                "resolved_at": resolution_time,
            })
        except Exception as exc:
            LOGGER.warning("skipping malformed closed market: %s", exc)
    return result


def _already_backfilled(condition_id: str) -> bool:
    """Check resolved.jsonl for an existing backfill entry for this market."""
    if not RESOLVED_PATH.exists():
        return False
    with RESOLVED_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                b = json.loads(line.strip())
                if b.get("condition_id") == condition_id and b.get("source") == "backfill":
                    return True
            except json.JSONDecodeError:
                pass
    return False


def run_backfill(days_back: int = 14, limit: int = 30) -> int:
    """
    Score recently closed markets and write synthetic resolved bets.
    Returns number of bets written.

    IMPORTANT: Gemini fetches news NOW (via search grounding), not at market
    close — this has look-ahead bias and will overstate AI accuracy. Use backfill
    data only to bootstrap the validator. For a bias-free read on the *quant*
    signal, use `--backtest`, which replays CLOB price history point-in-time.
    """
    markets = _fetch_recently_closed(days_back=days_back, limit=limit)
    LOGGER.info("backfill: found %s recently closed markets", len(markets))

    reset_token_usage()
    written = 0

    for market in markets:
        condition_id = market.get("condition_id", "")
        if _already_backfilled(condition_id):
            LOGGER.debug("skipping already-backfilled market %s", condition_id)
            continue

        try:
            score = score_market(market)  # Gemini fetches its own news via search grounding
            if not score or score.get("recommended_outcome") is None:
                continue

            correct = (
                str(score.get("recommended_outcome") or "").lower()
                == str(market["resolved_outcome"]).lower()
            )
            entry_price = max(float(score.get("current_price") or 0.5), 0.01)
            pnl = round(10.0 * (1.0 / entry_price - 1.0), 4) if correct else -10.0

            entry = {
                "bet_id": str(uuid.uuid4()),
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "question": market["question"],
                "url": market["url"],
                "condition_id": condition_id,
                "category": market.get("category", "other"),
                "recommended_outcome": score.get("recommended_outcome"),
                "current_price": entry_price,
                "fair_value_estimate": float(score.get("fair_value_estimate") or 0.0),
                "edge": float(score.get("edge") or 0.0),
                "confidence": score.get("confidence", "low"),
                "reasoning": score.get("reasoning", ""),
                "hypothetical_usdc": 10.0,
                "hypothetical_tokens": round(10.0 / entry_price, 2),
                "end_date": market.get("end_date", ""),
                "status": "resolved",
                "resolved_outcome": market["resolved_outcome"],
                "resolved_at": market["resolved_at"],
                "pnl_usdc": pnl,
                "correct": correct,
                "source": "backfill",  # marks look-ahead bias; weight lower than live bets
            }

            with RESOLVED_PATH.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry) + "\n")

            symbol = "✓" if correct else "✗"
            LOGGER.info("%s %s → predicted %s, actual %s",
                        symbol, market["question"][:50],
                        score.get("recommended_outcome"), market["resolved_outcome"])
            written += 1

        except Exception as exc:
            LOGGER.warning("backfill failed for %s: %s", market.get("question", "")[:40], exc)

    LOGGER.info("backfill: wrote %s resolved bets to resolved.jsonl", written)
    return written
