"""Full-pipeline backtest using cached AI scores, quant fusion and risk sizing."""

from __future__ import annotations

import json
import logging
import math
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from statistics import mean, pstdev
from typing import Any, Iterator

import requests

try:
    from .categories import categorize
    from .config import BASE_DIR, GAMMA_BASE
    from .market_data import fetch_price_history
    from .report import build_report
    from .scorer import read_cached_score, score_market
    from .utils import retry_with_backoff
except ImportError:  # pragma: no cover
    from categories import categorize
    from config import BASE_DIR, GAMMA_BASE
    from market_data import fetch_price_history
    from report import build_report
    from scorer import read_cached_score, score_market
    from utils import retry_with_backoff

LOGGER = logging.getLogger("backtest_full")

FULL_BACKTEST_PATH = BASE_DIR / "data" / "full_backtest.json"
_EMPTY_BOOK = {"bids": [], "asks": []}


@retry_with_backoff(max_retries=5, base_delay=1)
def _fetch_closed_gamma_markets(limit: int) -> list[dict[str, Any]]:
    response = requests.get(
        f"{GAMMA_BASE}/markets",
        params={"closed": "true", "limit": limit, "order": "volume", "ascending": "false"},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    return payload if isinstance(payload, list) else [payload]


def _parse_dt(value: Any) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _fetch_resolved_markets(days_back: int, limit: int) -> list[dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    try:
        raw = _fetch_closed_gamma_markets(limit)
    except Exception as exc:
        LOGGER.error("failed to fetch closed markets: %s", exc)
        return []

    markets: list[dict[str, Any]] = []
    for market in raw:
        try:
            resolution_time = market.get("resolutionTime") or market.get("resolution_time")
            resolved_at = _parse_dt(resolution_time)
            if resolved_at is None or resolved_at < cutoff:
                continue

            outcomes = json.loads(market.get("outcomes") or "[]")
            prices = [float(price) for price in json.loads(market.get("outcomePrices") or "[]")]
            token_ids = [str(token) for token in json.loads(market.get("clobTokenIds") or "[]")]
            if not outcomes or not token_ids or len(outcomes) != len(token_ids) or max(prices, default=0.0) < 0.99:
                continue

            markets.append(
                {
                    "id": market.get("id", market.get("conditionId", "")),
                    "condition_id": market.get("conditionId", market.get("condition_id", "")),
                    "question": market.get("question", ""),
                    "outcomes": outcomes,
                    "prices": prices,
                    "token_ids": token_ids,
                    "winner_index": prices.index(max(prices)),
                    "volume": float(market.get("volume", 0) or 0),
                    "end_date": market.get("endDate") or resolution_time,
                    "slug": market.get("slug", ""),
                    "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                    "category": categorize(market.get("question", "")),
                    "resolved_at": resolved_at.isoformat(),
                }
            )
        except Exception as exc:
            LOGGER.debug("skipping resolved market: %s", exc)
    return markets


def _clamp_price(value: Any) -> float:
    try:
        return max(0.01, min(0.99, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _entry_point(history: list[dict[str, Any]], entry_fraction: float) -> tuple[int, float, str]:
    if not history:
        return 0, 0.5, datetime.now(timezone.utc).date().isoformat()
    index = int(len(history) * entry_fraction)
    index = max(0, min(index, len(history) - 1))
    row = history[index]
    timestamp = row.get("t")
    if timestamp is not None:
        try:
            cache_date = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).date().isoformat()
        except (TypeError, ValueError, OSError):
            cache_date = datetime.now(timezone.utc).date().isoformat()
    else:
        cache_date = datetime.now(timezone.utc).date().isoformat()
    return index, _clamp_price(row.get("p")), cache_date


def _synthetic_book(price: float, volume: float) -> dict[str, list[dict[str, float]]]:
    size = max(volume / max(price, 0.01), 1_000.0)
    return {
        "bids": [{"price": round(max(price - 0.01, 0.01), 4), "size": round(size, 2)}],
        "asks": [{"price": round(min(price + 0.01, 0.99), 4), "size": round(size, 2)}],
    }


def _proxy_ai_score(market: dict[str, Any]) -> dict[str, Any]:
    prices = [_clamp_price(price) for price in market.get("prices", [])]
    outcomes = market.get("outcomes") or []
    if not prices or not outcomes:
        return {
            "recommended_outcome": None,
            "recommended_outcome_index": None,
            "current_price": 0.0,
            "fair_value_estimate": 0.0,
            "probability": 0.0,
            "edge": 0.0,
            "confidence": 0.0,
            "confidence_level": "low",
            "direction": "SKIP",
            "news_supports_bet": False,
            "reasoning": "no historical AI cache or market prices available",
        }

    index = max(range(len(prices)), key=lambda idx: prices[idx])
    current_price = prices[index]
    fair_value = min(0.99, current_price + 0.04)
    edge = fair_value - current_price
    return {
        "recommended_outcome": outcomes[index] if edge > 0.02 else None,
        "recommended_outcome_index": index if edge > 0.02 else None,
        "current_price": current_price,
        "fair_value_estimate": fair_value,
        "probability": fair_value,
        "edge": round(edge, 4),
        "confidence": 60.0,
        "confidence_level": "medium",
        "direction": "BUY" if edge > 0.02 else "SKIP",
        "news_supports_bet": edge > 0.02,
        "reasoning": "historical proxy because no cached Gemini response was available",
        "condition_id": market.get("condition_id", ""),
        "category": market.get("category", "other"),
    }


def _cached_or_live_score(market: dict[str, Any], *, allow_live_ai: bool) -> dict[str, Any] | None:
    cached = read_cached_score(market, max_age_hours=None)
    if cached is not None:
        return cached
    if not allow_live_ai:
        return _proxy_ai_score(market)
    return score_market(market)


@contextmanager
def _patched_pipeline(
    histories_by_token: dict[str, list[dict[str, Any]]],
    books_by_token: dict[str, dict[str, list[dict[str, float]]]],
    stats: dict[str, float],
    *,
    allow_live_ai: bool,
) -> Iterator[None]:
    try:
        from . import pipeline
    except ImportError:  # pragma: no cover
        import pipeline  # type: ignore

    original_history = pipeline.fetch_price_history
    original_book = pipeline.fetch_order_book
    original_stats = pipeline.fetch_market_stats
    original_score = pipeline.score_market

    pipeline.fetch_price_history = lambda token_id: histories_by_token.get(str(token_id), [])
    pipeline.fetch_order_book = lambda token_id: books_by_token.get(str(token_id), _EMPTY_BOOK)
    pipeline.fetch_market_stats = lambda _condition_id: stats
    pipeline.score_market = lambda market: _cached_or_live_score(market, allow_live_ai=allow_live_ai)
    try:
        yield
    finally:
        pipeline.fetch_price_history = original_history
        pipeline.fetch_order_book = original_book
        pipeline.fetch_market_stats = original_stats
        pipeline.score_market = original_score


def _settle(stake: float, price: float, win: bool) -> float:
    if stake <= 0:
        return 0.0
    if not win:
        return -stake
    return stake * (1.0 / max(price, 0.01) - 1.0)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trades = [row for row in rows if row.get("stake_usdc", 0.0) > 0]
    returns = [row["pnl_usdc"] / row["stake_usdc"] for row in trades if row.get("stake_usdc", 0.0) > 0]
    total_pnl = sum(row.get("pnl_usdc", 0.0) for row in trades)
    sharpe = 0.0
    if len(returns) > 1 and pstdev(returns) > 1e-12:
        sharpe = mean(returns) / pstdev(returns) * math.sqrt(len(returns))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "decisions": len(rows),
        "trades": len(trades),
        "win_rate": round(sum(1 for row in trades if row.get("win")) / len(trades), 4) if trades else 0.0,
        "total_pnl_usdc": round(total_pnl, 4),
        "sharpe_ratio": round(sharpe, 4),
    }


def run_full_backtest(
    days_back: int = 30,
    limit: int = 60,
    entry_fraction: float = 0.6,
    *,
    allow_live_ai: bool = False,
) -> dict[str, Any]:
    """Replay resolved markets through analyze_market, report filtering and risk."""
    try:
        from . import pipeline
    except ImportError:  # pragma: no cover
        import pipeline  # type: ignore

    markets = _fetch_resolved_markets(days_back, limit)
    rows: list[dict[str, Any]] = []

    for market in markets:
        histories = {token_id: fetch_price_history(token_id) for token_id in market["token_ids"]}
        if not any(histories.values()):
            continue

        entry_prices: list[float] = []
        cache_date = datetime.now(timezone.utc).date().isoformat()
        clipped_histories: dict[str, list[dict[str, Any]]] = {}
        books: dict[str, dict[str, list[dict[str, float]]]] = {}

        for token_id in market["token_ids"]:
            history = histories.get(token_id, [])
            index, price, entry_date = _entry_point(history, entry_fraction)
            entry_prices.append(price)
            cache_date = entry_date
            clipped_histories[token_id] = history[: index + 1]
            books[token_id] = _synthetic_book(price, market.get("volume", 0.0))

        entry_market = {
            **market,
            "prices": entry_prices,
            "cache_date": cache_date,
            "volume_24h": market.get("volume", 0.0),
        }
        stats = {"volume_24h": float(market.get("volume", 0.0) or 0.0), "open_interest": 0.0}

        with _patched_pipeline(clipped_histories, books, stats, allow_live_ai=allow_live_ai):
            score = pipeline.analyze_market(entry_market)

        if score is None:
            rows.append(
                {
                    "condition_id": market["condition_id"],
                    "question": market["question"],
                    "cache_date": cache_date,
                    "recommended_outcome": None,
                    "stake_usdc": 0.0,
                    "pnl_usdc": 0.0,
                    "win": None,
                }
            )
            continue

        scored = [{**entry_market, **score, "top_news": []}]
        report = build_report([entry_market], scored, token_usage={})
        opportunity = report["opportunities"][0] if report["opportunities"] else None
        if opportunity is None:
            rows.append(
                {
                    "condition_id": market["condition_id"],
                    "question": market["question"],
                    "cache_date": cache_date,
                    "recommended_outcome": score.get("recommended_outcome"),
                    "recommended_outcome_index": score.get("recommended_outcome_index"),
                    "edge": score.get("edge", 0.0),
                    "stake_usdc": 0.0,
                    "pnl_usdc": 0.0,
                    "win": None,
                }
            )
            continue

        index = opportunity.get("recommended_outcome_index")
        if index is None:
            index = market["outcomes"].index(opportunity["recommended_outcome"])
        win = int(index) == int(market["winner_index"])
        stake = float(opportunity.get("stake_usdc") or 0.0)
        price = _clamp_price(opportunity.get("current_price"))
        pnl = _settle(stake, price, win)
        rows.append(
            {
                **opportunity,
                "cache_date": cache_date,
                "winner_index": market["winner_index"],
                "win": win,
                "pnl_usdc": round(pnl, 4),
                "return_on_stake": round(pnl / stake, 4) if stake > 0 else 0.0,
            }
        )

    summary = _summary(rows)
    payload = {"summary": summary, "trades": rows}
    FULL_BACKTEST_PATH.parent.mkdir(exist_ok=True)
    FULL_BACKTEST_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n" + "=" * 58)
    print(" FULL BACKTEST — AI + QUANT + RISK")
    print("=" * 58)
    print(f" Decisions : {summary['decisions']}")
    print(f" Trades    : {summary['trades']}")
    print(f" Win Rate  : {summary['win_rate'] * 100:.1f}%")
    print(f" Sharpe    : {summary['sharpe_ratio']:+.2f}")
    print(f" P&L       : ${summary['total_pnl_usdc']:+.2f}")
    print(f" Saved to  : {FULL_BACKTEST_PATH}")
    print("=" * 58 + "\n")
    return payload
