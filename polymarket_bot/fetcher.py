from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import requests

try:
    from .config import GAMMA_BASE, MARKET_FILTERS, DISABLED_CATEGORIES
    from .categories import categorize
except ImportError:  # pragma: no cover
    from config import GAMMA_BASE, MARKET_FILTERS, DISABLED_CATEGORIES
    from categories import categorize

LOGGER = logging.getLogger("fetcher")


def _parse_end_date(raw_value: str) -> datetime:
    normalized = raw_value.replace("Z", "+00:00")
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def fetch_markets() -> list[dict]:
    all_raw: list[dict] = []
    seen_ids: set[str] = set()

    limit = MARKET_FILTERS["limit"]
    pages = 3
    for offset in range(0, limit * pages, limit):
        try:
            response = requests.get(
                f"{GAMMA_BASE}/markets",
                params={
                    "active": "true",
                    "closed": "false",
                    "limit": MARKET_FILTERS["limit"],
                    "offset": offset,
                    "order": "volume",
                    "ascending": "false",
                },
                timeout=30,
            )
            response.raise_for_status()
            page = response.json()
            if not page:
                break
            for market in page:
                mid = market.get("id")
                if mid and mid not in seen_ids:
                    seen_ids.add(mid)
                    all_raw.append(market)
        except Exception as exc:
            LOGGER.warning("failed to fetch page at offset %s: %s", offset, exc)
            break

    now = datetime.now(timezone.utc)
    dropped_volume = 0
    dropped_end_date = 0
    dropped_outcomes = 0
    dropped_category = 0
    filtered: list[dict] = []

    for market in all_raw:
        try:
            end_date = _parse_end_date(market["endDate"])
            outcomes = json.loads(market["outcomes"])
            prices = [float(price) for price in json.loads(market["outcomePrices"])]
            try:
                token_ids = [str(t) for t in json.loads(market.get("clobTokenIds") or "[]")]
            except (TypeError, ValueError, json.JSONDecodeError):
                token_ids = []
            normalized = {
                "id": market["id"],
                "condition_id": market["conditionId"],
                "question": market["question"],
                "outcomes": outcomes,
                "prices": prices,
                "token_ids": token_ids,
                "volume": float(market.get("volume", 0) or 0),
                "end_date": market["endDate"],
                "slug": market.get("slug", ""),
                "url": f"https://polymarket.com/event/{market.get('slug', '')}",
                "category": categorize(market.get("question", "")),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            LOGGER.warning("skipping malformed market: %s", exc)
            continue

        if normalized["volume"] < MARKET_FILTERS["min_volume"]:
            dropped_volume += 1
            continue

        days_to_close = (end_date - now).total_seconds() / 86400
        if days_to_close > MARKET_FILTERS["max_days_to_close"] or days_to_close <= 0:
            dropped_end_date += 1
            continue

        if len(normalized["outcomes"]) > MARKET_FILTERS["max_outcomes"]:
            dropped_outcomes += 1
            continue

        if normalized["category"] in DISABLED_CATEGORIES:
            dropped_category += 1
            continue

        filtered.append(normalized)

    LOGGER.info("dropped %s markets for low volume", dropped_volume)
    LOGGER.info("dropped %s markets for end date window", dropped_end_date)
    LOGGER.info("dropped %s markets for too many outcomes", dropped_outcomes)
    if DISABLED_CATEGORIES:
        LOGGER.info("dropped %s markets for disabled categories", dropped_category)
    return filtered
