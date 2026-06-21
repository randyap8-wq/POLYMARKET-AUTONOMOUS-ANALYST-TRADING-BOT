from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

from tavily import TavilyClient

try:
    from .config import (
        TAVILY_API_KEY,
        NEWS_WINDOW_MIN_DAYS,
        NEWS_WINDOW_MAX_DAYS,
        COUNTER_EVIDENCE_ENABLED,
    )
except ImportError:  # pragma: no cover
    from config import (
        TAVILY_API_KEY,
        NEWS_WINDOW_MIN_DAYS,
        NEWS_WINDOW_MAX_DAYS,
        COUNTER_EVIDENCE_ENABLED,
    )

LOGGER = logging.getLogger("news")


def _days_until(end_date: str | None) -> int | None:
    if not end_date:
        return None
    try:
        end = datetime.fromisoformat(str(end_date).replace("Z", "+00:00"))
        if end.tzinfo is None:
            end = end.replace(tzinfo=timezone.utc)
        return (end - datetime.now(timezone.utc)).days
    except (ValueError, TypeError):
        return None


def news_window_days(end_date: str | None) -> int:
    """Pick a Tavily search window.

    Fast-moving markets (closing soon) stay near the minimum window so the model
    only sees fresh, decision-relevant news. Slower-moving markets that close
    further out widen the window (up to the configured max) so enough context is
    captured.
    """
    days_to_close = _days_until(end_date)
    if days_to_close is None:
        return NEWS_WINDOW_MIN_DAYS
    return max(NEWS_WINDOW_MIN_DAYS, min(days_to_close, NEWS_WINDOW_MAX_DAYS))


def _normalize(results: list[dict], stance: str) -> list[dict]:
    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("content", item.get("snippet", "")),
            "url": item.get("url", ""),
            "published_date": item.get("published_date", item.get("date", "")),
            "stance": stance,
        }
        for item in results
    ]


def _search(client: TavilyClient, query: str, days: int, max_results: int) -> list[dict]:
    try:
        response = client.search(
            query=query,
            search_depth="advanced",
            max_results=max_results,
            days=days,
            include_answer=False,
        )
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.error("tavily search failed: %s", exc)
        return []
    finally:
        time.sleep(1)
    return response.get("results", [])


def fetch_news(
    question: str,
    end_date: str | None = None,
    *,
    counter_evidence: bool | None = None,
) -> list[dict]:
    if not TAVILY_API_KEY:
        LOGGER.warning("TAVILY_API_KEY is not configured; skipping news search")
        return []

    if counter_evidence is None:
        counter_evidence = COUNTER_EVIDENCE_ENABLED

    days = news_window_days(end_date)
    client = TavilyClient(api_key=TAVILY_API_KEY)

    supporting = _normalize(_search(client, question, days, max_results=5)[:3], "supporting")

    counter: list[dict] = []
    if counter_evidence:
        counter_query = (
            f"{question} unlikely OR \"will not\" OR fails OR delayed OR "
            f"denied OR evidence against"
        )
        counter = _normalize(_search(client, counter_query, days, max_results=4)[:2], "counter")

    # Merge, de-duplicating by URL while keeping supporting items first.
    merged: list[dict] = []
    seen: set[str] = set()
    for item in supporting + counter:
        key = item["url"] or item["title"]
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        merged.append(item)

    return merged
