from __future__ import annotations

import logging
import time

from tavily import TavilyClient

try:
    from .config import TAVILY_API_KEY
except ImportError:  # pragma: no cover
    from config import TAVILY_API_KEY

LOGGER = logging.getLogger("news")


def fetch_news(question: str) -> list[dict]:
    if not TAVILY_API_KEY:
        LOGGER.warning("TAVILY_API_KEY is not configured; skipping news search")
        return []

    client = TavilyClient(api_key=TAVILY_API_KEY)
    try:
        response = client.search(
            query=question,
            search_depth="advanced",
            max_results=5,
            days=3,
            include_answer=False,
        )
    except Exception as exc:  # pragma: no cover - depends on external service
        LOGGER.error("tavily search failed: %s", exc)
        return []
    finally:
        time.sleep(1)

    results = response.get("results", [])[:3]
    return [
        {
            "title": item.get("title", ""),
            "snippet": item.get("content", item.get("snippet", "")),
            "url": item.get("url", ""),
            "published_date": item.get("published_date", item.get("date", "")),
        }
        for item in results
    ]
