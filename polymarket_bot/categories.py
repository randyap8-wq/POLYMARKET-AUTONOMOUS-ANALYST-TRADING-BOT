"""Lightweight keyword-based market categorisation.

Categorising markets lets the validator break down win rate per category and
lets operators disable weak categories (see ``DISABLED_CATEGORIES`` in config).
The classifier is intentionally simple and dependency-free: it scans the market
question for category keywords in priority order and returns the first match.
"""

from __future__ import annotations

import re

OTHER = "other"

# Ordered by priority: the first category whose keywords match wins. More
# specific / higher-signal categories are listed first so, e.g., "Bitcoin
# election odds" classifies as crypto before politics only if crypto keywords
# appear — order here breaks ties when multiple categories match.
_CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    (
        "crypto",
        (
            "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
            "dogecoin", "blockchain", "binance", "stablecoin", "altcoin",
            "memecoin", "nft", "coinbase",
        ),
    ),
    (
        "politics",
        (
            "election", "president", "presidential", "senate", "congress",
            "vote", "governor", "primary", "parliament", "prime minister",
            "impeach", "nominee", "ballot", "referendum", "cabinet",
            "secretary of", "supreme court", "candidate",
        ),
    ),
    (
        "economics",
        (
            "fed", "federal reserve", "inflation", "gdp", "interest rate",
            "recession", "unemployment", "cpi", "jobs report", "rate cut",
            "rate hike", "economy", "tariff", "debt ceiling", "treasury",
        ),
    ),
    (
        "sports",
        (
            "nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball",
            "super bowl", "world cup", "playoff", "ufc", "tennis", "olympics",
            "champion", "championship", "premier league", "la liga", "f1",
            "grand prix", "match", "tournament",
        ),
    ),
    (
        "entertainment",
        (
            "movie", "box office", "oscar", "oscars", "grammy", "album",
            "celebrity", "song", "award", "billboard", "netflix", "emmy",
            "rotten tomatoes", "concert", "tour",
        ),
    ),
    (
        "science_tech",
        (
            "openai", "spacex", "nasa", "rocket", "launch", "vaccine",
            "gpt", "llm", "ai model", "artificial intelligence", "tesla",
            "apple", "nvidia", "iphone", "satellite", "fusion",
        ),
    ),
    (
        "weather",
        (
            "hurricane", "temperature", "rainfall", "snowfall", "storm",
            "heatwave", "tornado", "el nino", "la nina", "wildfire",
        ),
    ),
]


def categorize(question: str | None) -> str:
    """Return a coarse category label for a market question."""
    if not question:
        return OTHER

    text = question.lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        for keyword in keywords:
            # Word-boundary match so "eth" doesn't fire on "whether".
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, text):
                return category
    return OTHER
