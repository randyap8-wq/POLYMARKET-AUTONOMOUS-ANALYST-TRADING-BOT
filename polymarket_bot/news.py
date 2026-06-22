"""news.py — DEPRECATED.

News retrieval is now handled internally by Gemini's Google Search grounding in
``scorer.py``. This module is kept as an importable stub so older call sites
(``main``, ``backfill``) and tests keep working during the migration away from
the paid Tavily dependency.
"""

from __future__ import annotations

import logging

LOGGER = logging.getLogger("news")


def fetch_news(
    question: str,
    end_date: str | None = "",
    *,
    counter_evidence: bool | None = None,
) -> list[dict]:
    """Stub — Gemini handles news retrieval internally via search grounding."""
    LOGGER.debug("news.fetch_news called (stub) — Gemini handles search internally")
    return []
