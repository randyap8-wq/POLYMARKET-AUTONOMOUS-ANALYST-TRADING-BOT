from __future__ import annotations

import unittest

from polymarket_bot import news
from polymarket_bot.news import fetch_news


class NewsStubTests(unittest.TestCase):
    """news.py is a deprecated stub; Gemini handles search grounding now."""

    def test_fetch_news_returns_empty(self):
        self.assertEqual(fetch_news("Will X happen?"), [])

    def test_fetch_news_accepts_legacy_signature(self):
        # Old call sites pass an end_date and counter_evidence kwarg.
        self.assertEqual(
            fetch_news("Will X happen?", "2026-06-30T00:00:00Z", counter_evidence=True),
            [],
        )

    def test_module_exposes_fetch_news(self):
        self.assertTrue(hasattr(news, "fetch_news"))


if __name__ == "__main__":
    unittest.main()
