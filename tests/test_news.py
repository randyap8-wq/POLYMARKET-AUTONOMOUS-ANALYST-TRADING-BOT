from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from polymarket_bot import news
from polymarket_bot.news import news_window_days, fetch_news


class NewsWindowTests(unittest.TestCase):
    def test_window_clamps_to_min_for_soon_closing(self):
        soon = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
        self.assertEqual(news_window_days(soon), news.NEWS_WINDOW_MIN_DAYS)

    def test_window_widens_for_slow_market(self):
        mid = (datetime.now(timezone.utc) + timedelta(days=12, hours=1)).isoformat()
        self.assertEqual(news_window_days(mid), 12)

    def test_window_caps_at_max(self):
        far = (datetime.now(timezone.utc) + timedelta(days=120)).isoformat()
        self.assertEqual(news_window_days(far), news.NEWS_WINDOW_MAX_DAYS)

    def test_window_defaults_when_no_date(self):
        self.assertEqual(news_window_days(None), news.NEWS_WINDOW_MIN_DAYS)


class FetchNewsTests(unittest.TestCase):
    def setUp(self):
        sleep_patch = patch.object(news.time, "sleep", lambda *a, **k: None)
        sleep_patch.start()
        self.addCleanup(sleep_patch.stop)

    @patch.object(news, "TAVILY_API_KEY", "fake-key")
    @patch.object(news, "TavilyClient")
    def test_counter_evidence_runs_second_query_and_tags_stance(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.search.side_effect = [
            {"results": [{"title": "Yes likely", "url": "https://a", "content": "support"}]},
            {"results": [{"title": "But maybe not", "url": "https://b", "content": "doubt"}]},
        ]

        results = fetch_news("Will X happen?", None, counter_evidence=True)

        self.assertEqual(client.search.call_count, 2)
        stances = {item["stance"] for item in results}
        self.assertEqual(stances, {"supporting", "counter"})

    @patch.object(news, "TAVILY_API_KEY", "fake-key")
    @patch.object(news, "TavilyClient")
    def test_counter_evidence_disabled_runs_single_query(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.search.return_value = {
            "results": [{"title": "Yes likely", "url": "https://a", "content": "support"}]
        }

        results = fetch_news("Will X happen?", None, counter_evidence=False)

        self.assertEqual(client.search.call_count, 1)
        self.assertTrue(all(item["stance"] == "supporting" for item in results))

    @patch.object(news, "TAVILY_API_KEY", "fake-key")
    @patch.object(news, "TavilyClient")
    def test_deduplicates_by_url(self, mock_client_cls):
        client = mock_client_cls.return_value
        client.search.side_effect = [
            {"results": [{"title": "Same", "url": "https://dup", "content": "s"}]},
            {"results": [{"title": "Same", "url": "https://dup", "content": "c"}]},
        ]

        results = fetch_news("Will X happen?", None, counter_evidence=True)

        self.assertEqual(len(results), 1)

    @patch.object(news, "TAVILY_API_KEY", "fake-key")
    @patch.object(news, "TavilyClient")
    def test_counter_query_uses_focused_negation(self, mock_client_cls):
        client = mock_client_cls.return_value
        queries = []

        def fake_search(query, **kwargs):
            queries.append(query)
            return {"results": []}

        client.search.side_effect = fake_search

        fetch_news("Will X happen?", None, counter_evidence=True)

        counter_calls = [q for q in queries if "reasons why NOT" in q]
        self.assertTrue(len(counter_calls) >= 1)
        self.assertFalse(any("evidence against" in q for q in queries))

    @patch.object(news, "TAVILY_API_KEY", "")
    def test_missing_key_returns_empty(self):
        self.assertEqual(fetch_news("anything"), [])


if __name__ == "__main__":
    unittest.main()
