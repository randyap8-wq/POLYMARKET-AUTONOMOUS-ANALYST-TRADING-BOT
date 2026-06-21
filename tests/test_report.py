from __future__ import annotations

import unittest

from polymarket_bot.report import build_report


class BuildReportTests(unittest.TestCase):
    def test_build_report_filters_and_ranks_opportunities(self):
        markets = [{"id": "m1"}, {"id": "m2"}]
        scored = [
            {
                "question": "Best market",
                "url": "https://example.com/1",
                "recommended_outcome": "Yes",
                "current_price": 0.25,
                "fair_value_estimate": 0.45,
                "edge": 0.20,
                "confidence": "high",
                "reasoning": "Strong signal.",
                "news_supports_bet": True,
                "top_news": [{"title": "Headline", "url": "https://news", "published_date": "2026-06-21"}],
                "volume": 50000,
                "end_date": "2026-06-30T00:00:00Z",
                "condition_id": "c1",
            },
            {
                "question": "Below threshold",
                "url": "https://example.com/2",
                "recommended_outcome": "No",
                "current_price": 0.40,
                "fair_value_estimate": 0.45,
                "edge": 0.05,
                "confidence": "high",
                "reasoning": "Too small.",
                "news_supports_bet": True,
                "top_news": [],
                "volume": 25000,
                "end_date": "2026-06-30T00:00:00Z",
                "condition_id": "c2",
            },
        ]

        report = build_report(markets, scored)

        self.assertEqual(report["markets_scanned"], 2)
        self.assertEqual(report["markets_scored"], 2)
        self.assertEqual(report["opportunities_found"], 1)
        self.assertEqual(report["opportunities"][0]["rank"], 1)
        self.assertEqual(report["opportunities"][0]["recommended_outcome"], "Yes")
        self.assertEqual(report["opportunities"][0]["top_news"][0]["date"], "2026-06-21")


if __name__ == "__main__":
    unittest.main()
