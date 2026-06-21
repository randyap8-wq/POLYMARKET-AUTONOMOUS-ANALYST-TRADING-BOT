from __future__ import annotations

import unittest

from polymarket_bot.report import build_report, _kelly_fraction


def _scored(**overrides):
    base = {
        "question": "Market",
        "url": "https://example.com",
        "recommended_outcome": "Yes",
        "current_price": 0.25,
        "fair_value_estimate": 0.45,
        "edge": 0.20,
        "confidence": "high",
        "reasoning": "Strong signal.",
        "news_supports_bet": True,
        "category": "politics",
        "counter_evidence_considered": True,
        "top_news": [],
        "volume": 50000,
        "end_date": "2026-06-30T00:00:00Z",
        "condition_id": "c1",
    }
    base.update(overrides)
    return base


class BuildReportTests(unittest.TestCase):
    def test_build_report_filters_and_ranks_opportunities(self):
        markets = [{"id": "m1"}, {"id": "m2"}]
        scored = [
            _scored(question="Best market", top_news=[{"title": "Headline", "url": "https://news", "published_date": "2026-06-21"}]),
            _scored(question="Below threshold", edge=0.05, current_price=0.40, condition_id="c2"),
        ]

        report = build_report(markets, scored)

        self.assertEqual(report["markets_scanned"], 2)
        self.assertEqual(report["markets_scored"], 2)
        self.assertEqual(report["opportunities_found"], 1)
        self.assertEqual(report["opportunities"][0]["rank"], 1)
        self.assertEqual(report["opportunities"][0]["recommended_outcome"], "Yes")
        self.assertEqual(report["opportunities"][0]["category"], "politics")
        self.assertEqual(report["opportunities"][0]["top_news"][0]["published_date"], "2026-06-21")

    def test_extreme_price_opportunities_are_filtered(self):
        scored = [
            _scored(question="Too cheap", current_price=0.02),
            _scored(question="Too rich", current_price=0.98, condition_id="c2"),
            _scored(question="Just right", current_price=0.30, condition_id="c3"),
        ]

        report = build_report([{"id": "m1"}], scored)

        self.assertEqual(report["opportunities_found"], 1)
        self.assertEqual(report["opportunities"][0]["question"], "Just right")

    def test_token_usage_is_passed_through(self):
        usage = {"total_tokens": 1234, "cost_usd": 0.05, "calls": 3}
        report = build_report([], [], token_usage=usage)
        self.assertEqual(report["token_usage"], usage)

    def test_kelly_fraction_positive_when_edge_real(self):
        self.assertGreater(_kelly_fraction(0.25, 0.45), 0.0)

    def test_kelly_fraction_zero_without_edge(self):
        self.assertEqual(_kelly_fraction(0.50, 0.50), 0.0)


if __name__ == "__main__":
    unittest.main()
