from __future__ import annotations

import unittest

from polymarket_bot import scorer
from polymarket_bot.scorer import score_market, reset_token_usage, get_token_usage


class ScoreMarketTests(unittest.TestCase):
    def test_score_market_returns_neutral_pick_when_news_missing(self):
        market = {
            "condition_id": "cond-1",
            "question": "Will it rain?",
            "outcomes": ["Yes", "No"],
            "prices": [0.4, 0.6],
            "category": "weather",
        }

        score = score_market(market, [])

        self.assertIsNotNone(score)
        self.assertIsNone(score["recommended_outcome"])
        self.assertEqual(score["edge"], 0.0)
        self.assertFalse(score["news_supports_bet"])
        self.assertEqual(score["condition_id"], "cond-1")
        self.assertEqual(score["category"], "weather")


class TokenUsageTests(unittest.TestCase):
    def setUp(self):
        reset_token_usage()
        self.addCleanup(reset_token_usage)

    def test_record_usage_accumulates_and_costs(self):
        scorer._record_usage(
            scorer.DEEPSEEK_MODEL,
            {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
        )
        usage = get_token_usage()

        self.assertEqual(usage["calls"], 1)
        self.assertEqual(usage["prompt_tokens"], 1_000_000)
        self.assertEqual(usage["total_tokens"], 2_000_000)
        expected = (
            scorer.DEEPSEEK_PRICING[scorer.DEEPSEEK_MODEL]["input"]
            + scorer.DEEPSEEK_PRICING[scorer.DEEPSEEK_MODEL]["output"]
        )
        self.assertAlmostEqual(usage["cost_usd"], round(expected, 6), places=6)

    def test_reset_zeroes_usage(self):
        scorer._record_usage(scorer.DEEPSEEK_MODEL, {"prompt_tokens": 10, "completion_tokens": 5})
        reset_token_usage()
        usage = get_token_usage()
        self.assertEqual(usage["calls"], 0)
        self.assertEqual(usage["total_tokens"], 0)
        self.assertEqual(usage["cost_usd"], 0.0)

    def test_record_usage_ignores_missing(self):
        scorer._record_usage(scorer.DEEPSEEK_MODEL, None)
        self.assertEqual(get_token_usage()["calls"], 0)


if __name__ == "__main__":
    unittest.main()

