from __future__ import annotations

import unittest

from polymarket_bot.scorer import score_market


class ScoreMarketTests(unittest.TestCase):
    def test_score_market_returns_neutral_pick_when_news_missing(self):
        market = {
            "condition_id": "cond-1",
            "question": "Will it rain?",
            "outcomes": ["Yes", "No"],
            "prices": [0.4, 0.6],
        }

        score = score_market(market, [])

        self.assertIsNotNone(score)
        self.assertIsNone(score["recommended_outcome"])
        self.assertEqual(score["edge"], 0.0)
        self.assertFalse(score["news_supports_bet"])
        self.assertEqual(score["condition_id"], "cond-1")


if __name__ == "__main__":
    unittest.main()
