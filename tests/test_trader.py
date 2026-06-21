from __future__ import annotations

import unittest
from unittest.mock import patch

from polymarket_bot.trader import place_bet


class PlaceBetTests(unittest.TestCase):
    @patch("polymarket_bot.trader._append_trade_log")
    @patch("polymarket_bot.trader._best_ask_price", return_value=0.30)
    @patch(
        "polymarket_bot.trader._fetch_market_tokens",
        return_value=[{"outcome": "Yes", "token_id": "token-123"}],
    )
    def test_place_bet_returns_dry_run_result(self, *_mocks):
        market = {"condition_id": "cond-1", "question": "Will it rain?"}
        score = {
            "recommended_outcome": "Yes",
            "current_price": 0.29,
            "edge": 0.20,
        }

        result = place_bet(market, score)

        self.assertIsNotNone(result)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["token_id"], "token-123")


if __name__ == "__main__":
    unittest.main()
