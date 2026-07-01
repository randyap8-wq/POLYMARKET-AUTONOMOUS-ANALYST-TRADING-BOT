from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

from polymarket_bot import market_data


def _resp(payload):
    mock = Mock()
    mock.json.return_value = payload
    mock.raise_for_status.return_value = None
    return mock


class PriceHistoryTests(unittest.TestCase):
    @patch("polymarket_bot.market_data.requests.get")
    def test_parses_and_sorts_history(self, mock_get):
        mock_get.return_value = _resp(
            {"history": [{"t": 2, "p": "0.20"}, {"t": 1, "p": "0.10"}]}
        )
        history = market_data.fetch_price_history("tok")
        self.assertEqual([pt["t"] for pt in history], [1.0, 2.0])
        self.assertEqual(history[0]["p"], 0.10)

    @patch("polymarket_bot.market_data.requests.get")
    def test_failure_returns_empty(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        self.assertEqual(market_data.fetch_price_history("tok"), [])

    def test_empty_token_returns_empty(self):
        self.assertEqual(market_data.fetch_price_history(""), [])


class OrderBookTests(unittest.TestCase):
    @patch("polymarket_bot.market_data.requests.get")
    def test_parses_levels_as_floats(self, mock_get):
        mock_get.return_value = _resp(
            {
                "bids": [{"price": "0.49", "size": "100"}],
                "asks": [{"price": "0.51", "size": "200"}],
            }
        )
        book = market_data.fetch_order_book("tok")
        self.assertEqual(book["bids"][0]["price"], 0.49)
        self.assertEqual(book["asks"][0]["size"], 200.0)

    @patch("polymarket_bot.market_data.requests.get")
    def test_failure_returns_empty_book(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        book = market_data.fetch_order_book("tok")
        self.assertEqual(book, {"bids": [], "asks": []})

    @patch("polymarket_bot.market_data.requests.get")
    def test_skips_malformed_levels(self, mock_get):
        mock_get.return_value = _resp({"bids": [{"price": "x"}], "asks": []})
        book = market_data.fetch_order_book("tok")
        self.assertEqual(book["bids"], [])


class MarketStatsTests(unittest.TestCase):
    @patch("polymarket_bot.market_data.requests.get")
    def test_parses_activity_stats(self, mock_get):
        mock_get.return_value = _resp({"volume24hr": "12345.6", "openInterest": "789"})
        stats = market_data.fetch_market_stats("cond")
        self.assertEqual(stats["volume_24h"], 12345.6)
        self.assertEqual(stats["open_interest"], 789.0)

    @patch("polymarket_bot.market_data.requests.get")
    def test_stats_failure_returns_empty(self, mock_get):
        mock_get.side_effect = RuntimeError("boom")
        self.assertEqual(market_data.fetch_market_stats("cond"), {})


if __name__ == "__main__":
    unittest.main()
