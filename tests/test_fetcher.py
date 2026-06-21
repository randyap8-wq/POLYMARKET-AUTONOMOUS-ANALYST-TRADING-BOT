from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

from polymarket_bot.fetcher import fetch_markets


class FetchMarketsTests(unittest.TestCase):
    @patch("polymarket_bot.fetcher.requests.get")
    @patch("polymarket_bot.fetcher.datetime", wraps=datetime)
    def test_fetch_markets_normalizes_and_filters(self, mock_datetime, mock_get):
        fixed_now = datetime(2026, 6, 21, tzinfo=timezone.utc)
        future = (fixed_now + timedelta(days=5)).isoformat()
        far_future = (fixed_now + timedelta(days=90)).isoformat()
        payload = [
            {
                "id": "keep",
                "conditionId": "cond-1",
                "question": "Keep me?",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["0.45", "0.55"]),
                "volume": "15000",
                "endDate": future,
                "slug": "keep-me",
            },
            {
                "id": "low-volume",
                "conditionId": "cond-2",
                "question": "Too small",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["0.4", "0.6"]),
                "volume": "9999",
                "endDate": future,
                "slug": "too-small",
            },
            {
                "id": "too-far",
                "conditionId": "cond-3",
                "question": "Too far away",
                "outcomes": json.dumps(["Yes", "No"]),
                "outcomePrices": json.dumps(["0.4", "0.6"]),
                "volume": "20000",
                "endDate": far_future,
                "slug": "too-far",
            },
        ]
        mock_response = Mock()
        mock_response.json.return_value = payload
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        mock_datetime.now.return_value = fixed_now

        markets = fetch_markets()

        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0]["id"], "keep")
        self.assertEqual(markets[0]["condition_id"], "cond-1")
        self.assertEqual(markets[0]["prices"], [0.45, 0.55])
        self.assertEqual(markets[0]["url"], "https://polymarket.com/event/keep-me")


if __name__ == "__main__":
    unittest.main()
