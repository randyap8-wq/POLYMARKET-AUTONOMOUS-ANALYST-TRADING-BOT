from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from polymarket_bot import backfill


class BackfillTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1],
            prefix=".backfill-test-",
        )
        self.addCleanup(self.tempdir.cleanup)
        self.resolved_path = Path(self.tempdir.name) / "resolved.jsonl"

        resolved_patch = patch.object(backfill, "RESOLVED_PATH", self.resolved_path)
        resolved_patch.start()
        self.addCleanup(resolved_patch.stop)

    def _mock_markets_response(self, mock_get, markets):
        mock_response = Mock()
        mock_response.json.return_value = markets
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

    def _market(self, **overrides):
        market = {
            "id": "market-1",
            "conditionId": "cond-1",
            "question": "Will the test pass?",
            "slug": "will-the-test-pass",
            "volume": "123.45",
            "endDate": "2026-06-20T00:00:00Z",
            "resolutionTime": (
                datetime.now(timezone.utc) - timedelta(days=1)
            ).isoformat().replace("+00:00", "Z"),
            "outcomes": json.dumps(["No", "Yes"]),
            "outcomePrices": json.dumps(["0.0", "1.0"]),
        }
        market.update(overrides)
        return market

    @patch("polymarket_bot.backfill.requests.get")
    def test_fetch_recently_closed_returns_clean_recent_resolutions(self, mock_get):
        self._mock_markets_response(
            mock_get,
            [
                self._market(),
                self._market(
                    id="market-2",
                    conditionId="cond-2",
                    outcomePrices=json.dumps(["0.45", "0.55"]),
                ),
                self._market(id="market-3", conditionId="cond-3", resolutionTime=None),
            ],
        )

        markets = backfill._fetch_recently_closed(days_back=14, limit=10)

        self.assertEqual(len(markets), 1)
        self.assertEqual(markets[0]["condition_id"], "cond-1")
        self.assertEqual(markets[0]["resolved_outcome"], "Yes")

    def test_run_backfill_writes_resolved_entry(self):
        market = {
            "condition_id": "cond-1",
            "question": "Will the test pass?",
            "url": "https://polymarket.com/event/will-the-test-pass",
            "category": "other",
            "resolved_outcome": "Yes",
            "resolved_at": "2026-06-21T00:00:00Z",
            "end_date": "2026-06-20T00:00:00Z",
        }
        score = {
            "recommended_outcome": "Yes",
            "current_price": 0.25,
            "fair_value_estimate": 0.8,
            "edge": 0.55,
            "confidence": "high",
            "reasoning": "Test fixture.",
        }

        with (
            patch("polymarket_bot.backfill._fetch_recently_closed", return_value=[market]),
            patch("polymarket_bot.backfill.fetch_news", return_value=[]),
            patch("polymarket_bot.backfill.score_market", return_value=score),
            patch("polymarket_bot.backfill.reset_token_usage", return_value=None),
        ):
            count = backfill.run_backfill()

        self.assertEqual(count, 1)
        entries = self._read_resolved_entries()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["source"], "backfill")
        self.assertEqual(entries[0]["status"], "resolved")
        self.assertTrue(entries[0]["correct"])

    def test_run_backfill_skips_already_backfilled_market(self):
        market = {
            "condition_id": "cond-dup",
            "question": "Will the duplicate test pass?",
            "url": "https://polymarket.com/event/will-the-duplicate-test-pass",
            "category": "other",
            "resolved_outcome": "Yes",
            "resolved_at": "2026-06-21T00:00:00Z",
            "end_date": "2026-06-20T00:00:00Z",
        }
        score = {
            "recommended_outcome": "Yes",
            "current_price": 0.5,
            "fair_value_estimate": 0.7,
            "edge": 0.2,
            "confidence": "medium",
            "reasoning": "Test fixture.",
        }

        with (
            patch("polymarket_bot.backfill._fetch_recently_closed", return_value=[market]),
            patch("polymarket_bot.backfill.fetch_news", return_value=[]),
            patch("polymarket_bot.backfill.score_market", return_value=score),
            patch("polymarket_bot.backfill.reset_token_usage", return_value=None),
        ):
            first_count = backfill.run_backfill()
            second_count = backfill.run_backfill()

        self.assertEqual(first_count, 1)
        self.assertEqual(second_count, 0)
        self.assertEqual(len(self._read_resolved_entries()), 1)

    def _read_resolved_entries(self):
        with self.resolved_path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh]


if __name__ == "__main__":
    unittest.main()
