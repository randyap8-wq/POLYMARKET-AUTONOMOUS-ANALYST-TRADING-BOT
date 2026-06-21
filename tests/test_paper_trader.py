from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from polymarket_bot import paper_trader


class CheckResolutionsTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory(
            dir=Path(__file__).resolve().parents[1],
            prefix=".paper-trader-test-",
        )
        self.addCleanup(self.tempdir.cleanup)
        self.paper_bets_path = Path(self.tempdir.name) / "paper_bets.jsonl"
        self.resolved_path = Path(self.tempdir.name) / "resolved.jsonl"

        paper_patch = patch.object(paper_trader, "PAPER_BETS_PATH", self.paper_bets_path)
        resolved_patch = patch.object(paper_trader, "RESOLVED_PATH", self.resolved_path)
        paper_patch.start()
        resolved_patch.start()
        self.addCleanup(paper_patch.stop)
        self.addCleanup(resolved_patch.stop)

    def _pending_bet(self, **overrides):
        bet = {
            "bet_id": "bet-1",
            "recorded_at": "2026-06-21T00:00:00+00:00",
            "question": "Will test pass?",
            "url": "https://polymarket.com/event/test",
            "condition_id": "cond-1",
            "recommended_outcome": "Yes",
            "recommended_outcome_index": 1,
            "current_price": 0.25,
            "fair_value_estimate": 0.6,
            "edge": 0.35,
            "confidence": "high",
            "reasoning": "Test fixture.",
            "top_news": [],
            "hypothetical_usdc": 10.0,
            "hypothetical_tokens": 40.0,
            "end_date": "2026-06-30T00:00:00Z",
            "status": "pending",
            "resolved_outcome": None,
            "resolved_at": None,
            "pnl_usdc": None,
            "correct": None,
        }
        bet.update(overrides)
        return bet

    def _write_bet(self, bet):
        with self.paper_bets_path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(bet) + "\n")

    def _mock_market(self, mock_get, market):
        mock_response = Mock()
        mock_response.json.return_value = [market]
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

    @patch("polymarket_bot.paper_trader.requests.get")
    def test_correct_winning_bet_resolves(self, mock_get):
        self._write_bet(self._pending_bet())
        self._mock_market(
            mock_get,
            {
                "closed": True,
                "resolutionTime": "2026-06-22T00:00:00Z",
                "outcomePrices": json.dumps(["0.0", "1.0"]),
                "outcomes": json.dumps(["No", "Yes"]),
            },
        )

        count = paper_trader.check_resolutions()

        self.assertEqual(count, 1)
        resolved = paper_trader.load_resolved()
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["status"], "resolved")
        self.assertTrue(resolved[0]["correct"])
        self.assertEqual(resolved[0]["pnl_usdc"], 30.0)

    @patch("polymarket_bot.paper_trader.requests.get")
    def test_wrong_bet_resolves_with_loss(self, mock_get):
        self._write_bet(self._pending_bet(recommended_outcome="No", hypothetical_usdc=12.5))
        self._mock_market(
            mock_get,
            {
                "closed": True,
                "resolutionTime": "2026-06-22T00:00:00Z",
                "outcomePrices": json.dumps(["0.0", "1.0"]),
                "outcomes": json.dumps(["No", "Yes"]),
            },
        )

        count = paper_trader.check_resolutions()

        self.assertEqual(count, 1)
        resolved = paper_trader.load_resolved()
        self.assertFalse(resolved[0]["correct"])
        self.assertEqual(resolved[0]["pnl_usdc"], -12.5)

    @patch("polymarket_bot.paper_trader.requests.get")
    def test_closed_market_without_winner_price_does_not_resolve(self, mock_get):
        self._write_bet(self._pending_bet())
        self._mock_market(
            mock_get,
            {
                "closed": True,
                "resolutionTime": "2026-06-22T00:00:00Z",
                "outcomePrices": json.dumps(["0.45", "0.55"]),
                "outcomes": json.dumps(["No", "Yes"]),
            },
        )

        count = paper_trader.check_resolutions()

        self.assertEqual(count, 0)
        self.assertFalse(self.resolved_path.exists())
        self.assertEqual(paper_trader._load_paper_bets()[0]["status"], "pending")

    @patch("polymarket_bot.paper_trader.requests.get")
    def test_paper_bets_rewritten_with_resolved_status(self, mock_get):
        self._write_bet(self._pending_bet())
        self._mock_market(
            mock_get,
            {
                "closed": True,
                "resolutionTime": "2026-06-22T00:00:00Z",
                "outcomePrices": json.dumps(["0.0", "1.0"]),
                "outcomes": json.dumps(["No", "Yes"]),
            },
        )

        paper_trader.check_resolutions()

        bets = paper_trader._load_paper_bets()
        self.assertEqual(len(bets), 1)
        self.assertEqual(bets[0]["status"], "resolved")

    def test_duplicate_paper_bet_for_same_condition_id_is_skipped(self):
        market = {"condition_id": "cond-dup", "question": "Will it rain?", "url": ""}
        score = {"current_price": 0.4, "recommended_outcome": "Yes"}
        first = paper_trader.record_paper_bet(market, score, [])
        second = paper_trader.record_paper_bet(market, score, [])
        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(paper_trader._load_paper_bets()), 1)


if __name__ == "__main__":
    unittest.main()
