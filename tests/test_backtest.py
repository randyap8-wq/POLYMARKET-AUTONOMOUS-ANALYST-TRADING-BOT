from __future__ import annotations

import unittest

from polymarket_bot.backtest import evaluate_market


def _rising():
    return [{"t": i, "p": round(0.30 + i * 0.02, 4)} for i in range(30)]


def _falling():
    return [{"t": i, "p": round(0.70 - i * 0.02, 4)} for i in range(30)]


def _flat():
    return [{"t": i, "p": 0.50} for i in range(30)]


class EvaluateMarketTests(unittest.TestCase):
    def test_rising_winner_pays_off(self):
        signals = evaluate_market([_rising(), _falling()], winner_index=0)
        ups = [s for s in signals if s["index"] == 0 and s["direction"] == "up"]
        self.assertTrue(ups)
        self.assertTrue(ups[0]["win"])
        self.assertGreater(ups[0]["pnl"], 0)

    def test_rising_loser_loses_one_unit(self):
        signals = evaluate_market([_rising(), _falling()], winner_index=1)
        ups = [s for s in signals if s["index"] == 0 and s["direction"] == "up"]
        self.assertTrue(ups)
        self.assertFalse(ups[0]["win"])
        self.assertEqual(ups[0]["pnl"], -1.0)

    def test_flat_market_fires_no_signal(self):
        self.assertEqual(evaluate_market([_flat(), _flat()], winner_index=0), [])

    def test_short_history_is_skipped(self):
        short = [{"t": i, "p": 0.30 + i * 0.05} for i in range(4)]
        self.assertEqual(evaluate_market([short, short], winner_index=0), [])

    def test_signal_is_point_in_time(self):
        # A series that rises then crashes at the very end. With entry partway
        # through, the late crash must not leak into the entry decision.
        series = [{"t": i, "p": round(0.30 + i * 0.02, 4)} for i in range(20)]
        series += [{"t": 20 + i, "p": 0.05} for i in range(5)]
        signals = evaluate_market([series, _falling()], winner_index=0)
        ups = [s for s in signals if s["index"] == 0 and s["direction"] == "up"]
        self.assertTrue(ups)  # entry sees only the rising portion


if __name__ == "__main__":
    unittest.main()
