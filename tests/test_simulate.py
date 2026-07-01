from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from polymarket_bot import simulate
from polymarket_bot.simulate import run_simulation


class SimulationTests(unittest.TestCase):
    def test_run_simulation_returns_ranked_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "full_backtest.json"
            path.write_text(
                json.dumps(
                    {
                        "trades": [
                            {
                                "recommended_outcome": "Yes",
                                "current_price": 0.4,
                                "probability": 0.7,
                                "win": True,
                            },
                            {
                                "recommended_outcome": "Yes",
                                "current_price": 0.6,
                                "probability": 0.7,
                                "win": False,
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(simulate, "SIMULATION_PATH", Path(tmp) / "risk_simulation.json"):
                result = run_simulation(
                    {"kelly_fraction": [0.25], "max_portfolio_exposure_usdc": [50], "max_bet_usdc": [10]},
                    backtest_path=path,
                )

        self.assertEqual(len(result["results"]), 1)
        self.assertIn("sharpe_ratio", result["best"])


if __name__ == "__main__":
    unittest.main()
