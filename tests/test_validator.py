from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from polymarket_bot import validator


class ByCategoryTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        perf_patch = patch.object(
            validator, "PERFORMANCE_PATH", Path(self.tempdir.name) / "performance.json"
        )
        perf_patch.start()
        self.addCleanup(perf_patch.stop)

    def _bet(self, category, correct):
        return {
            "category": category,
            "correct": correct,
            "pnl_usdc": 5.0 if correct else -10.0,
            "edge": 0.12,
            "confidence": "high",
        }

    def test_report_includes_by_category_breakdown(self):
        resolved = [
            self._bet("politics", True),
            self._bet("politics", True),
            self._bet("sports", False),
            self._bet("sports", False),
        ]

        with patch.object(validator, "load_resolved", return_value=resolved):
            report = validator.generate_performance_report()

        self.assertIn("by_category", report)
        self.assertEqual(report["by_category"]["politics"]["count"], 2)
        self.assertEqual(report["by_category"]["politics"]["win_rate"], 1.0)
        self.assertEqual(report["by_category"]["sports"]["win_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
