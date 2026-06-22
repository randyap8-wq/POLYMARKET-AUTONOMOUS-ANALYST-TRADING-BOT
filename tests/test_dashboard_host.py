from __future__ import annotations

import inspect
import unittest
from unittest.mock import Mock, patch

from polymarket_bot import dashboard


class DashboardHostTests(unittest.TestCase):
    def test_default_host_is_localhost(self):
        self.assertEqual(
            inspect.signature(dashboard.run_dashboard).parameters["host"].default,
            "127.0.0.1",
        )

    @patch("uvicorn.run")
    @patch.object(dashboard, "create_app", return_value=Mock())
    def test_warns_when_bound_to_non_loopback(self, _app, _run):
        with self.assertLogs(dashboard.LOGGER, level="WARNING") as cm:
            dashboard.run_dashboard(host="0.0.0.0", port=8080)
        self.assertTrue(any("UNAUTHENTICATED" in m for m in cm.output))

    @patch("uvicorn.run")
    @patch.object(dashboard, "create_app", return_value=Mock())
    def test_no_warning_on_localhost(self, _app, _run):
        with patch.object(dashboard.LOGGER, "warning") as warning:
            dashboard.run_dashboard(host="127.0.0.1", port=8080)
        self.assertFalse(
            any("UNAUTHENTICATED" in str(call) for call in warning.call_args_list)
        )


if __name__ == "__main__":
    unittest.main()
