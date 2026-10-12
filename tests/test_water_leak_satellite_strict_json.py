"""Strict JSON parsing tests for water-leak evidence boundaries."""
from __future__ import annotations

import unittest

from tools.water_leak_satellite.strict_json import loads


class StrictJsonTests(unittest.TestCase):
    def test_valid_json_parses(self):
        self.assertEqual(loads('{"ok": true, "count": 2}'), {"ok": True, "count": 2})

    def test_duplicate_top_level_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            loads('{"result": "queued", "result": "passed"}')

    def test_duplicate_nested_key_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            loads('{"properties": {"datetime": "a", "datetime": "b"}}')

    def test_non_finite_constants_rejected(self):
        for token in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(token=token), self.assertRaisesRegex(ValueError, "non-standard JSON numeric constant"):
                loads('{"value": ' + token + '}')


if __name__ == "__main__":
    unittest.main()
