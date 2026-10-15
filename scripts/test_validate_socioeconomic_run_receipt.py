#!/usr/bin/env python3
"""Unit tests for strict JSON parsing in the socioeconomic receipt validator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from validate_socioeconomic_run_receipt import ValidationFailure, load_json


class StrictJsonLoaderTests(unittest.TestCase):
    def load_text(self, text: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.json"
            path.write_text(text, encoding="utf-8")
            return load_json(path)

    def test_unique_object_keys_are_accepted(self) -> None:
        self.assertEqual(self.load_text('{"schema":"receipt-v1","count":2}'), {
            "schema": "receipt-v1",
            "count": 2,
        })

    def test_duplicate_top_level_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationFailure, "duplicate JSON object key"):
            self.load_text('{"claim_ceiling":"synthetic_only","claim_ceiling":"production"}')

    def test_duplicate_nested_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationFailure, "duplicate JSON object key"):
            self.load_text('{"authority_boundary":{"can_settle":false,"can_settle":true}}')

    def test_non_finite_numeric_constants_are_rejected(self) -> None:
        for value in ("NaN", "Infinity", "-Infinity"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationFailure, "non-standard numeric constant"):
                    self.load_text('{"metric_value":' + value + '}')

    def test_float_overflow_and_underflow_are_rejected(self) -> None:
        for value in ("1e400", "1e-4000"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValidationFailure, "(overflows|underflows) float range"):
                    self.load_text('{"metric_value":' + value + '}')

    def test_malformed_json_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationFailure, "cannot load strict JSON"):
            self.load_text('{"missing_value":')


if __name__ == "__main__":
    unittest.main()
