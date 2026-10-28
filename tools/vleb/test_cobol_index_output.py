#!/usr/bin/env python3
"""Integration-check COBOL index JSON against schema and expected source anchors.

This is a test harness, not production parsing logic.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tools/vleb/cobol-index/Cargo.toml"
FIXTURE = ROOT / "tools/vleb/cobol-index/fixtures/settlement.cbl"
SCHEMA_PATH = ROOT / "tools/vleb/cobol-index/index-output.schema.json"
SCHEMA = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
Draft202012Validator.check_schema(SCHEMA)
VALIDATOR = Draft202012Validator(SCHEMA, format_checker=FormatChecker())


def run_indexer(source: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "cargo", "run", "--locked", "--quiet",
            "--manifest-path", str(MANIFEST),
            "--", "--format", "fixed", str(source),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )


def validate_json(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    errors = sorted(VALIDATOR.iter_errors(value), key=lambda error: list(map(str, error.absolute_path)))
    if errors:
        details = "\n".join(
            f"{'.'.join(map(str, error.absolute_path)) or '$'}: {error.message}"
            for error in errors
        )
        raise AssertionError("COBOL index output violates schema:\n" + details)
    return value


class CobolIndexOutputTests(unittest.TestCase):
    def test_fixture_output_is_schema_valid_and_contains_expected_anchors(self) -> None:
        first = run_indexer(FIXTURE)
        self.assertEqual(first.returncode, 0, first.stderr)
        report = validate_json(first.stdout)
        self.assertEqual(report["program_id"]["name"], "SETTLE01")
        self.assertEqual([row["name"] for row in report["divisions"]], [
            "IDENTIFICATION", "DATA", "PROCEDURE"
        ])
        copies = {row["name"]: row for row in report["copy_references"]}
        self.assertIn("BANK-REC", copies)
        self.assertTrue(copies["AMOUNT-REC"]["has_replacing_clause"])
        items = {row["name"]: row for row in report["data_items"]}
        self.assertEqual(items["WS-AMOUNT"]["picture"], "S9(7)V99")
        self.assertEqual(items["WS-AMOUNT"]["usage"], "COMP-3")
        self.assertEqual(items["WS-AMOUNT-TEXT"]["redefines"], "WS-AMOUNT")
        self.assertTrue(any(row["name"] == "FD:SETTLEMENT-IN" for row in report["file_descriptions"]))
        self.assertEqual([row["kind"] for row in report["embedded_blocks"]], ["SQL"])
        self.assertEqual(report["diagnostics"], [])

    def test_serialized_output_is_deterministic(self) -> None:
        first = run_indexer(FIXTURE)
        second = run_indexer(FIXTURE)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        self.assertEqual(first.stdout, second.stdout)

    def test_bad_fixed_columns_return_schema_valid_diagnostics_and_nonzero(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-cobol-json-test-") as directory:
            path = Path(directory) / "malformed.cbl"
            path.write_text(
                "       IDENTIFICATION DIVISION.\n"
                "       PROGRAM-ID. BADFIX.\n"
                "       DISPLAY CAFÉ.\n",
                encoding="utf-8",
            )
            output = run_indexer(path)
        self.assertEqual(output.returncode, 2, output.stderr)
        report = validate_json(output.stdout)
        self.assertIn("non_ascii_fixed_format_record", [row["code"] for row in report["diagnostics"]])

    def test_schema_rejects_unknown_output_fields(self) -> None:
        report = validate_json(run_indexer(FIXTURE).stdout)
        report["unsupported_semantics_claim"] = True
        errors = list(VALIDATOR.iter_errors(report))
        self.assertTrue(any("unsupported_semantics_claim" in str(error) for error in errors))

    def test_schema_rejects_source_spans_in_identification_area(self) -> None:
        report = validate_json(run_indexer(FIXTURE).stdout)
        report["program_id"]["location"]["column_end"] = 73
        errors = list(VALIDATOR.iter_errors(report))
        self.assertTrue(any("column_end" in str(error) for error in errors))


if __name__ == "__main__":
    unittest.main(verbosity=2)
