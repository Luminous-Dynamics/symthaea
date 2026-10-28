#!/usr/bin/env python3
"""Adversarial tests for the trusted VLEB witness execution recorder."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("record_witness_execution.py")
SCHEMA_PATH = Path(__file__).with_name("witness-execution-receipt.schema.json")
SPEC = importlib.util.spec_from_file_location("vleb_record_witness_execution", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load witness execution recorder")
recorder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = recorder
SPEC.loader.exec_module(recorder)

RUN_ENV = {
    "GITHUB_REPOSITORY": "Luminous-Dynamics/symthaea",
    "VLEB_REPOSITORY_ID": "123456789",
    "GITHUB_RUN_ID": "90001",
    "GITHUB_RUN_ATTEMPT": "2",
    "GITHUB_SHA": "4" * 40,
    "GITHUB_REF": "refs/heads/research/test",
    "VLEB_CANDIDATE_CHECKOUT_REVISION": "2" * 40,
    "VLEB_WORKFLOW_SHA": "1" * 40,
    "GITHUB_WORKFLOW_REF": "Luminous-Dynamics/symthaea/.github/workflows/vleb-confirmatory.yml@refs/heads/main",
    "GITHUB_TOKEN": "must-not-reach-verifier",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


class WitnessRecorderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vleb-witness-recorder-")
        self.root = Path(self.tmp.name)
        (self.root / "legacy.cbl").write_text("       COPY BANK-REC.\n", encoding="ascii")
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def run_recorder(self, *, command_code: str, timeout_ms: int = 1000, result: str = "supports") -> tuple[int, dict]:
        args = [
            "--artifact-root", str(self.root),
            "--task-id", "vleb-private-task-0001",
            "--claim-id", "COPY.BANK.REC",
            "--source-path", "legacy.cbl",
            "--verifier-kind", "source_index",
            "--verifier-id", "test-source-indexer",
            "--verifier-sha256", "d" * 64,
            "--observed-result", result,
            "--runtime-image-id", "sha256:" + "a" * 64,
            "--network-mode", "none",
            "--network-mode-observed", "true",
            "--working-directory", ".",
            "--timeout-ms", str(timeout_ms),
            "--receipt-path", "claims/witness.json",
            "--stdout-path", "logs/stdout.txt",
            "--stderr-path", "logs/stderr.txt",
            "--output-artifact", "index.json",
            "--",
            sys.executable, "-c", command_code,
        ]
        self.stdout = io.StringIO()
        self.stderr = io.StringIO()
        with patch.dict(os.environ, RUN_ENV, clear=True), contextlib.redirect_stdout(self.stdout), contextlib.redirect_stderr(self.stderr):
            code = recorder.main(args)
        stdout_text = self.stdout.getvalue()
        self.assertTrue(stdout_text or self.stderr.getvalue(), "recorder must emit either a receipt summary or an actionable error")
        if code == 0:
            summary = json.loads(stdout_text)
            receipt_path = self.root / summary["receipt_path"]
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["receipt_sha256"], sha(receipt_path.read_bytes()))
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
            errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(receipt))
            self.assertEqual(errors, [])
            return code, receipt
        return code, {}

    def test_records_command_run_identity_and_output_digests(self) -> None:
        code, receipt = self.run_recorder(
            command_code="import os; from pathlib import Path; Path('index.json').write_text('{\\\"copy_references\\\":[\\\"BANK-REC\\\"]}'); print(os.getenv('GITHUB_TOKEN', 'absent'))"
        )
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assertEqual(receipt["run_binding"]["run_id"], "90001")
        self.assertEqual(receipt["run_binding"]["run_attempt"], 2)
        self.assertEqual(receipt["verifier"]["sha256"], "d" * 64)
        self.assertEqual(receipt["source_sha256"], sha((self.root / "legacy.cbl").read_bytes()))
        self.assertEqual(receipt["observed_result"], "supports")
        outputs = {row["path"]: row for row in receipt["outputs"]}
        self.assertEqual(outputs["index.json"]["sha256"], sha((self.root / "index.json").read_bytes()))
        self.assertEqual(outputs["logs/stdout.txt"]["sha256"], sha((self.root / "logs/stdout.txt").read_bytes()))
        self.assertEqual(outputs["logs/stderr.txt"]["sha256"], sha((self.root / "logs/stderr.txt").read_bytes()))
        self.assertNotIn("GITHUB_TOKEN", receipt["invocation"])
        self.assertIn("absent", (self.root / "logs/stdout.txt").read_text(encoding="utf-8"))
        self.assertNotIn("must-not-reach-verifier", (self.root / "logs/stdout.txt").read_text(encoding="utf-8"))

    def test_timeout_forces_inconclusive_result_and_preserves_streams(self) -> None:
        code, receipt = self.run_recorder(
            command_code="from pathlib import Path; import time; Path('index.json').write_text('{}'); time.sleep(1)", timeout_ms=10, result="supports"
        )
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assertTrue(receipt["process"]["timed_out"])
        self.assertIsNone(receipt["process"]["exit_code"])
        self.assertEqual(receipt["observed_result"], "inconclusive")

    def test_nonzero_verifier_exit_cannot_be_recorded_as_support(self) -> None:
        code, receipt = self.run_recorder(
            command_code="from pathlib import Path; Path('index.json').write_text('{}'); raise SystemExit(3)",
            result="supports",
        )
        self.assertEqual(code, 0, self.stderr.getvalue())
        self.assertEqual(receipt["process"]["exit_code"], 3)
        self.assertEqual(receipt["observed_result"], "inconclusive")

    def test_missing_trusted_run_context_fails_before_command_executes(self) -> None:
        marker = self.root / "should-not-exist.txt"
        args = [
            "--artifact-root", str(self.root),
            "--task-id", "vleb-private-task-0001",
            "--claim-id", "COPY.BANK.REC",
            "--source-path", "legacy.cbl",
            "--verifier-kind", "source_index",
            "--verifier-id", "test-source-indexer",
            "--verifier-sha256", "d" * 64,
            "--observed-result", "supports",
            "--runtime-image-id", "sha256:" + "a" * 64,
            "--network-mode", "none",
            "--network-mode-observed", "true",
            "--receipt-path", "claims/witness.json",
            "--stdout-path", "logs/stdout.txt",
            "--stderr-path", "logs/stderr.txt",
            "--", sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()",
        ]
        with patch.dict(os.environ, {}, clear=True), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(self.stderr):
            code = recorder.main(args)
        self.assertEqual(code, 2)
        self.assertFalse(marker.exists())
        self.assertFalse((self.root / "claims/witness.json").exists())

    def test_path_traversal_is_rejected_without_receipt(self) -> None:
        args = [
            "--artifact-root", str(self.root),
            "--task-id", "vleb-private-task-0001",
            "--claim-id", "COPY.BANK.REC",
            "--source-path", "../outside.cbl",
            "--verifier-kind", "source_index",
            "--verifier-id", "test-source-indexer",
            "--verifier-sha256", "d" * 64,
            "--observed-result", "supports",
            "--runtime-image-id", "sha256:" + "a" * 64,
            "--network-mode", "none",
            "--network-mode-observed", "true",
            "--receipt-path", "claims/witness.json",
            "--stdout-path", "logs/stdout.txt",
            "--stderr-path", "logs/stderr.txt",
            "--", sys.executable, "-c", "print('should not run')",
        ]
        with patch.dict(os.environ, RUN_ENV, clear=True), contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(self.stderr):
            code = recorder.main(args)
        self.assertEqual(code, 2)
        self.assertFalse((self.root / "claims/witness.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
