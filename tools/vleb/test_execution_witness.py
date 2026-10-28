#!/usr/bin/env python3
"""Adversarial tests for evaluator-observed execution witness receipts."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load_module("vleb_run_execution_witness", ROOT / "tools/vleb/run_execution_witness.py")
auditor = load_module("vleb_audit_execution_witness", ROOT / "tools/vleb/audit_execution_witness.py")
SCHEMA_PATH = ROOT / "tools/vleb/execution-witness-receipt.schema.json"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class WitnessFixture:
    def __init__(self, *, expected_exit_code: int = 0, process_exit_code: int = 0) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vleb-execution-witness-")
        self.root = Path(self.tmp.name)
        self.source_path = self.root / "source.cbl"
        self.source_raw = b"       COPY ACCOUNT-REC.\n"
        self.source_path.write_bytes(self.source_raw)
        self.source_sha = sha(self.source_raw)
        self.executable_sha = sha(Path(sys.executable).resolve(strict=True).read_bytes())
        code = (
            "import pathlib,sys; "
            "print(pathlib.Path('source.cbl').read_text(), end=''); "
            "print('stderr-witness-sentinel', file=sys.stderr); "
            f"sys.exit({process_exit_code})"
        )
        self.plan = {
            "schema_version": "vleb-execution-witness-plan-v1",
            "task_id": "vleb-runtime-task-0001",
            "claim_id": "TX.ACCOUNT.BALANCE",
            "repository": {"full_name": "Luminous-Dynamics/symthaea", "id": 123456789},
            "run": {"run_id": "987654321", "attempt": 2},
            "candidate_revision": "a" * 40,
            "source_sha256": self.source_sha,
            "runtime": {
                "platform": "linux",
                "immutable_image_id": "sha256:" + "b" * 64,
                "network_mode": "none",
            },
            "verifier": {
                "id": "python-test-command",
                "executable_path": str(Path(sys.executable).resolve(strict=True)),
                "executable_sha256": self.executable_sha,
            },
            "invocation": {
                "argv": [str(Path(sys.executable).resolve(strict=True)), "-c", code],
                "working_directory": ".",
                "environment": {"PYTHONIOENCODING": "utf-8"},
                "timeout_seconds": 10,
                "expected_exit_code": expected_exit_code,
            },
            "source_path": "source.cbl",
            "inputs": [{"path": "source.cbl", "sha256": self.source_sha}],
            "outputs": {"stdout_path": "outputs/stdout.txt", "stderr_path": "outputs/stderr.txt"},
        }
        self.plan_raw = jbytes(self.plan)
        self.plan_path = self.root / "plan.json"
        self.plan_path.write_bytes(self.plan_raw)
        self.plan_sha = sha(self.plan_raw)
        self.receipt_path = "receipts/execution.json"
        self.runner_sha = sha(Path(runner.__file__).read_bytes())
        self.auditor_sha = sha(Path(auditor.__file__).read_bytes())
        self.schema_sha = sha(SCHEMA_PATH.read_bytes())
        self.identity = runner.expected_identity_from_plan(self.plan)
        self.result = runner.record_execution(
            plan_path=Path("plan.json"),
            evidence_root=self.root,
            receipt_path=self.receipt_path,
            expected_plan_sha256=self.plan_sha,
            expected_runner_sha256=self.runner_sha,
            expected_verifier_sha256=self.executable_sha,
            trusted_identity=self.identity,
        )

    def audit(self, **overrides: Any) -> dict[str, Any]:
        args = {
            "receipt_path": Path(self.receipt_path),
            "plan_path": Path("plan.json"),
            "schema_path": SCHEMA_PATH,
            "evidence_root": self.root,
            "expected_receipt_sha256": self.result["receipt_sha256"],
            "expected_auditor_sha256": self.auditor_sha,
            "expected_schema_sha256": self.schema_sha,
            "expected_runner_sha256": self.runner_sha,
            "expected_plan_sha256": self.plan_sha,
            "expected_verifier_sha256": self.executable_sha,
            "trusted_identity": copy.deepcopy(self.identity),
        }
        args.update(overrides)
        return auditor.audit_execution_receipt(**args)

    def close(self) -> None:
        self.tmp.cleanup()


class ExecutionWitnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fx = WitnessFixture()

    def tearDown(self) -> None:
        self.fx.close()

    def test_plan_schema_and_recorded_execution_are_valid(self) -> None:
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)
        receipt = self.fx.result["receipt"]
        errors = list(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(receipt))
        self.assertEqual(errors, [])
        self.assertTrue(self.fx.result["execution_contract_passed"])
        self.assertEqual(receipt["invocation"]["exit_code"], 0)

    def test_independent_auditor_reproduces_hash_and_identity_chain(self) -> None:
        result = self.fx.audit()
        self.assertTrue(result["evidence_valid"], result["issues"])
        self.assertTrue(result["execution_contract_passed"])
        self.assertEqual(result["task_id"], self.fx.plan["task_id"])
        self.assertEqual(result["run_attempt"], self.fx.plan["run"]["attempt"])

    def test_receipt_replay_into_another_run_attempt_is_rejected(self) -> None:
        identity = copy.deepcopy(self.fx.identity)
        identity["run"]["attempt"] += 1
        result = self.fx.audit(trusted_identity=identity)
        self.assertFalse(result["evidence_valid"], result)
        self.assertIn("plan_identity_mismatch", {row["code"] for row in result["issues"]})

    def test_changed_input_after_execution_is_rejected(self) -> None:
        self.fx.source_path.write_bytes(b"       COPY OTHER-REC.\n")
        result = self.fx.audit()
        self.assertFalse(result["evidence_valid"], result)
        self.assertIn("input_artifact_digest_mismatch", {row["code"] for row in result["issues"]})

    def test_changed_stdout_after_execution_is_rejected(self) -> None:
        out = self.fx.root / self.fx.plan["outputs"]["stdout_path"]
        out.write_bytes(b"edited result\n")
        result = self.fx.audit()
        self.assertFalse(result["evidence_valid"], result)
        self.assertIn("output_digest_mismatch", {row["code"] for row in result["issues"]})

    def test_plan_tampering_is_rejected_even_with_receipt_unchanged(self) -> None:
        self.fx.plan_path.write_bytes(self.fx.plan_raw + b" ")
        result = self.fx.audit()
        self.assertFalse(result["evidence_valid"], result)
        self.assertIn("plan_pin_mismatch", {row["code"] for row in result["issues"]})

    def test_exit_mismatch_remains_auditable_but_is_not_a_pass(self) -> None:
        fx = WitnessFixture(expected_exit_code=0, process_exit_code=7)
        try:
            result = fx.audit()
            self.assertTrue(result["evidence_valid"], result["issues"])
            self.assertFalse(result["execution_contract_passed"])
            self.assertEqual(result["observed_exit_code"], 7)
        finally:
            fx.close()

    def test_verifier_digest_mismatch_is_rejected_by_recorder(self) -> None:
        with self.assertRaisesRegex(ValueError, "externally pinned executable digest"):
            runner.record_execution(
                plan_path=Path("plan.json"),
                evidence_root=self.fx.root,
                receipt_path="other/receipt.json",
                expected_plan_sha256=self.fx.plan_sha,
                expected_runner_sha256=self.fx.runner_sha,
                expected_verifier_sha256="0" * 64,
                trusted_identity=self.fx.identity,
            )

    def test_receipt_path_traversal_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "traversal|canonical relative path"):
            runner.record_execution(
                plan_path=Path("plan.json"),
                evidence_root=self.fx.root,
                receipt_path="../escape.json",
                expected_plan_sha256=self.fx.plan_sha,
                expected_runner_sha256=self.fx.runner_sha,
                expected_verifier_sha256=self.fx.executable_sha,
                trusted_identity=self.fx.identity,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
