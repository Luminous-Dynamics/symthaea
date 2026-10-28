from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from audit_witness_receipt import validate_receipt


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def canonical_sha(value) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return sha(encoded)


class WitnessReceiptAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="vleb-witness-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "input.dat").write_bytes(b"account=123\namount=10\n")
        (self.root / "output.dat").write_bytes(b"balance=90\n")
        (self.root / "stdout.log").write_bytes(b"supports\n")
        (self.root / "stderr.log").write_bytes(b"")
        (self.root / "source.cbl").write_bytes(b"       01 BALANCE PIC 9(5).\n")
        self.schema_path = Path(__file__).with_name("witness-receipt.schema.json")
        source_bytes = (self.root / "source.cbl").read_bytes()
        self.receipt = {
            "schema_version": "vleb-evaluator-witness-receipt-v1",
            "receipt_id": "receipt:test-001",
            "producer": {
                "repository_id": "1136141775",
                "repository_full_name": "Luminous-Dynamics/symthaea",
                "workflow_path": ".github/workflows/vleb-ledger-reference.yml",
                "workflow_sha": "a" * 40,
                "workflow_ref": "refs/heads/locked-evaluator",
                "run_id": "40690000123",
                "run_attempt": 2,
                "event": "workflow_dispatch",
            },
            "task_id": "COBOL.SETTLEMENT.001",
            "claim_id": "BALANCE.PIC",
            "source": {
                "path": "source.cbl",
                "sha256": sha(source_bytes),
                "spans": [{
                    "line_start": 1,
                    "line_end": 1,
                    "column_start": 8,
                    "column_end": 27,
                    "excerpt_sha256": sha(source_bytes.splitlines()[0][7:27]),
                }],
            },
            "verifier": {"kind": "independent_execution", "id": "ledger-reference-v1", "sha256": "b" * 64},
            "invocation": {
                "argv": ["/opt/vleb/bin/verify-claim", "--scenario", "settlement-001"],
                "cwd": "/workspace/task",
                "environment": [{"name": "TZ", "value_sha256": sha(b"UTC")}],
            },
            "execution": {
                "status": "completed", "exit_code": 0, "duration_ms": 123,
                "stdout_sha256": sha(b"supports\n"), "stderr_sha256": sha(b""),
                "stdout_path": "stdout.log", "stderr_path": "stderr.log",
            },
            "runtime": {
                "image_digest": "sha256:" + "c" * 64,
                "toolchain_id": "cobol-runtime-1.2.0",
                "toolchain_sha256": "d" * 64,
                "os": "linux",
                "architecture": "x86_64",
                "network_mode": "disabled",
            },
            "inputs": [{"name": "scenario", "path": "input.dat", "sha256": sha((self.root / "input.dat").read_bytes())}],
            "outputs": [
                {"name": "execution-output", "path": "output.dat", "sha256": sha((self.root / "output.dat").read_bytes())},
                {"name": "stdout", "path": "stdout.log", "sha256": sha((self.root / "stdout.log").read_bytes())},
                {"name": "stderr", "path": "stderr.log", "sha256": sha((self.root / "stderr.log").read_bytes())},
            ],
            "claim_result": "supports",
        }
        self.receipt_path = self.root / "receipt.json"
        self.write_receipt()

    def write_receipt(self) -> None:
        self.receipt_path.write_text(json.dumps(self.receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def audit(self, **overrides):
        self.write_receipt()
        values = {
            "expected_receipt_sha256": sha(self.receipt_path.read_bytes()),
            "expected_schema_sha256": sha(self.schema_path.read_bytes()),
            "expected_repository_id": "1136141775",
            "expected_repository_full_name": "Luminous-Dynamics/symthaea",
            "expected_workflow_path": ".github/workflows/vleb-ledger-reference.yml",
            "expected_workflow_sha": "a" * 40,
            "expected_workflow_ref": "refs/heads/locked-evaluator",
            "expected_event": "workflow_dispatch",
            "expected_run_id": "40690000123",
            "expected_run_attempt": 2,
            "expected_task_id": "COBOL.SETTLEMENT.001",
            "expected_claim_id": "BALANCE.PIC",
            "expected_source_sha256": self.receipt["source"]["sha256"],
            "expected_source_spans_sha256": canonical_sha(self.receipt["source"]["spans"]),
            "expected_invocation_sha256": canonical_sha(self.receipt["invocation"]),
            "expected_inputs_sha256": canonical_sha(self.receipt["inputs"]),
            "expected_verifier_kind": "independent_execution",
            "expected_verifier_id": "ledger-reference-v1",
            "expected_verifier_sha256": "b" * 64,
            "expected_runtime_image_digest": "sha256:" + "c" * 64,
            "expected_toolchain_sha256": "d" * 64,
        }
        values.update(overrides)
        return validate_receipt(self.receipt_path, self.schema_path, self.root, **values)

    def test_exact_pins_and_artifact_bytes_pass(self) -> None:
        result = self.audit()
        self.assertTrue(result["eligible"], result["issues"])
        self.assertEqual(result["claim_result"], "supports")

    def test_old_run_id_is_rejected_as_replay(self) -> None:
        result = self.audit(expected_run_id="40689999999")
        self.assertIn("run_id_mismatch", {issue["code"] for issue in result["issues"]})

    def test_old_run_attempt_is_rejected_as_replay(self) -> None:
        result = self.audit(expected_run_attempt=1)
        self.assertIn("run_attempt_mismatch", {issue["code"] for issue in result["issues"]})

    def test_cross_task_receipt_is_rejected(self) -> None:
        result = self.audit(expected_task_id="OTHER.TASK.999")
        self.assertIn("task_id_mismatch", {issue["code"] for issue in result["issues"]})

    def test_cross_claim_receipt_is_rejected(self) -> None:
        result = self.audit(expected_claim_id="OTHER.CLAIM")
        self.assertIn("claim_id_mismatch", {issue["code"] for issue in result["issues"]})

    def test_workflow_commit_must_match_external_pin(self) -> None:
        result = self.audit(expected_workflow_sha="e" * 40)
        self.assertIn("workflow_sha_mismatch", {issue["code"] for issue in result["issues"]})

    def test_source_revision_must_match_external_pin(self) -> None:
        result = self.audit(expected_source_sha256="f" * 64)
        self.assertIn("source_digest_mismatch", {issue["code"] for issue in result["issues"]})

    def test_source_spans_must_match_evaluator_claim_inventory_pin(self) -> None:
        result = self.audit(expected_source_spans_sha256="e" * 64)
        self.assertIn("source_spans_pin_mismatch", {issue["code"] for issue in result["issues"]})

    def test_exact_invocation_must_match_evaluator_pin(self) -> None:
        result = self.audit(expected_invocation_sha256="e" * 64)
        self.assertIn("invocation_pin_mismatch", {issue["code"] for issue in result["issues"]})

    def test_input_inventory_must_match_evaluator_task_pin(self) -> None:
        result = self.audit(expected_inputs_sha256="e" * 64)
        self.assertIn("input_inventory_pin_mismatch", {issue["code"] for issue in result["issues"]})

    def test_workflow_reference_and_event_are_pinned(self) -> None:
        result = self.audit(expected_workflow_ref="refs/heads/untrusted")
        self.assertIn("workflow_ref_mismatch", {issue["code"] for issue in result["issues"]})
        result = self.audit(expected_event="pull_request")
        self.assertIn("event_mismatch", {issue["code"] for issue in result["issues"]})

    def test_repository_name_is_bound_with_numeric_identity(self) -> None:
        result = self.audit(expected_repository_full_name="attacker/fork")
        self.assertIn("repository_full_name_mismatch", {issue["code"] for issue in result["issues"]})

    def test_verifier_identity_and_digest_are_pinned(self) -> None:
        result = self.audit(expected_verifier_id="untrusted-verifier")
        self.assertIn("verifier_id_mismatch", {issue["code"] for issue in result["issues"]})
        result = self.audit(expected_verifier_sha256="e" * 64)
        self.assertIn("verifier_digest_mismatch", {issue["code"] for issue in result["issues"]})

    def test_runtime_and_toolchain_are_pinned(self) -> None:
        result = self.audit(expected_runtime_image_digest="sha256:" + "e" * 64)
        self.assertIn("runtime_image_mismatch", {issue["code"] for issue in result["issues"]})
        result = self.audit(expected_toolchain_sha256="e" * 64)
        self.assertIn("toolchain_digest_mismatch", {issue["code"] for issue in result["issues"]})

    def test_artifact_tampering_is_rejected(self) -> None:
        (self.root / "output.dat").write_bytes(b"forged result\n")
        result = self.audit()
        self.assertIn("artifact_digest_mismatch", {issue["code"] for issue in result["issues"]})

    def test_path_traversal_is_rejected(self) -> None:
        self.receipt["inputs"][0]["path"] = "../outside.dat"
        result = self.audit()
        codes = {issue["code"] for issue in result["issues"]}
        self.assertTrue({"receipt_schema_invalid", "artifact_path_invalid"} & codes)

    def test_symlink_artifact_is_rejected(self) -> None:
        outside = self.root.parent / (self.root.name + "-outside")
        outside.write_bytes(b"external\n")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        (self.root / "linked.dat").symlink_to(outside)
        self.receipt["inputs"][0]["path"] = "linked.dat"
        self.receipt["inputs"][0]["sha256"] = sha(b"external\n")
        result = self.audit()
        self.assertIn("artifact_path_invalid", {issue["code"] for issue in result["issues"]})

    def test_timeout_cannot_support_or_contradict(self) -> None:
        self.receipt["execution"]["status"] = "timed_out"
        self.receipt["execution"]["exit_code"] = None
        result = self.audit()
        self.assertIn("incomplete_execution_claimed_as_decisive", {issue["code"] for issue in result["issues"]})

    def test_source_bytes_are_rehashed(self) -> None:
        (self.root / "source.cbl").write_bytes(b"       01 BALANCE PIC 9(9).\n")
        result = self.audit()
        codes = {issue["code"] for issue in result["issues"]}
        self.assertIn("source_artifact_digest_mismatch", codes)
        self.assertIn("source_excerpt_digest_mismatch", codes)

    def test_source_span_digest_is_recomputed(self) -> None:
        self.receipt["source"]["spans"][0]["excerpt_sha256"] = "e" * 64
        result = self.audit()
        self.assertIn("source_excerpt_digest_mismatch", {issue["code"] for issue in result["issues"]})

    def test_schema_and_receipt_digest_pins_are_mandatory(self) -> None:
        result = self.audit(expected_schema_sha256="e" * 64)
        self.assertIn("schema_digest_mismatch", {issue["code"] for issue in result["issues"]})
        result = self.audit(expected_receipt_sha256="e" * 64)
        self.assertIn("receipt_digest_mismatch", {issue["code"] for issue in result["issues"]})

    def test_execution_log_digest_must_match_hashed_output_artifact(self) -> None:
        self.receipt["execution"]["stdout_sha256"] = "e" * 64
        result = self.audit()
        self.assertIn("execution_log_artifact_binding_mismatch", {issue["code"] for issue in result["issues"]})

    def test_execution_log_artifact_must_be_in_output_inventory(self) -> None:
        self.receipt["execution"]["stderr_path"] = "missing-stderr.log"
        result = self.audit()
        self.assertIn("execution_log_artifact_missing", {issue["code"] for issue in result["issues"]})

    def test_unknown_receipt_fields_fail_closed(self) -> None:
        self.receipt["agent_assertion"] = "trust me"
        result = self.audit()
        self.assertIn("receipt_schema_invalid", {issue["code"] for issue in result["issues"]})

    def test_missing_input_file_is_unresolved_not_accepted(self) -> None:
        (self.root / "input.dat").unlink()
        result = self.audit()
        self.assertIn("artifact_path_invalid", {issue["code"] for issue in result["issues"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)
