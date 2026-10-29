#!/usr/bin/env python3
"""Adversarial tests for VLEB shift-event evidence consistency checks."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("audit_shift_event_evidence.py")
SPEC = importlib.util.spec_from_file_location("vleb_audit_shift_event_evidence", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import shift-evidence auditor")
auditor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = auditor
SPEC.loader.exec_module(auditor)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def bind_payload(receipt: dict[str, Any]) -> None:
    payload = dict(receipt)
    payload.pop("receipt_payload_sha256", None)
    receipt["receipt_payload_sha256"] = auditor.canonical_digest(payload)


class ShiftEvidenceFixture:
    PROTOCOL = "a" * 64
    REVISION = "b" * 40
    IMAGE = "sha256:" + "c" * 64

    def __init__(self) -> None:
        self.scenario = {
            "scenario_id": "shift-001",
            "task_family": "legacy",
            "change_family": "requirement_update",
            "intervention_position": 4,
            "task_contract_before_sha256": sha(b"contract-before"),
            "task_contract_after_sha256": sha(b"contract-after"),
            "shift_payload_sha256": sha(b"locked-shift-payload"),
            "initial_workspace_sha256": sha(b"initial-workspace"),
            "initial_checkpoint_sha256": sha(b"initial-checkpoint"),
            "initial_revision": self.REVISION,
            "initial_image_id": self.IMAGE,
            "required_check_ids": ["contract-current", "regression-tests"],
            "stale_artifact_sha256s": [sha(b"old-test-report"), sha(b"old-build-report")],
        }
        self.scenarios = [self.scenario]
        self.seeds = [7, 13]
        self.manifest = {
            "schema_version": "vleb-shift-evidence-v1",
            "cohort_id": "shift-evidence-pilot-001",
            "protocol_sha256": self.PROTOCOL,
            "scenarios": self.scenarios,
            "seeds": self.seeds,
            "receipts": [],
        }
        for seed in self.seeds:
            for condition in ("clean", "shifted"):
                run_id = f"run-{seed}-{condition}"
                event_id = f"event-{seed}"
                final_workspace = sha(f"final-workspace:{run_id}".encode())
                event = None if condition == "clean" else {
                    "event_id": event_id,
                    "event_kind": "requirement_update",
                    "sequence_number": 4,
                    "intervention_position": self.scenario["intervention_position"],
                    "applied": True,
                    "payload_sha256": self.scenario["shift_payload_sha256"],
                    "pre_workspace_sha256": self.scenario["initial_workspace_sha256"],
                    "post_workspace_sha256": sha(f"after-shift:{seed}".encode()),
                    "transition_diff_sha256": sha(f"diff:{seed}".encode()),
                    "runner_event_log_sha256": sha(f"runner-log:{seed}".encode()),
                    "observed_monotonic_ns": 123456 + seed,
                    "event_source": "trusted_runner",
                }
                stale = [] if condition == "clean" else [
                    {"artifact_sha256": artifact, "disposition": "invalidated", "reason_event_id": event_id}
                    for artifact in self.scenario["stale_artifact_sha256s"]
                ]
                receipt = {
                    "receipt_payload_sha256": "",
                    "scenario_id": "shift-001",
                    "seed": seed,
                    "condition": condition,
                    "run_id": run_id,
                    "attempt_number": 1,
                    "scenario_sha256": auditor.canonical_digest(self.scenario),
                    "protocol_sha256": self.PROTOCOL,
                    "task_contract_sha256": self.scenario[
                        "task_contract_after_sha256" if condition == "shifted" else "task_contract_before_sha256"
                    ],
                    "initial_workspace_sha256": self.scenario["initial_workspace_sha256"],
                    "initial_checkpoint_sha256": self.scenario["initial_checkpoint_sha256"],
                    "initial_revision": self.REVISION,
                    "initial_image_id": self.IMAGE,
                    "model_id": "agent-a",
                    "model_version": "v1",
                    "scaffold_sha256": sha(b"scaffold"),
                    "initial_execution_profile_sha256": sha(b"initial-profile"),
                    "initial_network_mode": "none",
                    "post_shift_execution_profile_sha256": None if condition == "clean" else sha(b"post-profile"),
                    "post_shift_network_mode": None if condition == "clean" else "none",
                    "final_workspace_sha256": final_workspace,
                    "outcome": "pass",
                    "runner_event": event,
                    "stale_evidence_disposition": stale,
                    "validation_results": [
                        {"check_id": check, "input_workspace_sha256": final_workspace,
                         "status": "pass", "evidence_sha256": sha(f"{run_id}:{check}".encode())}
                        for check in self.scenario["required_check_ids"]
                    ],
                    "audit_report_sha256": sha(f"audit:{run_id}".encode()),
                }
                bind_payload(receipt)
                self.manifest["receipts"].append(receipt)

    def analyze(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        value = self.manifest if manifest is None else manifest
        return auditor.analyze_manifest(
            value, input_sha256=sha(jbytes(value)),
            expected_protocol_sha256=self.PROTOCOL,
            expected_verifier_sha256=sha(MODULE_PATH.read_bytes()),
        )


class ShiftEventEvidenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = ShiftEvidenceFixture()

    def refresh_payload(self, receipt: dict[str, Any]) -> None:
        bind_payload(receipt)

    def test_valid_manifest_checks_pairs_and_remains_ineligible(self) -> None:
        result = self.fixture.analyze()
        self.assertTrue(result["valid"])
        self.assertTrue(result["evidence_integrity_consistent"])
        self.assertTrue(result["runner_attestation_required"])
        self.assertFalse(result["score_eligible"])
        self.assertEqual(result["scenario_count"], 1)
        self.assertEqual(result["seed_count"], 2)
        self.assertEqual(result["expected_receipt_count"], 4)
        self.assertEqual(result["observed_receipt_count"], 4)

    def test_missing_receipt_fails_closed(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["receipts"].pop()
        with self.assertRaisesRegex(ValueError, "complete locked scenario"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_duplicate_condition_cell_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["receipts"][-1]["condition"] = "clean"
        self.refresh_payload(manifest["receipts"][-1])
        with self.assertRaisesRegex(ValueError, "duplicate receipt cell"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_bad_receipt_payload_digest_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["receipts"][0]["outcome"] = "fail"
        with self.assertRaisesRegex(ValueError, "receipt payload digest"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_shift_payload_must_match_locked_scenario(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["runner_event"]["payload_sha256"] = "f" * 64
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "observed shift payload differs"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_event_must_be_runner_sourced_and_applied(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["runner_event"]["event_source"] = "agent"
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "trusted_runner"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["runner_event"]["applied"] = False
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "trusted_runner"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_stale_evidence_inventory_must_be_complete_and_event_bound(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["stale_evidence_disposition"].pop()
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "do not cover the locked stale-artifact inventory"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["stale_evidence_disposition"][0]["reason_event_id"] = "different-event"
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "not bound to the observed shift event"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_all_required_checks_must_be_present_and_bind_final_state(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["validation_results"].pop()
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "cover every required check"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["validation_results"][0]["input_workspace_sha256"] = "f" * 64
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "other than the pinned final state"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_malformed_enum_fields_fail_closed_without_type_errors(self) -> None:
        cases = [
            ("runner_event", "event_kind", ["requirement_update"], "event_kind"),
            ("stale_evidence_disposition", "disposition", ["invalidated"], "disposition"),
            ("validation_results", "status", ["pass"], "status is invalid"),
        ]
        for collection_key, field, value, expected in cases:
            with self.subTest(field=field):
                manifest = copy.deepcopy(self.fixture.manifest)
                shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
                if collection_key == "runner_event":
                    shifted[collection_key][field] = value
                else:
                    shifted[collection_key][0][field] = value
                self.refresh_payload(shifted)
                with self.assertRaisesRegex(ValueError, expected):
                    auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_model_profile_cannot_drift_across_seeds(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        row = next(row for row in manifest["receipts"] if row["condition"] == "clean" and row["seed"] == 13)
        row["initial_execution_profile_sha256"] = sha(b"drifted-initial-profile")
        self.refresh_payload(row)
        with self.assertRaisesRegex(ValueError, "drifts in model/scaffold/initial profile across seeds"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_post_shift_profile_cannot_drift_across_seeds(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        row = next(row for row in manifest["receipts"] if row["condition"] == "shifted" and row["seed"] == 13)
        row["post_shift_execution_profile_sha256"] = sha(b"drifted-post-profile")
        self.refresh_payload(row)
        with self.assertRaisesRegex(ValueError, "drifts in post-shift profile/network mode across seeds"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_shift_position_must_match_locked_schedule(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["runner_event"]["intervention_position"] = 8
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "intervention position differs"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_pass_cannot_hide_a_failed_required_check(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["validation_results"][0]["status"] = "fail"
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "claims pass despite"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_paired_starting_identity_must_match(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        shifted = next(row for row in manifest["receipts"] if row["condition"] == "shifted")
        shifted["initial_execution_profile_sha256"] = "f" * 64
        self.refresh_payload(shifted)
        with self.assertRaisesRegex(ValueError, "clean/shifted pair does not share"):
            auditor.validate_manifest(manifest, self.fixture.PROTOCOL)

    def test_duplicate_json_keys_and_nonfinite_numbers_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            auditor.strict_json_loads(b'{"cohort_id":"a","cohort_id":"b"}')
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            auditor.strict_json_loads(b'{"cost_usd":NaN}')

    def test_cli_checks_all_external_pins(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-shift-evidence-") as temporary:
            root = Path(temporary)
            source = root / "manifest.json"
            output = root / "report.json"
            source.write_bytes(jbytes(self.fixture.manifest))
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--input", str(source),
                 "--expected-input-sha256", sha(source.read_bytes()),
                 "--expected-protocol-sha256", self.fixture.PROTOCOL,
                 "--expected-verifier-sha256", sha(MODULE_PATH.read_bytes()),
                 "--output", str(output)],
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads(completed.stdout)
            self.assertTrue(report["valid"])
            self.assertTrue(report["runner_attestation_required"])
            self.assertFalse(report["score_eligible"])
            self.assertEqual(report, json.loads(output.read_text(encoding="utf-8")))

    def test_cli_wrong_manifest_pin_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-shift-evidence-reject-") as temporary:
            source = Path(temporary) / "manifest.json"
            source.write_bytes(jbytes(self.fixture.manifest))
            completed = subprocess.run(
                [sys.executable, str(MODULE_PATH), "--input", str(source),
                 "--expected-input-sha256", "f" * 64,
                 "--expected-protocol-sha256", self.fixture.PROTOCOL,
                 "--expected-verifier-sha256", sha(MODULE_PATH.read_bytes())],
                capture_output=True, text=True, check=False, timeout=30,
            )
            self.assertEqual(completed.returncode, 1)
            report = json.loads(completed.stdout)
            self.assertFalse(report["valid"])
            self.assertFalse(report["evidence_integrity_consistent"])
            self.assertTrue(report["runner_attestation_required"])
            self.assertFalse(report["score_eligible"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
