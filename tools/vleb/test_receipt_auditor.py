#!/usr/bin/env python3
"""Adversarial unit tests for the VLEB independent evidence auditor.

All artifacts are synthetic. Passing this suite validates auditor logic on
fixtures; it does not certify a live benchmark run or verify runner provenance.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import tempfile
import unittest
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("audit_receipt.py")
SPEC = importlib.util.spec_from_file_location("vleb_audit_receipt", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load audit_receipt.py")
audit_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit_module
SPEC.loader.exec_module(audit_module)
AuditInputs = audit_module.AuditInputs


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class AuditBundle:
    """Create a synthetic, internally consistent audit bundle."""

    IMAGE_ID = "sha256:" + ("f" * 64)
    REVISION = "a" * 40
    ACTIVE = ["VLEB-LDG-001", "VLEB-LDG-002", "VLEB-LDG-004", "VLEB-LDG-007"]
    MUTATIONS = {
        "VLEB-LDG-001": ("duplicate_processing", "repeated_exact_duplicate_is_idempotent_for_arbitrary_multiplicity"),
        "VLEB-LDG-002": ("other_semantic", "conflicting_transaction_id_is_rejected"),
        "VLEB-LDG-004": ("rounding_or_truncation", "arithmetic_overflow_is_rejected"),
        "VLEB-LDG-007": ("other_semantic", "deterministic_generated_ledgers_match_independent_reference_reducer"),
    }
    EFFECTS = ["stdout_bytes", "exit_code", "control_totals", "numeric_semantics", "idempotency"]
    LOGICAL = [
        ("logical.stdout", "stdout_bytes_contract"),
        ("logical.exit", "exit_code_contract"),
        ("logical.totals", "control_totals_contract"),
        ("logical.numeric", "numeric_semantics_contract"),
        ("logical.idempotency", "idempotency_contract"),
    ]

    def __init__(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="vleb-auditor-tests-")
        self.base = Path(self.tmp.name)
        self.evidence = self.base / "evidence"
        (self.evidence / "logs").mkdir(parents=True)
        self.task_path = self.base / "task.json"
        self.task_schema_path = self.base / "task.schema.json"
        self.receipt_schema_path = self.base / "receipt.schema.json"
        self.protocol_path = self.base / "protocol.md"
        self.patch_path = self.base / "candidate.patch"
        self.catalog_path = self.base / "mutations.json"
        self.receipt_path = self.base / "receipt.json"
        self.write(self.protocol_path, b"Synthetic frozen protocol v1\n")
        self.write(self.patch_path, b"diff --git a/ledger.rs b/ledger.rs\n+synthetic candidate\n")
        self.write(self.task_schema_path, jbytes({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}))
        self.write(self.receipt_schema_path, jbytes({"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object"}))

        self.write(self.evidence / "logs/candidate-tests.log", self.candidate_test_log())
        self.write(self.evidence / "logs/command.stdout", b"tests completed\n")
        self.write(self.evidence / "logs/command.stderr", b"")
        self.write(self.evidence / "logs/baseline.log", self.baseline_log())
        self.write(self.evidence / "logs/reference.log", self.reference_log())
        self.write(self.evidence / "logs/reference.patch", b"diff --git a/ledger.rs b/ledger.rs\n+reference implementation\n")
        self.write(self.evidence / "logs/test-bundle.zip", b"synthetic test bundle identity\n")
        self.write(self.evidence / "logs/differential.json", b'{"synthetic":true}\n')
        self.mutation_log_paths: dict[str, tuple[str, str]] = {}
        for mid, (_, target) in self.MUTATIONS.items():
            compile_log = (
                b"Compiling synthetic vleb fixture\n"
                b"Finished test profile [unoptimized + debuginfo] target(s) in 0.01s\n"
                b"  Executable unittests src/lib.rs (target/debug/deps/vleb-123)\n"
            )
            test_log = f"running 1 test\ntest {target} ... FAILED\ntest result: FAILED. 0 passed; 1 failed; 0 ignored\n".encode()
            compile_name, test_name = f"logs/{mid}.compile.log", f"logs/{mid}.test.log"
            self.write(self.evidence / compile_name, compile_log)
            self.write(self.evidence / test_name, test_log)
            self.mutation_log_paths[mid] = (compile_name, test_name)

        ref_paths = {
            "baseline_log": "logs/baseline.log",
            "reference_log": "logs/reference.log",
            "reference_patch": "logs/reference.patch",
            "test_bundle": "logs/test-bundle.zip",
        }
        task = {
            "schema_version": "vleb-task-v1",
            "task_id": "vleb-private-auditor-001",
            "classification": {"exposure": "private_confirmatory", "reviewer_count": 1},
            "environment": {
                "platform": "linux",
                "network_mode": "none",
                "toolchains": [{"name": "rustc", "version": "1.96.0", "identity_sha256": sha(b"rustc-1.96.0-binary")}],
            },
            "behavior_contract": {
                "task_statement": "The evaluator must account for every declared logical behavior.",
                "invariants": ["Every declared observable is covered by a required executable test."],
                "allowed_behavior_changes": [],
                "observable_effects": self.EFFECTS,
                "requirement_traceability": [
                    {
                        "requirement_id": "TRACE.STATEMENT",
                        "source_kind": "task_statement",
                        "source_text": "The evaluator must account for every declared logical behavior.",
                        "required_test_ids": [self.LOGICAL[0][0]],
                        "observable_effects": self.EFFECTS,
                    },
                    {
                        "requirement_id": "TRACE.INVARIANT.001",
                        "source_kind": "invariant",
                        "source_text": "Every declared observable is covered by a required executable test.",
                        "required_test_ids": [logical for logical, _ in self.LOGICAL],
                        "observable_effects": self.EFFECTS,
                    },
                    *[
                        {
                            "requirement_id": "TRACE.EFFECT." + effect.upper(),
                            "source_kind": "observable_effect",
                            "source_text": effect,
                            "required_test_ids": [logical],
                            "observable_effects": [effect],
                        }
                        for effect, (logical, _) in zip(self.EFFECTS, self.LOGICAL)
                    ],
                ],
            },
            "oracle_contract": {
                "required_test_ids": [logical for logical, _ in self.LOGICAL],
                "test_execution_map": [{"test_id": logical, "execution_ids": [execution]} for logical, execution in self.LOGICAL],
                "baseline_empty_patch_must_fail": True,
                "reference_patch_must_pass": True,
                "reference_evidence_artifacts": ref_paths,
                "reference_evidence_sha256": {key: sha((self.evidence / path).read_bytes()) for key, path in ref_paths.items()},
                "oracle_review": {"independent_reviewer_count": 1, "mutation_equivalence_reviewed": True},
            },
        }
        self.write(self.task_path, jbytes(task))
        catalog = {
            "schema_version": "vleb-mutation-catalog-v1",
            "task_id": task["task_id"],
            "active_campaign": {"version": "vleb-ledger-mutants-v1", "required_mutation_ids": list(self.ACTIVE)},
            "mutations": [
                {"mutation_id": mid, "mutation_class": kind, "expected_equivalence": "non_equivalent", "expected_killer_test": target}
                for mid, (kind, target) in self.MUTATIONS.items()
            ],
        }
        self.write(self.catalog_path, jbytes(catalog))

        toolchain = {"name": "rustc", "version": "1.96.0", "identity_sha256": sha(b"rustc-1.96.0-binary")}
        mutation_rows = []
        for mid, (kind, target) in self.MUTATIONS.items():
            compile_name, test_name = self.mutation_log_paths[mid]
            compile_raw = (self.evidence / compile_name).read_bytes()
            test_raw = (self.evidence / test_name).read_bytes()
            mutation_rows.append({
                "mutation_id": mid, "mutation_class": kind, "compilable": True, "compile_exit_code": 0, "test_exit_code": 101,
                "equivalence_status": "non_equivalent", "observed_verdict": "rejected",
                "evidence_sha256": sha(compile_raw + b"\0" + test_raw),
                "compile_log_artifact": compile_name, "test_log_artifact": test_name,
                "compile_log_sha256": sha(compile_raw), "test_log_sha256": sha(test_raw),
                "expected_killer_test": target, "failed_test_names": [target],
                "review_notes": "independently reviewed synthetic mutant",
            })
        checks = []
        for effect in self.EFFECTS:
            value = sha(("expected:" + effect).encode())
            detail_name = "logs/differential.json"
            detail_raw = (self.evidence / detail_name).read_bytes()
            checks.append({
                "check_id": "diff-" + effect, "observable": effect, "status": "equal",
                "expected_sha256": value, "observed_sha256": value, "mismatch_count": 0,
                "details_sha256": sha(detail_raw), "details_artifact": detail_name,
            })

        command_stdout = (self.evidence / "logs/command.stdout").read_bytes()
        command_stderr = (self.evidence / "logs/command.stderr").read_bytes()
        candidate_log = (self.evidence / "logs/candidate-tests.log").read_bytes()
        self.receipt = {
            "schema_version": "vleb-evidence-v1", "run_id": "synthetic-audit-run",
            "task_id": task["task_id"], "task_contract_sha256": sha(self.task_path.read_bytes()),
            "protocol_sha256": sha(self.protocol_path.read_bytes()), "candidate_patch_sha256": sha(self.patch_path.read_bytes()),
            "attempt_number": 1,
            "agent": {
                "model_id": "synthetic-agent", "model_version": "test", "scaffold_sha256": sha(b"synthetic-scaffold"),
                "tool_policy": {"filesystem_scope": "task_checkout", "container_network": "none", "browser_allowed": False, "external_retrieval_allowed": False},
            },
            "environment": {
                "platform": "linux", "toolchains": [toolchain],
                "runtime_identity": {"container_or_host_id": "synthetic-container", "configured_image": "fixture-image",
                                     "immutable_image_id": self.IMAGE_ID, "network_mode": "none", "network_mode_observed": True},
                "checkout_identity": {"repository": "Luminous-Dynamics/symthaea", "expected_revision": self.REVISION,
                                      "observed_revision": self.REVISION, "clean_worktree": True, "candidate_row_sha256": sha(b"candidate row")},
                "sanitization": {"git_metadata_exit_code": 0, "hooks_disabled": True, "unexpected_refs_checked": True,
                                 "unreachable_objects_checked": True, "ignored_cache_policy": "removed"},
                "trust_boundary": "independently_inspected",
            },
            "commands": [{
                "command_id": "candidate-test-suite", "argv": ["cargo", "test", "--locked"], "working_directory": "/workspace",
                "exit_code": 0, "expected_exit_code": 0, "timed_out": False,
                "stdout_sha256": sha(command_stdout), "stderr_sha256": sha(command_stderr),
                "stdout_artifact": "logs/command.stdout", "stderr_artifact": "logs/command.stderr", "duration_ms": 1000,
            }],
            "required_test_ids": [logical for logical, _ in self.LOGICAL],
            "test_results": [{
                "test_id": logical, "status": "passed", "exit_code": 0, "log_sha256": sha(candidate_log),
                "log_artifact": "logs/candidate-tests.log", "duration_ms": 1000, "notes": "synthetic positive control",
            } for logical, _ in self.LOGICAL],
            "mutation_results": mutation_rows, "differential_checks": checks,
            "verifier": {
                "verifier_id": "synthetic-auditor-test", "verifier_version": "test",
                "verifier_binary_sha256": sha(b"synthetic-verifier-binary"), "receipt_recomputed": True,
                "required_test_coverage": 1, "environment_match": "match", "verdict": "pass", "reason_codes": [],
                "auditor_output_sha256": sha(b"synthetic-auditor-output"),
            },
            "outcome": "pass", "unresolved_reasons": [], "abstention_reason": None,
            "timing": {"started_at": "2026-10-28T08:00:00Z", "ended_at": "2026-10-28T08:00:01Z", "wall_clock_ms": 1000},
            "cost": {"model_cost_usd": 0, "tool_cost_usd": 0, "compute_cost_usd": 0, "currency": "USD", "cost_accounting_complete": True},
        }
        self.write_receipt()
        self.inputs = self.make_inputs()

    def write(self, path: Path, raw: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)

    def candidate_test_log(self) -> bytes:
        lines = ["running 5 tests"]
        lines.extend(f"test {execution} ... ok" for _, execution in self.LOGICAL)
        lines.append("test result: ok. 5 passed; 0 failed; 0 ignored")
        return ("\n".join(lines) + "\n").encode()

    def baseline_log(self) -> bytes:
        return (
            "running 5 tests\n"
            "test stdout_bytes_contract ... ok\n"
            "test exit_code_contract ... ok\n"
            "test control_totals_contract ... FAILED\n"
            "test numeric_semantics_contract ... FAILED\n"
            "test idempotency_contract ... FAILED\n"
            "test result: FAILED. 2 passed; 3 failed; 0 ignored\n"
        ).encode()

    def reference_log(self) -> bytes:
        return self.candidate_test_log()

    def make_inputs(self):
        return AuditInputs(
            receipt=self.receipt_path, task_contract=self.task_path, task_schema=self.task_schema_path,
            receipt_schema=self.receipt_schema_path, protocol=self.protocol_path, candidate_patch=self.patch_path,
            mutation_catalog=self.catalog_path, evidence_root=self.evidence, expected_revision=self.REVISION,
            expected_image_id=self.IMAGE_ID, expected_task_sha256=sha(self.task_path.read_bytes()),
            expected_protocol_sha256=sha(self.protocol_path.read_bytes()), expected_candidate_patch_sha256=sha(self.patch_path.read_bytes()),
            expected_auditor_sha256=audit_module.sha256_path(MODULE_PATH),
            expected_verifier_binary_sha256=sha(b"synthetic-verifier-binary"),
        )

    def write_receipt(self) -> None:
        self.write(self.receipt_path, jbytes(self.receipt))

    def refresh_task_hash(self) -> None:
        self.receipt["task_contract_sha256"] = sha(self.task_path.read_bytes())
        self.inputs = self.make_inputs()
        self.write_receipt()

    def run_audit(self):
        self.write_receipt()
        return audit_module.audit_bundle(self.inputs)

    def close(self) -> None:
        self.tmp.cleanup()


class ReceiptAuditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = AuditBundle()

    def tearDown(self) -> None:
        self.bundle.close()

    def assert_issue(self, result: dict[str, Any], code: str) -> None:
        codes = {issue["code"] for issue in result["issues"]}
        self.assertIn(code, codes, result)
        self.assertFalse(result["eligible"])

    def test_valid_synthetic_bundle_passes(self) -> None:
        result = self.bundle.run_audit()
        self.assertTrue(result["eligible"], result["issues"])
        self.assertEqual(result["verdict"], "pass")

    def test_task_digest_must_match_out_of_band_pin(self) -> None:
        self.bundle.inputs = replace(self.bundle.inputs, expected_task_sha256="0" * 64)
        self.assert_issue(self.bundle.run_audit(), "trusted_pin_mismatch")

    def test_protocol_digest_must_match_out_of_band_pin(self) -> None:
        self.bundle.inputs = replace(self.bundle.inputs, expected_protocol_sha256="0" * 64)
        self.assert_issue(self.bundle.run_audit(), "trusted_pin_mismatch")

    def test_candidate_patch_digest_must_match_out_of_band_pin(self) -> None:
        self.bundle.inputs = replace(self.bundle.inputs, expected_candidate_patch_sha256="0" * 64)
        self.assert_issue(self.bundle.run_audit(), "trusted_pin_mismatch")

    def test_auditor_binary_must_match_approved_digest(self) -> None:
        self.bundle.inputs = replace(self.bundle.inputs, expected_auditor_sha256="0" * 64)
        self.assert_issue(self.bundle.run_audit(), "auditor_identity_mismatch")

    def test_verifier_binary_identity_must_match_out_of_band_pin(self) -> None:
        self.bundle.inputs = replace(self.bundle.inputs, expected_verifier_binary_sha256="0" * 64)
        self.assert_issue(self.bundle.run_audit(), "receipt_verifier_identity_mismatch")

    def test_receipt_task_hash_tampering_is_rejected(self) -> None:
        self.bundle.receipt["task_contract_sha256"] = "1" * 64
        self.assert_issue(self.bundle.run_audit(), "trusted_pin_mismatch")

    def test_path_traversal_is_rejected(self) -> None:
        self.bundle.receipt["commands"][0]["stdout_artifact"] = "../outside.txt"
        self.assert_issue(self.bundle.run_audit(), "unsafe_evidence_path")

    def test_symlink_evidence_is_rejected(self) -> None:
        target = self.bundle.evidence / "logs/candidate-tests.log"
        link = self.bundle.evidence / "logs/candidate-link.log"
        link.symlink_to(target)
        self.bundle.receipt["test_results"][0]["log_artifact"] = "logs/candidate-link.log"
        self.assert_issue(self.bundle.run_audit(), "unsafe_evidence_path")

    def test_missing_execution_mapping_is_rejected(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["oracle_contract"]["test_execution_map"].pop()
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "execution_map_incomplete")

    def test_scored_task_must_trace_every_declared_invariant(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["behavior_contract"]["requirement_traceability"] = [
            row for row in task["behavior_contract"]["requirement_traceability"]
            if row["source_kind"] != "invariant"
        ]
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "requirement_source_coverage_incomplete")

    def test_traceability_cannot_reference_undeclared_test(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["behavior_contract"]["requirement_traceability"][0]["required_test_ids"] = ["logical.not.declared"]
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "requirement_unknown_test_id")
        self.assert_issue(self.bundle.run_audit(), "requirement_test_inventory_incomplete")

    def test_traceability_must_cover_every_required_test(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        target = self.bundle.LOGICAL[-1][0]
        for row in task["behavior_contract"]["requirement_traceability"]:
            row["required_test_ids"] = [value for value in row["required_test_ids"] if value != target]
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "requirement_test_inventory_incomplete")

    def test_traceability_source_text_must_match_frozen_contract(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["behavior_contract"]["requirement_traceability"][0]["source_text"] = "a more convenient requirement"
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "requirement_source_unmatched")

    def test_traceability_observables_must_be_declared(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["behavior_contract"]["requirement_traceability"][0]["observable_effects"] = ["invented_effect"]
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "requirement_observable_mismatch")

    def test_requirement_ids_must_be_unique(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        trace = task["behavior_contract"]["requirement_traceability"]
        trace[1]["requirement_id"] = trace[0]["requirement_id"]
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "requirement_id_duplicate")

    def test_wrong_runtime_image_is_rejected(self) -> None:
        self.bundle.receipt["environment"]["runtime_identity"]["immutable_image_id"] = "sha256:" + ("0" * 64)
        self.assert_issue(self.bundle.run_audit(), "runtime_identity_mismatch")

    def test_public_task_cannot_be_scored(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["classification"]["exposure"] = "public_development"
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "task_not_score_eligible")

    def test_incomplete_active_mutation_inventory_is_rejected(self) -> None:
        self.bundle.receipt["mutation_results"] = self.bundle.receipt["mutation_results"][:-1]
        self.assert_issue(self.bundle.run_audit(), "mutation_inventory_mismatch")

    def test_unrelated_failure_does_not_kill_a_mutant(self) -> None:
        mid = "VLEB-LDG-001"
        _, target = self.bundle.MUTATIONS[mid]
        compile_name, test_name = self.bundle.mutation_log_paths[mid]
        compile_raw = (self.bundle.evidence / compile_name).read_bytes()
        mutant_log = f"running 2 tests\ntest {target} ... ok\ntest unrelated_test ... FAILED\ntest result: FAILED. 1 passed; 1 failed\n".encode()
        self.bundle.write(self.bundle.evidence / test_name, mutant_log)
        row = next(row for row in self.bundle.receipt["mutation_results"] if row["mutation_id"] == mid)
        row["test_log_sha256"] = sha(mutant_log)
        row["evidence_sha256"] = sha(compile_raw + b"\0" + mutant_log)
        row["failed_test_names"] = ["unrelated_test"]
        self.assert_issue(self.bundle.run_audit(), "mutation_not_killed_by_target")

    def test_external_attestation_is_not_implicitly_trusted(self) -> None:
        self.bundle.receipt["environment"]["trust_boundary"] = "externally_attested"
        self.assert_issue(self.bundle.run_audit(), "trust_boundary_unverified")

    def test_timing_mismatch_is_rejected(self) -> None:
        self.bundle.receipt["timing"]["wall_clock_ms"] = 999999
        self.assert_issue(self.bundle.run_audit(), "timing_evidence_mismatch")

    def test_candidate_log_with_any_failed_test_is_rejected_even_if_mapped_tests_pass(self) -> None:
        new_log = self.bundle.candidate_test_log().decode().replace(
            "test result: ok. 5 passed; 0 failed; 0 ignored",
            "test unrelated_test ... FAILED\ntest result: FAILED. 5 passed; 1 failed",
        ).encode()
        self.bundle.write(self.bundle.evidence / "logs/candidate-tests.log", new_log)
        for row in self.bundle.receipt["test_results"]:
            row["log_sha256"] = sha(new_log)
        self.assert_issue(self.bundle.run_audit(), "candidate_test_suite_not_clean")

    def test_duplicate_catalog_ids_are_rejected(self) -> None:
        catalog = json.loads(self.bundle.catalog_path.read_text())
        catalog["mutations"].append(copy.deepcopy(catalog["mutations"][0]))
        self.bundle.write(self.bundle.catalog_path, jbytes(catalog))
        self.assert_issue(self.bundle.run_audit(), "duplicate_catalog_mutation_id")

    def test_non_equal_or_not_applicable_observables_do_not_cover_contract(self) -> None:
        for row in self.bundle.receipt["differential_checks"]:
            row["status"] = "not_applicable"
        self.assert_issue(self.bundle.run_audit(), "differential_effect_coverage_incomplete")

    def test_baseline_must_fail_preregistered_behavior_test(self) -> None:
        raw = (
            "running 2 tests\n"
            "test unrelated_test ... FAILED\n"
            "test unrelated_other ... ok\n"
            "test result: FAILED. 1 passed; 1 failed\n"
        ).encode()
        self.bundle.write(self.bundle.evidence / "logs/baseline.log", raw)
        task = json.loads(self.bundle.task_path.read_text())
        task["oracle_contract"]["reference_evidence_sha256"]["baseline_log"] = sha(raw)
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "baseline_control_not_demonstrated")

    def test_duplicate_execution_mapping_is_rejected(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["oracle_contract"]["test_execution_map"][1]["execution_ids"] = [self.bundle.LOGICAL[0][1]]
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "execution_mapping_reused")

    def test_toolchain_identity_mismatch_is_rejected(self) -> None:
        task = json.loads(self.bundle.task_path.read_text())
        task["environment"]["toolchains"][0]["identity_sha256"] = sha(b"other compiler")
        self.bundle.write(self.bundle.task_path, jbytes(task))
        self.bundle.refresh_task_hash()
        self.assert_issue(self.bundle.run_audit(), "toolchain_identity_mismatch")

    def test_mapped_test_duplicate_leaf_names_are_ambiguous(self) -> None:
        parsed = audit_module.parse_test_log(b"test alpha::same_name ... ok\ntest beta::same_name ... ok\n")
        self.assertEqual(parsed["same_name"], "CONFLICT")

    def test_differential_check_ids_must_be_unique(self) -> None:
        duplicate = copy.deepcopy(self.bundle.receipt["differential_checks"][0])
        duplicate["observable"] = "idempotency"
        self.bundle.receipt["differential_checks"].append(duplicate)
        self.assert_issue(self.bundle.run_audit(), "duplicate_differential_check_id")


if __name__ == "__main__":
    unittest.main(verbosity=2)
