#!/usr/bin/env python3
"""Schema and semantic invariants for VLEB trusted run manifests."""
from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = ROOT / "tools/vleb/run-manifest.schema.json"


def valid_manifest() -> dict[str, Any]:
    h = lambda char: char * 64
    git = lambda char: char * 40
    return {
        "schema_version": "vleb-run-manifest-v1",
        "run_id": "90001",
        "run_kind": "confirmatory",
        "repository": {
            "full_name": "Luminous-Dynamics/symthaea",
            "workflow_path": ".github/workflows/vleb-confirmatory.yml",
            "workflow_ref": "refs/heads/main",
            "workflow_sha": git("1"),
            "repository_id": 123456789,
            "run_id": "90001",
            "run_attempt": 1,
            "event": "workflow_dispatch",
            "expected_head_sha": git("2"),
        },
        "task": {
            "task_id": "vleb-private-task-0001",
            "exposure": "private_confirmatory",
            "contract_path": "tasks/task.json",
            "contract_sha256": h("a"),
            "independent_reviewer_count": 2,
            "baseline_empty_patch_failed": True,
            "reference_solution_passed": True,
        },
        "protocol": {"path": "protocol.md", "sha256": h("b")},
        "candidate": {
            "patch_path": "candidate.patch",
            "patch_sha256": h("c"),
            "checkout_revision": git("2"),
            "clean_worktree": True,
        },
        "runtime": {
            "platform": "linux",
            "immutable_image_id": "sha256:" + h("d"),
            "network_mode": "none",
            "network_mode_observed": True,
        },
        "toolchains": [{"name": "rustc", "version": "1.96.0", "identity_sha256": h("e")}],
        "auditor": {
            "source_path": "tools/vleb/audit_receipt.py",
            "source_sha256": h("f"),
            "verifier_binary_sha256": h("1"),
            "task_schema_path": "tools/vleb/task-contract.schema.json",
            "task_schema_sha256": h("a"),
            "receipt_schema_path": "tools/vleb/evidence-receipt.schema.json",
            "receipt_schema_sha256": h("b"),
        },
        "mutation_campaign": {
            "catalog_path": "mutations.json",
            "catalog_sha256": h("2"),
            "active_mutation_ids": ["M1", "M2", "M3", "M4"],
            "results_path": "mutations/results.json",
            "results_sha256": h("3"),
            "compilable": 4,
            "killed": 4,
            "survived": 0,
            "unresolved": 0,
        },
        "evidence_bundle": {
            "path": "evidence-bundle.zip",
            "sha256": h("4"),
            "receipt_path": "receipt.json",
            "receipt_sha256": h("5"),
            "artifact_count": 9,
        },
        "provenance": {
            "attestation_required": True,
            "provider": "github_actions_sigstore",
            "subject_path": "evidence-bundle.zip",
            "signer_repository": "Luminous-Dynamics/symthaea",
            "signer_workflow": ".github/workflows/vleb-confirmatory.yml",
            "predicate_type": "https://slsa.dev/provenance/v1",
        },
    }


def semantic_errors(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    repo = manifest.get("repository", {})
    if manifest.get("run_id") != repo.get("run_id"):
        errors.append("run_id_mismatch")
    if manifest.get("candidate", {}).get("checkout_revision") != repo.get("expected_head_sha"):
        errors.append("candidate_head_mismatch")
    if manifest.get("run_kind") == "confirmatory":
        if manifest.get("task", {}).get("exposure") not in ("private_holdout", "private_confirmatory"):
            errors.append("confirmatory_task_not_private")
        if manifest.get("task", {}).get("independent_reviewer_count", 0) < 1:
            errors.append("confirmatory_task_unreviewed")
        if manifest.get("task", {}).get("baseline_empty_patch_failed") is not True:
            errors.append("confirmatory_baseline_control_missing")
        if manifest.get("task", {}).get("reference_solution_passed") is not True:
            errors.append("confirmatory_reference_control_missing")
        provenance = manifest.get("provenance", {})
        if provenance.get("attestation_required") is not True:
            errors.append("confirmatory_attestation_not_required")
        if provenance.get("provider") != "github_actions_sigstore":
            errors.append("confirmatory_attestation_provider_invalid")
        if provenance.get("signer_repository") != repo.get("full_name"):
            errors.append("signer_repository_mismatch")
        if provenance.get("signer_workflow") != repo.get("workflow_path"):
            errors.append("signer_workflow_mismatch")
        if manifest.get("runtime", {}).get("network_mode") == "none" and manifest.get("runtime", {}).get("network_mode_observed") is not True:
            errors.append("offline_network_not_observed")
    campaign = manifest.get("mutation_campaign", {})
    active = campaign.get("active_mutation_ids", [])
    counts = [campaign.get("killed", 0), campaign.get("survived", 0), campaign.get("unresolved", 0)]
    if len(active) != len(set(active)):
        errors.append("duplicate_active_mutation_id")
    if sum(counts) != len(active):
        errors.append("mutation_accounting_inconsistent")
    if campaign.get("killed", 0) + campaign.get("survived", 0) > campaign.get("compilable", 0):
        errors.append("mutation_compilable_count_inconsistent")
    if manifest.get("evidence_bundle", {}).get("path") != manifest.get("provenance", {}).get("subject_path"):
        errors.append("attested_subject_mismatch")
    return errors


class RunManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(cls.schema)
        cls.validator = Draft202012Validator(cls.schema, format_checker=FormatChecker())

    def assert_invalid(self, manifest: dict[str, Any], semantic_code: str | None = None) -> None:
        schema_errors = list(self.validator.iter_errors(manifest))
        sem_errors = semantic_errors(manifest)
        self.assertTrue(schema_errors or sem_errors, "invalid manifest unexpectedly accepted")
        if semantic_code:
            self.assertIn(semantic_code, sem_errors)

    def test_confirmatory_manifest_positive_control(self) -> None:
        manifest = valid_manifest()
        self.assertEqual(list(self.validator.iter_errors(manifest)), [])
        self.assertEqual(semantic_errors(manifest), [])

    def test_development_smoke_can_be_unsigned_but_not_confirmatory(self) -> None:
        manifest = valid_manifest()
        manifest["run_kind"] = "development_smoke"
        manifest["task"]["exposure"] = "public_development"
        manifest["task"]["independent_reviewer_count"] = 0
        manifest["provenance"]["attestation_required"] = False
        manifest["provenance"]["provider"] = "none"
        manifest["provenance"]["signer_workflow"] = ".github/workflows/vleb-confirmatory.yml"
        self.assertEqual(list(self.validator.iter_errors(manifest)), [])
        self.assertEqual(semantic_errors(manifest), [])

    def test_confirmatory_public_task_is_rejected(self) -> None:
        m = valid_manifest()
        m["task"]["exposure"] = "public_development"
        self.assert_invalid(m, "confirmatory_task_not_private")

    def test_confirmatory_without_attestation_is_rejected(self) -> None:
        m = valid_manifest()
        m["provenance"]["attestation_required"] = False
        self.assert_invalid(m, "confirmatory_attestation_not_required")

    def test_unreviewed_task_is_rejected(self) -> None:
        m = valid_manifest()
        m["task"]["independent_reviewer_count"] = 0
        self.assert_invalid(m, "confirmatory_task_unreviewed")

    def test_run_identity_mismatch_is_rejected(self) -> None:
        m = valid_manifest()
        m["repository"]["run_id"] = "different-run"
        self.assert_invalid(m, "run_id_mismatch")

    def test_candidate_revision_must_match_trusted_expected_head(self) -> None:
        m = valid_manifest()
        m["candidate"]["checkout_revision"] = "4" * 40
        self.assert_invalid(m, "candidate_head_mismatch")

    def test_mutation_counts_must_reconcile(self) -> None:
        m = valid_manifest()
        m["mutation_campaign"]["killed"] = 3
        self.assert_invalid(m, "mutation_accounting_inconsistent")

    def test_duplicate_active_mutation_id_is_rejected(self) -> None:
        m = valid_manifest()
        m["mutation_campaign"]["active_mutation_ids"].append("M1")
        self.assert_invalid(m, "duplicate_active_mutation_id")

    def test_wrong_signer_repository_is_rejected(self) -> None:
        m = valid_manifest()
        m["provenance"]["signer_repository"] = "untrusted/fork"
        self.assert_invalid(m, "signer_repository_mismatch")

    def test_wrong_signer_workflow_is_rejected(self) -> None:
        m = valid_manifest()
        m["provenance"]["signer_workflow"] = ".github/workflows/other.yml"
        self.assert_invalid(m, "signer_workflow_mismatch")

    def test_attested_subject_must_be_the_evidence_bundle(self) -> None:
        m = valid_manifest()
        m["provenance"]["subject_path"] = "other.zip"
        self.assert_invalid(m, "attested_subject_mismatch")

    def test_confirmatory_baseline_and_reference_controls_are_required(self) -> None:
        baseline = valid_manifest()
        baseline["task"]["baseline_empty_patch_failed"] = False
        self.assert_invalid(baseline, "confirmatory_baseline_control_missing")
        reference = valid_manifest()
        reference["task"]["reference_solution_passed"] = False
        self.assert_invalid(reference, "confirmatory_reference_control_missing")

    def test_confirmatory_run_observes_offline_network_policy(self) -> None:
        m = valid_manifest()
        m["runtime"]["network_mode_observed"] = False
        self.assert_invalid(m)


if __name__ == "__main__":
    unittest.main(verbosity=2)
