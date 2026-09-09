#!/usr/bin/env python3
"""Regression tests for TrustedMainRootReceiptV2-bound witness V3."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import evidence_contract_witness as v1  # noqa: E402
import evidence_contract_witness_v3 as v3  # noqa: E402

SHA = "1" * 40
TREE = "2" * 40
QUALIFIER = "a" * 64
WORKFLOW = "b" * 64
TRUSTED_SHA = "6" * 40
TRUSTED_TREE = "7" * 40
ARCHIVE = "9" * 64
REPOSITORY_ID = 1136141775
RUN_ID = 123
RUN_ATTEMPT = 1
ARTIFACT_NAME = f"evidence-contract-qualification-{RUN_ID}-{RUN_ATTEMPT}"
REPOSITORY = "Luminous-Dynamics/symthaea"


def receipt_bytes(**overrides: str) -> bytes:
    values = {
        "schema": v1.SCHEMA,
        "status": "PASS",
        "exit_code": "0",
        "terminal_stage": "none",
        "scope": "software-contract-only",
        "scientific_authority": "none",
        "full_repository_ci": "independent",
        "baseline_generation": "forbidden",
        "environment_authority": "observed-not-capsule-qualified",
        "receipt_attestation": "none",
        "provider_metadata_basis": "ambient-runtime",
        "qualified_sha": SHA,
        "expected_sha": SHA,
        "committed_tree": TREE,
        "source_state": "clean-exact-checkout-postflight",
        "execution_provider": "github-actions",
        "runner_label": "ubuntu-24.04",
        "runner_os": "Linux",
        "runner_arch": "X64",
        "runner_image_os": "ubuntu24",
        "runner_image_version": "20260901.1.0",
        "os_release_sha256": "c" * 64,
        "kernel_release": "6.11.0-1018-azure",
        "rustc_release": "1.96.0",
        "rustc_commit_hash": "3" * 40,
        "rustc_host": "x86_64-unknown-linux-gnu",
        "cargo_version": "cargo 1.96.0",
        "checkout_action_sha": "4" * 40,
        "upload_artifact_action_sha": "5" * 40,
        "cargo_lock_sha256": "d" * 64,
        "workspace_manifest_sha256": "e" * 64,
        "rust_toolchain_sha256": "f" * 64,
        "qualifier_script_sha256": QUALIFIER,
        "workflow_sha256": WORKFLOW,
        "authority_integration_targets": "none",
        "github_event_name": "pull_request",
        "github_repository": REPOSITORY,
        "github_workflow_ref": "Luminous-Dynamics/symthaea/.github/workflows/evidence-contract.yml@refs/pull/1119/merge",
        "github_job": "evidence-contract",
        "github_run_id": str(RUN_ID),
        "github_run_attempt": str(RUN_ATTEMPT),
    }
    values.update(overrides)
    return "".join(f"{key}\t{values[key]}\n" for key in v1.RECEIPT_KEYS).encode()


def root_receipt_v2(**overrides) -> dict:
    trusted_recipe = v1.compute_recipe_id(QUALIFIER, WORKFLOW)
    payload = {
        "schema": v3.ROOT_RECEIPT_SCHEMA,
        "repository": REPOSITORY,
        "repository_id": REPOSITORY_ID,
        "target_ref": "refs/heads/main",
        "root_sha": TRUSTED_SHA,
        "root_tree": TRUSTED_TREE,
        "root_subject_id": "sha256:" + "1" * 64,
        "protection_policy_id": "sha256:" + "2" * 64,
        "protection_verification_id": "sha256:" + "3" * 64,
        "effective_rules_verification_id": "sha256:" + "4" * 64,
        "enforcement_evidence_id": "sha256:" + "5" * 64,
        "trusted_evidence_selection_id": "",
        "ruleset_id": 4242,
        "bypass_policy_id": "sha256:" + "6" * 64,
        "review_assurance": "pr-mediated-no-required-approval",
        "root_harness_identity": trusted_recipe,
        "v1_root_receipt_id": "sha256:" + "7" * 64,
        "admission_disposition": "AdmittedHistoricalRoot",
        "evidence_selection_basis": "trusted-phase-independent-expected-ids",
        "historical_scope": "exact-root-only",
        "current_admission": "not-evaluated",
        "receipt_attestation": "none",
        "self_hosted_runner_activation": "not-authorized",
        "scientific_authority": "none",
        "bootstrap_authority": "none",
    }
    payload.update(overrides)
    if not payload.get("trusted_evidence_selection_id"):
        payload["trusted_evidence_selection_id"] = v3._domain_id(v3.SELECTION_DOMAIN, {
            "schema": v3.SELECTION_SCHEMA,
            "protection_policy_id": payload["protection_policy_id"],
            "protection_verification_id": payload["protection_verification_id"],
            "effective_rules_verification_id": payload["effective_rules_verification_id"],
            "root_subject_id": payload["root_subject_id"],
            "enforcement_evidence_id": payload["enforcement_evidence_id"],
        })
    payload["root_receipt_id"] = v3._domain_id(v3.ROOT_RECEIPT_DOMAIN, payload)
    return payload


def evaluate(*, root=None, bootstrap=False, candidate_qualifier=QUALIFIER, candidate_workflow=WORKFLOW, **kwargs):
    if root is None and not bootstrap:
        root = root_receipt_v2()
    base = dict(
        receipt_bytes=receipt_bytes(
            qualifier_script_sha256=candidate_qualifier,
            workflow_sha256=candidate_workflow,
        ),
        expected_subject_sha=SHA,
        expected_subject_tree=TREE,
        expected_run_id=RUN_ID,
        expected_run_attempt=RUN_ATTEMPT,
        expected_run_conclusion="success",
        expected_workflow_id=354015802,
        expected_artifact_id=456,
        expected_artifact_name=ARTIFACT_NAME,
        expected_artifact_archive_sha256=ARCHIVE,
        expected_repository=REPOSITORY,
        expected_repository_id=REPOSITORY_ID,
        expected_candidate_qualifier_sha256=candidate_qualifier,
        expected_candidate_workflow_sha256=candidate_workflow,
        expected_authority_integration_targets="none",
        bootstrap_no_predecessor=bootstrap,
    )
    if not bootstrap:
        base.update(
            trusted_root_receipt_v2=root,
            expected_trusted_root_receipt_id=root["root_receipt_id"] if root is not None else None,
            trusted_predecessor_sha=TRUSTED_SHA,
            trusted_predecessor_tree=TRUSTED_TREE,
            trusted_qualifier_sha256=QUALIFIER,
            trusted_workflow_sha256=WORKFLOW,
        )
    base.update(kwargs)
    return v3.evaluate_witness_v3(**base)


class WitnessV3Tests(unittest.TestCase):
    def test_v2_root_enables_narrow_witness(self):
        result = evaluate()
        self.assertEqual(result["schema"], v3.WITNESS_SCHEMA)
        self.assertEqual(result["witness_disposition"], "FocusedSoftwareContractWitnessed")
        self.assertEqual(result["trusted_root_receipt_relation"], "IndependentlySelectedAdmittedRootBound")
        self.assertEqual(result["trusted_root_receipt_schema"], v3.ROOT_RECEIPT_SCHEMA)
        self.assertEqual(result["trusted_root_receipt_id"], root_receipt_v2()["root_receipt_id"])
        self.assertEqual(result["trusted_evidence_selection_id"], root_receipt_v2()["trusted_evidence_selection_id"])
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["current_admission"], "not-evaluated")

    def test_old_v1_root_receipt_schema_is_rejected(self):
        observed = root_receipt_v2(schema="symthaea.github-trusted-main-root-receipt.v1")
        with self.assertRaisesRegex(v3.WitnessV3Error, "expected .*root-receipt.v2"):
            evaluate(root=observed)

    def test_forged_root_receipt_content_id_rejects(self):
        observed = root_receipt_v2()
        observed["root_receipt_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(v3.WitnessV3Error, "root_receipt_id: content ID mismatch"):
            evaluate(root=observed, expected_trusted_root_receipt_id=observed["root_receipt_id"])

    def test_forged_evidence_selection_id_rejects(self):
        observed = root_receipt_v2(trusted_evidence_selection_id="sha256:" + "0" * 64)
        with self.assertRaisesRegex(v3.WitnessV3Error, "trusted_evidence_selection_id: content ID mismatch"):
            evaluate(root=observed)

    def test_root_cannot_select_its_own_expected_id(self):
        observed = root_receipt_v2(protection_policy_id="sha256:" + "e" * 64)
        trusted = root_receipt_v2()
        with self.assertRaisesRegex(v3.WitnessV3Error, "independently expected root receipt ID"):
            evaluate(root=observed, expected_trusted_root_receipt_id=trusted["root_receipt_id"])

    def test_missing_root_receipt_fails_closed(self):
        with self.assertRaisesRegex(v3.WitnessV3Error, "requires TrustedMainRootReceiptV2"):
            v3.evaluate_witness_v3(
                receipt_bytes=receipt_bytes(),
                expected_subject_sha=SHA, expected_subject_tree=TREE,
                expected_run_id=RUN_ID, expected_run_attempt=RUN_ATTEMPT,
                expected_run_conclusion="success", expected_workflow_id=354015802,
                expected_artifact_id=456, expected_artifact_name=ARTIFACT_NAME,
                expected_artifact_archive_sha256=ARCHIVE,
                expected_repository=REPOSITORY, expected_repository_id=REPOSITORY_ID,
                expected_candidate_qualifier_sha256=QUALIFIER,
                expected_candidate_workflow_sha256=WORKFLOW,
                expected_authority_integration_targets="none",
            )

    def test_root_sha_mismatch_rejects(self):
        observed = root_receipt_v2(root_sha="8" * 40)
        with self.assertRaisesRegex(v3.WitnessV3Error, "predecessor SHA/tree mismatch"):
            evaluate(root=observed)

    def test_root_tree_mismatch_rejects(self):
        observed = root_receipt_v2(root_tree="8" * 40)
        with self.assertRaisesRegex(v3.WitnessV3Error, "predecessor SHA/tree mismatch"):
            evaluate(root=observed)

    def test_harness_must_match_independently_observed_recipe(self):
        observed = root_receipt_v2(root_harness_identity="sha256:" + "f" * 64)
        with self.assertRaisesRegex(v3.WitnessV3Error, "harness identity"):
            evaluate(root=observed)

    def test_recipe_change_remains_conformance_only(self):
        result = evaluate(candidate_qualifier="0" * 64)
        self.assertEqual(result["witness_disposition"], "RecipeChangedConformanceOnly")
        self.assertEqual(result["trusted_root_receipt_relation"], "IndependentlySelectedAdmittedRootBound")

    def test_bootstrap_remains_predecessor_free(self):
        result = evaluate(root=None, bootstrap=True)
        self.assertEqual(result["witness_disposition"], "BootstrapNoPredecessor")
        self.assertEqual(result["trusted_root_receipt_relation"], "BootstrapNoPredecessor")
        self.assertIsNone(result["trusted_root_receipt_id"])
        self.assertIsNone(result["trusted_evidence_selection_id"])

    def test_bootstrap_cannot_smuggle_root_receipt(self):
        with self.assertRaisesRegex(v3.WitnessV3Error, "bootstrap cannot include"):
            v3.evaluate_witness_v3(
                receipt_bytes=receipt_bytes(),
                expected_subject_sha=SHA, expected_subject_tree=TREE,
                expected_run_id=RUN_ID, expected_run_attempt=RUN_ATTEMPT,
                expected_run_conclusion="success", expected_workflow_id=354015802,
                expected_artifact_id=456, expected_artifact_name=ARTIFACT_NAME,
                expected_artifact_archive_sha256=ARCHIVE,
                expected_repository=REPOSITORY, expected_repository_id=REPOSITORY_ID,
                expected_candidate_qualifier_sha256=QUALIFIER,
                expected_candidate_workflow_sha256=WORKFLOW,
                expected_authority_integration_targets="none",
                trusted_root_receipt_v2=root_receipt_v2(),
                bootstrap_no_predecessor=True,
            )

    def test_unknown_root_field_rejects(self):
        observed = root_receipt_v2()
        observed["future_authority"] = True
        with self.assertRaisesRegex(v3.WitnessV3Error, "unknown fields"):
            evaluate(root=observed)

    def test_root_identity_changes_witness_identity(self):
        first = evaluate()
        observed = root_receipt_v2(protection_policy_id="sha256:" + "e" * 64)
        second = evaluate(root=observed)
        self.assertNotEqual(first["trusted_root_receipt_id"], second["trusted_root_receipt_id"])
        self.assertNotEqual(first["witness_id"], second["witness_id"])


if __name__ == "__main__":
    unittest.main()
