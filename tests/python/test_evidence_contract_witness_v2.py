#!/usr/bin/env python3
"""Regression tests for root-receipt-bound evidence-contract witness V2."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import evidence_contract_witness as v1  # noqa: E402
import evidence_contract_witness_v2 as v2  # noqa: E402

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


def root_receipt(**overrides) -> dict:
    trusted_recipe = v1.compute_recipe_id(QUALIFIER, WORKFLOW)
    payload = {
        "schema": v2.ROOT_RECEIPT_SCHEMA,
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
        "ruleset_id": 4242,
        "bypass_policy_id": "sha256:" + "6" * 64,
        "review_assurance": "pr-mediated-no-required-approval",
        "root_harness_identity": trusted_recipe,
        "admission_disposition": "AdmittedHistoricalRoot",
        "non_admission_reasons": [],
        "historical_scope": "exact-root-only",
        "current_admission": "not-evaluated",
        "receipt_attestation": "none",
        "self_hosted_runner_activation": "not-authorized",
        "scientific_authority": "none",
        "bootstrap_authority": "none",
    }
    payload.update(overrides)
    payload["root_receipt_id"] = v2._domain_id(v2.ROOT_RECEIPT_DOMAIN, payload)
    return payload


def evaluate(*, root=root_receipt(), bootstrap=False, candidate_qualifier=QUALIFIER, candidate_workflow=WORKFLOW, **kwargs):
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
            trusted_root_receipt=root,
            expected_trusted_root_receipt_id=root["root_receipt_id"] if root is not None else None,
            trusted_predecessor_sha=TRUSTED_SHA,
            trusted_predecessor_tree=TRUSTED_TREE,
            trusted_qualifier_sha256=QUALIFIER,
            trusted_workflow_sha256=WORKFLOW,
        )
    base.update(kwargs)
    return v2.evaluate_witness_v2(**base)


class RootReceiptBindingTests(unittest.TestCase):
    def test_admitted_root_enables_narrow_v2_witness(self):
        result = evaluate()
        self.assertEqual(result["schema"], v2.WITNESS_SCHEMA)
        self.assertEqual(result["witness_disposition"], "FocusedSoftwareContractWitnessed")
        self.assertEqual(result["trusted_root_receipt_relation"], "AdmittedRootBound")
        self.assertEqual(result["trusted_root_receipt_id"], root_receipt()["root_receipt_id"])
        self.assertRegex(result["legacy_v1_witness_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertRegex(result["witness_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["current_admission"], "not-evaluated")

    def test_missing_root_receipt_fails_closed(self):
        with self.assertRaisesRegex(v2.WitnessV2Error, "requires an admitted trusted root receipt"):
            evaluate(root=None)

    def test_self_consistent_root_receipt_cannot_choose_its_own_trusted_id(self):
        observed = root_receipt(protection_policy_id="sha256:" + "e" * 64)
        with self.assertRaisesRegex(v2.WitnessV2Error, "independently expected root receipt ID"):
            evaluate(root=observed, expected_trusted_root_receipt_id=root_receipt()["root_receipt_id"])

    def test_root_receipt_content_id_forgery_rejects(self):
        observed = root_receipt()
        observed["root_receipt_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(v2.WitnessV2Error, "content ID mismatch"):
            evaluate(root=observed)

    def test_non_admitted_root_receipt_rejects(self):
        observed = root_receipt(
            admission_disposition="PendingEnforcementEvidence",
            non_admission_reasons=["EnforcementEvidenceMissing"],
            enforcement_evidence_id=None,
        )
        with self.assertRaisesRegex(v2.WitnessV2Error, "enforcement evidence ID|not admitted"):
            evaluate(root=observed)

    def test_root_sha_mismatch_rejects(self):
        observed = root_receipt(root_sha="8" * 40)
        with self.assertRaisesRegex(v2.WitnessV2Error, "predecessor SHA/tree mismatch"):
            evaluate(root=observed)

    def test_root_tree_mismatch_rejects(self):
        observed = root_receipt(root_tree="8" * 40)
        with self.assertRaisesRegex(v2.WitnessV2Error, "predecessor SHA/tree mismatch"):
            evaluate(root=observed)

    def test_repository_identity_mismatch_rejects(self):
        observed = root_receipt(repository="SomebodyElse/symthaea")
        with self.assertRaisesRegex(v2.WitnessV2Error, "repository identity mismatch"):
            evaluate(root=observed)

    def test_postbootstrap_harness_must_match_trusted_recipe_bytes(self):
        observed = root_receipt(root_harness_identity="sha256:" + "f" * 64)
        with self.assertRaisesRegex(v2.WitnessV2Error, "harness identity"):
            evaluate(root=observed)

    def test_prebootstrap_harness_cannot_authorize_ordinary_witness(self):
        observed = root_receipt(root_harness_identity="none-pre-bootstrap")
        with self.assertRaisesRegex(v2.WitnessV2Error, "post-bootstrap harness identity"):
            evaluate(root=observed)

    def test_recipe_change_remains_conformance_only_even_with_admitted_root(self):
        changed = "0" * 64
        result = evaluate(candidate_qualifier=changed)
        self.assertEqual(result["witness_disposition"], "RecipeChangedConformanceOnly")
        self.assertEqual(result["trusted_root_receipt_relation"], "AdmittedRootBound")

    def test_root_receipt_identity_changes_v2_witness_identity(self):
        first = evaluate()
        observed = root_receipt(protection_policy_id="sha256:" + "e" * 64)
        second = evaluate(root=observed)
        self.assertNotEqual(first["trusted_root_receipt_id"], second["trusted_root_receipt_id"])
        self.assertNotEqual(first["witness_id"], second["witness_id"])

    def test_bootstrap_still_requires_no_predecessor(self):
        result = evaluate(bootstrap=True, root=None)
        self.assertEqual(result["witness_disposition"], "BootstrapNoPredecessor")
        self.assertEqual(result["trusted_root_receipt_relation"], "BootstrapNoPredecessor")
        self.assertIsNone(result["trusted_root_receipt_id"])

    def test_bootstrap_cannot_smuggle_expected_root_id_only(self):
        with self.assertRaisesRegex(v2.WitnessV2Error, "bootstrap cannot include"):
            v2.evaluate_witness_v2(
                receipt_bytes=receipt_bytes(),
                expected_subject_sha=SHA, expected_subject_tree=TREE,
                expected_run_id=RUN_ID, expected_run_attempt=RUN_ATTEMPT,
                expected_run_conclusion="success", expected_workflow_id=354015802,
                expected_artifact_id=456, expected_artifact_name=ARTIFACT_NAME,
                expected_artifact_archive_sha256=ARCHIVE, expected_repository=REPOSITORY,
                expected_repository_id=REPOSITORY_ID,
                expected_candidate_qualifier_sha256=QUALIFIER,
                expected_candidate_workflow_sha256=WORKFLOW,
                expected_authority_integration_targets="none",
                expected_trusted_root_receipt_id=root_receipt()["root_receipt_id"],
                bootstrap_no_predecessor=True,
            )

    def test_bootstrap_cannot_smuggle_root_receipt(self):
        with self.assertRaisesRegex(v2.WitnessV2Error, "bootstrap cannot include"):
            v2.evaluate_witness_v2(
                receipt_bytes=receipt_bytes(),
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
                expected_candidate_qualifier_sha256=QUALIFIER,
                expected_candidate_workflow_sha256=WORKFLOW,
                expected_authority_integration_targets="none",
                trusted_root_receipt=root_receipt(),
                bootstrap_no_predecessor=True,
            )

    def test_unknown_root_receipt_field_rejects(self):
        observed = root_receipt()
        observed["future_authority"] = True
        with self.assertRaisesRegex(v2.WitnessV2Error, "unknown fields"):
            evaluate(root=observed)


if __name__ == "__main__":
    unittest.main()
