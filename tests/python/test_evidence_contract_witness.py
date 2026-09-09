#!/usr/bin/env python3
"""Regression tests for the data-only focused evidence-contract witness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import evidence_contract_witness as witness  # noqa: E402

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


def receipt_bytes(**overrides: str) -> bytes:
    values = {
        "schema": witness.SCHEMA,
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
        "github_repository": "Luminous-Dynamics/symthaea",
        "github_workflow_ref": (
            "Luminous-Dynamics/symthaea/.github/workflows/"
            "evidence-contract.yml@refs/pull/1119/merge"
        ),
        "github_job": "evidence-contract",
        "github_run_id": str(RUN_ID),
        "github_run_attempt": str(RUN_ATTEMPT),
    }
    values.update(overrides)
    return "".join(f"{key}\t{values[key]}\n" for key in witness.RECEIPT_KEYS).encode()


def evaluate(data: bytes, **kwargs):
    base = dict(
        receipt_bytes=data,
        expected_subject_sha=SHA,
        expected_subject_tree=TREE,
        expected_run_id=RUN_ID,
        expected_run_attempt=RUN_ATTEMPT,
        expected_run_conclusion="success",
        expected_workflow_id=354015802,
        expected_artifact_id=456,
        expected_artifact_name=ARTIFACT_NAME,
        expected_artifact_archive_sha256=ARCHIVE,
        expected_repository="Luminous-Dynamics/symthaea",
        expected_repository_id=REPOSITORY_ID,
        expected_candidate_qualifier_sha256=QUALIFIER,
        expected_candidate_workflow_sha256=WORKFLOW,
        expected_authority_integration_targets="none",
        trusted_predecessor_sha=TRUSTED_SHA,
        trusted_predecessor_tree=TRUSTED_TREE,
        trusted_qualifier_sha256=QUALIFIER,
        trusted_workflow_sha256=WORKFLOW,
    )
    base.update(kwargs)
    return witness.evaluate_witness(**base)


class ReceiptParserTests(unittest.TestCase):
    def test_valid_receipt_parses(self):
        parsed = witness.parse_receipt_bytes(receipt_bytes())
        self.assertEqual(parsed["status"], "PASS")
        self.assertEqual(tuple(parsed), witness.RECEIPT_KEYS)

    def test_duplicate_key_rejected(self):
        with self.assertRaisesRegex(witness.WitnessError, "duplicate key|canonical order"):
            data = receipt_bytes() + b"status\tPASS\n"
            witness.parse_receipt_bytes(data)

    def test_unknown_key_rejected(self):
        with self.assertRaisesRegex(witness.WitnessError, "unknown keys"):
            data = receipt_bytes() + b"future_authority\ttrue\n"
            witness.parse_receipt_bytes(data)

    def test_extra_tsv_column_rejected(self):
        data = receipt_bytes().replace(b"schema\t", b"schema\textra\t", 1)
        with self.assertRaisesRegex(witness.WitnessError, "exactly one TSV separator"):
            witness.parse_receipt_bytes(data)

    def test_missing_terminal_lf_rejected(self):
        with self.assertRaisesRegex(witness.WitnessError, "must end with LF"):
            witness.parse_receipt_bytes(receipt_bytes().rstrip(b"\n"))

    def test_reordered_fields_rejected(self):
        lines = receipt_bytes().splitlines(keepends=True)
        lines[0], lines[1] = lines[1], lines[0]
        with self.assertRaisesRegex(witness.WitnessError, "canonical order"):
            witness.parse_receipt_bytes(b"".join(lines))

    def test_unicode_line_separator_rejected(self):
        data = receipt_bytes(kernel_release="ok\u2028evil")
        with self.assertRaisesRegex(witness.WitnessError, "Unicode line separators"):
            witness.parse_receipt_bytes(data)

    def test_pass_with_nonzero_exit_rejected(self):
        with self.assertRaisesRegex(witness.WitnessError, "PASS requires exit_code=0"):
            witness.parse_receipt_bytes(receipt_bytes(exit_code="1"))

    def test_pass_with_unclean_source_rejected(self):
        with self.assertRaisesRegex(witness.WitnessError, "clean-exact-checkout-postflight"):
            witness.parse_receipt_bytes(receipt_bytes(source_state="tracked-modifications-present"))

    def test_authority_target_list_must_be_sorted_unique(self):
        with self.assertRaisesRegex(witness.WitnessError, "sorted and unique"):
            witness.parse_receipt_bytes(
                receipt_bytes(authority_integration_targets="z_target,a_target")
            )


class WitnessBindingTests(unittest.TestCase):
    def test_matching_predecessor_recipe_produces_narrow_witness(self):
        result = evaluate(receipt_bytes())
        self.assertEqual(result["recipe_relation"], "BaseIdentical")
        self.assertEqual(result["witness_disposition"], "FocusedSoftwareContractWitnessed")
        self.assertEqual(result["repository_id"], REPOSITORY_ID)
        self.assertEqual(result["provider_artifact_name"], ARTIFACT_NAME)
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertRegex(result["witness_id"], r"^sha256:[0-9a-f]{64}$")

    def test_subject_sha_substitution_fails_closed(self):
        with self.assertRaisesRegex(witness.WitnessError, "subject SHA"):
            evaluate(receipt_bytes(), expected_subject_sha="8" * 40)

    def test_subject_tree_substitution_fails_closed(self):
        with self.assertRaisesRegex(witness.WitnessError, "receipt tree"):
            evaluate(receipt_bytes(), expected_subject_tree="8" * 40)

    def test_run_identity_substitution_fails_closed(self):
        with self.assertRaisesRegex(witness.WitnessError, "run ID"):
            evaluate(receipt_bytes(), expected_run_id=124, expected_artifact_name="evidence-contract-qualification-124-1")

    def test_artifact_name_must_bind_run_and_attempt(self):
        with self.assertRaisesRegex(witness.WitnessError, "bind exact run/attempt"):
            evaluate(receipt_bytes(), expected_artifact_name="evidence-contract-qualification-other")

    def test_repository_id_changes_trust_root_identity(self):
        first = evaluate(receipt_bytes())
        second = evaluate(receipt_bytes(), expected_repository_id=REPOSITORY_ID + 1)
        self.assertNotEqual(first["trusted_root_snapshot_id"], second["trusted_root_snapshot_id"])
        self.assertNotEqual(first["witness_id"], second["witness_id"])

    def test_provider_non_success_cannot_witness_candidate_pass(self):
        with self.assertRaisesRegex(witness.WitnessError, "PASS conflicts"):
            evaluate(receipt_bytes(), expected_run_conclusion="failure")

    def test_self_reported_recipe_hash_cannot_substitute_for_observed_bytes(self):
        with self.assertRaisesRegex(witness.WitnessError, "qualifier hash"):
            evaluate(receipt_bytes(), expected_candidate_qualifier_sha256="0" * 64)

    def test_self_reported_workflow_hash_cannot_substitute_for_observed_bytes(self):
        with self.assertRaisesRegex(witness.WitnessError, "workflow hash"):
            evaluate(receipt_bytes(), expected_candidate_workflow_sha256="0" * 64)

    def test_candidate_recipe_change_is_conformance_only(self):
        changed = "0" * 64
        result = evaluate(
            receipt_bytes(qualifier_script_sha256=changed),
            expected_candidate_qualifier_sha256=changed,
        )
        self.assertEqual(result["recipe_relation"], "RecipeChanged")
        self.assertEqual(result["witness_disposition"], "RecipeChangedConformanceOnly")

    def test_authority_target_discovery_must_match_trusted_observation(self):
        with self.assertRaisesRegex(witness.WitnessError, "authority target list"):
            evaluate(
                receipt_bytes(authority_integration_targets="butlin_x_authority_regression"),
                expected_authority_integration_targets="none",
            )

    def test_bootstrap_is_explicitly_non_authorizing(self):
        result = witness.evaluate_witness(
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
            expected_repository="Luminous-Dynamics/symthaea",
            expected_repository_id=REPOSITORY_ID,
            expected_candidate_qualifier_sha256=QUALIFIER,
            expected_candidate_workflow_sha256=WORKFLOW,
            expected_authority_integration_targets="none",
            bootstrap_no_predecessor=True,
        )
        self.assertEqual(result["recipe_relation"], "BootstrapNoPredecessor")
        self.assertEqual(result["witness_disposition"], "BootstrapNoPredecessor")
        self.assertIsNone(result["trusted_recipe_id"])
        self.assertIsNone(result["trusted_root_snapshot_id"])

    def test_bootstrap_cannot_smuggle_trusted_inputs(self):
        with self.assertRaisesRegex(witness.WitnessError, "cannot include"):
            witness.evaluate_witness(
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
                expected_repository="Luminous-Dynamics/symthaea",
                expected_repository_id=REPOSITORY_ID,
                expected_candidate_qualifier_sha256=QUALIFIER,
                expected_candidate_workflow_sha256=WORKFLOW,
                expected_authority_integration_targets="none",
                bootstrap_no_predecessor=True,
                trusted_predecessor_sha=TRUSTED_SHA,
            )

    def test_candidate_fail_is_distinct_from_provider_non_success(self):
        failure_receipt = receipt_bytes(
            status="FAIL",
            exit_code="1",
            terminal_stage="butlin_structural_contract",
            source_state="clean-exact-checkout",
        )
        source_failure = evaluate(failure_receipt, expected_run_conclusion="failure")
        cancelled = evaluate(failure_receipt, expected_run_conclusion="cancelled")
        self.assertEqual(source_failure["witness_disposition"], "CandidateFailed")
        self.assertEqual(cancelled["witness_disposition"], "ProviderNonSuccess")

    def test_recipe_identity_is_separate_from_predecessor_snapshot_identity(self):
        recipe_id = witness.compute_recipe_id(QUALIFIER, WORKFLOW)
        self.assertEqual(recipe_id, witness.compute_recipe_id(QUALIFIER, WORKFLOW))
        root_a = witness.compute_trust_root_snapshot_id(REPOSITORY_ID, TRUSTED_SHA, TRUSTED_TREE, recipe_id)
        root_b = witness.compute_trust_root_snapshot_id(REPOSITORY_ID, "8" * 40, TRUSTED_TREE, recipe_id)
        self.assertNotEqual(root_a, root_b)

    def test_receipt_identity_changes_with_runtime_metadata_not_recipe_identity(self):
        first = evaluate(receipt_bytes())
        second = evaluate(receipt_bytes(runner_image_version="20260902.1.0"))
        self.assertNotEqual(first["candidate_receipt_id"], second["candidate_receipt_id"])
        self.assertEqual(first["candidate_recipe_id"], second["candidate_recipe_id"])

    def test_provider_archive_identity_is_separate_from_receipt_identity(self):
        first = evaluate(receipt_bytes())
        second = evaluate(receipt_bytes(), expected_artifact_archive_sha256="8" * 64)
        self.assertEqual(first["candidate_receipt_id"], second["candidate_receipt_id"])
        self.assertNotEqual(first["provider_artifact_archive_sha256"], second["provider_artifact_archive_sha256"])
        self.assertNotEqual(first["witness_id"], second["witness_id"])


if __name__ == "__main__":
    unittest.main()
