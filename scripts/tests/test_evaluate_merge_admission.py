import copy
import unittest

from scripts.ci.evaluate_merge_admission import (
    Decision,
    canonical_json,
    evaluate,
    git_blob_oid,
)


BASE = "1" * 40
HEAD = "2" * 40
TREE = "3" * 40
CI_BLOB = "4" * 40


class MergeAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "schema": "symthaea.merge-admission-policy.v1",
            "policy_path": "scripts/ci/merge_admission_policy_v1.json",
            "repository": "Luminous-Dynamics/symthaea",
            "target_branch": "main",
            "enforcement_ready": False,
            "decision_default": "incomplete",
            "head_change_invalidates": True,
            "base_change_invalidates": True,
            "unknown_evidence_default": "reject",
            "control_plane": {
                "mode": "exact_base_equivalence",
                "paths": [
                    ".github/workflows/ci.yml",
                    "scripts/ci/merge_admission_policy_v1.json",
                    "scripts/ci/required_ci_job_manifest_v1.json",
                ],
                "candidate_changes_require_independent_bootstrap": True,
            },
            "full_integration": {
                "workflow_path": ".github/workflows/ci.yml",
                "required_job_manifest_path": "scripts/ci/required_ci_job_manifest_v1.json",
                "accepted_events": ["pull_request", "workflow_dispatch"],
                "required_status": "completed",
                "required_conclusion": "success",
                "require_exact_head": True,
                "require_exact_base": True,
                "require_no_required_job_skips": True,
            },
            "focused_evidence_can_substitute_for_full_integration": False,
            "tier1_can_substitute_for_full_integration": False,
        }
        self.manifest = {
            "schema": "symthaea.required-ci-job-manifest.v1",
            "workflow_path": ".github/workflows/ci.yml",
            "workflow_blob_sha": CI_BLOB,
            "complete": True,
            "profiles": {
                "pull_request": {
                    "event": "pull_request",
                    "top_level_job_ids": ["test", "clippy"],
                    "families": [
                        {
                            "job_id": "test",
                            "api_name_regex": r"Test \(default features\)",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "clippy",
                            "api_name_regex": "Clippy",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                    ],
                },
                "workflow_dispatch": {
                    "event": "workflow_dispatch",
                    "top_level_job_ids": ["test", "clippy"],
                    "families": [
                        {
                            "job_id": "test",
                            "api_name_regex": r"Test \(default features\)",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "clippy",
                            "api_name_regex": "Clippy",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                    ],
                },
            },
        }
        self.policy_bytes = canonical_json(self.policy)
        self.manifest_bytes = canonical_json(self.manifest)
        policy_blob = git_blob_oid(self.policy_bytes, hex_length=40)
        manifest_blob = git_blob_oid(self.manifest_bytes, hex_length=40)
        self.observation = {
            "schema": "symthaea.merge-admission-observation.v1",
            "repository": "Luminous-Dynamics/symthaea",
            "target_branch": "main",
            "current_base_sha": BASE,
            "candidate_head_sha": HEAD,
            "candidate_tree_sha": TREE,
            "control_plane": [
                {
                    "path": ".github/workflows/ci.yml",
                    "base_blob_sha": CI_BLOB,
                    "candidate_blob_sha": CI_BLOB,
                },
                {
                    "path": "scripts/ci/merge_admission_policy_v1.json",
                    "base_blob_sha": policy_blob,
                    "candidate_blob_sha": policy_blob,
                },
                {
                    "path": "scripts/ci/required_ci_job_manifest_v1.json",
                    "base_blob_sha": manifest_blob,
                    "candidate_blob_sha": manifest_blob,
                },
            ],
            "full_integration": {
                "workflow_path": ".github/workflows/ci.yml",
                "workflow_blob_sha": CI_BLOB,
                "run_id": 123,
                "run_attempt": 1,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "head_sha": HEAD,
                "base_sha": BASE,
                "job_census_complete": True,
                "job_census": [
                    {
                        "job_id": 1001,
                        "name": "Test (default features)",
                        "status": "completed",
                        "conclusion": "success",
                        "skipped": False,
                    },
                    {
                        "job_id": 1002,
                        "name": "Clippy",
                        "status": "completed",
                        "conclusion": "success",
                        "skipped": False,
                    },
                ],
            },
        }

    def decision(self, observation=None, policy=None, manifest=None):
        policy = policy or self.policy
        manifest = manifest or self.manifest
        policy_bytes = canonical_json(policy)
        manifest_bytes = canonical_json(manifest)

        obs = copy.deepcopy(observation or self.observation)
        by_path = {row["path"]: row for row in obs["control_plane"]}
        if "scripts/ci/merge_admission_policy_v1.json" in by_path:
            policy_blob = git_blob_oid(policy_bytes, hex_length=40)
            by_path["scripts/ci/merge_admission_policy_v1.json"]["base_blob_sha"] = policy_blob
            by_path["scripts/ci/merge_admission_policy_v1.json"]["candidate_blob_sha"] = policy_blob
        if "scripts/ci/required_ci_job_manifest_v1.json" in by_path:
            manifest_blob = git_blob_oid(manifest_bytes, hex_length=40)
            by_path["scripts/ci/required_ci_job_manifest_v1.json"]["base_blob_sha"] = manifest_blob
            by_path["scripts/ci/required_ci_job_manifest_v1.json"]["candidate_blob_sha"] = manifest_blob

        result = evaluate(
            policy,
            manifest,
            obs,
            policy_bytes=policy_bytes,
            manifest_bytes=manifest_bytes,
        )
        self.assertEqual(result.receipt["decision"], result.decision.value)
        self.assertFalse(result.receipt["enforcement_ready"])
        self.assertEqual(len(result.receipt["receipt_sha256"]), 64)
        self.assertEqual(len(result.receipt["evidence_binding_sha256"]), 64)
        self.assertEqual(len(result.receipt["required_job_manifest_sha256"]), 64)
        return result

    def test_exact_current_full_success_is_admitted(self) -> None:
        result = self.decision()
        self.assertIs(result.decision, Decision.ADMITTED)
        satisfaction = result.receipt["evidence_binding"]["full_integration"][
            "manifest_satisfaction"
        ]
        self.assertEqual(satisfaction["required_job_count"], 2)

    def test_run_attempt_is_part_of_evidence_identity(self) -> None:
        first = self.decision()
        changed = copy.deepcopy(self.observation)
        changed["full_integration"]["run_attempt"] = 2
        second = self.decision(changed)
        self.assertIs(second.decision, Decision.ADMITTED)
        self.assertNotEqual(
            first.receipt["evidence_binding_sha256"],
            second.receipt["evidence_binding_sha256"],
        )

    def test_missing_run_attempt_is_refused(self) -> None:
        obs = copy.deepcopy(self.observation)
        del obs["full_integration"]["run_attempt"]
        with self.assertRaisesRegex(ValueError, "run_attempt"):
            self.decision(obs)

    def test_incomplete_job_census_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census_complete"] = False
        self.assertIs(self.decision(obs).decision, Decision.INCOMPLETE)

    def test_complete_census_missing_required_family_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"] = obs["full_integration"]["job_census"][:1]
        self.assertIs(self.decision(obs).decision, Decision.INCOMPLETE)

    def test_complete_census_with_unmanifested_job_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"].append(
            {
                "job_id": 1003,
                "name": "Unexpected Candidate-Owned Gate",
                "status": "completed",
                "conclusion": "success",
                "skipped": False,
            }
        )
        self.assertIs(self.decision(obs).decision, Decision.REJECTED)

    def test_failed_required_job_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1]["conclusion"] = "failure"
        self.assertIs(self.decision(obs).decision, Decision.REJECTED)

    def test_skipped_success_required_job_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1].update(
            conclusion="skipped", skipped=True
        )
        self.assertIs(self.decision(obs).decision, Decision.INCOMPLETE)

    def test_allowed_skip_family_can_skip(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["pull_request"]["families"][1][
            "required_disposition"
        ] = "allowed_skip"
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1].update(
            conclusion="skipped", skipped=True
        )
        self.assertIs(self.decision(obs, manifest=manifest).decision, Decision.ADMITTED)

    def test_incomplete_manifest_cannot_admit(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        self.assertIs(self.decision(manifest=manifest).decision, Decision.INCOMPLETE)

    def test_overlapping_manifest_families_are_rejected_at_evaluation(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["pull_request"]["families"][0]["api_name_regex"] = ".*"
        self.assertIs(self.decision(manifest=manifest).decision, Decision.REJECTED)

    def test_prior_head_success_is_stale(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["head_sha"] = "6" * 40
        self.assertIs(self.decision(obs).decision, Decision.STALE)

    def test_prior_base_success_is_stale(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["base_sha"] = "7" * 40
        self.assertIs(self.decision(obs).decision, Decision.STALE)

    def test_changed_workflow_requires_independent_bootstrap(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][0]["candidate_blob_sha"] = "8" * 40
        self.assertIs(self.decision(obs).decision, Decision.BOOTSTRAP_REQUIRED)

    def test_changed_policy_requires_independent_bootstrap(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][1]["candidate_blob_sha"] = "9" * 40
        result = evaluate(
            self.policy,
            self.manifest,
            obs,
            policy_bytes=self.policy_bytes,
            manifest_bytes=self.manifest_bytes,
        )
        self.assertIs(result.decision, Decision.BOOTSTRAP_REQUIRED)

    def test_changed_manifest_requires_independent_bootstrap(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][2]["candidate_blob_sha"] = "a" * 40
        result = evaluate(
            self.policy,
            self.manifest,
            obs,
            policy_bytes=self.policy_bytes,
            manifest_bytes=self.manifest_bytes,
        )
        self.assertIs(result.decision, Decision.BOOTSTRAP_REQUIRED)

    def test_policy_bytes_must_match_target_base_blob(self) -> None:
        obs = copy.deepcopy(self.observation)
        changed = copy.deepcopy(self.policy)
        changed["full_integration"]["accepted_events"] = ["pull_request"]
        result = evaluate(
            changed,
            self.manifest,
            obs,
            policy_bytes=canonical_json(changed),
            manifest_bytes=self.manifest_bytes,
        )
        self.assertIs(result.decision, Decision.REJECTED)

    def test_manifest_bytes_must_match_target_base_blob(self) -> None:
        obs = copy.deepcopy(self.observation)
        changed = copy.deepcopy(self.manifest)
        changed["profiles"]["pull_request"]["families"][0]["min_instances"] = 0
        result = evaluate(
            self.policy,
            changed,
            obs,
            policy_bytes=self.policy_bytes,
            manifest_bytes=canonical_json(changed),
        )
        self.assertIs(result.decision, Decision.REJECTED)

    def test_manifest_must_bind_trusted_workflow_blob(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["workflow_blob_sha"] = "a" * 40
        self.assertIs(self.decision(manifest=manifest).decision, Decision.REJECTED)

    def test_cancelled_run_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["conclusion"] = "cancelled"
        self.assertIs(self.decision(obs).decision, Decision.INCOMPLETE)

    def test_in_progress_run_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["status"] = "in_progress"
        obs["full_integration"]["conclusion"] = None
        self.assertIs(self.decision(obs).decision, Decision.INCOMPLETE)

    def test_missing_full_integration_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"] = None
        self.assertIs(self.decision(obs).decision, Decision.INCOMPLETE)

    def test_duplicate_job_id_is_refused(self) -> None:
        obs = copy.deepcopy(self.observation)
        duplicate = copy.deepcopy(obs["full_integration"]["job_census"][0])
        duplicate["name"] = "Different Name"
        obs["full_integration"]["job_census"].append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate job_id"):
            self.decision(obs)

    def test_wrong_repository_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["repository"] = "attacker/fork"
        self.assertIs(self.decision(obs).decision, Decision.REJECTED)

    def test_unknown_observation_field_is_refused(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["mystery_authority"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            self.decision(obs)

    def test_v1_cannot_claim_enforcement_ready(self) -> None:
        policy = copy.deepcopy(self.policy)
        policy["enforcement_ready"] = True
        with self.assertRaisesRegex(ValueError, "enforcement_ready=false"):
            self.decision(policy=policy)

    def test_receipt_binds_job_census(self) -> None:
        first = self.decision()
        changed = copy.deepcopy(self.observation)
        changed["full_integration"]["job_census"][0]["job_id"] = 2001
        second = self.decision(changed)
        self.assertIs(second.decision, Decision.ADMITTED)
        self.assertNotEqual(
            first.receipt["evidence_binding_sha256"],
            second.receipt["evidence_binding_sha256"],
        )

    def test_receipt_binds_manifest_content(self) -> None:
        first = self.decision()
        changed_manifest = copy.deepcopy(self.manifest)
        changed_manifest["profiles"]["workflow_dispatch"]["families"][0][
            "min_instances"
        ] = 0
        second = self.decision(manifest=changed_manifest)
        self.assertIs(second.decision, Decision.ADMITTED)
        self.assertNotEqual(
            first.receipt["required_job_manifest_sha256"],
            second.receipt["required_job_manifest_sha256"],
        )
        self.assertNotEqual(first.receipt["receipt_sha256"], second.receipt["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
