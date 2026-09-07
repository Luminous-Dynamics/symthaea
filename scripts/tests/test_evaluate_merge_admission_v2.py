import copy
import unittest

from scripts.ci.evaluate_merge_admission_v2 import (
    Decision,
    canonical_json,
    evaluate,
    git_blob_sha,
)

BASE = "1" * 40
HEAD = "2" * 40
TREE = "3" * 40
CI_BLOB = "4" * 40


class MergeAdmissionV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.manifest = {
            "schema": "symthaea.required-ci-job-manifest.v1",
            "workflow_path": ".github/workflows/ci.yml",
            "workflow_blob_sha": CI_BLOB,
            "complete": True,
            "profiles": {
                "pull_request": {
                    "event": "pull_request",
                    "top_level_job_ids": ["test", "clippy", "optional"],
                    "families": [
                        {
                            "job_id": "test",
                            "api_name_regex": "^Test \\(default features\\)$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "clippy",
                            "api_name_regex": "^Clippy$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "optional",
                            "api_name_regex": "^Optional gate$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "allowed_skip",
                        },
                    ],
                },
                "workflow_dispatch": {
                    "event": "workflow_dispatch",
                    "top_level_job_ids": ["test", "clippy", "optional"],
                    "families": [
                        {
                            "job_id": "test",
                            "api_name_regex": "^Test \\(default features\\)$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "clippy",
                            "api_name_regex": "^Clippy$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "optional",
                            "api_name_regex": "^Optional gate$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "allowed_skip",
                        },
                    ],
                },
            },
        }
        self.manifest_blob = git_blob_sha(canonical_json(self.manifest))
        self.policy = {
            "schema": "symthaea.merge-admission-policy.v2",
            "repository": "Luminous-Dynamics/symthaea",
            "target_branch": "main",
            "policy_path": "scripts/ci/merge_admission_policy_v2.json",
            "enforcement_ready": False,
            "decision_default": "incomplete",
            "head_change_invalidates": True,
            "base_change_invalidates": True,
            "unknown_evidence_default": "reject",
            "control_plane": {
                "mode": "exact_base_equivalence",
                "paths": [
                    ".github/workflows/ci.yml",
                    "scripts/ci/merge_admission_policy_v2.json",
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
        self.policy_blob = git_blob_sha(canonical_json(self.policy))
        self.observation = {
            "schema": "symthaea.merge-admission-observation.v2",
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
                    "path": "scripts/ci/merge_admission_policy_v2.json",
                    "base_blob_sha": self.policy_blob,
                    "candidate_blob_sha": self.policy_blob,
                },
                {
                    "path": "scripts/ci/required_ci_job_manifest_v1.json",
                    "base_blob_sha": self.manifest_blob,
                    "candidate_blob_sha": self.manifest_blob,
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
                    {
                        "job_id": 1003,
                        "name": "Optional gate",
                        "status": "completed",
                        "conclusion": "skipped",
                        "skipped": True,
                    },
                ],
            },
        }

    def result(self, observation=None, manifest=None):
        result = evaluate(
            self.policy,
            observation or self.observation,
            manifest or self.manifest,
        )
        self.assertEqual(result.receipt["decision"], result.decision.value)
        self.assertFalse(result.receipt["enforcement_ready"])
        self.assertEqual(len(result.receipt["receipt_sha256"]), 64)
        self.assertEqual(len(result.receipt["evidence_binding_sha256"]), 64)
        return result

    def decision(self, observation=None, manifest=None):
        return self.result(observation, manifest).decision

    def test_exact_current_full_success_is_admitted(self):
        self.assertIs(self.decision(), Decision.ADMITTED)

    def test_missing_full_integration_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"] = None
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_prior_head_success_is_stale(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["head_sha"] = "6" * 40
        self.assertIs(self.decision(obs), Decision.STALE)

    def test_prior_base_success_is_stale(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["base_sha"] = "7" * 40
        self.assertIs(self.decision(obs), Decision.STALE)

    def test_changed_control_plane_requires_bootstrap(self):
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][0]["candidate_blob_sha"] = "8" * 40
        self.assertIs(self.decision(obs), Decision.BOOTSTRAP_REQUIRED)

    def test_loaded_policy_must_equal_target_base_blob(self):
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][1]["base_blob_sha"] = "c" * 40
        obs["control_plane"][1]["candidate_blob_sha"] = "c" * 40
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_wrong_workflow_blob_is_rejected(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["workflow_blob_sha"] = "a" * 40
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_incomplete_census_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census_complete"] = False
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_manifest_not_complete_is_incomplete(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        blob = git_blob_sha(canonical_json(manifest))
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][2]["base_blob_sha"] = blob
        obs["control_plane"][2]["candidate_blob_sha"] = blob
        self.assertIs(self.decision(obs, manifest), Decision.INCOMPLETE)

    def test_loaded_manifest_must_equal_target_base_blob(self):
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][2]["base_blob_sha"] = "a" * 40
        obs["control_plane"][2]["candidate_blob_sha"] = "a" * 40
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_manifest_must_bind_trusted_workflow_blob(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["workflow_blob_sha"] = "b" * 40
        blob = git_blob_sha(canonical_json(manifest))
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][2]["base_blob_sha"] = blob
        obs["control_plane"][2]["candidate_blob_sha"] = blob
        self.assertIs(self.decision(obs, manifest), Decision.REJECTED)

    def test_missing_required_family_instance_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"] = [
            job
            for job in obs["full_integration"]["job_census"]
            if job["name"] != "Clippy"
        ]
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_unexpected_job_is_rejected(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"].append(
            {
                "job_id": 1999,
                "name": "Mystery Gate",
                "status": "completed",
                "conclusion": "success",
                "skipped": False,
            }
        )
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_duplicate_job_id_is_rejected(self):
        obs = copy.deepcopy(self.observation)
        extra = copy.deepcopy(obs["full_integration"]["job_census"][0])
        extra["name"] = "Clippy"
        obs["full_integration"]["job_census"].append(extra)
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_required_skip_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1].update(
            conclusion="skipped", skipped=True
        )
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_allowed_skip_is_admitted(self):
        self.assertIs(self.decision(), Decision.ADMITTED)

    def test_required_failure_is_rejected(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1]["conclusion"] = "failure"
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_cancelled_run_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["conclusion"] = "cancelled"
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_unknown_observation_field_is_refused(self):
        obs = copy.deepcopy(self.observation)
        obs["mystery_authority"] = True
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            evaluate(self.policy, obs, self.manifest)

    def test_run_attempt_is_part_of_evidence_identity(self):
        first = self.result()
        changed = copy.deepcopy(self.observation)
        changed["full_integration"]["run_attempt"] = 2
        second = self.result(changed)
        self.assertNotEqual(
            first.receipt["evidence_binding_sha256"],
            second.receipt["evidence_binding_sha256"],
        )
        self.assertNotEqual(first.receipt["receipt_sha256"], second.receipt["receipt_sha256"])

    def test_census_digest_is_order_independent(self):
        first = self.result()
        changed = copy.deepcopy(self.observation)
        changed["full_integration"]["job_census"].reverse()
        second = self.result(changed)
        self.assertEqual(
            first.receipt["evidence_binding"]["full_integration"]["job_census_sha256"],
            second.receipt["evidence_binding"]["full_integration"]["job_census_sha256"],
        )
        self.assertEqual(
            first.receipt["evidence_binding_sha256"],
            second.receipt["evidence_binding_sha256"],
        )

    def test_manifest_change_changes_receipt_identity(self):
        first = self.result()
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["workflow_dispatch"]["families"][0]["max_instances"] = 2
        blob = git_blob_sha(canonical_json(manifest))
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][2]["base_blob_sha"] = blob
        obs["control_plane"][2]["candidate_blob_sha"] = blob
        second = self.result(obs, manifest)
        self.assertNotEqual(
            first.receipt["evidence_binding_sha256"],
            second.receipt["evidence_binding_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
