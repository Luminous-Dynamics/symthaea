import copy
import unittest

from scripts.ci.evaluate_merge_admission_v2 import git_blob_sha
from scripts.ci.validate_required_ci_job_manifest import validate


class RequiredCiJobManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = b"""name: CI
jobs:
  fmt:
    runs-on: ubuntu-latest
    steps: []
  test:
    runs-on: ubuntu-latest
    steps: []
"""
        self.workflow_blob = git_blob_sha(self.workflow)
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
        families = [
            {
                "job_id": "fmt",
                "api_name_regex": "^Format Check$",
                "min_instances": 1,
                "max_instances": 1,
                "required_disposition": "success",
            },
            {
                "job_id": "test",
                "api_name_regex": "^Test$",
                "min_instances": 1,
                "max_instances": 1,
                "required_disposition": "success",
            },
        ]
        self.manifest = {
            "schema": "symthaea.required-ci-job-manifest.v1",
            "workflow_path": ".github/workflows/ci.yml",
            "workflow_blob_sha": self.workflow_blob,
            "complete": True,
            "profiles": {
                "pull_request": {
                    "event": "pull_request",
                    "top_level_job_ids": ["fmt", "test"],
                    "families": copy.deepcopy(families),
                },
                "workflow_dispatch": {
                    "event": "workflow_dispatch",
                    "top_level_job_ids": ["fmt", "test"],
                    "families": copy.deepcopy(families),
                },
            },
        }

    def test_complete_exact_manifest_is_qualification_eligible(self) -> None:
        result = validate(self.workflow, self.policy, self.manifest)
        self.assertTrue(result["manifest_complete"])
        self.assertTrue(result["qualification_eligible"])
        self.assertEqual(result["workflow_job_count"], 2)
        self.assertTrue(result["profiles"]["pull_request"]["exact_workflow_job_census"])

    def test_incomplete_manifest_fails_by_default(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        manifest["profiles"]["pull_request"]["top_level_job_ids"] = []
        manifest["profiles"]["pull_request"]["families"] = []
        with self.assertRaisesRegex(ValueError, "intentionally incomplete"):
            validate(self.workflow, self.policy, manifest)

    def test_incomplete_manifest_can_be_audited_but_not_qualified(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        manifest["profiles"]["pull_request"]["top_level_job_ids"] = []
        manifest["profiles"]["pull_request"]["families"] = []
        result = validate(
            self.workflow,
            self.policy,
            manifest,
            allow_incomplete=True,
        )
        self.assertFalse(result["qualification_eligible"])
        self.assertFalse(result["profiles"]["pull_request"]["exact_workflow_job_census"])

    def test_complete_manifest_missing_workflow_job_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["pull_request"]["top_level_job_ids"] = ["fmt"]
        manifest["profiles"]["pull_request"]["families"] = [
            manifest["profiles"]["pull_request"]["families"][0]
        ]
        with self.assertRaisesRegex(ValueError, "does not census the exact workflow"):
            validate(self.workflow, self.policy, manifest)

    def test_workflow_blob_drift_fails(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        manifest["workflow_blob_sha"] = "a" * 40
        with self.assertRaisesRegex(ValueError, "workflow blob drift"):
            validate(self.workflow, self.policy, manifest)

    def test_event_profile_set_must_match_policy(self) -> None:
        manifest = copy.deepcopy(self.manifest)
        del manifest["profiles"]["workflow_dispatch"]
        with self.assertRaisesRegex(ValueError, "event profiles do not match policy"):
            validate(self.workflow, self.policy, manifest)


if __name__ == "__main__":
    unittest.main()
