import copy
import unittest

from scripts.ci.merge_admission_manifest_v2 import git_blob_id
from scripts.ci.validate_required_ci_job_manifest_v3 import validate

WORKFLOW = b"""name: CI
jobs:
  test:
    name: Test
    runs-on: ubuntu-latest
    steps: []
"""


class RequiredManifestValidatorV3Tests(unittest.TestCase):
    def setUp(self):
        self.policy = {
            "schema": "symthaea.merge-admission-policy.v3",
            "repository": "Luminous-Dynamics/symthaea",
            "target_branch": "main",
            "policy_path": "scripts/ci/merge_admission_policy_v3.json",
            "enforcement_ready": False,
            "decision_default": "incomplete",
            "head_change_invalidates": True,
            "base_change_invalidates": True,
            "unknown_evidence_default": "reject",
            "control_plane": {
                "mode": "exact_base_equivalence",
                "paths": [
                    ".github/workflows/ci.yml",
                    "scripts/ci/required_ci_job_manifest_v1.json",
                    "scripts/ci/merge_admission_manifest_v2.py",
                    "scripts/ci/validate_required_ci_job_manifest_v3.py",
                    "scripts/ci/merge_admission_policy_v3.json",
                ],
                "candidate_changes_require_independent_bootstrap": True,
            },
            "full_integration": {
                "workflow_path": ".github/workflows/ci.yml",
                "required_job_manifest_path": "scripts/ci/required_ci_job_manifest_v1.json",
                "accepted_events": ["pull_request"],
                "required_status": "completed",
                "required_conclusion": "success",
                "require_exact_head": True,
                "require_exact_base": True,
                "require_complete_job_census": True,
            },
            "focused_evidence_can_substitute_for_full_integration": False,
            "tier1_can_substitute_for_full_integration": False,
        }
        self.manifest = {
            "schema": "symthaea.required-ci-job-manifest.v1",
            "workflow_path": ".github/workflows/ci.yml",
            "workflow_blob_sha": git_blob_id(WORKFLOW),
            "complete": True,
            "profiles": {
                "pull_request": {
                    "event": "pull_request",
                    "top_level_job_ids": ["test"],
                    "families": [
                        {
                            "job_id": "test",
                            "api_name_regex": r"^Test$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        }
                    ],
                }
            },
        }

    def test_complete_manifest_is_qualification_eligible(self):
        result = validate(WORKFLOW, self.policy, self.manifest)
        self.assertTrue(result["qualification_eligible"])
        self.assertTrue(result["profiles"]["pull_request"]["exact_workflow_job_census"])

    def test_incomplete_manifest_refuses_qualification(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate(WORKFLOW, self.policy, manifest)

    def test_incomplete_manifest_can_be_inspected_only_in_staging_mode(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        result = validate(WORKFLOW, self.policy, manifest, allow_incomplete=True)
        self.assertFalse(result["qualification_eligible"])


if __name__ == "__main__":
    unittest.main()
