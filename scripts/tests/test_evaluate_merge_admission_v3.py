import copy
import unittest

from scripts.ci.evaluate_merge_admission_v3 import Decision, evaluate
from scripts.ci.merge_admission_manifest_v2 import canonical_json, git_blob_id

BASE = "1" * 40
HEAD = "2" * 40
TREE = "3" * 40

WORKFLOW = b"""name: CI
on: [pull_request, workflow_dispatch]
jobs:
  test:
    name: Test (default features)
    runs-on: ubuntu-latest
    steps: []
  clippy:
    name: Clippy
    runs-on: ubuntu-latest
    steps: []
  optional:
    name: Optional gate
    runs-on: ubuntu-latest
    steps: []
"""


class MergeAdmissionV3Tests(unittest.TestCase):
    def setUp(self) -> None:
        workflow_blob = git_blob_id(WORKFLOW)
        self.manifest = {
            "schema": "symthaea.required-ci-job-manifest.v1",
            "workflow_path": ".github/workflows/ci.yml",
            "workflow_blob_sha": workflow_blob,
            "complete": True,
            "profiles": {
                event: {
                    "event": event,
                    "top_level_job_ids": ["test", "clippy", "optional"],
                    "families": [
                        {
                            "job_id": "test",
                            "api_name_regex": r"^Test \(default features\)$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "clippy",
                            "api_name_regex": r"^Clippy$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "success",
                        },
                        {
                            "job_id": "optional",
                            "api_name_regex": r"^Optional gate$",
                            "min_instances": 1,
                            "max_instances": 1,
                            "required_disposition": "allowed_skip",
                        },
                    ],
                }
                for event in ("pull_request", "workflow_dispatch")
            },
        }
        manifest_raw = canonical_json(self.manifest)
        self.manifest_blob = git_blob_id(manifest_raw)
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
                    "scripts/ci/merge_admission_policy_v3.json",
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
                "require_complete_job_census": True,
            },
            "focused_evidence_can_substitute_for_full_integration": False,
            "tier1_can_substitute_for_full_integration": False,
        }
        policy_raw = canonical_json(self.policy)
        self.policy_blob = git_blob_id(policy_raw)
        self.observation = {
            "schema": "symthaea.merge-admission-observation.v3",
            "repository": "Luminous-Dynamics/symthaea",
            "target_branch": "main",
            "current_base_sha": BASE,
            "candidate_head_sha": HEAD,
            "candidate_tree_sha": TREE,
            "control_plane": [
                {"path": ".github/workflows/ci.yml", "base_blob_sha": workflow_blob, "candidate_blob_sha": workflow_blob},
                {"path": "scripts/ci/required_ci_job_manifest_v1.json", "base_blob_sha": self.manifest_blob, "candidate_blob_sha": self.manifest_blob},
                {"path": "scripts/ci/merge_admission_policy_v3.json", "base_blob_sha": self.policy_blob, "candidate_blob_sha": self.policy_blob},
            ],
            "full_integration": {
                "workflow_path": ".github/workflows/ci.yml",
                "workflow_blob_sha": workflow_blob,
                "run_id": 123,
                "run_attempt": 1,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "head_sha": HEAD,
                "base_sha": BASE,
                "job_census_proof": {
                    "reported_total_count": 3,
                    "pages_fetched": 1,
                    "terminal_page_observed": True,
                },
                "job_census": [
                    {"job_id": 1001, "name": "Test (default features)", "status": "completed", "conclusion": "success", "skipped": False},
                    {"job_id": 1002, "name": "Clippy", "status": "completed", "conclusion": "success", "skipped": False},
                    {"job_id": 1003, "name": "Optional gate", "status": "completed", "conclusion": "skipped", "skipped": True},
                ],
            },
        }

    def result(self, observation=None, manifest=None, workflow=WORKFLOW):
        return evaluate(self.policy, observation or self.observation, manifest or self.manifest, workflow)

    def decision(self, observation=None, manifest=None, workflow=WORKFLOW):
        result = self.result(observation, manifest, workflow)
        self.assertFalse(result.receipt["enforcement_ready"])
        self.assertEqual(result.receipt["decision"], result.decision.value)
        self.assertEqual(len(result.receipt["receipt_sha256"]), 64)
        return result.decision

    def test_exact_current_complete_success_is_admitted(self):
        self.assertIs(self.decision(), Decision.ADMITTED)

    def test_terminal_page_not_seen_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census_proof"]["terminal_page_observed"] = False
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_reported_count_contradiction_is_collector_invalid(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census_proof"]["reported_total_count"] = 4
        self.assertIs(self.decision(obs), Decision.COLLECTOR_INVALID)

    def test_duplicate_job_id_is_collector_invalid(self):
        obs = copy.deepcopy(self.observation)
        extra = copy.deepcopy(obs["full_integration"]["job_census"][0])
        extra["name"] = "Clippy"
        obs["full_integration"]["job_census"].append(extra)
        obs["full_integration"]["job_census_proof"]["reported_total_count"] = 4
        self.assertIs(self.decision(obs), Decision.COLLECTOR_INVALID)

    def test_skipped_flag_contradiction_is_collector_invalid(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][2]["skipped"] = False
        self.assertIs(self.decision(obs), Decision.COLLECTOR_INVALID)

    def test_incomplete_base_manifest_is_control_plane_invalid(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["complete"] = False
        manifest_raw = canonical_json(manifest)
        blob = git_blob_id(manifest_raw)
        obs = copy.deepcopy(self.observation)
        for row in obs["control_plane"]:
            if row["path"].endswith("required_ci_job_manifest_v1.json"):
                row["base_blob_sha"] = blob
                row["candidate_blob_sha"] = blob
        self.assertIs(self.decision(obs, manifest), Decision.CONTROL_PLANE_INVALID)

    def test_manifest_source_drift_is_control_plane_invalid(self):
        manifest = copy.deepcopy(self.manifest)
        manifest["profiles"]["pull_request"]["top_level_job_ids"].remove("optional")
        manifest["profiles"]["pull_request"]["families"] = [
            f for f in manifest["profiles"]["pull_request"]["families"] if f["job_id"] != "optional"
        ]
        blob = git_blob_id(canonical_json(manifest))
        obs = copy.deepcopy(self.observation)
        for row in obs["control_plane"]:
            if row["path"].endswith("required_ci_job_manifest_v1.json"):
                row["base_blob_sha"] = blob
                row["candidate_blob_sha"] = blob
        self.assertIs(self.decision(obs, manifest), Decision.CONTROL_PLANE_INVALID)

    def test_wrong_loaded_workflow_is_control_plane_invalid(self):
        self.assertIs(self.decision(workflow=WORKFLOW + b"\n# drift\n"), Decision.CONTROL_PLANE_INVALID)

    def test_candidate_control_plane_change_requires_bootstrap(self):
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][0]["candidate_blob_sha"] = "a" * 40
        self.assertIs(self.decision(obs), Decision.BOOTSTRAP_REQUIRED)

    def test_old_head_is_stale(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["head_sha"] = "6" * 40
        self.assertIs(self.decision(obs), Decision.STALE)

    def test_old_base_is_stale(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["base_sha"] = "7" * 40
        self.assertIs(self.decision(obs), Decision.STALE)

    def test_explicit_required_job_failure_is_rejected(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1]["conclusion"] = "failure"
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_pending_required_job_is_incomplete(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"][1].update(status="in_progress", conclusion=None)
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_unmanifested_runtime_job_is_control_plane_invalid(self):
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"].append(
            {"job_id": 1999, "name": "Mystery Gate", "status": "completed", "conclusion": "success", "skipped": False}
        )
        obs["full_integration"]["job_census_proof"]["reported_total_count"] = 4
        self.assertIs(self.decision(obs), Decision.CONTROL_PLANE_INVALID)

    def test_run_attempt_changes_evidence_identity(self):
        first = self.result()
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["run_attempt"] = 2
        second = self.result(obs)
        self.assertNotEqual(first.receipt["evidence_binding_sha256"], second.receipt["evidence_binding_sha256"])
        self.assertNotEqual(first.receipt["receipt_sha256"], second.receipt["receipt_sha256"])

    def test_census_order_does_not_change_identity(self):
        first = self.result()
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_census"].reverse()
        second = self.result(obs)
        self.assertEqual(first.receipt["evidence_binding_sha256"], second.receipt["evidence_binding_sha256"])

    def test_missing_control_plane_row_is_collector_invalid(self):
        obs = copy.deepcopy(self.observation)
        obs["control_plane"].pop()
        self.assertIs(self.decision(obs), Decision.COLLECTOR_INVALID)


if __name__ == "__main__":
    unittest.main()
