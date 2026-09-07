import copy
import unittest

from scripts.ci.evaluate_merge_admission import Decision, evaluate


BASE = "1" * 40
HEAD = "2" * 40
TREE = "3" * 40
CI_BLOB = "4" * 40
POLICY_BLOB = "5" * 40


class MergeAdmissionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = {
            "schema": "symthaea.merge-admission-policy.v1",
            "repository": "Luminous-Dynamics/symthaea",
            "target_branch": "main",
            "decision_default": "incomplete",
            "head_change_invalidates": True,
            "base_change_invalidates": True,
            "unknown_evidence_default": "reject",
            "control_plane": {
                "mode": "exact_base_equivalence",
                "paths": [
                    ".github/workflows/ci.yml",
                    "scripts/ci/merge_admission_policy_v1.json",
                ],
                "candidate_changes_require_independent_bootstrap": True,
            },
            "full_integration": {
                "workflow_path": ".github/workflows/ci.yml",
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
                    "base_blob_sha": POLICY_BLOB,
                    "candidate_blob_sha": POLICY_BLOB,
                },
            ],
            "full_integration": {
                "workflow_path": ".github/workflows/ci.yml",
                "workflow_blob_sha": CI_BLOB,
                "run_id": 123,
                "event": "pull_request",
                "status": "completed",
                "conclusion": "success",
                "head_sha": HEAD,
                "base_sha": BASE,
                "job_set_complete": True,
                "required_jobs": [
                    {
                        "name": "Test (default features)",
                        "status": "completed",
                        "conclusion": "success",
                        "skipped": False,
                    },
                    {
                        "name": "Clippy",
                        "status": "completed",
                        "conclusion": "success",
                        "skipped": False,
                    },
                ],
            },
        }

    def decision(self, observation=None):
        result = evaluate(self.policy, observation or self.observation)
        self.assertEqual(result.receipt["decision"], result.decision.value)
        self.assertEqual(len(result.receipt["receipt_sha256"]), 64)
        self.assertEqual(len(result.receipt["evidence_binding_sha256"]), 64)
        self.assertIn("unsigned policy-core disposition", result.receipt["caveat"])
        return result.decision

    def test_exact_current_full_success_is_admitted(self) -> None:
        self.assertIs(self.decision(), Decision.ADMITTED)

    def test_missing_full_integration_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"] = None
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_tier1_or_focused_cannot_substitute_for_full(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"] = None
        obs["tier1_green"] = True
        obs["focused_evidence_green"] = True
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_prior_head_success_is_stale(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["head_sha"] = "6" * 40
        self.assertIs(self.decision(obs), Decision.STALE)

    def test_prior_base_success_is_stale(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["base_sha"] = "7" * 40
        self.assertIs(self.decision(obs), Decision.STALE)

    def test_changed_workflow_requires_independent_bootstrap(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][0]["candidate_blob_sha"] = "8" * 40
        self.assertIs(self.decision(obs), Decision.BOOTSTRAP_REQUIRED)

    def test_changed_policy_requires_independent_bootstrap(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"][1]["candidate_blob_sha"] = "9" * 40
        self.assertIs(self.decision(obs), Decision.BOOTSTRAP_REQUIRED)

    def test_similarly_named_but_wrong_workflow_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["workflow_path"] = ".github/workflows/ci-copy.yml"
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_wrong_workflow_blob_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["workflow_blob_sha"] = "a" * 40
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_success_with_skipped_required_job_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["required_jobs"][1].update(
            status="completed", conclusion="skipped", skipped=True
        )
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_failed_required_job_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["required_jobs"][1]["conclusion"] = "failure"
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_cancelled_run_is_incomplete_not_failure_evidence(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["conclusion"] = "cancelled"
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_in_progress_run_is_incomplete(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["status"] = "in_progress"
        obs["full_integration"]["conclusion"] = None
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_collector_must_establish_complete_required_job_set(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["job_set_complete"] = False
        self.assertIs(self.decision(obs), Decision.INCOMPLETE)

    def test_duplicate_required_job_observation_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["full_integration"]["required_jobs"].append(
            copy.deepcopy(obs["full_integration"]["required_jobs"][0])
        )
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_missing_control_plane_observation_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"].pop()
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_duplicate_control_plane_observation_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["control_plane"].append(copy.deepcopy(obs["control_plane"][0]))
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_wrong_repository_is_rejected(self) -> None:
        obs = copy.deepcopy(self.observation)
        obs["repository"] = "attacker/fork"
        self.assertIs(self.decision(obs), Decision.REJECTED)

    def test_policy_digest_changes_when_policy_changes(self) -> None:
        first = evaluate(self.policy, self.observation)
        changed = copy.deepcopy(self.policy)
        changed["full_integration"]["accepted_events"] = ["pull_request"]
        second = evaluate(changed, self.observation)
        self.assertNotEqual(first.receipt["policy_sha256"], second.receipt["policy_sha256"])
        self.assertNotEqual(first.receipt["receipt_sha256"], second.receipt["receipt_sha256"])

    def test_receipt_binds_exact_workflow_run_identity(self) -> None:
        first = evaluate(self.policy, self.observation)
        changed = copy.deepcopy(self.observation)
        changed["full_integration"]["run_id"] = 124
        second = evaluate(self.policy, changed)
        self.assertIs(first.decision, Decision.ADMITTED)
        self.assertIs(second.decision, Decision.ADMITTED)
        self.assertNotEqual(first.receipt["evidence_binding_sha256"], second.receipt["evidence_binding_sha256"])
        self.assertNotEqual(first.receipt["receipt_sha256"], second.receipt["receipt_sha256"])

    def test_receipt_binds_required_job_observation_set(self) -> None:
        first = evaluate(self.policy, self.observation)
        changed = copy.deepcopy(self.observation)
        changed["full_integration"]["required_jobs"].append(
            {
                "name": "Integration Tests",
                "status": "completed",
                "conclusion": "success",
                "skipped": False,
            }
        )
        second = evaluate(self.policy, changed)
        self.assertIs(first.decision, Decision.ADMITTED)
        self.assertIs(second.decision, Decision.ADMITTED)
        self.assertNotEqual(first.receipt["evidence_binding_sha256"], second.receipt["evidence_binding_sha256"])
        self.assertNotEqual(first.receipt["receipt_sha256"], second.receipt["receipt_sha256"])


if __name__ == "__main__":
    unittest.main()
