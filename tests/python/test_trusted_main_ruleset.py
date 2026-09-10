#!/usr/bin/env python3
"""Regression tests for the trusted-main P0 ruleset renderer/verifier."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trusted_main_ruleset as p0  # noqa: E402


def policy() -> dict:
    return {
        "schema": p0.POLICY_SCHEMA,
        "repository": "Luminous-Dynamics/symthaea",
        "repository_id": 1136141775,
        "target_ref": "refs/heads/main",
        "ruleset_name": "trusted-main-p0",
        "required_enforcement": "active",
        "allowed_bypass_actors": [],
        "required_rules": ["deletion", "non_fast_forward", "pull_request"],
        "pull_request_policy": {
            "required_approving_review_count_min": 0,
            "required_review_thread_resolution": True,
            "require_code_owner_review": False,
            "require_last_push_approval": False,
            "dismiss_stale_reviews_on_push": False,
            "allowed_merge_methods": ["merge", "rebase", "squash"],
        },
        "p0_required_status_checks": [],
        "non_claims": [
            "does not activate a trusted self-hosted runner",
            "does not prove a negative direct-push test has been performed",
            "does not prove bypass identities are globally impossible",
            "does not prove organization or enterprise rules are absent",
            "does not prove the ruleset is applied on GitHub",
            "does not qualify any source code or scientific result",
            "does not replace later P1 qualification admission policy",
        ],
    }


def repository() -> dict:
    return {
        "id": 1136141775,
        "full_name": "Luminous-Dynamics/symthaea",
        "default_branch": "main",
    }


def branch() -> dict:
    return {"name": "main", "protected": True}


def ruleset() -> dict:
    return {
        "id": 4242,
        "name": "trusted-main-p0",
        "target": "branch",
        "source_type": "Repository",
        "source": "Luminous-Dynamics/symthaea",
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {
            "ref_name": {
                "include": ["refs/heads/main"],
                "exclude": [],
            }
        },
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {
                "type": "pull_request",
                "parameters": {
                    "allowed_merge_methods": ["merge", "rebase", "squash"],
                    "dismiss_stale_reviews_on_push": False,
                    "require_code_owner_review": False,
                    "require_last_push_approval": False,
                    "required_approving_review_count": 0,
                    "required_review_thread_resolution": True,
                },
            },
        ],
    }


def verify(*, p=None, r=None, b=None, repo=None):
    return p0.verify_readback(
        p or policy(),
        r or ruleset(),
        b or branch(),
        repo or repository(),
    )


class PolicyTests(unittest.TestCase):
    def test_policy_has_content_addressed_identity(self):
        pid = p0.policy_id(policy())
        self.assertRegex(pid, r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(pid, p0.policy_id(copy.deepcopy(policy())))

    def test_renderer_is_exact_p0_not_ci_admission(self):
        rendered = p0.render_ruleset(policy())
        self.assertEqual(rendered["enforcement"], "active")
        self.assertEqual(rendered["conditions"]["ref_name"]["include"], ["refs/heads/main"])
        self.assertEqual(rendered["conditions"]["ref_name"]["exclude"], [])
        self.assertEqual(rendered["bypass_actors"], [])
        self.assertEqual(
            [rule["type"] for rule in rendered["rules"]],
            ["deletion", "non_fast_forward", "pull_request"],
        )
        self.assertNotIn("required_status_checks", [r["type"] for r in rendered["rules"]])

    def test_policy_non_claims_are_canonical(self):
        broken = policy()
        broken["non_claims"] = list(reversed(broken["non_claims"]))
        with self.assertRaisesRegex(p0.PolicyError, "sorted and unique"):
            p0.normalize_policy(broken)

    def test_p0_rejects_status_checks_in_policy_manifest(self):
        broken = policy()
        broken["p0_required_status_checks"] = ["CI"]
        with self.assertRaisesRegex(p0.PolicyError, "intentionally requires none"):
            p0.normalize_policy(broken)


class ReadbackTests(unittest.TestCase):
    def test_exact_readback_satisfies_structural_p0(self):
        result = verify()
        self.assertEqual(result["disposition"], "P0StructurallySatisfied")
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["ruleset_id"], 4242)
        self.assertEqual(result["ruleset_source_type"], "Repository")
        self.assertEqual(result["ruleset_source"], "Luminous-Dynamics/symthaea")
        self.assertEqual(result["negative_push_test"], "not-evaluated")
        self.assertEqual(result["organization_rules"], "not-enumerated")
        self.assertRegex(result["verification_id"], r"^sha256:[0-9a-f]{64}$")

    def test_default_branch_token_is_acceptable_when_default_is_main(self):
        observed = ruleset()
        observed["conditions"]["ref_name"]["include"] = ["~DEFAULT_BRANCH"]
        self.assertEqual(verify(r=observed)["disposition"], "P0StructurallySatisfied")

    def test_current_unprotected_shape_rejects(self):
        observed_branch = {"name": "main", "protected": False}
        observed_ruleset = {
            "id": 4242,
            "name": "trusted-main-p0",
            "target": "branch",
            "source_type": "Repository",
            "source": "Luminous-Dynamics/symthaea",
            "enforcement": "disabled",
            "bypass_actors": [],
            "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
            "rules": [],
        }
        result = verify(r=observed_ruleset, b=observed_branch)
        self.assertEqual(result["disposition"], "P0Rejected")
        self.assertIn("BranchNotReportedProtected", result["violations"])
        self.assertIn("RulesetNotActive", result["violations"])
        self.assertIn("RequiredRuleCount:pull_request", result["violations"])

    def test_missing_ruleset_id_rejects(self):
        observed = ruleset()
        del observed["id"]
        self.assertIn("RulesetIdMissing", verify(r=observed)["violations"])

    def test_repository_owned_ruleset_source_is_required(self):
        observed = ruleset()
        del observed["source_type"]
        del observed["source"]
        violations = verify(r=observed)["violations"]
        self.assertIn("RulesetSourceTypeMismatch", violations)
        self.assertIn("RulesetSourceMismatch", violations)

    def test_same_named_organization_ruleset_cannot_satisfy_repository_p0(self):
        observed = ruleset()
        observed["source_type"] = "Organization"
        observed["source"] = "Luminous-Dynamics"
        result = verify(r=observed)
        self.assertEqual(result["disposition"], "P0Rejected")
        self.assertIn("RulesetSourceTypeMismatch", result["violations"])
        self.assertIn("RulesetSourceMismatch", result["violations"])

    def test_wrong_repository_ruleset_source_rejects(self):
        observed = ruleset()
        observed["source"] = "Luminous-Dynamics/other-repo"
        self.assertIn("RulesetSourceMismatch", verify(r=observed)["violations"])

    def test_hidden_bypass_visibility_is_not_empty_bypass(self):
        observed = ruleset()
        del observed["bypass_actors"]
        self.assertIn("BypassVisibilityUnavailable", verify(r=observed)["violations"])

    def test_unexpected_bypass_actor_rejects(self):
        observed = ruleset()
        observed["bypass_actors"] = [
            {"actor_id": 1, "actor_type": "RepositoryRole", "bypass_mode": "always"}
        ]
        self.assertIn("UnexpectedBypassActor", verify(r=observed)["violations"])

    def test_missing_each_required_rule_rejects(self):
        for rule_type in ("deletion", "non_fast_forward", "pull_request"):
            with self.subTest(rule_type=rule_type):
                observed = ruleset()
                observed["rules"] = [r for r in observed["rules"] if r["type"] != rule_type]
                self.assertIn(
                    f"RequiredRuleCount:{rule_type}",
                    verify(r=observed)["violations"],
                )

    def test_duplicate_required_rule_rejects(self):
        observed = ruleset()
        observed["rules"].append({"type": "deletion"})
        self.assertIn("RequiredRuleCount:deletion", verify(r=observed)["violations"])

    def test_wrong_ref_scope_rejects(self):
        observed = ruleset()
        observed["conditions"]["ref_name"]["include"] = ["refs/heads/develop"]
        self.assertIn("TargetRefNotIncluded", verify(r=observed)["violations"])

    def test_any_ref_exclusion_rejects_p0(self):
        for excluded in ("refs/heads/main", "refs/heads/*", "~DEFAULT_BRANCH"):
            with self.subTest(excluded=excluded):
                observed = ruleset()
                observed["conditions"]["ref_name"]["exclude"] = [excluded]
                self.assertIn(
                    "TargetRefExclusionPolicyDrift",
                    verify(r=observed)["violations"],
                )

    def test_required_status_checks_are_p0_policy_drift(self):
        observed = ruleset()
        observed["rules"].append({
            "type": "required_status_checks",
            "parameters": {"required_status_checks": [{"context": "CI"}]},
        })
        self.assertIn("P0StatusCheckPolicyDrift", verify(r=observed)["violations"])

    def test_required_workflow_is_p0_policy_drift(self):
        observed = ruleset()
        observed["rules"].append({"type": "workflows", "parameters": {"workflows": []}})
        self.assertIn("P0RequiredWorkflowPolicyDrift", verify(r=observed)["violations"])

    def test_stricter_review_count_is_accepted(self):
        observed = ruleset()
        observed["rules"][2]["parameters"]["required_approving_review_count"] = 1
        self.assertEqual(verify(r=observed)["disposition"], "P0StructurallySatisfied")

    def test_unresolved_threads_policy_is_required(self):
        observed = ruleset()
        observed["rules"][2]["parameters"]["required_review_thread_resolution"] = False
        self.assertIn(
            "PullRequestPolicyTooWeak:required_review_thread_resolution",
            verify(r=observed)["violations"],
        )

    def test_unknown_merge_method_rejects(self):
        observed = ruleset()
        observed["rules"][2]["parameters"]["allowed_merge_methods"] = ["merge", "octopus"]
        self.assertIn("UnexpectedMergeMethod", verify(r=observed)["violations"])

    def test_repository_identity_mismatch_rejects(self):
        observed_repo = repository()
        observed_repo["id"] += 1
        observed_repo["full_name"] = "SomebodyElse/symthaea"
        violations = verify(repo=observed_repo)["violations"]
        self.assertIn("RepositoryIdMismatch", violations)
        self.assertIn("RepositoryNameMismatch", violations)

    def test_default_branch_mismatch_rejects_default_token_semantics(self):
        observed_repo = repository()
        observed_repo["default_branch"] = "develop"
        observed = ruleset()
        observed["conditions"]["ref_name"]["include"] = ["~DEFAULT_BRANCH"]
        self.assertIn("DefaultBranchMismatch", verify(r=observed, repo=observed_repo)["violations"])


if __name__ == "__main__":
    unittest.main()
