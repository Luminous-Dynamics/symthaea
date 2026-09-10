#!/usr/bin/env python3
"""Regression tests for GitHub-rule-suite-derived P0 enforcement evidence."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trusted_main_effective_rules as effective_rules  # noqa: E402
import trusted_main_enforcement_evidence_v2 as enforcement  # noqa: E402
import trusted_main_ruleset as p0  # noqa: E402

REPO = "Luminous-Dynamics/symthaea"
REPO_SHORT = "symthaea"
REPO_ID = 1136141775
ROOT_SHA = "1" * 40
ROOT_TREE = "2" * 40
RULESET_ID = 4242


def policy() -> dict:
    return {
        "schema": p0.POLICY_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
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


def ruleset_readback() -> dict:
    return {
        "id": RULESET_ID,
        "name": "trusted-main-p0",
        "target": "branch",
        "source_type": "Repository",
        "source": REPO,
        "enforcement": "active",
        "bypass_actors": [],
        "conditions": {"ref_name": {"include": ["refs/heads/main"], "exclude": []}},
        "rules": [
            {"type": "deletion"},
            {"type": "non_fast_forward"},
            {"type": "pull_request", "parameters": {
                "allowed_merge_methods": ["merge", "rebase", "squash"],
                "dismiss_stale_reviews_on_push": False,
                "require_code_owner_review": False,
                "require_last_push_approval": False,
                "required_approving_review_count": 0,
                "required_review_thread_resolution": True,
            }},
        ],
    }


def structural() -> dict:
    return p0.verify_readback(
        policy(),
        ruleset_readback(),
        {"name": "main", "protected": True},
        {"id": REPO_ID, "full_name": REPO, "default_branch": "main"},
    )


def effective() -> dict:
    rules = [
        {
            "type": rule_type,
            "ruleset_source_type": "Repository",
            "ruleset_source": REPO,
            "ruleset_id": RULESET_ID,
        }
        for rule_type in ("deletion", "non_fast_forward", "pull_request")
    ]
    return effective_rules.verify_effective_rules(policy(), ruleset_readback(), rules)


def root_subject() -> dict:
    return {
        "schema": enforcement.ROOT_SUBJECT_SCHEMA,
        "repository": REPO,
        "repository_id": REPO_ID,
        "target_ref": "refs/heads/main",
        "root_sha": ROOT_SHA,
        "root_tree": ROOT_TREE,
        "observation_basis": "github-commit-readback",
    }


def rule_suite(rule_type: str, *, suite_id: int, after_digit: str, **overrides) -> dict:
    result = {
        "id": suite_id,
        "actor_id": 215346314,
        "actor_name": "Tristan-Stoltz-ERC",
        "before_sha": ROOT_SHA,
        "after_sha": after_digit * 40,
        "ref": "refs/heads/main",
        "repository_id": REPO_ID,
        "repository_name": REPO_SHORT,
        "pushed_at": "2026-09-09T20:45:00Z",
        "result": "fail",
        "evaluation_result": "fail",
        "rule_evaluations": [
            {
                "rule_source": {
                    "type": "ruleset",
                    "id": RULESET_ID,
                    "name": "trusted-main-p0",
                },
                "enforcement": "active",
                "result": "fail",
                "rule_type": rule_type,
                "details": "blocked by trusted-main-p0",
            }
        ],
    }
    result.update(overrides)
    return result


def direct_suite(**overrides) -> dict:
    return rule_suite("pull_request", suite_id=101, after_digit="3", **overrides)


def force_suite(**overrides) -> dict:
    return rule_suite("non_fast_forward", suite_id=102, after_digit="4", **overrides)


def deletion_suite(**overrides) -> dict:
    return rule_suite("deletion", suite_id=103, after_digit="0", **overrides)


def derive(*, direct=None, force=None, deletion=None, s=None, e=None):
    return enforcement.derive_enforcement_evidence(
        policy=policy(),
        structural_verification=structural() if s is None else s,
        effective_rules_verification=effective() if e is None else e,
        root_subject=root_subject(),
        direct_update_rule_suite=direct,
        force_push_rule_suite=force,
        deletion_rule_suite=deletion,
    )


class EnforcementEvidenceTests(unittest.TestCase):
    def test_no_behavioral_suites_is_configured_only_not_positive(self):
        result = derive()
        self.assertEqual(result["disposition"], "EnforcementConfiguredOnly")
        self.assertEqual(result["violations"], [])
        self.assertEqual(
            result["missing_operations"],
            ["deletion", "force_push", "ordinary_direct_update"],
        )
        self.assertEqual(result["bypass_assurance"], "not-fully-established")

    def test_one_valid_suite_is_partial_not_positive(self):
        result = derive(direct=direct_suite())
        self.assertEqual(result["disposition"], "EnforcementBehavioralEvidencePartial")
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["missing_operations"], ["deletion", "force_push"])

    def test_three_distinct_exact_rule_failures_are_behaviorally_corroborated(self):
        result = derive(direct=direct_suite(), force=force_suite(), deletion=deletion_suite())
        self.assertEqual(result["disposition"], "EnforcementBehaviorallyCorroborated")
        self.assertEqual(result["violations"], [])
        self.assertEqual(result["missing_operations"], [])
        self.assertEqual(
            result["bypass_assurance"],
            "no-configured-p0-bypass-and-distinct-non-bypass-failures-observed",
        )
        self.assertRegex(result["evidence_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertEqual(result["current_admission"], "not-evaluated")

    def test_suite_pass_is_rejected_not_missing(self):
        result = derive(
            direct=direct_suite(result="pass"),
            force=force_suite(),
            deletion=deletion_suite(),
        )
        self.assertEqual(result["disposition"], "EnforcementEvidenceRejected")
        self.assertIn("RuleSuiteDidNotFail:ordinary_direct_update", result["violations"])

    def test_bypass_is_rejected_as_behavioral_enforcement(self):
        result = derive(
            direct=direct_suite(result="bypass"),
            force=force_suite(),
            deletion=deletion_suite(),
        )
        self.assertEqual(result["disposition"], "EnforcementEvidenceRejected")
        self.assertIn("RuleSuiteDidNotFail:ordinary_direct_update", result["violations"])

    def test_wrong_rule_type_cannot_stand_in_for_force_push(self):
        wrong = force_suite()
        wrong["rule_evaluations"][0]["rule_type"] = "pull_request"
        result = derive(direct=direct_suite(), force=wrong, deletion=deletion_suite())
        self.assertEqual(result["disposition"], "EnforcementEvidenceRejected")
        self.assertIn("ExpectedRuleEvaluationCount:force_push", result["violations"])

    def test_wrong_ruleset_source_cannot_supply_rule_evaluation(self):
        wrong = direct_suite()
        wrong["rule_evaluations"][0]["rule_source"]["id"] = 9999
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("ExpectedRuleEvaluationCount:ordinary_direct_update", result["violations"])

    def test_inactive_rule_is_rejected(self):
        wrong = direct_suite()
        wrong["rule_evaluations"][0]["enforcement"] = "evaluate"
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("RuleNotActive:ordinary_direct_update", result["violations"])

    def test_rule_result_must_be_fail(self):
        wrong = direct_suite()
        wrong["rule_evaluations"][0]["result"] = "pass"
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("RuleDidNotReject:ordinary_direct_update", result["violations"])

    def test_all_operations_bind_same_exact_root_before_sha(self):
        wrong = force_suite(before_sha="8" * 40)
        result = derive(direct=direct_suite(), force=wrong, deletion=deletion_suite())
        self.assertIn("RootBeforeShaMismatch:force_push", result["violations"])

    def test_duplicate_rule_suite_cannot_inflate_multiple_operations(self):
        force = force_suite()
        force["id"] = 101
        result = derive(direct=direct_suite(), force=force, deletion=deletion_suite())
        self.assertIn("DuplicateRuleSuiteAcrossOperations", result["violations"])
        self.assertEqual(result["disposition"], "EnforcementEvidenceRejected")

    def test_repository_numeric_identity_is_authoritative(self):
        wrong = direct_suite(repository_id=999)
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("RepositoryIdMismatch:ordinary_direct_update", result["violations"])

    def test_repository_short_name_shape_matches_github_rule_suite(self):
        wrong = direct_suite(repository_name=REPO)
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("RepositoryNameMismatch:ordinary_direct_update", result["violations"])

    def test_actor_identity_is_required_for_behavioral_corroboration(self):
        wrong = direct_suite(actor_id=None)
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("ActorIdentityMissing:ordinary_direct_update", result["violations"])

    def test_provider_timestamp_is_required_but_not_chronology_authority(self):
        wrong = direct_suite(pushed_at="not-a-time")
        result = derive(direct=wrong, force=force_suite(), deletion=deletion_suite())
        self.assertIn("ChronologyMalformed:ordinary_direct_update", result["violations"])
        self.assertEqual(result["chronology_authority"], "provider-timestamp-observed-only")

    def test_forged_structural_verification_id_rejects_before_behavior(self):
        s = structural()
        s["verification_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(enforcement.EnforcementEvidenceError, "structural.verification_id: content ID mismatch"):
            derive(direct=direct_suite(), force=force_suite(), deletion=deletion_suite(), s=s)

    def test_forged_effective_verification_id_rejects_before_behavior(self):
        e = effective()
        e["verification_id"] = "sha256:" + "0" * 64
        with self.assertRaisesRegex(enforcement.EnforcementEvidenceError, "effective.verification_id: content ID mismatch"):
            derive(direct=direct_suite(), force=force_suite(), deletion=deletion_suite(), e=e)

    def test_evidence_identity_is_deterministic(self):
        first = derive(direct=direct_suite(), force=force_suite(), deletion=deletion_suite())
        second = derive(direct=direct_suite(), force=force_suite(), deletion=deletion_suite())
        self.assertEqual(first["evidence_id"], second["evidence_id"])


if __name__ == "__main__":
    unittest.main()
