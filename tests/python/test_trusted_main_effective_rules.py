#!/usr/bin/env python3
"""Regression tests for trusted-main effective-rule verification."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import trusted_main_effective_rules as effective  # noqa: E402
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


def ruleset() -> dict:
    return {
        "id": 4242,
        "name": "trusted-main-p0",
        "source_type": "Repository",
        "source": "Luminous-Dynamics/symthaea",
        "enforcement": "active",
    }


def projected_rule(rule_type: str, *, ruleset_id: int = 4242, source_type: str = "Repository", source: str = "Luminous-Dynamics/symthaea") -> dict:
    return {
        "type": rule_type,
        "ruleset_source_type": source_type,
        "ruleset_source": source,
        "ruleset_id": ruleset_id,
    }


def active_rules() -> list[dict]:
    return [
        projected_rule("deletion"),
        projected_rule("non_fast_forward"),
        projected_rule("pull_request"),
        # Unrelated inherited defense-in-depth rules must not contaminate the
        # repository-owned P0 projection.
        projected_rule(
            "commit_message_pattern",
            ruleset_id=7777,
            source_type="Organization",
            source="Luminous-Dynamics",
        ),
    ]


class EffectiveRulesTests(unittest.TestCase):
    def test_exact_projection_satisfies_p0(self):
        result = effective.verify_effective_rules(policy(), ruleset(), active_rules())
        self.assertEqual(result["disposition"], "P0EffectiveRulesSatisfied")
        self.assertEqual(result["violations"], [])
        self.assertEqual(
            result["observed_p0_rule_types"],
            ["deletion", "non_fast_forward", "pull_request"],
        )
        self.assertRegex(result["verification_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(result["negative_push_test"], "not-evaluated")

    def test_missing_each_required_rule_rejects(self):
        for missing in ("deletion", "non_fast_forward", "pull_request"):
            with self.subTest(missing=missing):
                rules = [r for r in active_rules() if not (r["ruleset_id"] == 4242 and r["type"] == missing)]
                result = effective.verify_effective_rules(policy(), ruleset(), rules)
                self.assertIn(f"EffectiveRuleCount:{missing}", result["violations"])

    def test_duplicate_required_rule_rejects(self):
        rules = active_rules() + [projected_rule("deletion")]
        result = effective.verify_effective_rules(policy(), ruleset(), rules)
        self.assertIn("EffectiveRuleCount:deletion", result["violations"])

    def test_extra_rule_from_p0_rejects_policy_drift(self):
        rules = active_rules() + [projected_rule("required_status_checks")]
        result = effective.verify_effective_rules(policy(), ruleset(), rules)
        self.assertIn("UnexpectedEffectiveRuleFromP0:required_status_checks", result["violations"])

    def test_same_named_org_rule_cannot_supply_repository_p0_projection(self):
        observed = ruleset()
        observed["source_type"] = "Organization"
        observed["source"] = "Luminous-Dynamics"
        result = effective.verify_effective_rules(policy(), observed, active_rules())
        self.assertIn("RulesetSourceTypeMismatch", result["violations"])
        self.assertIn("RulesetSourceMismatch", result["violations"])

    def test_effective_rule_source_must_match_repository(self):
        rules = active_rules()
        rules[0] = projected_rule("deletion", source_type="Organization", source="Luminous-Dynamics")
        result = effective.verify_effective_rules(policy(), ruleset(), rules)
        self.assertIn("EffectiveRuleSourceTypeMismatch:deletion", result["violations"])
        self.assertIn("EffectiveRuleSourceMismatch:deletion", result["violations"])

    def test_rules_from_other_rulesets_are_ignored_for_p0_count(self):
        rules = active_rules() + [projected_rule("deletion", ruleset_id=9999)]
        result = effective.verify_effective_rules(policy(), ruleset(), rules)
        self.assertEqual(result["disposition"], "P0EffectiveRulesSatisfied")

    def test_inactive_detailed_ruleset_rejects_even_if_projection_claims_rules(self):
        observed = ruleset()
        observed["enforcement"] = "disabled"
        result = effective.verify_effective_rules(policy(), observed, active_rules())
        self.assertIn("RulesetNotActive", result["violations"])

    def test_malformed_effective_rule_rejects(self):
        with self.assertRaisesRegex(effective.EffectiveRulesError, "object required"):
            effective.verify_effective_rules(policy(), ruleset(), ["not-an-object"])


if __name__ == "__main__":
    unittest.main()
