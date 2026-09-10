#!/usr/bin/env python3
"""Regression tests for exact P0 ruleset version-state verification."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_ruleset as p0  # noqa: E402
import trusted_main_ruleset_version_state as version_state  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402

RULESET_ID = 4242


def version(*, version_id: int = 7, updated_at: str = "2026-09-09T21:00:00Z") -> dict:
    rendered = p0.render_ruleset(fx.policy())
    return {
        "version_id": version_id,
        "actor": {"id": 215346314, "type": "User"},
        "updated_at": updated_at,
        "state": {
            "id": RULESET_ID,
            "name": rendered["name"],
            "target": rendered["target"],
            "source_type": "Repository",
            "source": fx.REPO,
            "enforcement": rendered["enforcement"],
            "bypass_actors": copy.deepcopy(rendered["bypass_actors"]),
            "conditions": copy.deepcopy(rendered["conditions"]),
            "rules": copy.deepcopy(rendered["rules"]),
        },
    }


class RulesetVersionStateTests(unittest.TestCase):
    def test_exact_reviewed_p0_version_is_narrowly_satisfied(self):
        result = version_state.verify_version_state(version(), fx.policy())
        self.assertEqual(result["disposition"], "P0RulesetVersionStateSatisfied")
        self.assertEqual(result["ruleset_id"], RULESET_ID)
        self.assertEqual(result["version_id"], 7)
        self.assertEqual(result["provider_order_authority"], "github-provider-observed-only")
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertRegex(result["version_state_id"], r"^sha256:[0-9a-f]{64}$")

    def test_identity_is_deterministic(self):
        first = version_state.verify_version_state(version(), fx.policy())
        second = version_state.verify_version_state(version(), fx.policy())
        self.assertEqual(first["version_state_id"], second["version_state_id"])

    def test_rule_and_merge_method_order_canonicalize(self):
        first = version_state.verify_version_state(version(), fx.policy())
        reordered = version()
        reordered["state"]["rules"].reverse()
        for rule in reordered["state"]["rules"]:
            if rule["type"] == "pull_request":
                rule["parameters"]["allowed_merge_methods"].reverse()
        second = version_state.verify_version_state(reordered, fx.policy())
        self.assertEqual(first["version_state_id"], second["version_state_id"])

    def test_version_id_changes_identity(self):
        first = version_state.verify_version_state(version(version_id=7), fx.policy())
        second = version_state.verify_version_state(version(version_id=8), fx.policy())
        self.assertNotEqual(first["version_state_id"], second["version_state_id"])

    def test_provider_actor_change_changes_identity(self):
        first = version_state.verify_version_state(version(), fx.policy())
        changed = version()
        changed["actor"]["id"] = 99
        second = version_state.verify_version_state(changed, fx.policy())
        self.assertNotEqual(first["version_state_id"], second["version_state_id"])

    def test_changed_bypass_policy_rejects_same_ruleset_id(self):
        changed = version()
        changed["state"]["bypass_actors"] = [
            {"actor_id": 123, "actor_type": "Team", "bypass_mode": "always"}
        ]
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "bypass_actors"):
            version_state.verify_version_state(changed, fx.policy())

    def test_changed_pr_approval_policy_rejects(self):
        changed = version()
        for rule in changed["state"]["rules"]:
            if rule["type"] == "pull_request":
                rule["parameters"]["required_approving_review_count"] = 1
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "rules"):
            version_state.verify_version_state(changed, fx.policy())

    def test_extra_rule_is_different_policy_lineage(self):
        changed = version()
        changed["state"]["rules"].append({"type": "required_signatures"})
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "unexpected P0 rule"):
            version_state.verify_version_state(changed, fx.policy())

    def test_organization_owned_state_cannot_substitute(self):
        changed = version()
        changed["state"]["source_type"] = "Organization"
        changed["state"]["source"] = "Luminous-Dynamics"
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "repository-owned"):
            version_state.verify_version_state(changed, fx.policy())

    def test_evaluate_mode_cannot_equal_active_p0(self):
        changed = version()
        changed["state"]["enforcement"] = "evaluate"
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "enforcement"):
            version_state.verify_version_state(changed, fx.policy())

    def test_target_exclusion_drift_rejects(self):
        changed = version()
        changed["state"]["conditions"]["ref_name"]["exclude"] = ["refs/heads/main"]
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "conditions"):
            version_state.verify_version_state(changed, fx.policy())

    def test_malformed_provider_timestamp_rejects(self):
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "updated_at"):
            version_state.verify_version_state(version(updated_at="yesterday"), fx.policy())

    def test_unknown_outer_version_field_rejects(self):
        changed = version()
        changed["future_semantics"] = True
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "closed"):
            version_state.verify_version_state(changed, fx.policy())

    def test_unknown_state_field_rejects(self):
        changed = version()
        changed["state"]["future_semantics"] = True
        with self.assertRaisesRegex(version_state.RulesetVersionStateError, "exact P0 semantic state schema"):
            version_state.verify_version_state(changed, fx.policy())

    def test_provider_timestamp_changes_identity_without_becoming_external_chronology(self):
        first = version_state.verify_version_state(version(updated_at="2026-09-09T21:00:00Z"), fx.policy())
        second = version_state.verify_version_state(version(updated_at="2026-09-09T21:00:01Z"), fx.policy())
        self.assertNotEqual(first["version_state_id"], second["version_state_id"])
        self.assertEqual(second["chronology_authority"], "not-externally-anchored")


if __name__ == "__main__":
    unittest.main()
