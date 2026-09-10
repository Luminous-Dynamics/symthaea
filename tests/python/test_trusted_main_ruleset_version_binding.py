#!/usr/bin/env python3
"""Regression tests for exact closed-interval P0 ruleset-version binding."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_enforcement_evidence_v2 as enforcement_v2  # noqa: E402
import trusted_main_enforcement_evidence_v3 as enforcement_v3  # noqa: E402
import trusted_main_ruleset as p0  # noqa: E402
import trusted_main_ruleset_history as history  # noqa: E402
import trusted_main_ruleset_version_binding as binding  # noqa: E402
import trusted_main_ruleset_version_state_v2 as version_state_v2  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402
import test_trusted_main_enforcement_evidence_v3 as v3fx  # noqa: E402
import test_trusted_main_ruleset_version_state as state_fx  # noqa: E402

RULESET_ID = 4242
ACTOR_ID = 215346314


def history_item(version_id: int, updated_at: str, *, actor_id: int = ACTOR_ID) -> dict:
    return {
        "version_id": version_id,
        "actor": {"id": actor_id, "type": "User"},
        "updated_at": updated_at,
    }


def verified_history(specs: list[tuple[int, str, int]] | None = None) -> dict:
    if specs is None:
        specs = [
            (6, "2026-09-09T20:00:00Z", ACTOR_ID),
            (7, "2026-09-09T20:30:00Z", ACTOR_ID),
            (8, "2026-09-09T22:00:00Z", ACTOR_ID),
        ]
    # Present provider page newest-first to ensure the verifier's chronology is
    # derived from strict instants, not accidental fixture list order.
    items = [history_item(v, t, actor_id=a) for v, t, a in reversed(specs)]
    pages = [{"page": 1, "items": items}, {"page": 2, "items": []}]
    capture = {
        "schema": history.CAPTURE_SCHEMA,
        "repository": fx.REPO,
        "repository_id": fx.REPO_ID,
        "ruleset_id": RULESET_ID,
        "per_page": 100,
        "first_pass": copy.deepcopy(pages),
        "second_pass": copy.deepcopy(pages),
        "observation_basis": "github-ruleset-history-double-read",
    }
    return history.verify_history_capture(capture, fx.policy(), expected_ruleset_id=RULESET_ID)


def exact_state(*, version_id: int = 7, updated_at: str = "2026-09-09T20:30:00Z") -> dict:
    return version_state_v2.verify_version_state_v2(
        state_fx.version(version_id=version_id, updated_at=updated_at),
        fx.policy(),
    )


def enforcement_bundle(*, direct=None, force=None, deletion=None) -> tuple[dict, dict]:
    direct = fx.direct_suite() if direct is None else direct
    force = fx.force_suite() if force is None else force
    deletion = fx.deletion_suite() if deletion is None else deletion
    base = enforcement_v2.derive_enforcement_evidence(
        policy=fx.policy(),
        structural_verification=fx.structural(),
        effective_rules_verification=fx.effective(),
        root_subject=fx.root_subject(),
        direct_update_rule_suite=direct,
        force_push_rule_suite=force,
        deletion_rule_suite=deletion,
    )
    if base["disposition"] != "EnforcementBehaviorallyCorroborated":
        raise AssertionError("binding fixture requires V2 positive enforcement observations")
    ids = {
        operation: enforcement_v3.rule_suite_observation_id(observation)
        for operation, observation in base["operation_observations"].items()
    }
    kwargs = v3fx.expected_kwargs()
    kwargs.update({
        "expected_direct_update_observation_id": ids["ordinary_direct_update"],
        "expected_force_push_observation_id": ids["force_push"],
        "expected_deletion_observation_id": ids["deletion"],
    })
    enforced = enforcement_v3.derive_enforcement_evidence_v3(
        policy=fx.policy(),
        structural_verification=fx.structural(),
        effective_rules_verification=fx.effective(),
        root_subject=fx.root_subject(),
        direct_update_rule_suite=direct,
        force_push_rule_suite=force,
        deletion_rule_suite=deletion,
        **kwargs,
    )
    selected = {
        "schema": binding.OBSERVATIONS_SCHEMA,
        "observations": copy.deepcopy(base["operation_observations"]),
    }
    return enforced, selected


def derive(*, h=None, state=None, enforcement=None, observations=None, **selectors) -> dict:
    h = verified_history() if h is None else h
    state = exact_state() if state is None else state
    if enforcement is None or observations is None:
        default_enforcement, default_observations = enforcement_bundle()
        enforcement = default_enforcement if enforcement is None else enforcement
        observations = default_observations if observations is None else observations
    values = {
        "policy": fx.policy(),
        "history_verification": h,
        "selected_version_state": state,
        "enforcement_evidence_v3": enforcement,
        "selected_observations": observations,
        "expected_history_id": h["history_id"],
        "expected_version_state_id": state["version_state_id"],
        "expected_enforcement_evidence_id": enforcement["evidence_id"],
    }
    values.update(selectors)
    return binding.derive_version_binding(**values)


class RulesetVersionBindingTests(unittest.TestCase):
    def test_all_selected_attempts_inside_exact_closed_p0_interval_succeed_narrowly(self):
        result = derive()
        self.assertEqual(result["schema"], binding.SCHEMA)
        self.assertEqual(result["disposition"], "P0RulesetVersionBindingOnly")
        self.assertEqual(result["selected_version_id"], 7)
        self.assertEqual(result["selected_interval"]["predecessor_version_id"], 6)
        self.assertEqual(result["selected_interval"]["successor_version_id"], 8)
        self.assertEqual(
            result["selected_interval"]["boundary_assurance"],
            "closed-predecessor-and-successor-version-boundaries",
        )
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")
        self.assertEqual(result["capture_authentication"], "none")
        self.assertEqual(result["operator_authentication"], "none")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertRegex(result["binding_id"], r"^sha256:[0-9a-f]{64}$")
        self.assertEqual(
            {entry["resolved_version_id"] for entry in result["attempt_resolutions"].values()},
            {7},
        )

    def test_binding_identity_is_deterministic(self):
        self.assertEqual(derive()["binding_id"], derive()["binding_id"])

    def test_selected_version_start_equality_is_inclusive(self):
        enforced, selected = enforcement_bundle(
            direct=fx.direct_suite(pushed_at="2026-09-09T20:30:00Z")
        )
        result = derive(enforcement=enforced, observations=selected)
        self.assertEqual(
            result["attempt_resolutions"]["ordinary_direct_update"]["resolved_version_id"], 7
        )

    def test_successor_timestamp_equality_resolves_to_successor_and_rejects(self):
        enforced, selected = enforcement_bundle(
            force=fx.force_suite(pushed_at="2026-09-09T22:00:00Z")
        )
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "force_push.*version 8"):
            derive(enforcement=enforced, observations=selected)

    def test_attempt_before_selected_start_resolves_to_predecessor_and_rejects(self):
        enforced, selected = enforcement_bundle(
            direct=fx.direct_suite(pushed_at="2026-09-09T20:15:00Z")
        )
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "ordinary_direct_update.*version 6"):
            derive(enforcement=enforced, observations=selected)

    def test_missing_predecessor_boundary_fails_closed(self):
        h = verified_history([
            (7, "2026-09-09T20:30:00Z", ACTOR_ID),
            (8, "2026-09-09T22:00:00Z", ACTOR_ID),
        ])
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "no predecessor boundary"):
            derive(h=h)

    def test_missing_successor_boundary_fails_closed(self):
        h = verified_history([
            (6, "2026-09-09T20:00:00Z", ACTOR_ID),
            (7, "2026-09-09T20:30:00Z", ACTOR_ID),
        ])
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "no successor boundary"):
            derive(h=h)

    def test_change_and_change_back_cannot_masquerade_as_continuous_old_version(self):
        h = verified_history([
            (6, "2026-09-09T20:00:00Z", ACTOR_ID),
            (7, "2026-09-09T20:30:00Z", ACTOR_ID),
            (8, "2026-09-09T21:00:00Z", ACTOR_ID),
            (9, "2026-09-09T21:30:00Z", ACTOR_ID),
            (10, "2026-09-09T23:00:00Z", ACTOR_ID),
        ])
        pushed = "2026-09-09T21:45:00Z"
        enforced, selected = enforcement_bundle(
            direct=fx.direct_suite(pushed_at=pushed),
            force=fx.force_suite(pushed_at=pushed),
            deletion=fx.deletion_suite(pushed_at=pushed),
        )
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "version 9, not selected P0 version 7"):
            derive(h=h, enforcement=enforced, observations=selected)

    def test_selected_version_must_exist_in_complete_history(self):
        h = verified_history([
            (6, "2026-09-09T20:00:00Z", ACTOR_ID),
            (8, "2026-09-09T22:00:00Z", ACTOR_ID),
            (9, "2026-09-09T23:00:00Z", ACTOR_ID),
        ])
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "absent or non-unique"):
            derive(h=h)

    def test_history_summary_actor_must_match_exact_version_state(self):
        h = verified_history([
            (6, "2026-09-09T20:00:00Z", ACTOR_ID),
            (7, "2026-09-09T20:30:00Z", 99),
            (8, "2026-09-09T22:00:00Z", ACTOR_ID),
        ])
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "provider_actor_id"):
            derive(h=h)

    def test_selected_observation_bytes_cannot_drift_from_v3_identity(self):
        enforced, selected = enforcement_bundle()
        selected["observations"]["force_push"]["pushed_at"] = "2026-09-09T20:46:00Z"
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "force_push.*V3-selected"):
            derive(enforcement=enforced, observations=selected)

    def test_regex_shaped_but_calendar_invalid_suite_time_rejects_at_binding_layer(self):
        bad_time = "2026-99-99T25:61:61Z"
        enforced, selected = enforcement_bundle(
            direct=fx.direct_suite(pushed_at=bad_time)
        )
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "invalid UTC calendar instant"):
            derive(enforcement=enforced, observations=selected)

    def test_history_selector_is_independent(self):
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "history_verification.*independently selected"):
            derive(expected_history_id="sha256:" + "0" * 64)

    def test_version_state_selector_is_independent(self):
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "version_state_v2.*independently selected"):
            derive(expected_version_state_id="sha256:" + "0" * 64)

    def test_enforcement_selector_is_independent(self):
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "enforcement_v3.*independently selected"):
            derive(expected_enforcement_evidence_id="sha256:" + "0" * 64)

    def test_changed_selected_version_changes_binding_identity(self):
        first = derive()
        h2 = verified_history([
            (7, "2026-09-09T20:00:00Z", ACTOR_ID),
            (8, "2026-09-09T20:30:00Z", ACTOR_ID),
            (9, "2026-09-09T22:00:00Z", ACTOR_ID),
        ])
        state2 = exact_state(version_id=8, updated_at="2026-09-09T20:30:00Z")
        second = derive(h=h2, state=state2)
        self.assertNotEqual(first["selected_version_state_id"], second["selected_version_state_id"])
        self.assertNotEqual(first["binding_id"], second["binding_id"])

    def test_provider_order_never_upgrades_to_external_chronology(self):
        result = derive()
        self.assertEqual(result["provider_order_authority"], "github-provider-valid-utc-instants-only")
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")
        self.assertNotIn("ChronologySatisfied", result.values())

    def test_recomputed_history_tamper_still_fails_independent_selection(self):
        h = verified_history()
        changed = copy.deepcopy(h)
        changed["versions"][0]["provider_actor_id"] = 99
        payload = dict(changed)
        del payload["history_id"]
        changed["history_id"] = history._content_id(payload)
        with self.assertRaisesRegex(binding.RulesetVersionBindingError, "history_verification.*independently selected"):
            derive(h=changed, expected_history_id=h["history_id"])


if __name__ == "__main__":
    unittest.main()
