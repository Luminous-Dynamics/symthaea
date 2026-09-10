#!/usr/bin/env python3
"""Composition regressions for V3 authority semantics at the interval binder."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_enforcement_evidence_v3 as v3  # noqa: E402
import trusted_main_ruleset_version_binding as binding  # noqa: E402
import test_trusted_main_ruleset_version_binding as fx  # noqa: E402


def rehash(value: dict) -> dict:
    changed = copy.deepcopy(value)
    payload = dict(changed)
    payload.pop("evidence_id", None)
    changed["evidence_id"] = v3._content_id(v3.DOMAIN, payload)
    return changed


def rederive_selection(value: dict) -> dict:
    changed = copy.deepcopy(value)
    payload = {
        "schema": v3.SELECTION_SCHEMA,
        "structural_verification_id": changed["structural_verification_id"],
        "effective_rules_verification_id": changed["effective_rules_verification_id"],
        "root_subject_id": changed["root_subject_id"],
        "rule_suite_ids": changed["selected_rule_suite_ids"],
        "rule_suite_observation_ids": changed["selected_rule_suite_observation_ids"],
        "actor_id": changed["selected_actor_id"],
    }
    changed["trusted_enforcement_selection_id"] = v3._content_id(v3.SELECTION_DOMAIN, payload)
    return rehash(changed)


class RulesetVersionBindingAuthorityTests(unittest.TestCase):
    def test_rehashed_v3_chronology_inflation_cannot_enter_binding(self):
        enforcement, observations = fx.enforcement_bundle()
        enforcement["chronology_authority"] = "externally-anchored"
        enforcement = rehash(enforcement)
        with self.assertRaisesRegex(
            binding.RulesetVersionBindingError,
            "enforcement_v3: revalidation failed:.*chronology_authority",
        ):
            fx.derive(enforcement=enforcement, observations=observations)

    def test_outer_rehash_cannot_hide_v3_selection_identity_tamper(self):
        enforcement, observations = fx.enforcement_bundle()
        enforcement["trusted_enforcement_selection_id"] = "sha256:" + "0" * 64
        enforcement = rehash(enforcement)
        with self.assertRaisesRegex(
            binding.RulesetVersionBindingError,
            "enforcement_v3: revalidation failed:.*trusted selection identity mismatch",
        ):
            fx.derive(enforcement=enforcement, observations=observations)

    def test_rederived_duplicate_attempt_selection_cannot_enter_binding(self):
        enforcement, observations = fx.enforcement_bundle()
        enforcement["selected_rule_suite_ids"]["force_push"] = 101
        enforcement = rederive_selection(enforcement)
        with self.assertRaisesRegex(
            binding.RulesetVersionBindingError,
            "enforcement_v3: revalidation failed:.*IDs must be distinct",
        ):
            fx.derive(enforcement=enforcement, observations=observations)

    def test_bypass_assurance_must_match_selected_policy(self):
        enforcement, observations = fx.enforcement_bundle()
        enforcement["bypass_assurance"] = "not-fully-established"
        enforcement = rehash(enforcement)
        # This label is a valid V3 ceiling in the abstract, but not for the
        # selected no-bypass P0 policy. Composition must reject the mismatch.
        self.assertEqual(v3.validate_enforcement_evidence_v3(enforcement), enforcement)
        with self.assertRaisesRegex(
            binding.RulesetVersionBindingError,
            "bypass assurance is inconsistent with selected P0 policy",
        ):
            fx.derive(enforcement=enforcement, observations=observations)


if __name__ == "__main__":
    unittest.main()
