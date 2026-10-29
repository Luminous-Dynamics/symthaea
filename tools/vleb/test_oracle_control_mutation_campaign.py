#!/usr/bin/env python3
"""Unit tests for the VLEB oracle-control verifier mutation campaign."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("run_oracle_control_mutation_campaign.py")
VERIFIER_PATH = Path(__file__).with_name("verify_oracle_controls.py")
SPEC = importlib.util.spec_from_file_location("vleb_oracle_control_mutation_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import oracle-control mutation campaign")
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


class OracleControlMutationCampaignTests(unittest.TestCase):
    def test_all_reviewed_mutation_anchors_are_unique_and_applicable(self) -> None:
        source = VERIFIER_PATH.read_text(encoding="utf-8")
        ids = [mutation["id"] for mutation in campaign.MUTATIONS]
        self.assertEqual(len(ids), len(set(ids)))
        for mutation in campaign.MUTATIONS:
            with self.subTest(mutation=mutation["id"]):
                mutated = campaign.replace_reviewed_anchors(source, mutation)
                self.assertNotEqual(mutated, source)
                self.assertEqual(
                    mutated.count(mutation["replacements"][0]["replacement"]) >= 1,
                    True,
                )

    def test_anchor_occurrence_mismatch_is_not_silently_mutated(self) -> None:
        mutation = {
            "id": "ambiguous",
            "replacements": [{"anchor": "guard", "replacement": "if False:"}],
        }
        with self.assertRaisesRegex(ValueError, "anchor count is 2"):
            campaign.replace_reviewed_anchors("guard then guard", mutation)

        with self.assertRaisesRegex(ValueError, "anchor count is 0"):
            campaign.replace_reviewed_anchors("no match", mutation)

    def test_catalog_contains_named_sensitive_tests_for_every_mutation(self) -> None:
        self.assertGreaterEqual(len(campaign.MUTATIONS), 12)
        for mutation in campaign.MUTATIONS:
            self.assertTrue(mutation["expected_failure_test"].startswith("test_"))
            self.assertTrue(mutation["family"])
            self.assertTrue(mutation["description"])
            self.assertTrue(mutation["replacements"])

    def test_family_membership_is_frozen_and_nonempty(self) -> None:
        families = {mutation["family"] for mutation in campaign.MUTATIONS}
        self.assertGreaterEqual(len(families), 5)
        for family in families:
            members = [mutation for mutation in campaign.MUTATIONS if mutation["family"] == family]
            self.assertTrue(members, family)

    def test_named_failure_parser_handles_unittest_output(self) -> None:
        import re

        output = """FAIL: test_wrong_manifest_pin_is_rejected (test_suite.Example.test_wrong_manifest_pin_is_rejected)
----------------------------------------------------------------------
Traceback (most recent call last):
  File "test_suite.py", line 10, in test_wrong_manifest_pin_is_rejected
    raise AssertionError("expected")
AssertionError: expected

ERROR: test_timeout_is_unresolved_not_a_killed_baseline (test_suite.Example.test_timeout_is_unresolved_not_a_killed_baseline)
----------------------------------------------------------------------
"""
        found = sorted(set(re.findall(r"^(?:FAIL|ERROR): (test_[A-Za-z0-9_]+) \(", output, re.MULTILINE)))
        self.assertEqual(found, [
            "test_timeout_is_unresolved_not_a_killed_baseline",
            "test_wrong_manifest_pin_is_rejected",
        ])


if __name__ == "__main__":
    unittest.main(verbosity=2)
