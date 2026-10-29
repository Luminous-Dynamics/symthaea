#!/usr/bin/env python3
"""Unit tests for the shift-event auditor mutation campaign catalog."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("run_shift_event_mutation_campaign.py")
AUDITOR_PATH = Path(__file__).with_name("audit_shift_event_evidence.py")
SPEC = importlib.util.spec_from_file_location("vleb_shift_event_mutation_campaign", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import shift-event mutation campaign")
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


class ShiftEventMutationCampaignTests(unittest.TestCase):
    def test_every_reviewed_anchor_is_unique_and_actionable(self) -> None:
        source = AUDITOR_PATH.read_text(encoding="utf-8")
        mutation_ids = [row["id"] for row in campaign.MUTATIONS]
        self.assertEqual(len(mutation_ids), len(set(mutation_ids)))
        for mutation in campaign.MUTATIONS:
            with self.subTest(mutation=mutation["id"]):
                changed = campaign.replace_reviewed_anchors(source, mutation)
                self.assertNotEqual(changed, source)

    def test_ambiguous_and_missing_anchors_are_rejected(self) -> None:
        mutation = {"id": "demo", "replacements": [{"anchor": "guard", "replacement": "if False:"}]}
        with self.assertRaisesRegex(ValueError, "anchor count is 2"):
            campaign.replace_reviewed_anchors("guard then guard", mutation)
        with self.assertRaisesRegex(ValueError, "anchor count is 0"):
            campaign.replace_reviewed_anchors("no anchor", mutation)

    def test_each_mutation_has_a_named_regression_target(self) -> None:
        source = (Path(__file__).with_name("test_audit_shift_event_evidence.py")).read_text(encoding="utf-8")
        self.assertGreaterEqual(len(campaign.MUTATIONS), 12)
        for mutation in campaign.MUTATIONS:
            with self.subTest(mutation=mutation["id"]):
                self.assertIn(f"def {mutation['expected_failure_test']}(", source)
                self.assertTrue(mutation["family"])
                self.assertTrue(mutation["description"])

    def test_guardrail_families_are_frozen_and_nonempty(self) -> None:
        families = {row["family"] for row in campaign.MUTATIONS}
        self.assertGreaterEqual(len(families), 5)
        for family in families:
            self.assertTrue(any(row["family"] == family for row in campaign.MUTATIONS))


if __name__ == "__main__":
    unittest.main(verbosity=2)
