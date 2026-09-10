#!/usr/bin/env python3
"""Regression for distinct normalized observation identities in enforcement V3."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_enforcement_evidence_v3 as v3  # noqa: E402
import test_trusted_main_enforcement_evidence_v3 as fx  # noqa: E402


class EnforcementEvidenceV3DistinctObservationTests(unittest.TestCase):
    def test_rederived_selection_cannot_reuse_one_observation_for_two_operations(self):
        changed = fx.derive()
        changed["selected_rule_suite_observation_ids"]["force_push"] = (
            changed["selected_rule_suite_observation_ids"]["deletion"]
        )
        changed = fx.rederive_selection(changed)
        with self.assertRaisesRegex(
            v3.EnforcementEvidenceV3Error,
            "selected_rule_suite_observation_ids: IDs must be distinct",
        ):
            v3.validate_enforcement_evidence_v3(changed)


if __name__ == "__main__":
    unittest.main()
