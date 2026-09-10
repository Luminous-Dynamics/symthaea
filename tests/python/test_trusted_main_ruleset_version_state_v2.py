#!/usr/bin/env python3
"""Regression tests for strict-time P0 ruleset version-state V2."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_provider_time as provider_time  # noqa: E402
import trusted_main_ruleset_version_state as v1  # noqa: E402
import trusted_main_ruleset_version_state_v2 as v2  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402
import test_trusted_main_ruleset_version_state as v1fx  # noqa: E402


class RulesetVersionStateV2Tests(unittest.TestCase):
    def test_exact_v1_state_with_valid_time_satisfies_v2(self):
        raw = v1fx.version(updated_at="2026-09-09T21:00:00Z")
        result = v2.verify_version_state_v2(raw, fx.policy())
        self.assertEqual(result["schema"], v2.SCHEMA)
        self.assertEqual(result["disposition"], "P0RulesetVersionStateSatisfied")
        self.assertEqual(result["provider_time_schema"], provider_time.SCHEMA)
        self.assertEqual(result["provider_order_authority"], "github-provider-valid-utc-instant-only")
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertEqual(result["scientific_authority"], "none")
        self.assertRegex(result["version_state_id"], r"^sha256:[0-9a-f]{64}$")

    def test_v1_identity_is_preserved_as_predecessor(self):
        raw = v1fx.version(updated_at="2026-09-09T21:00:00Z")
        predecessor = v1.verify_version_state(copy.deepcopy(raw), fx.policy())
        result = v2.verify_version_state_v2(raw, fx.policy())
        self.assertEqual(result["v1_version_state_id"], predecessor["version_state_id"])
        self.assertNotEqual(result["version_state_id"], predecessor["version_state_id"])

    def test_regex_shaped_but_calendar_invalid_time_rejects(self):
        for invalid in (
            "2026-99-09T21:00:00Z",
            "2026-09-31T21:00:00Z",
            "2026-09-09T25:00:00Z",
            "2026-09-09T21:61:00Z",
            "2026-09-09T21:00:61Z",
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(v2.RulesetVersionStateV2Error):
                    v2.verify_version_state_v2(v1fx.version(updated_at=invalid), fx.policy())

    def test_fractional_provider_time_has_exact_nanosecond_coordinate(self):
        raw = v1fx.version(updated_at="1970-01-01T00:00:00.123456789Z")
        result = v2.verify_version_state_v2(raw, fx.policy())
        self.assertEqual(result["provider_unix_nanos"], 123_456_789)

    def test_semantically_equal_fractional_instants_share_order_coordinate_but_not_record_identity(self):
        first = v2.verify_version_state_v2(
            v1fx.version(updated_at="2026-09-09T21:00:00.1Z"), fx.policy()
        )
        second = v2.verify_version_state_v2(
            v1fx.version(updated_at="2026-09-09T21:00:00.100000000Z"), fx.policy()
        )
        self.assertEqual(first["provider_unix_nanos"], second["provider_unix_nanos"])
        # The provider text is retained as observed evidence, so byte/content
        # identity remains distinct even when ordering coordinates are equal.
        self.assertNotEqual(first["v1_version_state_id"], second["v1_version_state_id"])
        self.assertNotEqual(first["version_state_id"], second["version_state_id"])

    def test_v2_identity_is_deterministic(self):
        raw = v1fx.version(updated_at="2026-09-09T21:00:00.000000001Z")
        first = v2.verify_version_state_v2(copy.deepcopy(raw), fx.policy())
        second = v2.verify_version_state_v2(copy.deepcopy(raw), fx.policy())
        self.assertEqual(first["version_state_id"], second["version_state_id"])

    def test_v1_semantic_rejection_still_rejects(self):
        changed = v1fx.version()
        changed["state"]["enforcement"] = "evaluate"
        with self.assertRaises(v1.RulesetVersionStateError):
            v2.verify_version_state_v2(changed, fx.policy())

    def test_timezone_offset_is_not_accepted_as_canonical_provider_time(self):
        # V1 rejects this syntax before V2. The stricter successor must not
        # broaden the accepted wire representation.
        with self.assertRaises(v1.RulesetVersionStateError):
            v2.verify_version_state_v2(
                v1fx.version(updated_at="2026-09-09T21:00:00+00:00"), fx.policy()
            )

    def test_provider_ordering_does_not_upgrade_chronology_authority(self):
        result = v2.verify_version_state_v2(v1fx.version(), fx.policy())
        self.assertEqual(result["provider_order_authority"], "github-provider-valid-utc-instant-only")
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")


if __name__ == "__main__":
    unittest.main()
