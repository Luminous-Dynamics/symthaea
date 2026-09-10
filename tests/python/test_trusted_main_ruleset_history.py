#!/usr/bin/env python3
"""Regression tests for stable complete trusted-main ruleset history readback."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "python"))

import trusted_main_ruleset_history as history  # noqa: E402
import test_trusted_main_enforcement_evidence_v2 as fx  # noqa: E402

RULESET_ID = 4242
ACTOR_ID = 215346314


def item(version_id: int, updated_at: str, *, actor_id: int = ACTOR_ID) -> dict:
    return {
        "version_id": version_id,
        "actor": {"id": actor_id, "type": "User"},
        "updated_at": updated_at,
    }


def pages(*, middle_time: str = "2026-09-09T21:00:00Z") -> list[dict]:
    # GitHub examples return newest first; the verifier must not depend on that
    # response order for chronology semantics.
    return [
        {
            "page": 1,
            "items": [
                item(8, "2026-09-09T22:00:00Z"),
                item(7, middle_time),
                item(6, "2026-09-09T20:00:00Z"),
            ],
        },
        {"page": 2, "items": []},
    ]


def capture() -> dict:
    return {
        "schema": history.CAPTURE_SCHEMA,
        "repository": fx.REPO,
        "repository_id": fx.REPO_ID,
        "ruleset_id": RULESET_ID,
        "per_page": 100,
        "first_pass": pages(),
        "second_pass": pages(),
        "observation_basis": "github-ruleset-history-double-read",
    }


def verify(value: dict | None = None) -> dict:
    return history.verify_history_capture(
        capture() if value is None else value,
        fx.policy(),
        expected_ruleset_id=RULESET_ID,
    )


class RulesetHistoryTests(unittest.TestCase):
    def test_two_complete_matching_passes_are_narrowly_positive(self):
        result = verify()
        self.assertEqual(result["schema"], history.SCHEMA)
        self.assertEqual(result["disposition"], "RulesetHistoryStableCompleteReadback")
        self.assertEqual(result["version_count"], 3)
        self.assertEqual([entry["version_id"] for entry in result["versions"]], [6, 7, 8])
        self.assertEqual(
            result["completeness_basis"],
            "two-full-passes-each-ending-in-explicit-empty-page",
        )
        self.assertEqual(result["capture_authentication"], "none")
        self.assertEqual(result["chronology_authority"], "not-externally-anchored")
        self.assertEqual(result["current_admission"], "not-evaluated")
        self.assertRegex(result["history_id"], r"^sha256:[0-9a-f]{64}$")

    def test_history_identity_is_deterministic(self):
        self.assertEqual(verify()["history_id"], verify()["history_id"])

    def test_missing_explicit_empty_terminal_page_rejects(self):
        changed = capture()
        changed["first_pass"] = changed["first_pass"][:1]
        with self.assertRaisesRegex(history.RulesetHistoryError, "explicit empty terminal"):
            verify(changed)

    def test_empty_page_before_terminal_rejects(self):
        changed = capture()
        changed["first_pass"] = [
            changed["first_pass"][0],
            {"page": 2, "items": []},
            {"page": 3, "items": [item(5, "2026-09-09T19:00:00Z")]},
            {"page": 4, "items": []},
        ]
        with self.assertRaisesRegex(history.RulesetHistoryError, "empty page before terminal"):
            verify(changed)

    def test_noncontiguous_page_numbers_reject(self):
        changed = capture()
        changed["second_pass"][1]["page"] = 3
        with self.assertRaisesRegex(history.RulesetHistoryError, "contiguous pagination"):
            verify(changed)

    def test_duplicate_version_id_rejects(self):
        changed = capture()
        changed["first_pass"][0]["items"][0]["version_id"] = 7
        with self.assertRaisesRegex(history.RulesetHistoryError, "duplicate version_id"):
            verify(changed)

    def test_equal_provider_instants_are_ambiguous(self):
        changed = capture()
        changed["first_pass"][0]["items"][0]["updated_at"] = "2026-09-09T21:00:00Z"
        with self.assertRaisesRegex(history.RulesetHistoryError, "interval ordering is ambiguous"):
            verify(changed)

    def test_calendar_invalid_provider_time_rejects(self):
        changed = capture()
        changed["first_pass"][0]["items"][1]["updated_at"] = "2026-99-99T25:61:61Z"
        with self.assertRaisesRegex(history.RulesetHistoryError, "invalid UTC calendar instant"):
            verify(changed)

    def test_second_pass_content_drift_rejects(self):
        changed = capture()
        changed["second_pass"][0]["items"][1]["actor"]["id"] = 99
        with self.assertRaisesRegex(history.RulesetHistoryError, "changed or pagination drifted"):
            verify(changed)

    def test_second_pass_version_insertion_rejects(self):
        changed = capture()
        changed["second_pass"][0]["items"].insert(
            0, item(9, "2026-09-09T23:00:00Z")
        )
        with self.assertRaisesRegex(history.RulesetHistoryError, "changed or pagination drifted"):
            verify(changed)

    def test_repository_identity_mismatch_rejects(self):
        changed = capture()
        changed["repository_id"] = 999
        with self.assertRaisesRegex(history.RulesetHistoryError, "repository identity mismatch"):
            verify(changed)

    def test_ruleset_selector_is_independent(self):
        with self.assertRaisesRegex(history.RulesetHistoryError, "ruleset ID mismatch"):
            history.verify_history_capture(capture(), fx.policy(), expected_ruleset_id=999)

    def test_derived_history_revalidation_detects_tamper(self):
        result = verify()
        changed = copy.deepcopy(result)
        changed["versions"][1]["provider_actor_id"] = 99
        with self.assertRaisesRegex(history.RulesetHistoryError, "content identity mismatch"):
            history.validate_verified_history(changed)

    def test_derived_time_coordinate_tamper_rejects_before_identity(self):
        result = verify()
        changed = copy.deepcopy(result)
        changed["versions"][1]["provider_unix_nanos"] += 1
        with self.assertRaisesRegex(history.RulesetHistoryError, "time coordinate mismatch"):
            history.validate_verified_history(changed)


if __name__ == "__main__":
    unittest.main()
