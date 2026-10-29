#!/usr/bin/env python3
"""Regression tests for the research-only bitemporal reference resolver."""
from __future__ import annotations

from copy import deepcopy
from datetime import date
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bitemporal_reference as br  # noqa: E402


def load_fixtures():
    event_rows, _ = br.load_jsonl(HERE / "fixtures" / "events.jsonl")
    query_rows, _ = br.load_jsonl(HERE / "fixtures" / "queries.jsonl")
    return br.validate_events(event_rows), br.validate_queries(query_rows)


class BitemporalReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events, cls.queries = load_fixtures()
        cls.query_by_id = {query["query_id"]: query for query in cls.queries}

    def result(self, query_id, events=None):
        return br.resolve_query(
            self.events if events is None else events,
            self.query_by_id[query_id],
        )

    def test_all_annotated_fixture_cases_match(self):
        report = br.evaluate(self.events, self.queries)
        self.assertEqual(report["query_count"], 11)
        self.assertEqual(report["correct_count"], 11)
        self.assertEqual(report["exact_accuracy"], 1.0)

    def test_known_then_does_not_leak_later_correction(self):
        result = self.result("Q004")
        self.assertEqual(result["state"], "resolved")
        self.assertEqual(result["value"], "db-primary-a.internal")
        self.assertNotIn("E003", result["eligible_event_ids"])

    def test_corrected_history_uses_late_arriving_event(self):
        result = self.result("Q003")
        self.assertEqual(result["value"], "db-primary-c.internal")
        self.assertIn("E003", result["selected_event_ids"])
        self.assertIn("E001", result["shadowed_event_ids"])

    def test_future_valid_change_is_unknown_until_recorded(self):
        result = self.result("Q001")
        self.assertEqual(result["value"], "db-primary-a.internal")
        self.assertNotIn("E002", result["eligible_event_ids"])

    def test_recorded_at_cutoff_is_inclusive(self):
        query = deepcopy(self.query_by_id["Q011"])
        query["known_at"] = "2026-02-10T10:00:00Z"
        query["_known_at"] = br.parse_timestamp(query["known_at"], "test cutoff")
        result = br.resolve_query(self.events, query)
        self.assertEqual(result["value"], "db-primary-b.internal")

    def test_valid_time_end_is_exclusive(self):
        result = self.result("Q005")
        self.assertEqual(result["value"], "db-primary-b.internal")
        self.assertNotIn("E003", result["eligible_event_ids"])

    def test_authority_breaks_tie_at_same_effective_start(self):
        result = self.result("Q006")
        self.assertEqual(result["value"], "closed")
        self.assertEqual(result["selected_authority_rank"], 3)
        self.assertEqual(result["selected_event_ids"], ["E005"])
        self.assertIn("E006", result["shadowed_event_ids"])

    def test_authority_event_is_not_visible_before_recorded(self):
        result = self.result("Q007")
        self.assertEqual(result["value"], "open")
        self.assertNotIn("E005", result["eligible_event_ids"])

    def test_same_rank_disagreement_returns_conflict(self):
        result = self.result("Q008")
        self.assertEqual(result["state"], "conflict")
        self.assertIsNone(result["value"])
        self.assertEqual(result["conflicting_values"], ["120", "90"])
        self.assertEqual(result["selected_event_ids"], ["E008", "E009"])

    def test_same_rank_same_value_preserves_both_witnesses(self):
        result = self.result("Q010")
        self.assertEqual(result["state"], "resolved")
        self.assertEqual(result["value"], "true")
        self.assertEqual(result["selected_event_ids"], ["E010", "E011"])
        self.assertEqual([item["source"] for item in result["selected_evidence"]], ["service-owner", "audit-agent"])

    def test_selected_evidence_carries_time_and_authority_provenance(self):
        result = self.result("Q006")
        evidence = result["selected_evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["event_id"], "E005")
        self.assertEqual(evidence[0]["source"], "security")
        self.assertEqual(evidence[0]["authority_rank"], 3)
        self.assertEqual(evidence[0]["recorded_at"], "2026-03-01T08:00:00Z")

    def test_missing_evidence_returns_unknown_not_guess(self):
        result = self.result("Q009")
        self.assertEqual(result["state"], "unknown")
        self.assertIsNone(result["value"])
        self.assertEqual(result["selected_event_ids"], [])

    def test_event_input_order_does_not_change_query_resolution(self):
        forward = self.result("Q003", events=self.events)
        reverse = self.result("Q003", events=list(reversed(self.events)))
        self.assertEqual(forward, reverse)

    def test_duplicate_event_ids_fail_closed(self):
        rows = [
            {key: value for key, value in event.items() if not key.startswith("_")}
            for event in self.events
        ]
        rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            br.validate_events(rows)

    def test_boolean_authority_rank_is_rejected(self):
        raw = {key: value for key, value in self.events[0].items() if not key.startswith("_")}
        raw["authority_rank"] = True
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            br.validate_events([raw])

    def test_naive_recorded_at_is_rejected(self):
        raw = {key: value for key, value in self.events[0].items() if not key.startswith("_")}
        raw["recorded_at"] = "2026-01-01T09:00:00"
        with self.assertRaisesRegex(ValueError, "UTC offset"):
            br.validate_events([raw])

    def test_invalid_calendar_date_is_rejected(self):
        raw = {key: value for key, value in self.events[0].items() if not key.startswith("_")}
        raw["valid_from"] = "2026-02-30"
        with self.assertRaisesRegex(ValueError, "invalid calendar date"):
            br.validate_events([raw])

    def test_empty_or_reversed_validity_interval_is_rejected(self):
        raw = {key: value for key, value in self.events[0].items() if not key.startswith("_")}
        raw["valid_to"] = raw["valid_from"]
        with self.assertRaisesRegex(ValueError, "later than valid_from"):
            br.validate_events([raw])

    def test_duplicate_query_ids_fail_closed(self):
        rows = [
            {key: value for key, value in query.items() if not key.startswith("_")}
            for query in self.queries
        ]
        rows.append(deepcopy(rows[0]))
        with self.assertRaisesRegex(ValueError, "duplicate query_id"):
            br.validate_queries(rows)

    def test_non_string_expected_state_fails_as_validation_error(self):
        raw = {key: value for key, value in self.queries[0].items() if not key.startswith("_")}
        raw["expected_state"] = []
        with self.assertRaisesRegex(ValueError, "expected_state"):
            br.validate_queries([raw])


if __name__ == "__main__":
    unittest.main(verbosity=2)
