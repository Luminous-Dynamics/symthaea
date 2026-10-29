#!/usr/bin/env python3
"""Regression tests for temporal metamorphic checks."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bitemporal_reference as br  # noqa: E402
import metamorphic_temporal_audit as mta  # noqa: E402


def load_raw(path: Path):
    records, _ = br.load_jsonl(path)
    return records


class MetamorphicTemporalAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.events = load_raw(HERE / "fixtures" / "events.jsonl")
        cls.queries = load_raw(HERE / "fixtures" / "queries.jsonl")

    def test_default_metamorphic_audit_passes(self):
        report = mta.run_audit(self.events, self.queries)
        self.assertTrue(report["all_passed"], [
            item for item in report["checks"] if not item["passed"]
        ])
        self.assertEqual(report["failed_count"], 0)
        self.assertGreaterEqual(report["check_count"], 250)

    def test_report_includes_future_cutoff_invariant(self):
        report = mta.run_audit(self.events, self.queries, permutation_trials=1, query_order_trials=1)
        self.assertIn("future_record_cutoff_noninterference", report["invariant_summary"])
        self.assertEqual(
            report["invariant_summary"]["future_record_cutoff_noninterference"]["failed"],
            0,
        )

    def test_equal_authority_conflict_is_exercised(self):
        report = mta.run_audit(self.events, self.queries, permutation_trials=1, query_order_trials=1)
        conflict = report["invariant_summary"]["equal_authority_disagreement_abstains"]
        self.assertGreater(conflict["passed"], 0)
        self.assertEqual(conflict["failed"], 0)

    def test_lower_authority_disagreement_is_noninterfering(self):
        report = mta.run_audit(self.events, self.queries, permutation_trials=1, query_order_trials=1)
        result = report["invariant_summary"]["lower_authority_noninterference"]
        self.assertGreater(result["passed"], 0)
        self.assertEqual(result["failed"], 0)

    def test_same_value_witness_preserves_semantics(self):
        report = mta.run_audit(self.events, self.queries, permutation_trials=1, query_order_trials=1)
        result = report["invariant_summary"]["same_value_witness_preserves_answer"]
        self.assertGreater(result["passed"], 0)
        self.assertEqual(result["failed"], 0)

    def test_event_permutation_trials_are_deterministic(self):
        first = mta.run_audit(self.events, self.queries, permutation_trials=3, query_order_trials=2)
        second = mta.run_audit(self.events, self.queries, permutation_trials=3, query_order_trials=2)
        self.assertEqual(first, second)

    def test_different_seed_changes_trial_metadata_and_maybe_orders(self):
        first = mta.run_audit(self.events, self.queries, permutation_trials=3, query_order_trials=2, seed=4)
        second = mta.run_audit(self.events, self.queries, permutation_trials=3, query_order_trials=2, seed=5)
        self.assertNotEqual(first["seed"], second["seed"])
        self.assertTrue(first["all_passed"])
        self.assertTrue(second["all_passed"])

    def test_trial_bounds_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "permutation_trials"):
            mta.run_audit(self.events, self.queries, permutation_trials=0)
        with self.assertRaisesRegex(ValueError, "query_order_trials"):
            mta.run_audit(self.events, self.queries, query_order_trials=1001)

    def test_future_event_is_not_selected_or_listed_as_eligible(self):
        base_events = br.validate_events(self.events)
        queries = br.validate_queries(self.queries)
        query = next(item for item in queries if item["query_id"] == "Q006")
        future = {
            "event_id": "FUTURE",
            "subject": query["subject"],
            "attribute": query["attribute"],
            "value": "tampered",
            "valid_from": "1900-01-01",
            "valid_to": None,
            "recorded_at": "2099-12-31T23:59:59Z",
            "source": "future",
            "authority_rank": 999,
        }
        mutated = br.validate_events(self.events + [future])
        actual = br.resolve_query(mutated, query)
        self.assertNotIn("FUTURE", actual["eligible_event_ids"])
        self.assertNotIn("FUTURE", actual["selected_event_ids"])

    def test_audit_fails_if_resolver_signature_changes(self):
        # Negative control: inject a local wrapper that violates cutoff semantics.
        original = mta.br.resolve_query
        def bad_resolver(events, query):
            result = original(events, query)
            result = dict(result)
            if query["query_id"] == "Q001":
                result["value"] = "deliberately-corrupted"
            return result
        mta.br.resolve_query = bad_resolver
        try:
            report = mta.run_audit(self.events, self.queries, permutation_trials=1, query_order_trials=1)
            self.assertFalse(report["all_passed"])
            self.assertGreater(report["failed_count"], 0)
        finally:
            mta.br.resolve_query = original


if __name__ == "__main__":
    unittest.main(verbosity=2)
