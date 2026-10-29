#!/usr/bin/env python3
"""Regression tests for policy-pinned evidence admission and retractions."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import policy_pinned_ledger as ledger  # noqa: E402


def load_rows(name: str):
    rows, _ = ledger.load_jsonl(HERE / "fixtures" / "pinned-ledger" / name)
    return rows


class PolicyPinnedLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw_policies = load_rows("policies.jsonl")
        cls.raw_events = load_rows("events.jsonl")
        cls.raw_queries = load_rows("queries.jsonl")
        cls.policies = ledger.validate_policies(cls.raw_policies)
        cls.events = ledger.validate_events(cls.raw_events)
        cls.queries = ledger.validate_queries(cls.raw_queries)
        cls.admission = ledger.validate_admission(cls.events, cls.policies)
        cls.query_by_id = {query["query_id"]: query for query in cls.queries}

    def result(self, query_id: str, events=None, policies=None, admission=None):
        active_events = self.events if events is None else events
        active_policies = self.policies if policies is None else policies
        active_admission = self.admission if admission is None else admission
        return ledger.resolve_query(
            active_events, active_policies, active_admission,
            self.query_by_id[query_id],
        )

    def test_all_synthetic_queries_match_expected(self):
        report = ledger.evaluate(self.events, self.policies, self.admission, self.queries)
        self.assertEqual(report["correct_count"], 12)
        self.assertEqual(report["exact_fixture_accuracy"], 1.0)

    def test_event_does_not_supply_its_own_authority_rank(self):
        raw = deepcopy(self.raw_events[0])
        raw["authority_rank"] = 999999
        with self.assertRaisesRegex(ValueError, "exact schema mismatch"):
            ledger.validate_events([raw])

    def test_rank_is_taken_from_policy_visible_at_query_cutoff(self):
        before = self.result("Q002")
        after = self.result("Q003")
        self.assertEqual((before["policy_version"], before["value"]), (1, "db-primary-c.internal"))
        self.assertEqual((after["policy_version"], after["value"]), (2, "db-primary-d.internal"))

    def test_policy_record_time_boundary_is_inclusive(self):
        before = self.result("Q007")
        at = self.result("Q006")
        self.assertEqual(before["policy_version"], 1)
        self.assertEqual(at["policy_version"], 2)

    def test_latest_policy_does_not_leak_before_recorded_time(self):
        result = self.result("Q007")
        self.assertNotEqual(result["policy_version"], 2)
        self.assertEqual(result["value"], "db-primary-c.internal")

    def test_retraction_does_not_leak_before_its_recorded_time(self):
        before = self.result("Q008")
        at = self.result("Q009")
        self.assertEqual(before["value"], "db-primary-d.internal")
        self.assertIn("E004", before["selected_event_ids"])
        self.assertEqual(at["value"], "db-primary-c.internal")
        self.assertNotIn("E004", at["selected_event_ids"])
        self.assertIn("R005", at["applied_retractions"])

    def test_retraction_is_authorized_under_policy_at_record_time(self):
        record = self.admission["R005"]
        self.assertTrue(record["admitted"])
        self.assertEqual(record["record_policy_version"], 2)

    def test_unauthorized_retraction_cannot_remove_evidence(self):
        result = self.result("Q004")
        self.assertEqual(result["value"], "db-primary-c.internal")
        self.assertNotIn("R006", result["applied_retractions"])
        excluded = {item["event_id"]: item["reason"] for item in result["excluded_events"]}
        self.assertIn("R006", excluded)
        self.assertIn("lacked retraction authority", excluded["R006"])

    def test_unregistered_source_is_excluded(self):
        result = self.result("Q002")
        self.assertNotIn("E007", result["eligible_event_ids"])
        admission = {item["event_id"]: item for item in ledger.evaluate(
            self.events, self.policies, self.admission, self.queries
        )["event_admission"]}
        self.assertFalse(admission["E007"]["admitted"])
        self.assertIn("not registered", admission["E007"]["reason"])

    def test_out_of_scope_source_cannot_assert_endpoint(self):
        result = self.result("Q002")
        self.assertNotIn("E006", result["eligible_event_ids"])
        self.assertEqual(result["value"], "db-primary-c.internal")

    def test_current_policy_scope_is_checked_in_addition_to_admission(self):
        # Policy v2 still admits incident-review for endpoint. A modified policy
        # that removes that scope must not keep its evidence eligible.
        modified_raw = deepcopy(self.raw_policies)
        modified_raw[1]["sources"][1]["attributes"] = ["incident-note"]
        modified = ledger.validate_policies(modified_raw)
        admission = ledger.validate_admission(self.events, modified)
        result = self.result("Q003", policies=modified, admission=admission)
        self.assertNotIn("E004", result["eligible_event_ids"])
        self.assertEqual(result["value"], "db-primary-c.internal")

    def test_same_time_equal_rank_disagreement_abstains(self):
        extra = {
            "event_id": "E009",
            "kind": "assert",
            "subject": "db-prod",
            "attribute": "endpoint",
            "value": "db-primary-x.internal",
            "valid_from": "2026-01-15",
            "valid_to": None,
            "recorded_at": "2026-03-10T10:00:00Z",
            "source": "operations",
        }
        events = ledger.validate_events(self.raw_events + [extra])
        admission = ledger.validate_admission(events, self.policies)
        result = self.result("Q003", events=events, admission=admission)
        self.assertEqual(result["state"], "conflict")
        self.assertIsNone(result["value"])
        self.assertEqual(result["conflicting_values"], ["db-primary-c", "db-primary-d"]) if False else None
        self.assertEqual(set(result["conflicting_values"]), {"db-primary-c.internal", "db-primary-d.internal", "db-primary-x.internal"})

    def test_duplicate_event_id_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "duplicate event_id"):
            ledger.validate_events(self.raw_events + [deepcopy(self.raw_events[0])])

    def test_dangling_retraction_target_fails_closed(self):
        raw = deepcopy(self.raw_events)
        raw.append({
            "event_id": "R-DANGLING",
            "kind": "retract",
            "target_event_id": "MISSING",
            "recorded_at": "2026-04-02T00:00:00Z",
            "source": "operations",
        })
        with self.assertRaisesRegex(ValueError, "target must name an assertion"):
            ledger.validate_events(raw)

    def test_retraction_cannot_precede_target(self):
        raw = deepcopy(self.raw_events)
        raw.append({
            "event_id": "R-TOO-EARLY",
            "kind": "retract",
            "target_event_id": "E004",
            "recorded_at": "2026-03-01T00:00:00Z",
            "source": "operations",
        })
        with self.assertRaisesRegex(ValueError, "cannot precede target"):
            ledger.validate_events(raw)

    def test_duplicate_retractions_of_one_claim_fail_closed(self):
        raw = deepcopy(self.raw_events)
        raw.append({
            "event_id": "R-DUPLICATE",
            "kind": "retract",
            "target_event_id": "E004",
            "recorded_at": "2026-04-02T00:00:00Z",
            "source": "operations",
        })
        with self.assertRaisesRegex(ValueError, "multiple retractions"):
            ledger.validate_events(raw)

    def test_boolean_policy_rank_is_rejected(self):
        raw = deepcopy(self.raw_policies)
        raw[0]["sources"][0]["authority_rank"] = True
        with self.assertRaisesRegex(ValueError, "non-negative integer"):
            ledger.validate_policies(raw)

    def test_policy_versions_must_increase(self):
        raw = deepcopy(self.raw_policies)
        raw[1]["policy_version"] = 1
        with self.assertRaisesRegex(ValueError, "increase strictly"):
            ledger.validate_policies(raw)

    def test_policy_record_times_must_increase(self):
        raw = deepcopy(self.raw_policies)
        raw[1]["recorded_at"] = raw[0]["recorded_at"]
        with self.assertRaisesRegex(ValueError, "recorded_at must increase"):
            ledger.validate_policies(raw)

    def test_invalid_retraction_kind_and_type_are_rejected(self):
        raw = deepcopy(self.raw_events)
        raw[4]["kind"] = "maybe"
        with self.assertRaisesRegex(ValueError, "kind must be assert or retract"):
            ledger.validate_events(raw)

    def test_order_of_input_events_does_not_change_resolution(self):
        reversed_events = ledger.validate_events(list(reversed(self.raw_events)))
        reversed_admission = ledger.validate_admission(reversed_events, self.policies)
        first = self.result("Q003")
        second = self.result("Q003", events=reversed_events, admission=reversed_admission)
        self.assertEqual(first["state"], second["state"])
        self.assertEqual(first["value"], second["value"])
        self.assertEqual(first["selected_event_ids"], second["selected_event_ids"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
