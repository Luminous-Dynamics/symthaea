#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from decision_action_ledger import (
    ZERO_SHA256,
    append_record,
    canonical_json,
    decision_event,
    outcome_event,
    verify_ledger_bytes,
)


STATE = "a" * 64
POLICY = "b" * 64
OUTCOME = "c" * 64
RESULT_STATE = "d" * 64


def event(**overrides):
    fields = {
        "task_id": "task-1",
        "run_id": "seed-7",
        "state_sha256": STATE,
        "policy_sha256": POLICY,
        "disposition": "evaluate",
        "reason_code": "highest_decision_value_per_cost",
        "action_id": "action-1",
        "candidate_id": "candidate-1",
        "fidelity": "high",
        "estimated_cost_units": 5.0,
    }
    fields.update(overrides)
    return decision_event(**fields)


def completed_ledger() -> bytes:
    first = {
        "schema_version": "1.0.0",
        "sequence": 0,
        "previous_record_sha256": ZERO_SHA256,
        **event(),
    }
    first["record_sha256"] = hashlib.sha256(
        b"CASTING-DECISION-ACTION-LEDGER-V1\0" + canonical_json(first)
    ).hexdigest()
    second_event = outcome_event(
        first,
        actual_cost_units=6.5,
        outcome_status="succeeded",
        outcome_sha256=OUTCOME,
        resulting_state_sha256=RESULT_STATE,
        reason_code="evaluation_completed",
    )
    second = {
        "schema_version": "1.0.0",
        "sequence": 1,
        "previous_record_sha256": first["record_sha256"],
        **second_event,
    }
    second["record_sha256"] = hashlib.sha256(
        b"CASTING-DECISION-ACTION-LEDGER-V1\0" + canonical_json(second)
    ).hexdigest()
    return canonical_json(first) + b"\n" + canonical_json(second) + b"\n"


class DecisionActionLedgerTests(unittest.TestCase):
    def test_empty_ledger_is_valid_and_has_zero_head(self):
        result = verify_ledger_bytes(b"")
        self.assertTrue(result["valid"])
        self.assertEqual(result["record_count"], 0)
        self.assertEqual(result["last_record_sha256"], ZERO_SHA256)

    def test_append_decision_then_outcome_tracks_cost_overrun(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            decision = append_record(path, event())
            self.assertEqual(decision["sequence"], 0)
            self.assertEqual(decision["previous_record_sha256"], ZERO_SHA256)
            self.assertEqual(verify_ledger_bytes(path.read_bytes())["unresolved_action_count"], 1)

            result_event = outcome_event(
                decision,
                actual_cost_units=6.5,
                outcome_status="succeeded",
                outcome_sha256=OUTCOME,
                resulting_state_sha256=RESULT_STATE,
                reason_code="evaluation_completed",
            )
            outcome = append_record(path, result_event)
            self.assertEqual(outcome["sequence"], 1)
            summary = verify_ledger_bytes(path.read_bytes())
            self.assertEqual(summary["record_count"], 2)
            self.assertEqual(summary["unresolved_action_count"], 0)
            self.assertEqual(summary["total_estimated_action_cost_units"], 5.0)
            self.assertEqual(summary["total_actual_action_cost_units"], 6.5)
            self.assertEqual(summary["actual_minus_estimated_cost_units"], 1.5)

    def test_stop_and_abstain_are_audited_without_fake_evaluation_cost(self):
        stop = decision_event(
            task_id="task-1",
            run_id="seed-7",
            state_sha256=STATE,
            policy_sha256=POLICY,
            disposition="abstain",
            reason_code="risk_gate_unresolved",
        )
        self.assertIsNone(stop["action_id"])
        self.assertEqual(stop["estimated_cost_units"], 0.0)

    def test_rejects_costly_stop_event(self):
        with self.assertRaisesRegex(ValueError, "zero estimated cost"):
            decision_event(
                task_id="task-1", run_id="seed-7", state_sha256=STATE,
                policy_sha256=POLICY, disposition="stop", reason_code="target_met",
                estimated_cost_units=1.0,
            )

    def test_hash_chain_detects_byte_tampering(self):
        raw = completed_ledger()
        modified = raw.replace(b"6.5", b"7.5", 1)
        with self.assertRaisesRegex(ValueError, "record_sha256"):
            verify_ledger_bytes(modified)

    def test_hash_chain_detects_reordering(self):
        raw = completed_ledger()
        lines = raw.splitlines()
        with self.assertRaises(ValueError):
            verify_ledger_bytes(lines[1] + b"\n" + lines[0] + b"\n")

    def test_rejects_partial_trailing_record(self):
        with self.assertRaisesRegex(ValueError, "trailing partial record"):
            verify_ledger_bytes(completed_ledger().rstrip(b"\n"))

    def test_rejects_duplicate_json_keys(self):
        raw = completed_ledger().replace(b'"sequence":0', b'"sequence":0,"sequence":0', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            verify_ledger_bytes(raw)

    def test_rejects_outcome_without_decision(self):
        action_decision = event()
        result_event = outcome_event(
            {
                "record_type": "decision",
                **action_decision,
            },
            actual_cost_units=1,
            outcome_status="succeeded",
            outcome_sha256=OUTCOME,
            resulting_state_sha256=RESULT_STATE,
            reason_code="done",
        )
        only_outcome = {
            "schema_version": "1.0.0", "sequence": 0,
            "previous_record_sha256": ZERO_SHA256, **result_event,
        }
        only_outcome["record_sha256"] = hashlib.sha256(
            b"CASTING-DECISION-ACTION-LEDGER-V1\0" + canonical_json(only_outcome)
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, "does not reference a prior action decision"):
            verify_ledger_bytes(canonical_json(only_outcome) + b"\n")

    def test_rejects_second_outcome_for_same_action(self):
        raw = completed_ledger()
        second = json.loads(raw.splitlines()[1])
        third = dict(second)
        third["sequence"] = 2
        third["previous_record_sha256"] = second["record_sha256"]
        third["record_sha256"] = hashlib.sha256(
            b"CASTING-DECISION-ACTION-LEDGER-V1\0"
            + canonical_json({k: v for k, v in third.items() if k != "record_sha256"})
        ).hexdigest()
        duplicated = raw + canonical_json(third) + b"\n"
        with self.assertRaisesRegex(ValueError, "duplicate outcome"):
            verify_ledger_bytes(duplicated)

    def test_rejects_mismatched_estimated_cost_in_outcome(self):
        decision = event()
        with self.assertRaisesRegex(ValueError, "differs from its decision"):
            altered = outcome_event(
                decision,
                actual_cost_units=5,
                outcome_status="succeeded",
                outcome_sha256=OUTCOME,
                resulting_state_sha256=RESULT_STATE,
                reason_code="done",
            )
            altered["estimated_cost_units"] = 4.0
            # The helper's output is later checked in a full chained ledger.
            altered["record_type"] = "outcome"
            first = {
                "schema_version": "1.0.0", "sequence": 0,
                "previous_record_sha256": ZERO_SHA256, **decision,
            }
            first["record_sha256"] = hashlib.sha256(
                b"CASTING-DECISION-ACTION-LEDGER-V1\0" + canonical_json(first)
            ).hexdigest()
            second = {
                "schema_version": "1.0.0", "sequence": 1,
                "previous_record_sha256": first["record_sha256"], **altered,
            }
            second["record_sha256"] = hashlib.sha256(
                b"CASTING-DECISION-ACTION-LEDGER-V1\0"
                + canonical_json({k: v for k, v in second.items() if k != "record_sha256"})
            ).hexdigest()
            verify_ledger_bytes(canonical_json(first) + b"\n" + canonical_json(second) + b"\n")

    def test_append_refuses_corrupt_existing_ledger_without_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.jsonl"
            path.write_bytes(b"not valid json\n")
            before = path.read_bytes()
            with self.assertRaises(ValueError):
                append_record(path, event())
            self.assertEqual(path.read_bytes(), before)

    def test_rejects_extremely_large_actual_cost(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            outcome_event(
                {"record_type": "decision", **event()},
                actual_cost_units=10**10000,
                outcome_status="succeeded",
                outcome_sha256=OUTCOME,
                resulting_state_sha256=RESULT_STATE,
                reason_code="done",
            )

    def test_rejects_nonfinite_actual_cost(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            outcome_event(
                {"record_type": "decision", **event()},
                actual_cost_units=float("nan"),
                outcome_status="succeeded",
                outcome_sha256=OUTCOME,
                resulting_state_sha256=RESULT_STATE,
                reason_code="done",
            )


if __name__ == "__main__":
    unittest.main()
