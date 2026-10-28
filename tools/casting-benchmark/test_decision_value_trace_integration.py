#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from decision_action_ledger import (
    append_record,
    decision_event,
    outcome_event,
    verify_ledger_bytes,
)
from decision_value_gate import DecisionAction, choose_next_step
from discrete_decision_voi import ObservationAction, evaluate_action


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class DecisionGateLedgerIntegrationTests(unittest.TestCase):
    def test_risk_driven_escalation_is_traced_and_cost_overrun_is_visible(self):
        result = choose_next_step(
            [
                DecisionAction(
                    "objective-only", "candidate-a", "evaluate", "low",
                    estimated_cost_units=1.0,
                    decision_loss_reduction=10.0,
                    normalized_constraint_risk_reduction=0.0,
                ),
                DecisionAction(
                    "verify-boundary", "candidate-a", "escalate", "high",
                    estimated_cost_units=4.0,
                    decision_loss_reduction=3.0,
                    normalized_constraint_risk_reduction=0.5,
                ),
            ],
            remaining_cost_units=4.0,
            quality_target_met=False,
            risk_gate_satisfied=False,
            minimum_value_per_cost=0.1,
            minimum_risk_reduction=0.2,
        )
        self.assertEqual(result.disposition, "escalate")
        self.assertEqual(result.action_id, "verify-boundary")

        state_sha = sha(b"policy-visible-state-v1")
        policy_sha = sha(b"decision-gate-config-v1")
        decision = decision_event(
            task_id="casting-synthetic-task",
            run_id="paired-seed-17",
            state_sha256=state_sha,
            policy_sha256=policy_sha,
            disposition=result.disposition,
            reason_code=result.reason_code,
            action_id=result.action_id,
            candidate_id=result.candidate_id,
            fidelity=result.fidelity,
            estimated_cost_units=result.reserved_cost_units,
            decision_loss_reduction=result.decision_loss_reduction,
            normalized_constraint_risk_reduction=result.normalized_constraint_risk_reduction,
            value_per_cost=result.value_per_cost,
        )

        outcome_bytes = json.dumps(
            {"status": "succeeded", "observed_cost_units": 4.75},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        next_state_sha = sha(b"policy-visible-state-v2")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision-trace.jsonl"
            stored_decision = append_record(path, decision)
            result_event = outcome_event(
                stored_decision,
                actual_cost_units=4.75,
                outcome_status="succeeded",
                outcome_sha256=sha(outcome_bytes),
                resulting_state_sha256=next_state_sha,
                reason_code="evaluation_completed",
            )
            append_record(path, result_event)
            summary = verify_ledger_bytes(path.read_bytes())

        self.assertTrue(summary["valid"])
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["unresolved_action_count"], 0)
        self.assertEqual(summary["total_estimated_action_cost_units"], 4.0)
        self.assertEqual(summary["total_actual_action_cost_units"], 4.75)
        self.assertAlmostEqual(summary["actual_minus_estimated_cost_units"], 0.75)
        self.assertEqual(
            summary["records"][0]["normalized_constraint_risk_reduction"], 0.5
        )
        self.assertAlmostEqual(summary["records"][0]["value_per_cost"], 0.75)

    def test_exact_discrete_voi_selects_cost_effective_query_and_records_it(self):
        prior = (0.5, 0.5)
        losses = (
            (0.0, 10.0),
            (10.0, 0.0),
        )
        noisy = evaluate_action(
            prior,
            losses,
            ObservationAction("noisy", 2.0, ((0.9, 0.1), (0.1, 0.9))),
        )
        perfect = evaluate_action(
            prior,
            losses,
            ObservationAction("perfect", 5.0, ((1.0, 0.0), (0.0, 1.0))),
        )
        decision = choose_next_step(
            [
                DecisionAction(
                    noisy.action_id, "casting-design", "evaluate", "medium",
                    noisy.estimated_cost_units, noisy.expected_value_of_information, 0.0
                ),
                DecisionAction(
                    perfect.action_id, "casting-design", "escalate", "high",
                    perfect.estimated_cost_units, perfect.expected_value_of_information, 0.0
                ),
            ],
            remaining_cost_units=5.0,
            quality_target_met=False,
            risk_gate_satisfied=True,
            minimum_value_per_cost=0.5,
        )
        self.assertEqual(decision.action_id, "noisy")
        self.assertAlmostEqual(decision.value_per_cost, 2.0)

        decision_record = decision_event(
            task_id="analytic-voi-known-answer",
            run_id="seed-0",
            state_sha256=sha(b"visible-prior-and-loss-table"),
            policy_sha256=sha(b"discrete-voi-v1"),
            disposition=decision.disposition,
            reason_code=decision.reason_code,
            action_id=decision.action_id,
            candidate_id=decision.candidate_id,
            fidelity=decision.fidelity,
            estimated_cost_units=decision.reserved_cost_units,
            decision_loss_reduction=decision.decision_loss_reduction,
            normalized_constraint_risk_reduction=decision.normalized_constraint_risk_reduction,
            value_per_cost=decision.value_per_cost,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision-trace.jsonl"
            stored = append_record(path, decision_record)
            append_record(
                path,
                outcome_event(
                    stored,
                    actual_cost_units=2.25,
                    outcome_status="succeeded",
                    outcome_sha256=sha(b"synthetic-observation-y0"),
                    resulting_state_sha256=sha(b"updated-visible-belief"),
                    reason_code="query_completed",
                ),
            )
            summary = verify_ledger_bytes(path.read_bytes())
        self.assertEqual(summary["unresolved_action_count"], 0)
        self.assertAlmostEqual(summary["records"][0]["decision_loss_reduction"], 4.0)
        self.assertAlmostEqual(summary["actual_minus_estimated_cost_units"], 0.25)

    def test_abstention_is_audited_without_inventing_an_evaluation(self):
        decision = choose_next_step(
            [
                DecisionAction(
                    "objective-only", "candidate-a", "evaluate", "low", 1.0, 10.0, 0.0
                )
            ],
            remaining_cost_units=10.0,
            quality_target_met=False,
            risk_gate_satisfied=False,
            minimum_value_per_cost=0.0,
            minimum_risk_reduction=0.1,
        )
        self.assertEqual(decision.disposition, "abstain")
        record = decision_event(
            task_id="casting-synthetic-task",
            run_id="paired-seed-18",
            state_sha256=sha(b"state"),
            policy_sha256=sha(b"policy"),
            disposition=decision.disposition,
            reason_code=decision.reason_code,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision-trace.jsonl"
            append_record(path, record)
            summary = verify_ledger_bytes(path.read_bytes())
        self.assertEqual(summary["decision_count"], 1)
        self.assertEqual(summary["outcome_count"], 0)
        self.assertEqual(summary["total_estimated_action_cost_units"], 0.0)
        self.assertEqual(summary["unresolved_action_count"], 0)


if __name__ == "__main__":
    unittest.main()
