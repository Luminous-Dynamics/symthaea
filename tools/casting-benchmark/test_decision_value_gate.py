#!/usr/bin/env python3
from __future__ import annotations

import math
import unittest

from decision_value_gate import DecisionAction, choose_next_step


def action(
    action_id: str,
    *,
    kind: str = "evaluate",
    cost: float = 1.0,
    value: float = 1.0,
    risk: float = 0.0,
) -> DecisionAction:
    return DecisionAction(
        action_id=action_id,
        candidate_id="candidate-" + action_id,
        kind=kind,
        fidelity="high" if kind == "escalate" else "low",
        estimated_cost_units=cost,
        decision_loss_reduction=value,
        normalized_constraint_risk_reduction=risk,
    )


class DecisionValueGateTests(unittest.TestCase):
    def choose(self, actions, **kwargs):
        defaults = {
            "remaining_cost_units": 10.0,
            "quality_target_met": False,
            "risk_gate_satisfied": True,
            "minimum_value_per_cost": 0.0,
            "minimum_risk_reduction": 0.05,
        }
        defaults.update(kwargs)
        return choose_next_step(actions, **defaults)

    def test_selects_highest_decision_value_per_cost(self):
        result = self.choose([
            action("slow", cost=5, value=4),
            action("fast", cost=2, value=3),
        ])
        self.assertEqual(result.disposition, "evaluate")
        self.assertEqual(result.action_id, "fast")
        self.assertAlmostEqual(result.value_per_cost, 1.5)
        self.assertEqual(result.reserved_cost_units, 2.0)

    def test_unresolved_risk_forces_risk_reducing_action(self):
        result = self.choose(
            [
                action("objective-only", cost=1, value=10, risk=0),
                action("verify-boundary", kind="escalate", cost=4, value=3, risk=0.5),
            ],
            risk_gate_satisfied=False,
            minimum_risk_reduction=0.2,
        )
        self.assertEqual(result.disposition, "escalate")
        self.assertEqual(result.action_id, "verify-boundary")

    def test_abstains_when_risk_is_unresolved_and_no_action_reduces_it(self):
        result = self.choose(
            [action("cheap", value=100, risk=0)],
            risk_gate_satisfied=False,
        )
        self.assertEqual(result.disposition, "abstain")
        self.assertEqual(result.reason_code, "risk_gate_unresolved_no_qualifying_affordable_action")
        self.assertEqual(result.reserved_cost_units, 0.0)

    def test_abstains_when_only_risk_reducing_action_exceeds_budget(self):
        result = self.choose(
            [action("expensive", kind="escalate", cost=11, value=100, risk=0.8)],
            remaining_cost_units=10,
            risk_gate_satisfied=False,
        )
        self.assertEqual(result.disposition, "abstain")

    def test_stops_when_quality_target_and_risk_gate_are_satisfied(self):
        result = self.choose(
            [action("still-available", value=100)],
            quality_target_met=True,
            risk_gate_satisfied=True,
        )
        self.assertEqual(result.disposition, "stop")
        self.assertEqual(result.reason_code, "quality_target_and_risk_gate_satisfied")

    def test_stops_below_value_threshold_without_misreporting_target(self):
        result = self.choose(
            [action("weak", cost=4, value=0.2)],
            risk_gate_satisfied=True,
            minimum_value_per_cost=0.1,
        )
        self.assertEqual(result.disposition, "stop")
        self.assertFalse(result.quality_target_met)
        self.assertEqual(result.reason_code, "marginal_decision_value_below_threshold")

    def test_exact_budget_is_affordable(self):
        result = self.choose(
            [action("exact", cost=3, value=1)],
            remaining_cost_units=3,
        )
        self.assertEqual(result.action_id, "exact")
        self.assertEqual(result.reserved_cost_units, 3.0)

    def test_ties_are_deterministic(self):
        result = self.choose([action("z", value=2), action("a", value=2)])
        self.assertEqual(result.action_id, "a")

    def test_replication_is_a_supported_selected_action(self):
        result = self.choose([
            action("new-design", kind="evaluate", cost=2, value=1),
            action("repeat-measurement", kind="replicate", cost=1, value=0.8),
        ])
        self.assertEqual(result.disposition, "replicate")
        self.assertEqual(result.action_id, "repeat-measurement")

    def test_rejects_edge_whitespace_in_action_identity(self):
        with self.assertRaisesRegex(ValueError, "edge whitespace"):
            self.choose([DecisionAction(" action ", "candidate", "evaluate", "low", 1, 1, 0)])

    def test_unhashable_action_kind_is_rejected_cleanly(self):
        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            self.choose([action("bad-kind", kind=[] )])

    def test_serialized_decision_marks_proxy_only(self):
        result = self.choose([action("one", value=2)])
        receipt = result.to_dict()
        self.assertTrue(receipt["decision_proxy_only"])
        self.assertEqual(receipt["reserved_cost_units"], receipt["estimated_cost_units"])

    def test_rejects_zero_or_negative_cost(self):
        for bad_cost in (0, -1):
            with self.subTest(bad_cost=bad_cost):
                with self.assertRaisesRegex(ValueError, "strictly positive"):
                    self.choose([action("bad", cost=bad_cost)])

    def test_rejects_nonfinite_or_boolean_cost_and_value(self):
        for bad_cost in (math.inf, math.nan, True):
            with self.subTest(bad_cost=bad_cost):
                with self.assertRaises(ValueError):
                    self.choose([action("bad", cost=bad_cost)])
        with self.assertRaisesRegex(ValueError, "finite"):
            self.choose([action("bad-value", value=math.nan)])

    def test_rejects_extremely_large_integer_cost(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            self.choose([action("huge-cost", cost=10**10000)])

    def test_rejects_negative_value_and_invalid_risk_proxy(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            self.choose([action("negative-value", value=-0.1)])
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            self.choose([action("bad-risk", risk=1.1)])

    def test_rejects_duplicate_action_ids(self):
        with self.assertRaisesRegex(ValueError, "duplicate action_id"):
            self.choose([action("same"), action("same")])

    def test_rejects_unknown_action_kinds(self):
        with self.assertRaisesRegex(ValueError, "kind must be one of"):
            self.choose([action("bad-kind", kind="guess")])

    def test_rejects_negative_budget_and_threshold(self):
        with self.assertRaisesRegex(ValueError, "remaining_cost_units"):
            self.choose([action("ok")], remaining_cost_units=-1)
        with self.assertRaisesRegex(ValueError, "minimum_value_per_cost"):
            self.choose([action("ok")], minimum_value_per_cost=-1)

    def test_requires_boolean_gate_inputs(self):
        with self.assertRaisesRegex(ValueError, "must be booleans"):
            self.choose([action("ok")], risk_gate_satisfied=1)


if __name__ == "__main__":
    unittest.main()
