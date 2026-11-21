"""Known-answer and adversarial tests for the finite inspection policy oracle."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parent))
import semi_inspection_decision_reference as policy  # noqa: E402
from semi_inspection_decision_reference import InspectionDecisionError  # noqa: E402

FIXTURE = HERE.parents[1] / "fixtures" / "inspection-decision-known-answer-v1.json"


class InspectionDecisionReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.subject = json.loads(FIXTURE.read_text(encoding="utf-8"))

    def test_known_answer_exact_budgeted_optimum(self) -> None:
        report = policy.evaluate_inspection_decision(self.subject)
        expected_path = HERE.parents[1] / "fixtures" / "inspection-decision-known-answer-v1.expected.json"
        expected = json.loads(expected_path.read_text(encoding="utf-8"))
        self.assertEqual(report, expected)
        self.assertEqual(report["selected_candidate_ids"], ["A", "C", "F"])
        self.assertEqual(report["selected_inspection_minutes"], 45.0)
        self.assertEqual(report["selected_inspection_cost"], 16.0)

    def test_expected_value_policy_beats_matched_budget_risk_only_policy(self) -> None:
        report = policy.evaluate_inspection_decision(self.subject)
        self.assertEqual(report["risk_only_comparator"]["selected_candidate_ids"], ["A", "B", "D", "F"])
        self.assertGreater(
            report["selected_expected_net_value"],
            report["risk_only_comparator"]["expected_net_value"],
        )

    def test_group_capacity_and_budgets_are_never_exceeded(self) -> None:
        report = policy.evaluate_inspection_decision(self.subject)
        self.assertLessEqual(report["selected_inspection_minutes"], self.subject["budget"]["max_minutes"])
        self.assertLessEqual(report["selected_inspection_cost"], self.subject["budget"]["max_cash_cost"])
        groups = [
            row["group_id"] for row in report["candidate_ledger"]
            if row["candidate_id"] in report["selected_candidate_ids"]
        ]
        self.assertEqual(len(groups), len(set(groups)))

    def test_candidate_order_does_not_change_result(self) -> None:
        reordered = copy.deepcopy(self.subject)
        reordered["candidates"].reverse()
        self.assertEqual(
            policy.evaluate_inspection_decision(reordered),
            policy.evaluate_inspection_decision(self.subject),
        )

    def test_rejects_out_of_range_probability(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["candidates"][0]["risk_probability"] = 1.01
        with self.assertRaisesRegex(InspectionDecisionError, "risk_probability"):
            policy.evaluate_inspection_decision(subject)

    def test_rejects_negative_cost(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["candidates"][0]["inspection_cost"] = -1
        with self.assertRaisesRegex(InspectionDecisionError, "non-negative"):
            policy.evaluate_inspection_decision(subject)

    def test_rejects_duplicate_candidate_identity(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["candidates"][1]["candidate_id"] = subject["candidates"][0]["candidate_id"]
        with self.assertRaisesRegex(InspectionDecisionError, "duplicate"):
            policy.evaluate_inspection_decision(subject)

    def test_rejects_invalid_group_capacity(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["budget"]["max_per_group"] = 0
        with self.assertRaisesRegex(InspectionDecisionError, "positive integer"):
            policy.evaluate_inspection_decision(subject)

    def test_rejects_more_than_reference_limit(self) -> None:
        subject = copy.deepcopy(self.subject)
        subject["candidates"] = subject["candidates"] * 4
        with self.assertRaisesRegex(InspectionDecisionError, "maximum 18"):
            policy.evaluate_inspection_decision(subject)

    def test_empty_positive_value_set_returns_no_inspections(self) -> None:
        subject = copy.deepcopy(self.subject)
        for candidate in subject["candidates"]:
            candidate["avoidable_loss_per_defect"] = 0
        report = policy.evaluate_inspection_decision(subject)
        self.assertEqual(report["selected_candidate_ids"], [])
        self.assertEqual(report["selected_expected_net_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
