"""Synthetic controls for the metamorphic evaluator receipt auditor."""
from __future__ import annotations

import copy
import json
import unittest

from metamorphic_evaluator_audit import MetamorphicAuditError, audit_receipt
from metamorphic_evaluator_fixture import build_example


def encode(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")


class MetamorphicEvaluatorAuditTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan, self.receipt = build_example()
        self.plan_bytes = encode(self.plan)
        self.receipt_bytes = encode(self.receipt)

    def run_audit(self, plan=None, receipt=None):
        plan_value = self.plan if plan is None else plan
        receipt_value = self.receipt if receipt is None else receipt
        return audit_receipt(encode(plan_value), encode(receipt_value))

    def test_clean_seven_relation_fixture_passes(self) -> None:
        report = audit_receipt(self.plan_bytes, self.receipt_bytes)
        self.assertTrue(report["valid"])
        self.assertTrue(report["all_relations_pass"], report["pair_results"])
        self.assertEqual(report["planned_pair_count"], 7)
        self.assertEqual(report["observed_pair_count"], 7)
        self.assertEqual(report["passed_pair_count"], 7)
        self.assertEqual(report["failed_pair_count"], 0)
        self.assertEqual(len(report["results_by_relation_family"]), 7)
        self.assertTrue(report["fixed_receipt_only"])
        self.assertFalse(report["evaluator_execution_proven"])
        self.assertFalse(report["receipt_authenticity_proven"])
        self.assertFalse(report["causal_effect_identified"])
        self.assertFalse(report["generalized_accuracy_estimated"])
        self.assertFalse(report["construct_validity_proven"])
        self.assertTrue(report["external_anchor_required"])
        self.assertFalse(report["scoring_eligible_declared"])

    def test_report_is_deterministic_for_exact_input_bytes(self) -> None:
        first = encode(audit_receipt(self.plan_bytes, self.receipt_bytes))
        second = encode(audit_receipt(self.plan_bytes, self.receipt_bytes))
        self.assertEqual(first, second)

    def test_plan_commitment_tampering_is_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["pairs"][0]["relation_rule"] = "support_non_decreasing"
        with self.assertRaisesRegex(MetamorphicAuditError, "plan commitment mismatch"):
            audit_receipt(encode(plan), self.receipt_bytes)

    def test_receipt_must_bind_exact_plan_bytes(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["plan_sha256"] = "0" * 64
        with self.assertRaisesRegex(MetamorphicAuditError, "does not bind exact plan bytes"):
            self.run_audit(receipt=receipt)

    def test_missing_observation_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["observations"].pop()
        with self.assertRaisesRegex(MetamorphicAuditError, "missing pairs"):
            self.run_audit(receipt=receipt)

    def test_extra_observation_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["observations"].append(copy.deepcopy(receipt["observations"][0]))
        with self.assertRaisesRegex(MetamorphicAuditError, "duplicate observation pair_id"):
            self.run_audit(receipt=receipt)

    def test_duplicate_pair_plan_is_rejected(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["pairs"].append(copy.deepcopy(plan["pairs"][0]))
        with self.assertRaisesRegex(MetamorphicAuditError, "duplicate pair_id"):
            self.run_audit(plan=plan)

    def test_invariance_violation_fails_relation_without_making_receipt_malformed(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "invariance-format-001")
        target["variant"]["criterion_scores"][2]["score_millionths"] += 20_000
        report = self.run_audit(receipt=receipt)
        self.assertTrue(report["valid"])
        self.assertFalse(report["all_relations_pass"])
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "invariance-format-001")
        self.assertTrue(any("score drift exceeded tolerance" in issue for issue in failed["failures"]))

    def test_evidence_withdrawal_cannot_increase_support_or_affected_score(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "evidence-withdrawal-001")
        target["baseline"]["decision"] = "insufficient_evidence"
        target["variant"]["decision"] = "supported_within_scope"
        target["variant"]["criterion_scores"][1]["score_millionths"] = 950_000
        report = self.run_audit(receipt=receipt)
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "evidence-withdrawal-001")
        self.assertFalse(failed["passed"])
        self.assertTrue(any("support increased" in issue for issue in failed["failures"]))
        self.assertTrue(any("score increased after evidence withdrawal" in issue for issue in failed["failures"]))

    def test_locality_leakage_into_unaffected_criterion_is_detected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "criterion-locality-001")
        target["variant"]["criterion_scores"][2]["score_millionths"] -= 30_000
        report = self.run_audit(receipt=receipt)
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "criterion-locality-001")
        self.assertFalse(failed["passed"])
        self.assertTrue(any("unaffected criterion clarity" in issue for issue in failed["failures"]))

    def test_position_swap_must_preserve_candidate_identity(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "position-swap-001")
        target["variant"]["selected_candidate_id"] = "candidate-B"
        report = self.run_audit(receipt=receipt)
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "position-swap-001")
        self.assertFalse(failed["passed"])
        self.assertTrue(any("identity changed" in issue for issue in failed["failures"]))

    def test_mirrored_source_cannot_increase_unique_origin_count(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "provenance-mirror-001")
        target["variant"]["unique_source_group_count"] = 3
        report = self.run_audit(receipt=receipt)
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "provenance-mirror-001")
        self.assertFalse(failed["passed"])
        self.assertTrue(any("increased independent source-origin count" in issue for issue in failed["failures"]))

    def test_pointwise_and_pairwise_preferences_must_agree(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "pointwise-pairwise-001")
        target["variant"]["pairwise_preferred_candidate_id"] = "candidate-B"
        report = self.run_audit(receipt=receipt)
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "pointwise-pairwise-001")
        self.assertFalse(failed["passed"])
        self.assertTrue(any("disagrees with pointwise" in issue for issue in failed["failures"]))

    def test_counterfactual_must_match_declared_variant_decision(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        target = next(row for row in receipt["observations"] if row["pair_id"] == "unit-counterfactual-001")
        target["variant"]["decision"] = "supported_within_scope"
        report = self.run_audit(receipt=receipt)
        failed = next(row for row in report["pair_results"] if row["pair_id"] == "unit-counterfactual-001")
        self.assertFalse(failed["passed"])
        self.assertTrue(any("differs from declared expectation" in issue for issue in failed["failures"]))

    def test_unknown_fields_and_out_of_range_scores_are_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["observations"][0]["variant"]["untrusted_field"] = True
        with self.assertRaisesRegex(MetamorphicAuditError, "unknown fields"):
            self.run_audit(receipt=receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["observations"][0]["variant"]["criterion_scores"][0]["score_millionths"] = 1_000_001
        with self.assertRaisesRegex(MetamorphicAuditError, "out-of-range score"):
            self.run_audit(receipt=receipt)

    def test_unhashable_relation_decision_and_preference_values_fail_closed(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["pairs"][0]["expected_variant_decision"] = ["supported_within_scope"]
        with self.assertRaises(MetamorphicAuditError):
            self.run_audit(plan=plan)
        receipt = copy.deepcopy(self.receipt)
        receipt["observations"][0]["baseline"]["decision"] = ["supported_within_scope"]
        with self.assertRaises(MetamorphicAuditError):
            self.run_audit(receipt=receipt)
        receipt = copy.deepcopy(self.receipt)
        receipt["observations"][5]["baseline"]["pairwise_preferred_candidate_id"] = ["candidate-A"]
        with self.assertRaises(MetamorphicAuditError):
            self.run_audit(receipt=receipt)

    def test_cross_plan_evaluator_or_rubric_revision_is_rejected(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["rubric_sha256"] = "f" * 64
        with self.assertRaisesRegex(MetamorphicAuditError, "rubric digest differs"):
            self.run_audit(receipt=receipt)


if __name__ == "__main__":
    unittest.main()
