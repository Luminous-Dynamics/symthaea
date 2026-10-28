#!/usr/bin/env python3
"""Known-answer and fail-closed tests for the false-feasible promotion gate."""
from __future__ import annotations

import unittest

from promotion_risk_gate import (
    assess_promotion_risk,
    clopper_pearson_upper_bound,
)


class PromotionRiskGateTests(unittest.TestCase):
    def test_zero_failures_requires_59_independent_trials_for_five_percent_target(self) -> None:
        bound_58 = clopper_pearson_upper_bound(0, 58, confidence=0.95)
        bound_59 = clopper_pearson_upper_bound(0, 59, confidence=0.95)
        self.assertGreater(bound_58, 0.05)
        self.assertLessEqual(bound_59, 0.05)
        self.assertAlmostEqual(bound_59, 0.049507609888227, places=12)
        self.assertFalse(assess_promotion_risk(0, 59)["certified"])
        self.assertTrue(
            assess_promotion_risk(0, 59, independent_held_out=True)["certified"]
        )

    def test_eight_clean_promotions_are_not_safety_evidence_for_five_percent_risk(self) -> None:
        assessment = assess_promotion_risk(0, 8, risk_target=0.05, confidence=0.95)
        self.assertAlmostEqual(assessment["upper_confidence_bound"], 0.31234397806636776, places=12)
        self.assertFalse(assessment["certified"])
        self.assertEqual(assessment["decision"], "abstain")
        self.assertEqual(assessment["reason"], "insufficient_evidence_for_target")
        # Optimization-selected outcomes are not independent validation data;
        # even a numerically small bound cannot certify the policy's own trace.

    def test_observed_failure_blocks_gate(self) -> None:
        assessment = assess_promotion_risk(1, 59, risk_target=0.05, confidence=0.95)
        self.assertGreater(assessment["upper_confidence_bound"], 0.05)
        self.assertFalse(assessment["certified"])
        self.assertEqual(assessment["decision"], "abstain")

    def test_no_eligible_validation_is_not_a_pass(self) -> None:
        assessment = assess_promotion_risk(0, 0)
        self.assertIsNone(assessment["upper_confidence_bound"])
        self.assertFalse(assessment["certified"])
        self.assertEqual(assessment["decision"], "abstain")
        self.assertEqual(assessment["reason"], "no_validated_predicted_feasible_promotions")
        self.assertEqual(assessment["validation_source"], "optimization_trace")


    def test_optimization_trace_never_certifies_its_own_risk(self) -> None:
        assessment = assess_promotion_risk(
            0, 1000, risk_target=0.05, confidence=0.95,
            independent_held_out=False,
        )
        self.assertLessEqual(assessment["upper_confidence_bound"], 0.05)
        self.assertFalse(assessment["certified"])
        self.assertEqual(assessment["decision"], "abstain")
        self.assertEqual(
            assessment["reason"], "validation_not_independent_or_exchangeable"
        )

    def test_held_out_sample_with_failure_does_not_certify(self) -> None:
        assessment = assess_promotion_risk(
            1, 59, risk_target=0.05, confidence=0.95,
            independent_held_out=True,
        )
        self.assertFalse(assessment["certified"])
        self.assertEqual(assessment["validation_source"], "independent_held_out")

    def test_every_failure_returns_unit_upper_bound(self) -> None:
        self.assertEqual(clopper_pearson_upper_bound(9, 9), 1.0)
        assessment = assess_promotion_risk(9, 9, risk_target=0.05)
        self.assertFalse(assessment["certified"])
        self.assertEqual(assessment["observed_false_feasible_rate"], 1.0)

    def test_invalid_counts_and_probabilities_are_rejected(self) -> None:
        invalid_calls = (
            lambda: clopper_pearson_upper_bound(-1, 5),
            lambda: clopper_pearson_upper_bound(6, 5),
            lambda: clopper_pearson_upper_bound(True, 5),
            lambda: clopper_pearson_upper_bound(0, 0),
            lambda: clopper_pearson_upper_bound(0, 5, confidence=1.0),
            lambda: assess_promotion_risk(2, 1),
            lambda: assess_promotion_risk(0, 4, risk_target=0.0),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


if __name__ == "__main__":
    unittest.main()
