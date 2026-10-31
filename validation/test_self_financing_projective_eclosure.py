import copy
import json
import unittest
from fractions import Fraction

from self_financing_projective_eclosure import (
    build_self_financing_intersection_paths,
    select_self_financing_capacity_actions,
    verify_self_financing_sequence,
)


class SelfFinancingProjectiveEclosureTests(unittest.TestCase):
    def setUp(self):
        # One-step martingale construction:
        # under H1, X1,1 is 8 with probability 1/8 and 0 otherwise;
        # under H2, X2,2 is 16 with probability 1/16 and 0 otherwise.
        # Other increments are one. The fixture is the all-success path.
        self.increments = [
            [1, 8, 1],
            [1, 1, 16],
        ]
        self.active = [0b01, 0b11, 0b11]
        self.scores = [1, 1]
        self.alpha = Fraction(1, 2)
        self.values = [10, 8]
        self.costs = [1, 1]
        self.capacity = [0, 1, 2]
        self.thresholds = [2, 4, 8, 16, 32, 64]
        self.adjuster_increments = [1, 1, 1, 1, 1, 1]

    def test_self_financing_process_and_capacity_decisions(self):
        raw, shares = build_self_financing_intersection_paths(
            self.increments, self.active, self.scores
        )
        self.assertEqual(shares[0], (Fraction(1), Fraction(0)))
        self.assertEqual(shares[1], (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(shares[2], (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(raw[0b01], (Fraction(1), Fraction(9, 2), Fraction(9, 2)))
        self.assertEqual(raw[0b10], (Fraction(1), Fraction(1), Fraction(17, 2)))
        self.assertEqual(raw[0b11], (Fraction(1), Fraction(9, 2), Fraction(153, 4)))

        result = select_self_financing_capacity_actions(
            self.increments, self.active, self.scores, self.alpha,
            self.values, self.costs, self.capacity,
            self.thresholds, self.adjuster_increments,
        )
        self.assertEqual(
            [row["rejections"] for row in result["decisions"]],
            [[], [1], [1, 2]],
        )
        self.assertEqual(result["adjuster_integral_budget"], [63, 64])
        self.assertTrue(result["verification"]["valid"])
        self.assertTrue(result["verification"]["optimality_verified"])
        json.dumps(result, sort_keys=True, separators=(",", ":"))

    def test_future_priority_and_evidence_do_not_rewrite_past_receipts(self):
        # Coordinate 2 is not active until the final update, so its future
        # score and future evidence must not alter the first two decisions.
        active = [0b01, 0b01, 0b11]
        first = select_self_financing_capacity_actions(
            self.increments, active, [1, 1], self.alpha,
            self.values, self.costs, self.capacity,
            self.thresholds, self.adjuster_increments,
        )
        changed_increments = copy.deepcopy(self.increments)
        changed_increments[1][2] = 32
        second = select_self_financing_capacity_actions(
            changed_increments, active, [1, 999], self.alpha,
            self.values, self.costs, self.capacity,
            self.thresholds, self.adjuster_increments,
        )
        self.assertEqual(
            [r["rejections"] for r in first["decisions"][:2]],
            [r["rejections"] for r in second["decisions"][:2]],
        )
        self.assertEqual(
            [r["prefix_sha256"] for r in first["decisions"][:2]],
            [r["prefix_sha256"] for r in second["decisions"][:2]],
        )

    def test_independent_verifier_rejects_valid_but_suboptimal_selection(self):
        result = select_self_financing_capacity_actions(
            self.increments, self.active, self.scores, self.alpha,
            self.values, self.costs, self.capacity,
            self.thresholds, self.adjuster_increments,
        )
        tampered = copy.deepcopy(result["decisions"])
        # {1} remains certified at the final time but {1,2} has more value.
        tampered[2]["rejection_mask"] = 0b01
        tampered[2]["rejections"] = [1]
        tampered[2]["utility"] = [10, 1]
        tampered[2]["cost"] = [1, 1]
        tampered[2]["minimum_closure_slack"] = [0, 1]
        tampered[2]["worst_intersection"] = 0b01
        check = verify_self_financing_sequence(
            tampered,
            tuple(tuple(Fraction(v) for v in row) for row in self.increments),
            tuple(self.active),
            tuple(Fraction(v) for v in self.scores),
            self.alpha,
            tuple(Fraction(v) for v in self.values),
            tuple(Fraction(v) for v in self.costs),
            tuple(Fraction(v) for v in self.capacity),
            self.thresholds,
            self.adjuster_increments,
        )
        self.assertFalse(check["valid"])
        self.assertFalse(check["optimality_verified"])
        self.assertTrue(
            any(
                isinstance(f, tuple) and f[0] == "utility_optimality_failure"
                for f in check["failures"]
            )
        )

    def test_non_nested_active_sets_are_rejected(self):
        with self.assertRaises(ValueError):
            select_self_financing_capacity_actions(
                self.increments, [0b01, 0b11, 0b01], self.scores,
                self.alpha, self.values, self.costs, self.capacity,
                self.thresholds, self.adjuster_increments,
            )

    def test_dormant_increment_must_be_one(self):
        bad = copy.deepcopy(self.increments)
        bad[1][0] = 2
        with self.assertRaises(ValueError):
            select_self_financing_capacity_actions(
                bad, self.active, self.scores, self.alpha,
                self.values, self.costs, self.capacity,
                self.thresholds, self.adjuster_increments,
            )

    def test_negative_increment_and_bad_adjuster_fail_closed(self):
        bad = copy.deepcopy(self.increments)
        bad[0][1] = -1
        with self.assertRaises(ValueError):
            select_self_financing_capacity_actions(
                bad, self.active, self.scores, self.alpha,
                self.values, self.costs, self.capacity,
                self.thresholds, self.adjuster_increments,
            )
        with self.assertRaises(ValueError):
            select_self_financing_capacity_actions(
                self.increments, self.active, self.scores, self.alpha,
                self.values, self.costs, self.capacity,
                [2, 4], [2, 2],
            )

    def test_all_zero_scores_holds_cash_and_preserves_validity(self):
        result = select_self_financing_capacity_actions(
            self.increments, self.active, [0, 0], self.alpha,
            self.values, self.costs, self.capacity,
            self.thresholds, self.adjuster_increments,
        )
        self.assertTrue(all(
            share == [0, 1]
            for row in result["decisions"]
            for share in row["shares"]
        ))
        self.assertEqual(
            [row["rejections"] for row in result["decisions"]],
            [[], [], []],
        )
        self.assertTrue(result["verification"]["valid"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
