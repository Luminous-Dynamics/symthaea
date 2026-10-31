import unittest
from fractions import Fraction

from awaire_dynamic_projective_baseline import (
    build_intersection_local_paths,
    verify_intersection_local_collection,
)


class AwaireDynamicProjectiveBaselineTests(unittest.TestCase):
    def setUp(self):
        self.increments = [
            [1, 8, 1],
            [1, 1, 16],
        ]
        self.active = [0b01, 0b11, 0b11]
        self.scores = [1, 1]
        self.thresholds = [2, 4, 8, 16, 32, 64]
        self.adjuster_increments = [1, 1, 1, 1, 1, 1]

    def test_intersection_local_equal_mixture_and_projectivity(self):
        raw, shares = build_intersection_local_paths(
            self.increments, self.active, self.scores,
            share_policy="equal_active",
        )
        self.assertEqual(raw[0b01], (Fraction(1), Fraction(8), Fraction(8)))
        self.assertEqual(raw[0b10], (Fraction(1), Fraction(1), Fraction(16)))
        self.assertEqual(
            raw[0b11],
            (Fraction(1), Fraction(9, 2), Fraction(153, 4)),
        )
        self.assertEqual(shares[0b01][1], (Fraction(1), Fraction(0)))
        self.assertEqual(shares[0b10][1], (Fraction(0), Fraction(1)))
        self.assertEqual(
            shares[0b11][1], (Fraction(1, 2), Fraction(1, 2))
        )

        report = verify_intersection_local_collection(
            raw, self.active, self.thresholds, self.adjuster_increments
        )
        self.assertTrue(report["valid"], report)
        self.assertEqual(report["adjuster_integral_budget"], Fraction(63, 64))
        self.assertEqual(report["projective_checks"], 8)

    def test_awaire_past_wealth_policy_rebalances_predictably(self):
        raw, shares = build_intersection_local_paths(
            self.increments, self.active, self.scores,
            share_policy="past_wealth",
        )
        self.assertEqual(shares[0b11][1], (Fraction(1, 2), Fraction(1, 2)))
        self.assertEqual(
            shares[0b11][2], (Fraction(8, 9), Fraction(1, 9))
        )
        self.assertEqual(raw[0b01], (Fraction(1), Fraction(8), Fraction(8)))
        self.assertEqual(raw[0b10], (Fraction(1), Fraction(1), Fraction(16)))
        self.assertEqual(raw[0b11], (Fraction(1), Fraction(9, 2), Fraction(12)))

        report = verify_intersection_local_collection(
            raw, self.active, self.thresholds, self.adjuster_increments
        )
        self.assertTrue(report["valid"], report)

    def test_dynamic_local_wealth_portfolios_preserve_future_extension(self):
        # The active universe grows after time 0. A future member cannot
        # change the already-computed path for the current active restriction.
        raw, _ = build_intersection_local_paths(
            self.increments, self.active, self.scores,
            share_policy="past_wealth",
        )
        self.assertEqual(raw[0b11][0], raw[0b01][0])
        self.assertEqual(raw[0b11][1], raw[0b11 & 0b11][1])
        self.assertEqual(raw[0b10][0], Fraction(1))

    def test_bad_dormant_increment_and_non_nested_active_sets_fail(self):
        bad = [list(row) for row in self.increments]
        bad[1][0] = 2
        with self.assertRaises(ValueError):
            build_intersection_local_paths(
                bad, self.active, self.scores
            )
        with self.assertRaises(ValueError):
            build_intersection_local_paths(
                self.increments, [0b01, 0b11, 0b01], self.scores
            )

    def test_nonnegative_predictable_mixture_under_correlated_null_increments(self):
        # U is uniform on 16 points. X1=8 on the first 2 points;
        # X2=16 on the first point. Both have mean exactly one, and
        # their dependence is maximally nested rather than independent.
        outcomes = [
            (Fraction(8), Fraction(16), Fraction(1, 16)),
            (Fraction(8), Fraction(0), Fraction(1, 16)),
            (Fraction(0), Fraction(0), Fraction(14, 16)),
        ]
        ex1 = sum(x1 * probability for x1, _, probability in outcomes)
        ex2 = sum(x2 * probability for _, x2, probability in outcomes)
        self.assertEqual(ex1, 1)
        self.assertEqual(ex2, 1)

        # Any predictable convex combination for each intersection has
        # conditional expectation one under this arbitrary dependence.
        for p1 in (Fraction(0), Fraction(1, 4), Fraction(1, 2), Fraction(1)):
            p2 = 1 - p1
            expected_factor = sum(
                (p1 * x1 + p2 * x2) * probability
                for x1, x2, probability in outcomes
            )
            self.assertEqual(expected_factor, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
