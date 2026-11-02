import itertools
import random
import unittest
from fractions import Fraction as F

from anchored_compound_ebh import (
    anchor_slack,
    anchored_rejections,
    brute_largest_anchored_superset,
    closure_deficit,
    is_closure_certified,
    local_intersection_process,
    projective_weight_prefix,
    weighted_compound_ebh,
)


class AnchoredCompoundEBHTests(unittest.TestCase):
    def test_all_subset_closure_equals_positive_part_deficit(self):
        values = [F(0), F(1, 4), F(1, 2), F(1), F(2)]
        alphas = [F(1, 2), F(1, 3), F(1, 10)]
        deltas = [F(0), F(1, 10), F(1, 2), F(1)]
        checks = 0
        for n in range(1, 5):
            for scores in itertools.product(values, repeat=n):
                for mask in range(1 << n):
                    rejection = tuple(i for i in range(n) if mask >> i & 1)
                    for alpha in alphas:
                        for delta in deltas:
                            direct = is_closure_certified(
                                scores, alpha, delta, rejection
                            )
                            deficit_ok = (
                                closure_deficit(scores, alpha, rejection)
                                <= alpha * delta
                            )
                            self.assertEqual(direct, deficit_ok)
                            checks += 1
        self.assertEqual(checks, 133320)

    def test_fast_solver_matches_exhaustive_rejection_set_search(self):
        values = [F(0), F(1, 2), F(1), F(2), F(5)]
        alphas = [F(1, 2), F(1, 5), F(1, 10)]
        deltas = [F(0), F(1, 10), F(1, 2), F(1)]
        checks = 0
        for n in range(1, 5):
            for scores in itertools.product(values, repeat=n):
                for alpha in alphas:
                    for delta in deltas:
                        actual = anchored_rejections(scores, alpha, delta)
                        oracle = brute_largest_anchored_superset(
                            scores, alpha, delta
                        )
                        self.assertEqual(len(actual), len(oracle))
                        self.assertTrue(
                            set(weighted_compound_ebh(scores, alpha))
                            .issubset(actual)
                        )
                        self.assertTrue(
                            is_closure_certified(
                                scores, alpha, delta, actual
                            )
                        )
                        checks += 1
        self.assertEqual(checks, 9360)

    def test_randomized_solver_against_brute_force(self):
        rng = random.Random(10150)
        checks = 0
        alphas = [F(1, 2), F(1, 5), F(1, 10)]
        deltas = [F(0), F(1, 20), F(1, 5), F(1, 2)]
        for n in range(1, 9):
            for _ in range(100):
                scores = [
                    F(rng.randrange(0, 100), rng.randrange(1, 8))
                    for _ in range(n)
                ]
                for alpha in alphas:
                    for delta in deltas:
                        actual = anchored_rejections(scores, alpha, delta)
                        oracle = brute_largest_anchored_superset(
                            scores, alpha, delta
                        )
                        self.assertEqual(len(actual), len(oracle))
                        self.assertTrue(
                            set(weighted_compound_ebh(scores, alpha))
                            .issubset(actual)
                        )
                        self.assertTrue(
                            is_closure_certified(
                                scores, alpha, delta, actual
                            )
                        )
                        checks += 1
        self.assertEqual(checks, 9600)

    def test_baseline_is_preserved_and_strict_addition_is_possible(self):
        alpha, delta = F(1, 10), F(1, 5)
        scores = [F(199, 20), F(0)]
        baseline = weighted_compound_ebh(scores, alpha)
        anchored = anchored_rejections(scores, alpha, delta)
        self.assertEqual(baseline, ())
        self.assertEqual(anchored, (0,))
        self.assertTrue(is_closure_certified(scores, alpha, delta, anchored))

        scores = [F(11), F(6), F(0)]
        baseline = weighted_compound_ebh(scores, alpha)
        anchored = anchored_rejections(scores, alpha, delta)
        self.assertTrue(set(baseline).issubset(anchored))

    def test_ties_empty_baseline_delta_zero_and_delta_one(self):
        scores = [F(10), F(10), F(5), F(0)]
        alpha = F(1, 10)
        baseline = weighted_compound_ebh(scores, alpha)
        self.assertEqual(baseline, (0, 1, 2))
        self.assertEqual(anchored_rejections(scores, alpha, F(0)), baseline)
        full = anchored_rejections(scores, alpha, F(1))
        self.assertTrue(set(baseline).issubset(full))
        self.assertTrue(is_closure_certified(scores, alpha, F(1), full))

        low = [F(0), F(0), F(0)]
        self.assertEqual(weighted_compound_ebh(low, alpha), ())
        self.assertEqual(anchored_rejections(low, alpha, F(0)), ())
        self.assertTrue(
            is_closure_certified(
                low, alpha, F(1), anchored_rejections(low, alpha, F(1))
            )
        )

    def test_projective_global_weight_prefix_does_not_renormalize(self):
        mass = F(19, 20)
        short = projective_weight_prefix(5, mass)
        long = projective_weight_prefix(50, mass)
        self.assertEqual(short, long[:5])
        self.assertEqual(sum(short, F(0)), mass * F(5, 6))
        self.assertEqual(anchor_slack(mass), F(1, 20))
        self.assertGreater(
            sum(long, F(0)),
            sum(short, F(0))
        )
        self.assertLess(sum(long, F(0)), mass)

    def test_local_intersection_process_is_pathwise_monotone(self):
        delta = F(1, 20)
        path = [
            [F(0), F(0), F(0)],
            [F(1, 10), F(0), F(1, 20)],
            [F(1, 5), F(1, 20), F(1, 10)],
            [F(1, 3), F(1, 10), F(1, 5)],
        ]
        values = [
            local_intersection_process(row, (0, 2), delta)
            for row in path
        ]
        self.assertEqual(values, sorted(values))
        self.assertEqual(values[0], delta)

    def test_invalid_float_or_negative_inputs_fail_closed(self):
        with self.assertRaises(TypeError):
            anchored_rejections([0.1], F(1, 10), F(1, 20))
        with self.assertRaises(ValueError):
            anchored_rejections([F(-1)], F(1, 10), F(1, 20))
        with self.assertRaises(ValueError):
            anchored_rejections([F(1)], F(1, 10), F(2))
        with self.assertRaises(ValueError):
            projective_weight_prefix(3, F(1))


if __name__ == "__main__":
    unittest.main()
