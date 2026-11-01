import unittest
from fractions import Fraction

from countable_prior_weighted_eclosure import (
    canonical_intersection,
    elias_gamma_code,
    elias_gamma_length,
    enumerate_certified_rejection_sets,
    finite_prior_mass,
    intersection_code,
    intersection_code_length,
    intersection_prior_weight,
    nonempty_subsets,
    verify_finite_truth_path,
    verify_finite_prior_budget,
    verify_local_closure,
)


class CountablePriorWeightedEclosureTests(unittest.TestCase):
    def test_prefix_code_has_exact_kraft_mass_on_small_complete_prefix(self):
        # Every finite nonempty subset of positive integers has one code.
        codes = [intersection_code(s) for s in nonempty_subsets(range(1, 6))]
        self.assertEqual(len(codes), len(set(codes)))
        for left in codes:
            for right in codes:
                if left != right:
                    self.assertFalse(right.startswith(left))
        self.assertEqual(
            sum((Fraction(1, 1 << len(code)) for code in codes), Fraction(0)),
            Fraction(269, 512),
        )

    def test_gamma_code_lengths_and_intersection_prior_are_exact(self):
        self.assertEqual(elias_gamma_code(1), "1")
        self.assertEqual(elias_gamma_code(2), "010")
        self.assertEqual(elias_gamma_length(8), 7)
        self.assertEqual(intersection_code_length((1,)), 2)
        self.assertEqual(intersection_prior_weight((1,)), Fraction(1, 4))
        self.assertEqual(intersection_prior_weight((2,)), Fraction(1, 16))
        self.assertEqual(intersection_prior_weight((1, 2)), Fraction(1, 32))
        self.assertEqual(intersection_prior_weight((1, 3)), Fraction(1, 128))

    def test_prior_mass_of_finite_prefix_is_strictly_below_one(self):
        for n in range(1, 8):
            mass = finite_prior_mass(range(1, n + 1))
            self.assertGreater(mass, 0)
            self.assertLess(mass, 1)

    def test_intersection_canonicalization_rejects_bad_ids(self):
        self.assertEqual(canonical_intersection((3, 1, 2)), (1, 2, 3))
        for bad in ((), (0,), (-1,), (True,), (1, 1), (1.0,)):
            with self.subTest(bad=bad):
                with self.assertRaises((TypeError, ValueError)):
                    canonical_intersection(bad)

    def setUp(self):
        self.active = [1, 2, 3]
        self.true = {1, 3}
        self.alpha = Fraction(1, 2)
        # Exact deterministic paths. Stochastic validity is not claimed.
        self.processes = {
            s: [Fraction(512 + sum(s)), Fraction(1024 + sum(s)), Fraction(2048 + sum(s))]
            for s in nonempty_subsets(self.active)
        }

    def test_local_closure_checks_every_active_intersection(self):
        result = verify_local_closure(
            self.active, (1,), 1, self.processes, self.alpha
        )
        self.assertEqual(result["constraints_checked"], 7)
        self.assertTrue(result["certified"], result)

    def test_missing_intersection_process_fails_closed(self):
        incomplete = dict(self.processes)
        del incomplete[(1, 2, 3)]
        result = verify_local_closure(
            self.active, (1,), 1, incomplete, self.alpha
        )
        self.assertFalse(result["certified"])
        self.assertIn(
            ("missing_intersection_process", (1, 2, 3)), result["failures"]
        )

    def test_exhaustive_oracle_returns_only_locally_certified_sets(self):
        candidates = enumerate_certified_rejection_sets(
            self.active, 1, self.processes, self.alpha
        )
        self.assertIn((), candidates)
        for candidate in candidates:
            self.assertTrue(
                verify_local_closure(
                    self.active, candidate, 1, self.processes, self.alpha
                )["certified"]
            )
        self.assertEqual(len(candidates), len(set(candidates)))

    def test_finite_truth_path_is_dominated_by_prior_mixture(self):
        active_path = [(1,), (1, 2), (1, 2, 3)]
        rejection_path = [(), (1,), (1,)]
        result = verify_finite_truth_path(
            active_path, rejection_path, self.true, self.processes, self.alpha
        )
        self.assertTrue(result["valid"], result)
        self.assertTrue(all(row["dominated"] for row in result["path"]))
        self.assertLessEqual(result["sup_fdp"], result["sup_mixture_bound"])

    def test_random_true_intersection_is_included_without_coherence(self):
        # The active true-null set changes non-nestedly with the path. The
        # mixture uses fixed global intersection processes and no projectivity.
        active_path = [(1, 2), (2, 3), (1, 3)]
        rejection_path = [(1,), (2,), (3,)]
        result = verify_finite_truth_path(
            active_path, rejection_path, {1, 2, 3}, self.processes, self.alpha
        )
        self.assertTrue(result["valid"], result)

    def test_invalid_data_dependent_prior_selection_can_break_e_validity(self):
        # Each component is 2 with probability 1/2 and 0 otherwise.
        # Selecting the larger after observing both has expectation 3/2.
        values = [(2, 2), (2, 0), (0, 2), (0, 0)]
        fixed_mixture = [Fraction(x + y, 2) for x, y in values]
        selected_max = [max(x, y) for x, y in values]
        self.assertEqual(sum(fixed_mixture, Fraction(0)) / 4, Fraction(1))
        self.assertEqual(sum(selected_max, Fraction(0)) / 4, Fraction(3, 2))

    def test_prior_mass_over_one_is_rejected(self):
        good = verify_finite_prior_budget((1, 2))
        self.assertTrue(good["valid"], good)
        bad_prior = lambda _s: Fraction(1)
        bad = verify_finite_prior_budget((1, 2), bad_prior)
        self.assertFalse(bad["valid"])
        self.assertIn("prior_mass_budget_exceeded", [x[0] for x in bad["failures"]])

    def test_boundary_case_uses_exact_rational_comparison(self):
        # One hypothesis: FDP=1, alpha=1/2, pi=1/4, E=8 gives equality.
        one = {(1,): [Fraction(8)]}
        result = verify_local_closure((1,), (1,), 0, one, Fraction(1, 2))
        self.assertTrue(result["certified"], result)
        self.assertEqual(result["minimum_slack"], 0)

    def test_empty_rejection_has_zero_local_fdp(self):
        result = verify_local_closure(
            self.active, (), 0, self.processes, self.alpha
        )
        self.assertTrue(result["certified"], result)
        self.assertTrue(all(row["fdp"] == 0 for row in result["constraints"]))

    def test_invalid_rejection_outside_active_set_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "subset of the active family"):
            verify_local_closure(
                (1, 2), (3,), 0, self.processes, self.alpha
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
