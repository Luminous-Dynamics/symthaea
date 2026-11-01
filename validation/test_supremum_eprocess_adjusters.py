import math
import unittest

from supremum_eprocess_adjusters import (
    adjusted_intersection_paths,
    adjusted_running_maximum_path,
    admissible_power_adjuster,
    admissible_power_adjuster_area,
    fixed_prior_adjuster_portfolio,
    running_maximum,
)


class SupremumAdjusterTests(unittest.TestCase):
    def test_power_adjuster_normalization_for_beta_grid(self):
        for beta in (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.99):
            self.assertAlmostEqual(admissible_power_adjuster_area(beta), 1.0, places=12)
            self.assertEqual(admissible_power_adjuster(1.0, beta), 0.0)

    def test_known_square_root_adjuster(self):
        for value in (1.0, 1.21, 4.0, 9.0, 100.0):
            self.assertAlmostEqual(
                admissible_power_adjuster(value, 0.5),
                math.sqrt(value) - 1.0,
                places=12,
            )

    def test_adjusted_running_maximum_is_nondecreasing(self):
        raw = [1.0, 0.5, 2.0, 1.5, 16.0, 4.0]
        self.assertEqual(running_maximum(raw), (1.0, 1.0, 2.0, 2.0, 16.0, 16.0))
        adjusted = adjusted_running_maximum_path(raw, 0.5)
        self.assertEqual(adjusted, tuple(sorted(adjusted)))
        self.assertAlmostEqual(adjusted[-1], 3.0)

    def test_fixed_prior_portfolio_is_nondecreasing_and_reproducible(self):
        raw = [1.0, 0.8, 1.2, 4.0, 2.0, 25.0]
        first = fixed_prior_adjuster_portfolio(raw)
        second = fixed_prior_adjuster_portfolio(raw)
        self.assertEqual(first, second)
        self.assertEqual(first, tuple(sorted(first)))
        self.assertEqual(first[0], 0.0)
        self.assertGreater(first[-1], 0.0)

    def test_intersection_paths_preserve_keys_and_horizons(self):
        raw = {
            frozenset({"a"}): [1.0, 2.0, 3.0],
            frozenset({"a", "b"}): [1.0, 1.0, 5.0],
        }
        adjusted = adjusted_intersection_paths(raw)
        self.assertEqual(set(adjusted), set(raw))
        self.assertTrue(all(len(path) == 3 for path in adjusted.values()))
        self.assertTrue(all(tuple(sorted(path)) == path for path in adjusted.values()))

    def test_rejects_invalid_e_process_paths_and_priors(self):
        for raw in ([], [0.0, 2.0], [2.0, 3.0], [1.0, -1.0], [1.0, math.inf]):
            with self.assertRaises(ValueError):
                adjusted_running_maximum_path(raw, 0.5)
        with self.assertRaises(ValueError):
            fixed_prior_adjuster_portfolio(
                [1.0, 2.0], betas=(0.2, 0.2), weights=(0.5, 0.5)
            )
        with self.assertRaises(ValueError):
            fixed_prior_adjuster_portfolio(
                [1.0, 2.0], betas=(0.2, 0.5), weights=(0.5, 0.4)
            )


if __name__ == "__main__":
    unittest.main()
