import math
import unittest

from epsilon_frontier_approximation_benchmark import (
    make_epsilon_frontier_solver,
    prune_frontier_epsilon,
    run_benchmark,
    self_test,
)
from fixed_prior_minimax_policy_benchmark import prune_frontier
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior


class EpsilonFrontierApproximationTests(unittest.TestCase):
    def test_prior_candidates_normalized(self):
        for name in PRIOR_NAMES:
            p = build_prior(name)
            self.assertTrue(all(x > 0 for x in p))
            self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_epsilon_pruning_returns_a_nondominated_frontier(self):
        candidates = [
            ((0.1, 0.2), "a"),
            ((0.2, 0.1), "b"),
            ((0.3, 0.3), "dominated"),
            ((0.11, 0.21), "near-a"),
        ]
        frontier = prune_frontier_epsilon(candidates, 0.05)
        self.assertTrue(frontier)
        for i, a in enumerate(frontier):
            for j, b in enumerate(frontier):
                if i != j:
                    self.assertFalse(all(x <= y + 1e-12 for x, y in zip(a[0], b[0])) and
                                     any(x < y - 1e-12 for x, y in zip(a[0], b[0])))

    def test_approximation_can_reduce_frontier_size(self):
        candidates = [
            ((0.101, 0.201, 0.301), "a"),
            ((0.104, 0.204, 0.304), "near-a"),
            ((0.201, 0.101, 0.301), "b"),
            ((0.301, 0.301, 0.101), "c"),
        ]
        exact = prune_frontier(candidates)
        approx = prune_frontier_epsilon(candidates, 0.01)
        self.assertLessEqual(len(approx), len(exact))

    def test_full_small_grid_has_valid_exact_gap_certificate(self):
        result = run_benchmark(
            ("uniform", "high_spill_biased"),
            ("uniform", "chain_biased"),
            ("joint",),
            2,
            (0.0, 0.025),
        )
        self.assertEqual(len(result["rows"]), 8)
        self.assertTrue(all(row["worst_case_optimality_gap_bits"] >= -1e-9
                            for row in result["rows"]))

    def test_horizon_one_has_zero_minimax_gap_in_small_grid(self):
        result = run_benchmark(
            ("uniform", "high_spill_biased"),
            ("uniform", "chain_biased"),
            ("joint",),
            1,
            (0.0, 0.025),
        )
        self.assertTrue(all(abs(row["worst_case_optimality_gap_bits"]) < 1e-8
                            for row in result["rows"]))

    def test_zero_epsilon_is_exact_pruning(self):
        candidates = [((0.1, 0.2), "a"), ((0.2, 0.1), "b"), ((0.4, 0.4), "c")]
        self.assertEqual(prune_frontier_epsilon(candidates, 0.0), prune_frontier(candidates))

    def test_negative_epsilon_fails_closed(self):
        with self.assertRaises(ValueError):
            make_epsilon_frontier_solver("joint", ("uniform",), -0.1)

    def test_unknown_objective_fails_closed(self):
        with self.assertRaises(ValueError):
            make_epsilon_frontier_solver("unknown", ("uniform",), 0.1)

    def test_embedded_self_test(self):
        self.assertTrue(self_test())


if __name__ == "__main__":
    unittest.main()
