import math
import unittest

from epsilon_frontier_approximation_benchmark import (
    prune_frontier_epsilon,
    make_epsilon_frontier_solver,
    run_benchmark,
    self_test,
)
from fixed_prior_minimax_policy_benchmark import prune_frontier
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior


class EpsilonFrontierApproximationTests(unittest.TestCase):
    def test_zero_epsilon_is_exact_pruning(self):
        candidates = [((1.0, 2.0), "a"), ((0.5, 2.5), "b"), ((1.2, 2.2), "c")]
        self.assertEqual(prune_frontier_epsilon(candidates, 0.0), prune_frontier(candidates))

    def test_epsilon_pruning_returns_a_nondominated_frontier(self):
        candidates = [((0.101, 0.201), "a"), ((0.104, 0.204), "b"), ((0.201, 0.101), "c"), ((0.5, 0.5), "d")]
        result = prune_frontier_epsilon(candidates, 0.01)
        self.assertFalse(any(a[0] != b[0] and all(x <= y + 1e-12 for x, y in zip(a[0], b[0])) and any(x < y - 1e-12 for x, y in zip(a[0], b[0])) for a in result for b in result))

    def test_negative_epsilon_fails_closed(self):
        with self.assertRaises(ValueError):
            make_epsilon_frontier_solver("joint", ("uniform",), -0.1)

    def test_unknown_objective_fails_closed(self):
        with self.assertRaises(ValueError):
            make_epsilon_frontier_solver("unknown", ("uniform",), 0.01)

    def test_prior_candidates_normalized(self):
        for name in PRIOR_NAMES:
            prior = build_prior(name)
            self.assertTrue(all(p > 0.0 for p in prior))
            self.assertAlmostEqual(sum(prior), 1.0, places=12)

    def test_full_small_grid_has_valid_exact_gap_certificate(self):
        result = run_benchmark(("uniform", "high_spill_biased"), ("uniform", "chain_biased"),
                               ("joint",), 2, (0.0, 0.025))
        self.assertEqual(len(result["rows"]), 8)
        for row in result["rows"]:
            self.assertGreaterEqual(row["worst_case_optimality_gap_bits"], -1e-10)
            self.assertTrue(math.isfinite(row["approximate_policy_worst_case_ce_bits"]))

    def test_horizon_one_has_zero_minimax_gap_in_small_grid(self):
        result = run_benchmark(("uniform", "high_spill_biased"), ("uniform", "chain_biased"),
                               ("joint",), 1, (0.0, 0.05))
        gaps = [r["worst_case_optimality_gap_bits"] for r in result["rows"]]
        self.assertTrue(all(abs(gap) < 1e-8 for gap in gaps))

    def test_approximation_can_reduce_frontier_size(self):
        result = run_benchmark(("uniform", "chain_biased", "fork_biased"),
                               ("uniform",), ("kernel",), 2, (0.0, 0.05))
        exact = next(r for r in result["rows"] if r["epsilon_bits"] == 0.0 and r["horizon"] == 2)
        approx = next(r for r in result["rows"] if r["epsilon_bits"] == 0.05 and r["horizon"] == 2)
        self.assertLessEqual(approx["frontier_size_approx"], exact["frontier_size_exact"])

    def test_embedded_self_test(self):
        self.assertTrue(self_test())


if __name__ == "__main__":
    unittest.main()
