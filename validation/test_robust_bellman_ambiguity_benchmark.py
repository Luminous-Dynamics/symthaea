import math
import unittest

from robust_bellman_ambiguity_benchmark import (
    make_bellman_solver, eval_policy, run_benchmark, self_test,
)
from objective_matched_oracle_benchmark import ACTIONS, HYPOTHESES
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior


class RobustBellmanAmbiguityTests(unittest.TestCase):
    def test_prior_candidates_normalize(self):
        for name in PRIOR_NAMES:
            p = build_prior(name)
            self.assertEqual(len(p), len(HYPOTHESES))
            self.assertTrue(all(x > 0 for x in p))
            self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_bellman_action_and_value_are_finite(self):
        scenarios, value, action = make_bellman_solver("joint", ("uniform", "high_spill_biased"))
        p = tuple(build_prior("uniform"))
        self.assertIn(action(p, scenarios, 1), ACTIONS)
        self.assertTrue(math.isfinite(value(p, scenarios, 2)))

    def test_horizon_one_matches_one_step_minimax(self):
        a = eval_policy("robust_bellman_rectangular", "joint", "uniform", "uniform", horizon=1)
        b = eval_policy("robust_minimax_one_step", "joint", "uniform", "uniform", horizon=1)
        self.assertAlmostEqual(a["expected_terminal_cross_entropy_bits"],
                               b["expected_terminal_cross_entropy_bits"], places=9)

    def test_full_small_grid_row_count_and_finite_scores(self):
        result = run_benchmark(1, ("uniform", "high_spill_biased"), ("uniform", "chain_biased"),
                               ("uniform", "high_spill_biased", "chain_biased"),
                               ("robust_bellman_rectangular", "robust_minimax_one_step", "greedy_joint",
                                "round_robin", "random_uniform"), ("joint",))
        self.assertEqual(len(result["rows"]), 20)
        self.assertTrue(all(math.isfinite(row["expected_terminal_cross_entropy_bits"]) for row in result["rows"]))

    def test_prior_label_does_not_change_bellman_action_function(self):
        scenarios, _, action = make_bellman_solver("joint", ("uniform", "high_spill_biased"))
        p = tuple(build_prior("uniform"))
        self.assertEqual(action(p, scenarios, 2), action(p, scenarios, 2))

    def test_unknown_objective_fails_closed(self):
        with self.assertRaises(ValueError):
            make_bellman_solver("unknown", ("uniform",))

    def test_unknown_policy_fails_closed(self):
        with self.assertRaises(ValueError):
            eval_policy("unknown", "joint", "uniform", "uniform")

    def test_horizon_validation_fails_closed(self):
        with self.assertRaises(ValueError):
            run_benchmark(4)

    def test_embedded_self_test(self):
        self.assertTrue(self_test())


if __name__ == "__main__":
    unittest.main()
