import math
import unittest

from fixed_prior_minimax_policy_benchmark import (
    dominates, prune_frontier, make_frontier_solver, run_benchmark, self_test,
)
from objective_matched_oracle_benchmark import HYPOTHESES
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior


class FixedPriorMinimaxFrontierTests(unittest.TestCase):
    def test_candidate_priors_are_positive_and_normalized(self):
        for name in PRIOR_NAMES:
            p = build_prior(name)
            self.assertEqual(len(p), len(HYPOTHESES))
            self.assertTrue(all(x > 0 for x in p))
            self.assertAlmostEqual(sum(p), 1.0, places=12)

    def test_risk_vector_dominance(self):
        self.assertTrue(dominates((0.1, 0.2, 0.3), (0.2, 0.2, 0.4)))
        self.assertFalse(dominates((0.2, 0.2, 0.4), (0.1, 0.2, 0.3)))
        self.assertFalse(dominates((0.1, 0.3), (0.2, 0.2)))

    def test_pareto_pruning_removes_dominated_vectors(self):
        frontier = prune_frontier([
            ((1.0, 1.0), (0, ())),
            ((0.5, 1.2), (1, ())),
            ((1.2, 1.2), (2, ())),
        ])
        self.assertEqual(len(frontier), 2)
        self.assertFalse(any(dominates(a[0], b[0]) for a in frontier for b in frontier if a is not b))

    def test_frontier_exact_risk_vector_has_one_component_per_prior(self):
        scenarios, frontier, best, _ = make_frontier_solver("joint", ("uniform", "high_spill_biased"))
        vector, tree = best(tuple(build_prior("uniform")), 2)
        self.assertEqual(len(vector), 2)
        self.assertIsNotNone(tree)
        self.assertTrue(all(math.isfinite(x) for x in vector))
        self.assertTrue(frontier(tuple(build_prior("uniform")), scenarios, 2))

    def test_full_small_grid_counts_rows_and_scores(self):
        result = run_benchmark(1, ("uniform", "high_spill_biased"),
                               ("uniform", "chain_biased"), ("joint",))
        self.assertEqual(len(result["rows"]), 20)
        self.assertTrue(all(math.isfinite(r["selected_policy_cross_entropy_bits"]) for r in result["rows"]))

    def test_horizon_two_frontier_minimizes_worst_case_against_baselines(self):
        result = run_benchmark(2, ("uniform", "high_spill_biased"),
                               ("uniform", "chain_biased"), ("joint",))
        rows = result["rows"]
        for learner in ("uniform", "chain_biased"):
            for horizon in (1, 2):
                subset = [r for r in rows if r["learner_prior"] == learner and r["horizon"] == horizon]
                robust = next(r for r in subset if r["policy"] == "fixed_prior_minimax_frontier")
                for baseline in ("robust_minimax_one_step", "greedy_joint", "round_robin", "random_uniform"):
                    vals = [r["selected_policy_cross_entropy_bits"] for r in subset if r["policy"] == baseline]
                    self.assertLessEqual(robust["worst_case_cross_entropy_bits"], max(vals) + 1e-8)

    def test_frontier_does_not_accept_true_prior_as_an_action_input(self):
        _, _, best, _ = make_frontier_solver("joint", ("uniform", "high_spill_biased"))
        p = tuple(build_prior("uniform"))
        self.assertEqual(best(p, 2), best(p, 2))

    def test_invalid_horizon_fails_closed(self):
        with self.assertRaises(ValueError):
            run_benchmark(4)

    def test_invalid_objective_fails_closed(self):
        with self.assertRaises(ValueError):
            make_frontier_solver("unknown", ("uniform",))

    def test_embedded_self_test(self):
        self.assertTrue(self_test())


if __name__ == "__main__":
    unittest.main()
