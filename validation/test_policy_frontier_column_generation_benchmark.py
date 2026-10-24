"""Invariant tests for certified minimax policy-tree column generation."""
import math
import unittest

from objective_matched_oracle_benchmark import OBJECTIVES
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior
from fixed_prior_minimax_policy_benchmark import evaluate_tree_from_posteriors
from policy_frontier_column_generation_benchmark import (
    make_pricing_oracle, solve_column_generation, run_benchmark,
)


class PolicyFrontierColumnGenerationTests(unittest.TestCase):
    def test_rejects_unknown_objective(self):
        with self.assertRaises(ValueError):
            make_pricing_oracle("unknown", ("uniform",))

    def test_rejects_empty_candidate_set(self):
        with self.assertRaises(ValueError):
            make_pricing_oracle("joint", ())

    def test_rejects_invalid_weights(self):
        pricing = make_pricing_oracle("joint", ("uniform", "high_spill_biased"))
        with self.assertRaises(ValueError):
            pricing(tuple(build_prior("uniform")), 1, (1.0,))
        with self.assertRaises(ValueError):
            pricing(tuple(build_prior("uniform")), 1, (0.0, 0.0))
        with self.assertRaises(ValueError):
            pricing(tuple(build_prior("uniform")), 1, (-1.0, 2.0))

    def test_pricing_tree_has_independent_risk_vector(self):
        candidates = ("uniform", "high_spill_biased")
        learner = tuple(build_prior("chain_biased"))
        weights = (0.3, 0.7)
        result = make_pricing_oracle("joint", candidates)(learner, 2, weights)
        scores = tuple(
            evaluate_tree_from_posteriors(result["tree"], tuple(build_prior(n)), learner, "joint", 2)
            for n in candidates
        )
        self.assertTrue(all(math.isclose(a, b, abs_tol=1e-8) for a, b in zip(scores, result["risk_vector"])))
        scalar = sum(w * x for w, x in zip(weights, scores)) / sum(weights)
        self.assertTrue(math.isclose(scalar, result["scalarized_risk_bits"], abs_tol=1e-8))

    def test_all_objectives_pricing_yields_finite_risk(self):
        learner = tuple(build_prior("uniform"))
        for objective in OBJECTIVES:
            result = make_pricing_oracle(objective, ("uniform", "high_spill_biased"))(learner, 1, (0.5, 0.5))
            self.assertTrue(all(math.isfinite(x) for x in result["risk_vector"]))
            self.assertGreaterEqual(result["scalarized_risk_bits"], 0.0)

    def test_column_generation_matches_exhaustive_oracle(self):
        result = run_benchmark(("uniform", "high_spill_biased"),
                               ("uniform", "chain_biased"), ("joint",), (1, 2))
        self.assertEqual(len(result["rows"]), 4)
        for row in result["rows"]:
            cg = row["column_generation"]
            self.assertTrue(cg["converged"])
            self.assertLessEqual(cg["primal_dual_certificate_gap_bits"], 2e-8)
            self.assertLessEqual(abs(row["column_generation_minus_exact_bits"]), 2e-7)

    def test_horizon_two_kernel_frontier_compression(self):
        row = solve_column_generation("kernel", "uniform", PRIOR_NAMES, 2)
        _, frontier, _, _ = __import__("fixed_prior_minimax_policy_benchmark").make_frontier_solver("kernel", PRIOR_NAMES)
        exact_count = len(frontier(tuple(build_prior("uniform")), tuple(tuple(build_prior(n)) for n in PRIOR_NAMES), 2))
        self.assertLessEqual(row["policy_tree_columns"], 2)
        self.assertGreaterEqual(exact_count, row["policy_tree_columns"])
        self.assertTrue(row["converged"])

    def test_horizon_three_kernel_certificate_without_frontier_enumeration(self):
        row = solve_column_generation("kernel", "graph_kernel_correlated", PRIOR_NAMES, 3)
        self.assertTrue(row["converged"])
        self.assertEqual(row["status"], "CERTIFIED_COLUMN_GENERATION")
        self.assertLessEqual(row["primal_dual_certificate_gap_bits"], 2e-8)
        self.assertLessEqual(row["policy_tree_columns"], 3)

    def test_invalid_horizon_and_tolerance_fail_closed(self):
        with self.assertRaises(ValueError):
            solve_column_generation("joint", "uniform", ("uniform",), 4)
        with self.assertRaises(ValueError):
            solve_column_generation("joint", "uniform", ("uniform",), 1, tolerance=0.0)


if __name__ == "__main__":
    unittest.main()
