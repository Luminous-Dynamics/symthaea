import unittest
import numpy as np

from prior_set_coverage_double_oracle_benchmark import (
    OBJECTIVES, TOL, build_risk_tensor, candidate_priors, point_mass,
    prior_risk_matrix, run_benchmark, solve_double_oracle, solve_zero_sum,
)


class PriorCoverageDoubleOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensor, cls.priors, cls.trees = build_risk_tensor()

    def test_risk_is_linear_for_named_priors_all_objectives(self):
        for objective in OBJECTIVES:
            errors = self.tensor[objective]["linearity_checks"]
            self.assertLess(max(errors.values()), 2e-9)

    def test_full_simplex_value_not_below_six_scenario_value(self):
        for objective in OBJECTIVES:
            A = self.tensor[objective]["risk_by_policy_and_world"]
            six = solve_zero_sum(prior_risk_matrix(A, list(self.priors.values())))
            full = solve_zero_sum(A)
            self.assertGreaterEqual(full["value_bits"] + 2e-7, six["value_bits"])

    def test_world_lp_uses_extreme_vertex(self):
        for objective in OBJECTIVES:
            A = self.tensor[objective]["risk_by_policy_and_world"]
            rng = np.random.default_rng(909)
            mu = rng.dirichlet(np.ones(A.shape[0]))
            per_world = A.T @ mu
            for _ in range(100):
                q = rng.dirichlet(np.ones(A.shape[1]))
                self.assertLessEqual(float(np.dot(q, per_world)), float(np.max(per_world)) + 1e-9)

    def test_double_oracle_matches_full_simplex_exact_game(self):
        for objective in OBJECTIVES:
            A = self.tensor[objective]["risk_by_policy_and_world"]
            do = solve_double_oracle(A, self.priors, tolerance=TOL)
            full = solve_zero_sum(A)
            self.assertEqual(do["status"], "CERTIFIED_FINITE_GAME")
            self.assertLessEqual(do["certificate_gap_bits"], TOL + 2e-7)
            self.assertLessEqual(abs(do["robust_upper_bits"] - full["value_bits"]), 2e-7)
            self.assertLessEqual(abs(do["robust_lower_bits"] - full["value_bits"]), 2e-7)
            for step in do["history"]:
                self.assertLessEqual(step["robust_lower_bits"], full["value_bits"] + 2e-7)
                self.assertGreaterEqual(step["robust_upper_bits"], full["value_bits"] - 2e-7)

    def test_full_grid_has_three_objectives_and_certificate(self):
        result = run_benchmark()
        self.assertEqual(len(result["rows"]), 3)
        for row in result["rows"]:
            self.assertEqual(row["double_oracle"]["status"], "CERTIFIED_FINITE_GAME")
            self.assertLessEqual(row["double_oracle"]["certificate_gap_bits"], TOL + 2e-7)

    def test_rejects_invalid_zero_sum_input(self):
        with self.assertRaises(ValueError):
            solve_zero_sum(np.asarray([[]]))

    def test_point_mass_is_normalized(self):
        for index in range(16):
            q = point_mass(index)
            self.assertEqual(sum(q), 1.0)
            self.assertEqual(q[index], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
