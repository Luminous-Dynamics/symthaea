import unittest
import numpy as np

from constrained_prior_polytope_benchmark import (
    OBJECTIVES, ambiguity_polytope, maximize_linear_over_polytope,
    solve_constrained_minimax, run_benchmark,
)
from prior_set_coverage_double_oracle_benchmark import build_risk_tensor


class ConstrainedPriorPolytopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tensor, cls.priors, cls.trees = build_risk_tensor()
        cls.poly = ambiguity_polytope()

    def test_declared_polytope_has_feasible_witness(self):
        q = np.asarray(self.poly["feasible_witness"])
        self.assertAlmostEqual(float(q.sum()), 1.0, places=9)
        self.assertGreaterEqual(float(q.min()), self.poly["floor"] - 1e-9)
        self.assertLessEqual(float(np.max(self.poly["A_ub"] @ q - self.poly["b_ub"])), 1e-8)

    def test_world_oracle_matches_independent_objective(self):
        c = np.linspace(0.1, 1.6, 16)
        r = maximize_linear_over_polytope(c, self.poly)
        self.assertAlmostEqual(r["value_bits"], float(np.dot(c, r["prior"])), places=9)

    def test_constrained_game_closes_two_sided_interval(self):
        for objective in OBJECTIVES:
            A = self.tensor[objective]["risk_by_policy_and_world"]
            r = solve_constrained_minimax(A, self.poly)
            self.assertEqual(r["status"], "CERTIFIED_CONSTRAINED_POLYTOPE_GAME")
            self.assertLessEqual(r["gap_bits"], 2.1e-7)
            self.assertAlmostEqual(sum(r["policy_weights"]), 1.0, places=8)
            self.assertGreaterEqual(min(r["policy_weights"]), -1e-12)

    def test_result_grid_covers_all_objectives(self):
        r = run_benchmark()
        self.assertEqual([x["objective"] for x in r["rows"]], list(OBJECTIVES))
        self.assertTrue(all(x["constrained_polytope"]["status"] == "CERTIFIED_CONSTRAINED_POLYTOPE_GAME" for x in r["rows"]))

    def test_rejects_invalid_prior_floor(self):
        with self.assertRaises(ValueError):
            ambiguity_polytope(floor=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
