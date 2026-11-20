import unittest
from exact_finite_horizon_oracle import (
    ACTIONS, HYPOTHESES, POLICIES, R_VALUES, one_step_information,
    oracle_value, predictive, run_benchmark, self_test,
)

class ExactFiniteHorizonOracleTests(unittest.TestCase):
    def test_reduced_hypothesis_space_size(self):
        self.assertEqual(len(HYPOTHESES), 16)

    def test_predictive_distribution_normalizes(self):
        prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
        for action in ACTIONS:
            self.assertAlmostEqual(sum(p for _, p in predictive(prior, action)), 1.0, places=12)

    def test_one_step_information_is_nonnegative(self):
        prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
        for action in ACTIONS:
            self.assertTrue(all(v >= 0.0 for v in one_step_information(prior, action).values()))

    def test_oracle_weakly_dominates_policies_at_horizon_two(self):
        result = run_benchmark(max_horizon=2)
        for horizon in (1, 2):
            rows = [r for r in result["rows"] if r["horizon"] == horizon]
            oracle = next(r for r in rows if r["policy"] == "exact_bellman_oracle")
            self.assertAlmostEqual(oracle["regret_bits_vs_exact_oracle"], 0.0, places=10)
            self.assertTrue(all(r["regret_bits_vs_exact_oracle"] >= 0.0 for r in rows))

    def test_bellman_value_nonincreasing_with_budget(self):
        prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
        self.assertLessEqual(oracle_value(prior, 2), oracle_value(prior, 1) + 1e-10)

    def test_policy_values_are_deterministic(self):
        self.assertEqual(run_benchmark(2), run_benchmark(2))

    def test_embedded_self_test(self):
        self.assertTrue(self_test())

if __name__ == "__main__":
    unittest.main()
