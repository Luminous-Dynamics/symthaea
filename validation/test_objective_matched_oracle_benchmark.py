import unittest
from objective_matched_oracle_benchmark import (
    ACTIONS, HYPOTHESES, OBJECTIVES, POLICIES, information_gain,
    predictive, run_benchmark, self_test,
)

class ObjectiveMatchedOracleTests(unittest.TestCase):
    def test_hypothesis_space_and_observation_normalization(self):
        self.assertEqual(len(HYPOTHESES), 16)
        prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
        for action in ACTIONS:
            self.assertAlmostEqual(sum(p for _, p in predictive(prior, action)), 1.0, places=12)

    def test_each_objective_has_nonnegative_information_gain(self):
        prior = tuple([1.0 / len(HYPOTHESES)] * len(HYPOTHESES))
        for action in ACTIONS:
            for objective in OBJECTIVES:
                self.assertGreaterEqual(information_gain(prior, action, objective), 0.0)

    def test_matching_oracle_has_zero_regret_for_every_objective(self):
        result = run_benchmark(2)
        for horizon in (1, 2):
            for objective in OBJECTIVES:
                row = next(r for r in result['rows'] if r['horizon'] == horizon and r['policy'] == f'oracle_{objective}' and r['scored_objective'] == objective)
                self.assertAlmostEqual(row['regret_bits_vs_objective_matched_oracle'], 0.0, places=10)

    def test_no_policy_beats_objective_matched_oracle(self):
        result = run_benchmark(2)
        self.assertTrue(all(r['regret_bits_vs_objective_matched_oracle'] >= 0.0 for r in result['rows']))

    def test_cross_objective_policies_are_scored_on_every_objective(self):
        result = run_benchmark(1)
        self.assertEqual(len(result['rows']), len(POLICIES) * len(OBJECTIVES))
        for policy in POLICIES:
            self.assertEqual({r['scored_objective'] for r in result['rows'] if r['policy'] == policy}, set(OBJECTIVES))

    def test_horizon_three_regret_nonnegative_and_oracle_zero(self):
        result = run_benchmark(3)
        self.assertTrue(all(r['regret_bits_vs_objective_matched_oracle'] >= 0.0 for r in result['rows']))
        for objective in OBJECTIVES:
            row = next(r for r in result['rows'] if r['horizon'] == 3 and r['policy'] == f'oracle_{objective}' and r['scored_objective'] == objective)
            self.assertAlmostEqual(row['regret_bits_vs_objective_matched_oracle'], 0.0, places=10)

    def test_results_deterministic(self):
        self.assertEqual(run_benchmark(2), run_benchmark(2))

    def test_embedded_self_test(self):
        self.assertTrue(self_test())

if __name__ == '__main__':
    unittest.main()
