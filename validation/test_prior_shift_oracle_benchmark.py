import unittest
from prior_shift_oracle_benchmark import (
    PRIOR_NAMES, build_prior, run_benchmark, self_test,
)
from objective_matched_oracle_benchmark import HYPOTHESES, OBJECTIVES, POLICIES

class PriorShiftOracleTests(unittest.TestCase):
    def test_prior_variants_are_positive_and_normalized(self):
        for name in PRIOR_NAMES:
            prior = build_prior(name)
            self.assertEqual(len(prior), len(HYPOTHESES))
            self.assertTrue(all(p > 0 for p in prior))
            self.assertAlmostEqual(sum(prior), 1.0, places=12)

    def test_priors_are_not_all_identical(self):
        priors = [build_prior(name) for name in PRIOR_NAMES]
        self.assertEqual(len(set(priors)), len(PRIOR_NAMES))

    def test_each_matched_oracle_has_zero_regret(self):
        out = run_benchmark(2, prior_names=("uniform", "graph_kernel_correlated"))
        for prior in ("uniform", "graph_kernel_correlated"):
            for horizon in (1, 2):
                for objective in OBJECTIVES:
                    row = next(r for r in out['rows'] if r['prior']==prior and r['horizon']==horizon and r['policy']==f'oracle_{objective}' and r['scored_objective']==objective)
                    self.assertAlmostEqual(row['regret_bits_vs_matched_oracle'], 0.0, places=10)

    def test_no_policy_has_negative_regret(self):
        out = run_benchmark(2, prior_names=("chain_biased", "fork_biased"))
        self.assertTrue(all(r['regret_bits_vs_matched_oracle'] >= 0.0 for r in out['rows']))

    def test_horizon_three_prior_shift_changes_scored_rows(self):
        out = run_benchmark(3, prior_names=("uniform", "chain_biased"))
        self.assertEqual(len(out['rows']), 2 * 3 * len(POLICIES) * len(OBJECTIVES))
        uniform = next(r for r in out['rows'] if r['prior']=='uniform' and r['horizon']==3 and r['policy']=='greedy_graph' and r['scored_objective']=='graph')
        biased = next(r for r in out['rows'] if r['prior']=='chain_biased' and r['horizon']==3 and r['policy']=='greedy_graph' and r['scored_objective']=='graph')
        self.assertNotAlmostEqual(uniform['initial_entropy_bits'], biased['initial_entropy_bits'], places=5)

    def test_embedded_self_test(self):
        self.assertTrue(self_test())

if __name__ == '__main__':
    unittest.main()
