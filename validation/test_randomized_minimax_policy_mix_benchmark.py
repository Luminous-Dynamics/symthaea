import math
import unittest
from randomized_minimax_policy_mix_benchmark import solve_minimax_mixture, run_benchmark, self_test
from prior_shift_oracle_benchmark import PRIOR_NAMES, build_prior

class RandomizedMinimaxPolicyMixTests(unittest.TestCase):
    def test_simple_symmetric_game_mixes(self):
        r=solve_minimax_mixture([[0,1],[1,0]])
        self.assertAlmostEqual(r['worst_case_bits'],0.5,places=8)
        self.assertEqual(r['support_size'],2)
    def test_lp_weights_sum_to_one(self):
        r=solve_minimax_mixture([[0,2],[1,3],[2,0]])
        self.assertAlmostEqual(sum(r['weights']),1.0,places=9)
    def test_primal_dual_certificate_closes(self):
        r=solve_minimax_mixture([[0,1],[1,0],[0.6,0.6]])
        self.assertAlmostEqual(sum(r['scenario_dual_weights']),1.0,places=8)
        self.assertLessEqual(r['primal_dual_gap_bits'],1e-7)
        self.assertLessEqual(r['dual_lower_bound_bits'],r['worst_case_bits']+1e-8)
    def test_mixture_never_worse_than_deterministic_frontier(self):
        r=run_benchmark(('uniform','high_spill_biased'),('uniform','chain_biased'),('joint',),(1,2))
        for row in r['rows']:
            self.assertLessEqual(row['randomized_minimax']['worst_case_bits'],row['deterministic_worst_case_bits']+1e-8)
    def test_full_declared_prior_support_normalizes(self):
        for n in PRIOR_NAMES:
            p=build_prior(n); self.assertTrue(all(x>0 for x in p)); self.assertAlmostEqual(sum(p),1.0,places=12)
    def test_finite_scores(self):
        r=run_benchmark(('uniform','high_spill_biased'),('uniform',),('joint',),(1,))
        self.assertTrue(all(math.isfinite(x['randomized_minimax']['worst_case_bits']) for x in r['rows']))
    def test_invalid_risk_matrix_fails_closed(self):
        with self.assertRaises(ValueError): solve_minimax_mixture([])
    def test_nonfinite_scores_fail_closed(self):
        with self.assertRaises(ValueError): solve_minimax_mixture([[math.inf,0]])
    def test_embedded_self_test(self): self.assertTrue(self_test())

if __name__=='__main__': unittest.main()
