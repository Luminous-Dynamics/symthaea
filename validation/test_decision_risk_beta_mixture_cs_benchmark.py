import unittest
import numpy as np
import decision_risk_beta_mixture_cs_benchmark as b

class BetaMixtureConfidenceSequenceTests(unittest.TestCase):
    def test_evalue_is_minimized_near_empirical_mean(self):
        s, n = 52, 100
        m = s/n
        center = b.beta_mixture_log_evalue(m,s,n)
        self.assertLess(center,b.beta_mixture_log_evalue(m-0.1,s,n))
        self.assertLess(center,b.beta_mixture_log_evalue(m+0.1,s,n))

    def test_cs_contains_empirical_mean_and_narrows_with_more_observations(self):
        low1, high1 = b.beta_mixture_cs(50,100,alpha_cell=0.025)
        low2, high2 = b.beta_mixture_cs(500,1000,alpha_cell=0.025)
        self.assertLessEqual(low1,0.5); self.assertGreaterEqual(high1,0.5)
        self.assertLessEqual(low2,0.5); self.assertGreaterEqual(high2,0.5)
        self.assertLess(high2-low2, high1-low1)

    def test_zero_and_one_success_boundaries(self):
        lo0, hi0=b.beta_mixture_cs(0,128,alpha_cell=0.025)
        lon, hin=b.beta_mixture_cs(128,128,alpha_cell=0.025)
        self.assertEqual(lo0,0.0); self.assertEqual(hin,1.0)
        self.assertLess(hi0,0.2); self.assertGreater(lon,0.8)

    def test_finite_run_tracks_admission_and_false_admission(self):
        out=b.run(repeats=100,sample_sizes=(4096,8192,16384,32768))
        clean=out['results']['correct_model']
        self.assertEqual(clean['beta_mixture_e_process_cs']['admission_rate'],1.0)
        over=out['results']['clearly_over_regret_budget']
        self.assertEqual(over['beta_mixture_e_process_cs']['false_admission_rate'],0.0)
        self.assertEqual(out['protocol']['cells'],4)

if __name__=='__main__': unittest.main(verbosity=2)
