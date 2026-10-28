import unittest
import numpy as np
import groupwise_robust_coverage_benchmark as g
from decision_sufficient_ambiguity_benchmark import make_region, included, quantile_for, run

class DecisionSufficientAmbiguityTests(unittest.TestCase):
    def test_risk_space_region_is_centered_on_policy_risks(self):
        risk=np.asarray([[0.,1.,2.],[2.,1.,0.]])
        q=np.asarray([0.2,0.3,0.5])
        region=make_region(q,0.1,risk)
        self.assertEqual(region['A_ub'].shape,(4,3))
        self.assertTrue(included(q,risk,region))
        outside=np.asarray([0.8,0.1,0.1])
        self.assertFalse(included(outside,risk,region))

    def test_unbounded_region_admits_any_simplex_prior(self):
        geom=np.asarray([[1.,0.,1.]])
        region=make_region(np.asarray([0.2,0.3,0.5]),0.1,geom,unbounded=True)
        self.assertTrue(included(np.asarray([0.,1.,0.]),geom,region))

    def test_group_quantile_is_at_least_each_regime_quantile(self):
        rng=np.random.default_rng(4); alphas=g.meta_alphas(); samples=g.balanced_samples(rng,alphas,8,(12,)); geom=g.feature_matrix()
        rs=[quantile_for(samples,i,12,geom,0.8) for i in range(6)]
        rmax=max(rs)
        self.assertTrue(all(rmax >= r for r in rs))

    def test_small_run_emits_paired_methods_and_numerical_certificates(self):
        out=run(8,2,77,budgets=(60,120),targets=(0.9,))
        self.assertEqual(len(out['settings']),2)
        for s in out['settings']:
            self.assertEqual(set(s['methods']),{'feature_pooled','feature_group_max','risk_pooled','risk_group_max','full_prior'})
            self.assertGreaterEqual(s['methods']['risk_pooled']['coverage']['rate'],0.0)
            for v in s['methods'].values():
                self.assertTrue(np.isfinite(v['mean_regret_bits']))
                self.assertLess(v['max_certificate_gap_bits'],1e-5)

if __name__=='__main__': unittest.main()
