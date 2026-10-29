import unittest
import numpy as np
from groupwise_robust_coverage_benchmark import score_radius, paired_bootstrap_mean_ci, nested_estimates, max_group_radius, run
from groupwise_robust_coverage_benchmark import feature_matrix, meta_alphas, conformal_quantile

class GroupwiseRobustCoverageTests(unittest.TestCase):
    def test_conformal_quantile_bounds_and_order(self):
        scores=[0.1,0.2,0.3,0.4]
        self.assertEqual(conformal_quantile(scores,0.5),0.3)
        self.assertEqual(conformal_quantile(scores,0.99),1.0)

    def test_group_max_radius_dominates_each_group_radius(self):
        rng=np.random.default_rng(10); C=feature_matrix(); alphas=meta_alphas(); samples=[]
        for r,a in enumerate(alphas):
            for _ in range(8):
                q=rng.dirichlet(a); samples.append({'q':q,'regime':r,'est':nested_estimates(rng,q,(12,))})
        radii=[score_radius(samples,r,12,C,0.8) for r in range(len(alphas))]
        rmax=max_group_radius({i:x for i,x in enumerate(radii)})
        self.assertEqual(rmax,max(radii))
        self.assertTrue(all(rmax>=x for x in radii))
        with self.assertRaises(ValueError): max_group_radius({})

    def test_paired_bootstrap_ci_contains_fixed_identical_delta(self):
        ci=paired_bootstrap_mean_ci(np.zeros(20),seed=2,reps=200)
        self.assertEqual(ci,[0.0,0.0])

    def test_smoke_run_has_methods_and_finite_certificates(self):
        # Small smoke fixture; the full experiment is run separately for the report.
        out=run(n_cal_per_regime=8,n_test_per_regime=2,seed=77)
        self.assertEqual(len(out['settings']),6)
        for s in out['settings']:
            self.assertEqual(set(s['methods']),{'pooled_global','group_max_common_radius','regime_conditional_oracle','unrestricted_full_simplex'})
            self.assertAlmostEqual(s['radii']['group_max_common_radius'],max(s['radii']['group_radii_by_regime'].values()))
            for m in s['methods'].values():
                self.assertTrue(np.isfinite(m['mean_regret_bits']))
                self.assertLess(m['max_certificate_gap_bits'],1e-5)

if __name__=='__main__': unittest.main()
