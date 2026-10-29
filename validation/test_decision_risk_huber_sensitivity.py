import unittest
import numpy as np
import decision_risk_huber_sensitivity as b

class HuberSensitivityTests(unittest.TestCase):
    def test_clean_mean_interval_contains_huber_clean_target(self):
        # q=(1-rho)*p + rho*R, with R arbitrary in [0,1].
        for p in (0.0, 0.15, 0.37, 0.8, 1.0):
            for rho in (0.0, 0.01, 0.1, 0.4, 0.8):
                for contam in (0.0, 0.25, 0.75, 1.0):
                    actual_rho = rho * 0.8
                    q = (1-actual_rho)*p + actual_rho*contam
                    lo, hi = b.clean_mean_interval(q, q, rho)
                    self.assertLessEqual(lo-1e-12, p)
                    self.assertGreaterEqual(hi+1e-12, p)

    def test_invalid_contamination_bound_fails_validation(self):
        with self.assertRaises(ValueError): b.clean_mean_interval(0.2,0.3,1.0)

    def test_integrity_preflight_fails_closed(self):
        self.assertFalse(b.preflight(support_covered=False,identity_matches=True,contamination_bound_known=True))
        self.assertFalse(b.preflight(support_covered=True,identity_matches=False,contamination_bound_known=True))
        self.assertFalse(b.preflight(support_covered=True,identity_matches=True,contamination_bound_known=False))
        self.assertTrue(b.preflight(support_covered=True,identity_matches=True,contamination_bound_known=True))

    def test_unknown_bound_cannot_certify(self):
        streams=np.zeros((2,b.MAX_PER_CELL),dtype=np.int8)
        result=b.run_certifier(streams,np.array([0.0,0.0]),mode='huber_robust',rho_upper=None)
        self.assertEqual(result['status'],'refuse_unresolved')
        self.assertEqual(result['total_observations'],0)

if __name__=='__main__': unittest.main(verbosity=2)
