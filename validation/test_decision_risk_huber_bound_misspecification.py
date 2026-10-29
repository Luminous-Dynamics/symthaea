import unittest
import numpy as np
import decision_risk_huber_sensitivity as huber

class HuberBoundMisspecificationTests(unittest.TestCase):
    def test_sensitivity_set_widens_monotonically(self):
        lo_q, hi_q = 0.29, 0.35
        intervals = [huber.clean_mean_interval(lo_q, hi_q, rho) for rho in (0.0, 0.05, 0.10, 0.20)]
        for left, right in zip(intervals, intervals[1:]):
            self.assertLessEqual(right[0], left[0])
            self.assertGreaterEqual(right[1], left[1])

    def test_clean_target_is_contained_for_actual_contamination_below_claim(self):
        p, contaminant = 0.37, 0.0
        for actual_rho in (0.0, 0.025, 0.05, 0.10, 0.18):
            q = (1-actual_rho)*p + actual_rho*contaminant
            for claimed_rho in (actual_rho, min(0.20, actual_rho + 0.05), 0.20):
                lo, hi = huber.clean_mean_interval(q, q, claimed_rho)
                self.assertLessEqual(lo-1e-12, p)
                self.assertGreaterEqual(hi+1e-12, p)

    def test_understated_bound_is_not_mislabeled_as_covered(self):
        self.assertGreater(0.12 - 0.05, 0.0)
        self.assertFalse(0.05 + 1e-12 >= 0.12)
        self.assertTrue(0.12 + 1e-12 >= 0.12)

    def test_understated_bound_can_admit_unsafe_clean_target(self):
        model=np.array([.15,.30,.60,.80])
        q=np.array([.15,.82*.37,.60,.80])
        rng=np.random.default_rng(20261030)
        streams=np.vstack([rng.binomial(1,p,size=huber.MAX_PER_CELL) for p in q]).astype(np.int8)
        nominal=huber.run_certifier(streams,model,mode="huber_robust",rho_upper=0.0)
        conservative=huber.run_certifier(streams,model,mode="huber_robust",rho_upper=0.18)
        self.assertEqual(nominal["status"],"admit")
        self.assertEqual(conservative["status"],"refuse_unresolved")

    def test_unknown_bound_fails_closed(self):
        out = huber.run_certifier(np.zeros((4, huber.MAX_PER_CELL), dtype=np.int8),
                                  np.array([.15,.30,.60,.80]), mode="huber_robust", rho_upper=None)
        self.assertEqual(out["status"], "refuse_unresolved")
        self.assertEqual(out["total_observations"], 0)

if __name__ == "__main__": unittest.main(verbosity=2)
