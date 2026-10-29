import unittest
import numpy as np

from calibrated_prior_ambiguity_benchmark import (
    TARGET_COVERAGE, calibrated_polytope, conformal_radius,
    feature_matrix, feature_radius, meta_regime_alphas,
    run_benchmark,
)


class CalibratedPriorAmbiguityTests(unittest.TestCase):
    def test_split_conformal_order_statistic(self):
        self.assertEqual(conformal_radius([0.1, 0.2, 0.3, 0.4], 0.5), 0.3)
        self.assertEqual(conformal_radius([0.1, 0.2], 0.99), 1.0)

    def test_prior_center_is_in_its_calibrated_polytope(self):
        q = np.ones(16) / 16
        Q = calibrated_polytope(q, 0.1)
        self.assertLessEqual(float(np.max(Q["A_ub"] @ q - Q["b_ub"])), 1e-9)

    def test_feature_score_is_nonnegative_and_zero_for_same_prior(self):
        q = np.ones(16) / 16
        self.assertEqual(feature_radius(q, q), 0.0)
        self.assertGreaterEqual(feature_matrix().shape[0], 10)

    def test_meta_regimes_are_strictly_positive(self):
        for alpha in meta_regime_alphas():
            self.assertTrue(np.all(np.asarray(alpha) > 0))
            self.assertEqual(len(alpha), 16)

    def test_small_calibration_run_reports_coverage_and_closed_certificates(self):
        r = run_benchmark(n_calibration=40, n_test=12, seed=17, target_coverage=0.8, n_observations=60)
        self.assertEqual(set(r["metrics"]), {"graph", "kernel", "joint"})
        for m in r["metrics"].values():
            self.assertGreaterEqual(m["calibrated_set_empirical_coverage"], 0.0)
            self.assertLessEqual(m["calibrated_set_empirical_coverage"], 1.0)
            self.assertTrue(m["all_calibrated_cases_closed_certificate"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
