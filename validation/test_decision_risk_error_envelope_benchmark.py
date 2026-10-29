import unittest
import numpy as np
import decision_risk_error_envelope_benchmark as b

class RiskErrorEnvelopeTests(unittest.TestCase):
    def test_empirical_bernstein_and_hoeffding_radii_are_finite(self):
        rng = np.random.default_rng(7)
        x = 2.0 + rng.uniform(-0.05, 0.05, size=(128, 3, 4))
        radii = b.empirical_bernstein_cell_radii(x)
        self.assertEqual(radii.shape, (3, 4))
        self.assertTrue(np.all(np.isfinite(radii)))
        self.assertTrue(np.all(radii >= 0))
        self.assertGreater(b.hoeffding_cell_radius(128, 12), 0)
        self.assertGreater(b.hoeffding_cell_radius(64, 12), b.hoeffding_cell_radius(128, 12))

    def test_missing_out_of_range_and_bad_shape_fail_closed(self):
        good = np.ones((8, 2, 3))
        with self.assertRaises(ValueError):
            b.empirical_bernstein_cell_radii(np.full((8, 2, 3), np.nan))
        with self.assertRaises(ValueError):
            b.empirical_bernstein_cell_radii(np.full((8, 2, 3), 4.1))
        with self.assertRaises(ValueError):
            b.error_envelopes(np.ones((2, 3)), good[:, :1, :])
        with self.assertRaises(ValueError):
            b.empirical_bernstein_cell_radii(np.ones((1, 2, 3)))

    def test_fixed_seed_audit_bound_has_high_familywise_coverage(self):
        out = b.run(repeats=80, sample_sizes=(32, 512, 4096))
        for model in out['results'].values():
            for sample_size in model.values():
                eb = sample_size['empirical_bernstein']
                hoeffding = sample_size['hoeffding']
                self.assertGreaterEqual(eb['envelope_coverage_rate'], 0.85)
                self.assertGreaterEqual(hoeffding['envelope_coverage_rate'], 0.85)
        correct = out['results']['correct_model']['4096']
        self.assertGreater(correct['empirical_bernstein']['admission_rate_using_this_method'], 0.5)
        self.assertEqual(correct['hoeffding']['admission_rate_using_this_method'], 0.0)
        near = out['results']['risk_misspec_near_budget']['512']
        self.assertGreater(near['plugin']['unsafe_admission_rate_using_this_method'], 0.0)
        self.assertEqual(near['empirical_bernstein']['unsafe_admission_rate_using_this_method'], 0.0)

    def test_statistical_envelope_distinguishes_safe_small_error_from_large_error(self):
        out = b.run(repeats=80, sample_sizes=(8192,))
        results = out['results']
        eb_correct = results['correct_model']['8192']['empirical_bernstein']
        eb_large = results['risk_misspec_20pct']['8192']['empirical_bernstein']
        self.assertGreater(eb_correct['admission_rate_using_this_method'], 0.5)
        self.assertEqual(eb_large['admission_rate_using_this_method'], 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
