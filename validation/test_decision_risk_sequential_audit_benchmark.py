import unittest
import numpy as np
import decision_risk_sequential_audit_benchmark as b

class SequentialAuditTests(unittest.TestCase):
    def test_adjusted_bounds_are_wider_and_infinite_horizon_is_most_conservative(self):
        v = np.full((2,2), 0.25)
        for n in (64, 512, 4096):
            fixed = b.fixed_time_radius(v,n,4)
            finite = b.finite_look_radius(v,n,4,5)
            anytime = b.anytime_radius(v,n,4)
            self.assertTrue(np.all(finite > fixed))
            self.assertTrue(np.all(anytime > finite))

    def test_rejects_invalid_radius_inputs(self):
        with self.assertRaises(ValueError):
            b.empirical_bernstein_radius(np.array([[-1.0]]), 4, 2.0)
        with self.assertRaises(ValueError):
            b.fixed_time_radius(np.ones((2,2)), 8, 4, 1.0)
        with self.assertRaises(ValueError):
            b.finite_look_radius(np.ones((2,2)), 8, 4, 0)

    def test_fixed_seed_audit_reports_expected_precision_cost_tradeoff(self):
        out = b.run(repeats=100, sample_sizes=(4096,8192,16384,32768))
        clean = out['results']['correct_model']
        self.assertGreaterEqual(clean['fixed_time_repeated_peek']['any_admission_rate'],
                                clean['finite_look_alpha_spending']['any_admission_rate'])
        self.assertGreaterEqual(clean['finite_look_alpha_spending']['any_admission_rate'],
                                clean['anytime_alpha_spending']['any_admission_rate'])
        over = out['results']['clearly_over_regret_budget']
        self.assertEqual(over['anytime_alpha_spending']['false_admission_rate'], 0.0)
        self.assertEqual(over['finite_look_alpha_spending']['false_admission_rate'], 0.0)
        self.assertEqual(out['protocol']['cells'], 4)

if __name__ == '__main__':
    unittest.main(verbosity=2)
