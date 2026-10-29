import unittest
import numpy as np
import decision_risk_certified_fallback as f

class CertifiedFallbackTests(unittest.TestCase):
    def test_error_budget_bound_is_monotone_and_validates_inputs(self):
        self.assertAlmostEqual(f.certified_regret_bound(0.1, 0.02), 0.24, places=12)
        self.assertGreater(f.certified_regret_bound(0.1, 0.2), f.certified_regret_bound(0.1, 0.02))
        with self.assertRaises(ValueError):
            f.certified_regret_bound(-0.1, 0.0)
        with self.assertRaises(ValueError):
            f.certified_regret_bound(0.1, float('inf'))

    def test_fixed_seed_run_admits_correct_model_and_rejects_corruption(self):
        out = f.run()
        result = out['results']
        for method in ('pooled', 'group_max'):
            self.assertTrue(result['correct_model'][method]['admitted'])
            self.assertEqual(result['correct_model'][method]['fallback_rate'], 0.0)
            self.assertEqual(result['correct_model'][method]['conditional_regret_bound_violations'], 0)
            for condition in ('risk_misspec_5pct', 'risk_misspec_20pct', 'policy_row_identity_mismatch'):
                self.assertFalse(result[condition][method]['admitted'])
                self.assertEqual(result[condition][method]['fallback_rate'], 1.0)
                self.assertEqual(result[condition][method]['conditional_regret_bound_violations'], 0)

    def test_certified_bound_covers_every_covered_case_across_all_stress_conditions(self):
        out = f.run()
        for methods in out['results'].values():
            for method in methods.values():
                self.assertEqual(method['conditional_regret_bound_violations'], 0)
                self.assertLess(method['max_lp_certificate_gap_bits'], 1e-7)
                self.assertGreaterEqual(method['certified_regret_bound_bits'], 0.0)

if __name__ == '__main__':
    unittest.main(verbosity=2)
