import unittest
import numpy as np
import decision_sufficient_negative_controls as nc

class DecisionSufficientNegativeControlTests(unittest.TestCase):
    def test_risk_perturbation_is_reproducible_and_nonnegative(self):
        base = np.arange(1, 17, dtype=float).reshape(4, 4)
        a = nc.perturbed_risk(base, 0.2, 77)
        b = nc.perturbed_risk(base, 0.2, 77)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.all(a > 0.0))
        self.assertFalse(np.allclose(a, base))

    def test_unseen_regime_is_positive_and_stresses_farthest_risk_columns(self):
        alpha = nc.novel_regime_alpha()
        self.assertEqual(alpha.shape, (16,))
        self.assertTrue(np.all(alpha > 0.0))
        self.assertAlmostEqual(float(alpha.sum()), 1.028, places=8)
        self.assertEqual(int(np.argmax(alpha)), 9)
        self.assertEqual(float(alpha[9]), float(alpha[15]))

    def test_summarizer_counts_bound_violations_only_for_covered_cases(self):
        values = [
            {"covered": True, "regret_bits": 0.3, "radius": 0.1, "certificate_gap_bits": 1e-9},
            {"covered": True, "regret_bits": 0.05, "radius": 0.1, "certificate_gap_bits": 1e-9},
            {"covered": False, "regret_bits": 1.0, "radius": 0.1, "certificate_gap_bits": 1e-9},
        ]
        out = nc.summarize(values, include_bounds=True)
        self.assertEqual(out["bound_2r_violations"], 1)
        self.assertEqual(out["covered_n"], 2)
        self.assertAlmostEqual(out["max_covered_regret_over_2r"], 1.5)

    def test_full_falsification_run_reproduces_expected_failure_modes(self):
        out = nc.run()
        known = out["known_regime_results"]
        clean = known["correct_model"]["pooled"]["mean_regret_bits"]
        small = known["risk_misspec_5pct"]["pooled"]["mean_regret_bits"]
        large = known["risk_misspec_20pct"]["pooled"]["mean_regret_bits"]
        wrong_identity = known["policy_row_identity_mismatch"]["pooled"]["mean_regret_bits"]
        self.assertLess(clean, small)
        self.assertLess(small, large)
        self.assertLess(large, wrong_identity)
        self.assertEqual(known["correct_model"]["pooled"]["bound_2r_violations"], 0)
        self.assertGreater(known["policy_row_identity_mismatch"]["pooled"]["bound_2r_violations"], 0)
        self.assertGreater(known["policy_row_identity_mismatch"]["group_max"]["bound_2r_violations"], 0)
        ood = out["unseen_regime_results"]["correct_model"]
        self.assertLess(ood["pooled"]["coverage"], 0.90)
        self.assertGreaterEqual(ood["group_max"]["coverage"], ood["pooled"]["coverage"])
        self.assertLess(max(v[m]["max_certificate_gap_bits"] for v in known.values() for m in ("pooled", "group_max")), 1e-7)

if __name__ == "__main__":
    unittest.main(verbosity=2)
