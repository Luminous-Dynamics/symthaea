import unittest

from huber_identifiability_frontier import (
    build_witness, clean_mean_envelope, classify_certificate,
    evaluate_claim, observed_bernoulli_mean,
)


class HuberIdentifiabilityTests(unittest.TestCase):
    def test_exact_safe_and_unsafe_worlds_have_same_observed_law(self):
        witness = build_witness()
        self.assertTrue(witness["same_observed_distribution"])
        self.assertTrue(witness["world_safe"]["safe"])
        self.assertFalse(witness["world_unsafe"]["safe"])
        self.assertAlmostEqual(witness["world_safe"]["observed_mean"], 0.37)
        self.assertAlmostEqual(witness["world_unsafe"]["observed_mean"], 0.37)

    def test_5_percent_understatement_excludes_true_clean_mean_and_admits(self):
        witness = build_witness()["understated_bound_5pct"]
        self.assertEqual(witness["certificate_status"], "admit")
        self.assertTrue(witness["true_unsafe_mean_excluded"])
        self.assertGreater(0.43, witness["identified_interval"][1])

    def test_valid_15_percent_bound_contains_unsafe_world_and_abstains(self):
        witness = build_witness()["valid_bound_15pct"]
        self.assertTrue(witness["true_unsafe_mean_included"])
        self.assertEqual(witness["certificate_status"], "refuse_unresolved")

    def test_missing_or_invalid_provenance_fails_closed_before_sampling(self):
        for rho, provenance, scope in ((None, True, True), (0.05, False, True), (0.05, True, False)):
            result = evaluate_claim(
                observed_mean=0.37, model_mean=0.37, tolerance=0.05,
                rho_upper=rho, provenance_valid=provenance, scope_valid=scope,
            )
            self.assertEqual(result["status"], "refuse_unresolved")
            self.assertEqual(result["observations_used_for_certificate"], 0)
            self.assertIsNone(result["identified_clean_mean_interval"])

    def test_authenticated_but_wrong_bound_can_still_admit_unsafe_world(self):
        witness = build_witness()
        result = witness["authenticated_but_understated_bound"]
        self.assertTrue(witness["world_unsafe"]["true_model_error"] > 0.05)
        self.assertEqual(result["status"], "admit")
        self.assertTrue(result["attestation_confidence_is_not_verified_here"])
        self.assertAlmostEqual(result["composed_failure_budget"], 0.06)

    def test_authentication_confidence_budget_is_composed_conservatively(self):
        result = evaluate_claim(
            observed_mean=0.37, model_mean=0.37, tolerance=0.05,
            rho_upper=0.15, provenance_valid=True, scope_valid=True,
            attestation_confidence=0.99, alpha_statistical=0.05,
        )
        self.assertAlmostEqual(result["assumption_failure_probability"], 0.01)
        self.assertAlmostEqual(result["composed_failure_budget"], 0.06)
        self.assertTrue(result["attestation_confidence_is_not_verified_here"])

    def test_sensitivity_envelope_sharp_bounds(self):
        self.assertAlmostEqual(clean_mean_envelope(0.37, 0.05)[0], (0.37 - 0.05) / 0.95)
        self.assertAlmostEqual(clean_mean_envelope(0.37, 0.05)[1], 0.37 / 0.95)
        for p in (0.0, 0.1, 0.37, 0.43, 0.9, 1.0):
            for rho in (0.0, 0.05, 0.15, 0.5):
                q = observed_bernoulli_mean(p, rho, 0.0)
                lo, hi = clean_mean_envelope(q, rho)
                if rho <= 1.0 - 1e-12:
                    self.assertLessEqual(lo, p + 1e-12)
                    self.assertGreaterEqual(hi, p - 1e-12)

    def test_invalid_probabilities_fail(self):
        with self.assertRaises(ValueError): clean_mean_envelope(0.37, 1.0)
        with self.assertRaises(ValueError): observed_bernoulli_mean(0.4, 1.0, 0.0)
        with self.assertRaises(ValueError): clean_mean_envelope(-0.1, 0.05)

    def test_decision_requires_whole_interval_to_be_safe(self):
        self.assertEqual(classify_certificate(0.37, (0.34, 0.40), 0.05), "admit")
        self.assertEqual(classify_certificate(0.37, (0.43, 0.50), 0.05), "reject")
        self.assertEqual(classify_certificate(0.37, (0.30, 0.43), 0.05), "refuse_unresolved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
