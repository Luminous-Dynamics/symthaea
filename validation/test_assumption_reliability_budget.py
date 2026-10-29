import unittest

from assumption_reliability_budget import (
    assess_issuer, binomial_cdf, clopper_pearson_upper,
    compose_familywise_budget, run_drift_stress,
)


class AssumptionReliabilityBudgetTests(unittest.TestCase):
    def test_zero_failures_does_not_imply_zero_probability(self):
        upper = clopper_pearson_upper(0, 200, 0.05)
        self.assertAlmostEqual(upper, 1 - 0.05 ** (1 / 200), places=9)
        self.assertGreater(upper, 0.01)

    def test_exact_upper_is_monotone_in_observed_failures(self):
        bounds = [clopper_pearson_upper(k, 100, 0.05) for k in (0, 1, 5, 20, 100)]
        self.assertEqual(bounds, sorted(bounds))
        self.assertEqual(bounds[-1], 1.0)

    def test_more_independent_audits_narrow_zero_failure_bound(self):
        self.assertLess(clopper_pearson_upper(0, 1000, 0.05),
                        clopper_pearson_upper(0, 100, 0.05))

    def test_binomial_cdf_known_endpoints(self):
        self.assertEqual(binomial_cdf(3, 3, 0.4), 1.0)
        self.assertEqual(binomial_cdf(0, 3, 1.0), 0.0)
        self.assertAlmostEqual(binomial_cdf(0, 1, 0.25), 0.75)

    def test_risk_budget_composes_calibration_and_future_decisions(self):
        once = compose_familywise_budget(alpha_stat=0.005, alpha_cal=0.005,
                                         issuer_failure_upper=0.01, decisions=1)
        ten = compose_familywise_budget(alpha_stat=0.005, alpha_cal=0.005,
                                        issuer_failure_upper=0.01, decisions=10)
        self.assertAlmostEqual(once, 0.02)
        self.assertAlmostEqual(ten, 0.155)
        self.assertGreater(ten, once)

    def test_expired_wrong_scope_unqualified_and_missing_audit_fail_closed(self):
        base = dict(failures=0, audits=1000, alpha_cal=0.005, alpha_stat=0.005,
                    decisions=1, target_budget=0.05, evidence_scope="issuer/scope/1",
                    requested_scope="issuer/scope/1", expired=False,
                    audit_process_qualified=True)
        cases = [
            ({**base, "requested_scope": "issuer/scope/2"}, "scope_mismatch_or_missing"),
            ({**base, "expired": True}, "evidence_expired"),
            ({**base, "audit_process_qualified": False}, "audit_process_unqualified"),
            ({**base, "audits": 0}, "no_independent_audits"),
        ]
        for kwargs, expected in cases:
            result = assess_issuer(**kwargs)
            self.assertEqual(result["status"], "refuse_unresolved")
            self.assertEqual(result["reason"], expected)
            self.assertEqual(result["observations_used_for_certificate"], 0)

    def test_representative_scope_may_pass_only_if_budget_fits(self):
        result = assess_issuer(failures=0, audits=1000, alpha_cal=0.005,
            alpha_stat=0.005, decisions=1, target_budget=0.05,
            evidence_scope="issuer/scope/1", requested_scope="issuer/scope/1",
            expired=False, audit_process_qualified=True)
        self.assertEqual(result["status"], "assumption_budget_accepted")
        self.assertLessEqual(result["familywise_failure_budget_upper"], 0.05)
        self.assertGreater(result["issuer_failure_probability_upper"], 0.0)

    def test_drift_stress_exposes_historical_calibration_limit(self):
        result = run_drift_stress(repeats=300, audits=1000, seed=42)
        stable = result["results"]["future_rate_0.010"]
        shifted = result["results"]["future_rate_0.100"]
        self.assertGreater(shifted["undercoverage_rate"], 0.99)
        self.assertGreater(shifted["acceptance_rate"], 0.95)
        self.assertAlmostEqual(stable["acceptance_rate"], shifted["acceptance_rate"])

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError): clopper_pearson_upper(1, 0)
        with self.assertRaises(ValueError): clopper_pearson_upper(5, 3)
        with self.assertRaises(ValueError): clopper_pearson_upper(0, 10, 1.0)
        with self.assertRaises(ValueError): compose_familywise_budget(
            alpha_stat=0.1, alpha_cal=0.1, issuer_failure_upper=0.1, decisions=0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
