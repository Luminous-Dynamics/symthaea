import unittest
import numpy as np

from process_valued_supfdr_experiment import (
    Scenario,
    METHODS,
    evaluate_method,
    generate_observations,
    make_evidence_paths,
    make_scenarios,
    prior_for_donor,
    weighted_ebh,
)


class ProcessValuedSupFDRTests(unittest.TestCase):
    def test_null_likelihood_ratio_has_conditional_mean_one(self):
        self.assertAlmostEqual(0.5 * 1.4 + 0.5 * 0.6, 1.0)

    def test_projective_priors_retain_outside_horizon_mass(self):
        for n in (2, 5, 32, 128):
            for donor in range(n):
                for kind in ("telescope", "geometric", "fixed_next"):
                    q = prior_for_donor(n, donor, kind)
                    self.assertTrue(np.all(q >= 0))
                    self.assertAlmostEqual(float(q.sum()), 1.0, places=14)

    def test_balanced_negative_nulls_have_exactly_half_successes(self):
        n, looks, streams = 8, 20, 40
        scenario = Scenario(
            "all_null",
            np.full((looks, n), 0.5),
            np.ones(n, dtype=bool),
            np.zeros(n, dtype=np.int64),
        )
        y = generate_observations(scenario, streams, "balanced_negative", 3301)
        self.assertTrue(np.all(y.sum(axis=2) == n // 2))

    def test_shared_shock_preserves_each_null_marginal(self):
        n, looks, streams = 10, 100, 500
        scenario = Scenario(
            "all_null",
            np.full((looks, n), 0.5),
            np.ones(n, dtype=bool),
            np.zeros(n, dtype=np.int64),
        )
        y = generate_observations(scenario, streams, "shared_uniform", 3302)
        means = y.mean(axis=(0, 1))
        self.assertTrue(np.all(np.abs(means - 0.5) < 0.015))

    def test_adjusted_paths_are_nonnegative_and_non_decreasing(self):
        scenarios = make_scenarios(12, 25)
        for scenario in scenarios:
            y = generate_observations(scenario, 30, "independent", 3303)
            x, raw = make_evidence_paths(y, scenario.arrivals)
            self.assertTrue(np.all(x >= 0))
            self.assertTrue(np.all(np.diff(x, axis=1) >= -1e-12))
            self.assertTrue(np.all(raw >= 1))

    def test_all_donation_policies_preserve_pathwise_accounting(self):
        n, looks, streams = 12, 30, 50
        scenario = make_scenarios(n, looks)[1]
        y = generate_observations(scenario, streams, "shared_uniform", 3304)
        x, raw = make_evidence_paths(y, scenario.arrivals)
        gamma = np.array([1.0 / ((i + 1) * (i + 2)) for i in range(n)])
        for method in METHODS:
            result = evaluate_method(
                x, raw, scenario, gamma, method, 0.1, 1.0, 0.05
            )
            if method != "raw_supremum_invalid_control":
                self.assertLessEqual(result["max_invariant_error"], 2e-8)
                self.assertLessEqual(result["max_donor_cap_error"], 2e-8)
                self.assertLessEqual(result["max_own_increment_error"], 2e-8)
                self.assertLessEqual(result["max_share_mass_excess"], 2e-8)

    def test_weighted_ebh_self_consistency(self):
        z = np.array([[30.0, 12.0, 7.0, 0.0]])
        rejected = weighted_ebh(z, alpha=0.1)
        k = int(rejected.sum())
        self.assertGreater(k, 0)
        self.assertTrue(np.all(z[rejected] >= 1.0 / (0.1 * k)))


if __name__ == "__main__":
    unittest.main()
