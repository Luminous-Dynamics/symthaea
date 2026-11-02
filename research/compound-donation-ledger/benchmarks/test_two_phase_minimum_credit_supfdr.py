import unittest
from fractions import Fraction as F

import numpy as np

from two_phase_minimum_credit_supfdr import (
    METHODS,
    adjusted_evidence_path,
    ebh_rejections,
    generate_observations,
    make_scenario,
    minimum_credit_batch,
    simulate_policy,
)


class TwoPhaseMinimumCreditSupFDRTests(unittest.TestCase):
    def test_null_lr_factor_has_conditional_mean_one(self):
        self.assertEqual(F(1, 2) * F(7, 5) + F(1, 2) * F(3, 5), F(1))

    def test_a2_adjusted_evidence_is_nonnegative_and_monotone(self):
        scenario = make_scenario("late_bridge", 12, 20)
        observations = generate_observations(scenario, "shared_uniform", 101251)
        path = adjusted_evidence_path(observations, scenario["arrivals"])
        for i in range(12):
            values = [row[i] for row in path]
            self.assertTrue(all(value >= 0 for value in values))
            self.assertTrue(all(a <= b for a, b in zip(values, values[1:])))

    def test_exact_minimum_credit_enters_ebh(self):
        scores = [F(12), F(2), F(1), F(0)]
        alpha = F(1, 10)
        deficits = minimum_credit_batch(scores, alpha)
        for candidate, deficit in enumerate(deficits):
            updated = list(scores)
            updated[candidate] += deficit
            self.assertIn(candidate, ebh_rejections(updated, alpha))

    def test_two_phase_valid_process_preserves_invariants(self):
        scenario = make_scenario("immediate_bridge", 24, 25)
        observations = generate_observations(scenario, "independent", 101252)
        path = adjusted_evidence_path(observations, scenario["arrivals"])
        gamma = [F(1, (i + 1) * (i + 2)) for i in range(24)]
        for method in METHODS:
            result = simulate_policy(
                path, scenario, gamma, F(1, 10), method
            )
            self.assertEqual(result["max_prefix_error"], 0.0)
            self.assertEqual(result["max_donor_cap_error"], 0.0)
            self.assertEqual(result["max_score_decrease"], 0.0)
            self.assertGreaterEqual(result["reserve_volume"], 0.0)
            self.assertGreaterEqual(result["credited_volume"], 0.0)

    def test_two_phase_reserve_conservation_and_no_false_spending(self):
        scenario = make_scenario("no_useful", 24, 30)
        observations = generate_observations(scenario, "shared_uniform", 101253)
        path = adjusted_evidence_path(observations, scenario["arrivals"])
        gamma = [F(1, (i + 1) * (i + 2)) for i in range(24)]
        result = simulate_policy(
            path, scenario, gamma, F(1, 10), "two_phase_min_credit"
        )
        self.assertAlmostEqual(
            result["reserve_volume"],
            result["credited_volume"] + result["unassigned_or_pending_escrow"],
            places=12,
        )
        self.assertEqual(result["assignment_count"], 0)


if __name__ == "__main__":
    unittest.main()
