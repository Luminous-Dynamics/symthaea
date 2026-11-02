import unittest
from fractions import Fraction as F

from anchored_compound_ebh import (
    finite_envelope_slack,
    finite_envelope_weight_prefix,
    weighted_compound_ebh,
    anchored_rejections,
)
from anchored_compound_ebh_pilot import (
    DEPENDENCE,
    make_scenario,
    simulate_stream,
)


class AnchoredCompoundEBHPilotTests(unittest.TestCase):
    def test_finite_envelope_slack_is_fixed_by_maximum_family_size(self):
        for n in (1, 2, 10, 24, 100, 10_000):
            weights = finite_envelope_weight_prefix(n, F(1))
            self.assertEqual(sum(weights, F(0)), F(n, n + 1))
            self.assertEqual(finite_envelope_slack(n, F(1)), F(1, n + 1))

    def test_prefix_weights_do_not_renormalize_when_roster_is_dormant(self):
        n = 24
        weights = finite_envelope_weight_prefix(n, F(1))
        self.assertEqual(weights[:5], finite_envelope_weight_prefix(5, F(1)))
        self.assertEqual(sum(weights, F(0)), F(24, 25))

    def test_anchor_solver_preserves_finite_envelope_baseline(self):
        scores = [F(1, 2), F(3), F(10), F(0), F(1)]
        alpha = F(1, 10)
        delta = F(1, 6)
        baseline = set(weighted_compound_ebh(scores, alpha))
        anchored = set(anchored_rejections(scores, alpha, delta))
        self.assertTrue(baseline.issubset(anchored))

    def test_late_cluster_pilot_returns_all_methods(self):
        scenario = make_scenario("late_cluster", 24, 40)
        for dependence in DEPENDENCE:
            result = simulate_stream(
                scenario, dependence, 10150 + len(dependence),
                (F(1, 100), F(1, 20), F(1, 10), F(1, 5))
            )
            self.assertIn("finite_envelope_baseline_e_bh", result)
            self.assertIn("finite_envelope_anchored", result)
            self.assertLessEqual(
                result["finite_envelope_baseline_e_bh"]["terminal_power"],
                result["finite_envelope_anchored"]["terminal_power"] + 1e-12
            )

    def test_predeclared_arrival_schedule_is_independent_of_observations(self):
        scenario = make_scenario("late_cluster", 24, 40)
        self.assertEqual(scenario["arrivals"][16], 20)
        self.assertEqual(scenario["arrivals"][23], 20)
        self.assertEqual(scenario["null"][16], False)
        self.assertEqual(scenario["null"][23], False)


if __name__ == "__main__":
    unittest.main()
