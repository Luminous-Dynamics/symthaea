import unittest
from fractions import Fraction

from crc_finite_recertification import (
    composition_count,
    compositions,
    iterate_recertification,
    recertify_once,
    verify_controller,
    verify_recertification_step,
)


class FiniteExchangeabilityCrcTests(unittest.TestCase):
    def test_two_type_one_step_exact_rational(self):
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(1, 2), Fraction(1, 4)]
        slope = [Fraction(3, 10), Fraction(3, 10)]
        baseline = {state: Fraction(1) for state in compositions(n, 2)}

        before = verify_controller(baseline, n, alpha, base, slope)
        self.assertTrue(before["valid"], before)
        self.assertEqual(before["minimum_slack"], Fraction(0))
        self.assertEqual(before["augmented_states_checked"], 4)

        recertified = recertify_once(baseline, n, alpha, base, slope)
        self.assertEqual(recertified[(0, 2)], Fraction(4, 9))
        self.assertEqual(recertified[(1, 1)], Fraction(13, 18))
        self.assertEqual(recertified[(2, 0)], Fraction(1))

        check = verify_recertification_step(
            baseline, recertified, n, alpha, base, slope
        )
        self.assertTrue(check["valid"], check)
        self.assertTrue(check["pointwise_no_more_conservative"])
        self.assertEqual(check["risk"]["minimum_slack"], Fraction(0))

    def test_iterates_remain_valid_and_actions_decrease(self):
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(1, 2), Fraction(1, 4)]
        slope = [Fraction(3, 10), Fraction(3, 10)]
        baseline = {state: Fraction(1) for state in compositions(n, 2)}

        result = iterate_recertification(
            baseline, n, alpha, base, slope, max_iterations=8
        )
        self.assertEqual(len(result["history"]), 8)
        self.assertTrue(result["validity"]["valid"])
        changes = [row["sup_action_change"] for row in result["history"]]
        for previous, current in zip(changes, changes[1:]):
            self.assertLessEqual(current, previous * Fraction(2, 3))

    def test_three_type_n50_full_state_enumeration(self):
        n = 50
        alpha = Fraction(9, 20)
        base = [
            Fraction(9, 20),
            Fraction(7, 20),
            Fraction(1, 4),
        ]
        slope = [Fraction(1, 20)] * 3
        baseline = {state: Fraction(1) for state in compositions(n, 3)}

        self.assertEqual(len(baseline), 1326)
        self.assertEqual(composition_count(n, 3), 1326)
        self.assertEqual(composition_count(n + 1, 3), 1378)

        before = verify_controller(baseline, n, alpha, base, slope)
        self.assertTrue(before["valid"], before)
        self.assertEqual(before["augmented_states_checked"], 1378)

        recertified = recertify_once(baseline, n, alpha, base, slope)
        check = verify_recertification_step(
            baseline, recertified, n, alpha, base, slope
        )
        self.assertTrue(check["valid"], check)
        self.assertEqual(check["risk"]["augmented_states_checked"], 1378)
        self.assertEqual(sum(a < 1 for a in recertified.values()), 1325)
        self.assertEqual(check["risk"]["minimum_slack"], Fraction(0))

    def test_invalid_baseline_fails_closed(self):
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(1, 2), Fraction(1, 4)]
        slope = [Fraction(3, 10), Fraction(3, 10)]
        invalid = {state: Fraction(0) for state in compositions(n, 2)}

        report = verify_controller(invalid, n, alpha, base, slope)
        self.assertFalse(report["valid"])
        step_report = verify_recertification_step(
            invalid, invalid, n, alpha, base, slope
        )
        self.assertFalse(step_report["valid"])
        self.assertTrue(
            any(x[0] == "baseline_not_finite_exchangeability_valid"
                for x in step_report["failures"])
        )
        with self.assertRaises(ValueError):
            recertify_once(invalid, n, alpha, base, slope)

    def test_zero_loss_slope_can_be_recategorized_without_risk_change(self):
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(1, 5), Fraction(1, 10)]
        slope = [Fraction(0), Fraction(0)]
        baseline = {state: Fraction(1) for state in compositions(n, 2)}

        recertified = recertify_once(baseline, n, alpha, base, slope)
        self.assertTrue(all(action == 0 for action in recertified.values()))
        check = verify_recertification_step(
            baseline, recertified, n, alpha, base, slope
        )
        self.assertTrue(check["valid"], check)


if __name__ == "__main__":
    unittest.main(verbosity=2)
