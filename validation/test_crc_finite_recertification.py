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
    def test_paper_two_type_example_exact_rational(self):
        # Bai & Ramdas (2026), Section 3.4, Table 1:
        # ell(H, lambda)=1-lambda; ell(L, lambda)=(1-lambda)/2.
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(0), Fraction(0)]
        slope = [Fraction(1), Fraction(1, 2)]
        baseline = {
            (2, 0): Fraction(1, 2),   # HH
            (1, 1): Fraction(2, 5),   # HL
            (0, 2): Fraction(1, 4),   # LL
        }

        before = verify_controller(baseline, n, alpha, base, slope)
        self.assertTrue(before["valid"], before)
        self.assertEqual(before["minimum_slack"], Fraction(0))
        self.assertEqual(before["augmented_states_checked"], 4)

        recertified = recertify_once(baseline, n, alpha, base, slope)
        self.assertEqual(recertified[(2, 0)], Fraction(1, 2))
        self.assertEqual(recertified[(1, 1)], Fraction(23, 60))
        self.assertEqual(recertified[(0, 2)], Fraction(1, 5))

        check = verify_recertification_step(
            baseline, recertified, n, alpha, base, slope
        )
        self.assertTrue(check["valid"], check)
        self.assertTrue(check["pointwise_no_more_conservative"])
        self.assertEqual(check["risk"]["minimum_slack"], Fraction(0))

    def test_repeated_recursions_match_paper_geometric_limit(self):
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(0), Fraction(0)]
        slope = [Fraction(1), Fraction(1, 2)]
        baseline = {
            (2, 0): Fraction(1, 2),
            (1, 1): Fraction(2, 5),
            (0, 2): Fraction(1, 4),
        }
        limit = {
            (2, 0): Fraction(1, 2),
            (1, 1): Fraction(3, 8),
            (0, 2): Fraction(1, 8),
        }

        result = iterate_recertification(
            baseline, n, alpha, base, slope, max_iterations=8
        )
        self.assertEqual(len(result["history"]), 8)
        self.assertTrue(result["validity"]["valid"])
        initial_error = max(abs(baseline[state] - limit[state]) for state in baseline)
        final_error = max(abs(result["controller"][state] - limit[state]) for state in baseline)
        rho = Fraction(n, n + 1)
        self.assertLessEqual(final_error, rho**8 * initial_error)

    def test_loss_proportional_allocation_is_valid_and_more_aggressive(self):
        n = 2
        alpha = Fraction(1, 2)
        base = [Fraction(0), Fraction(0)]
        slope = [Fraction(1), Fraction(1, 2)]
        baseline = {
            (2, 0): Fraction(1, 2),
            (1, 1): Fraction(2, 5),
            (0, 2): Fraction(1, 4),
        }

        equal = recertify_once(baseline, n, alpha, base, slope)
        weighted = recertify_once(
            baseline, n, alpha, base, slope, allocation="loss_proportional"
        )
        self.assertEqual(weighted[(2, 0)], Fraction(1, 2))
        self.assertEqual(weighted[(1, 1)], Fraction(11, 29))
        self.assertEqual(weighted[(0, 2)], Fraction(1, 6))
        self.assertTrue(all(weighted[state] <= equal[state] for state in baseline))

        check = verify_recertification_step(
            baseline, weighted, n, alpha, base, slope,
            allocation="loss_proportional",
        )
        self.assertTrue(check["valid"], check)
        self.assertEqual(check["risk"]["minimum_slack"], Fraction(0))

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
