import unittest
from fractions import Fraction

from crc_discrete_admissibility import (
    coordinate_sweep_admissible,
    verify_grid_admissibility,
    verify_grid_controller,
)


class DiscreteCrcAdmissibilityTests(unittest.TestCase):
    def setUp(self):
        self.n = 2
        self.alpha = Fraction(1, 2)
        self.actions = [Fraction(i, 40) for i in range(41)]
        high_loss = [1 - a for a in self.actions]
        low_loss = [(1 - a) / 2 for a in self.actions]
        self.losses = [[high_loss, low_loss]]
        self.baseline = {
            (2, 0): 20,  # HH: 1/2
            (1, 1): 16,  # HL: 2/5
            (0, 2): 10,  # LL: 1/4
        }

    def test_paper_example_reaches_pointwise_admissible_grid_controller(self):
        before = verify_grid_controller(
            self.baseline, self.n, [self.alpha], self.losses
        )
        self.assertTrue(before["valid"], before)
        self.assertEqual(before["augmented_states_checked"], 4)

        result = coordinate_sweep_admissible(
            self.baseline, self.n, [self.alpha], self.losses
        )
        self.assertEqual(
            result["controller"],
            {(2, 0): 20, (1, 1): 15, (0, 2): 5},
        )
        self.assertTrue(result["validity"]["valid"])
        self.assertTrue(result["admissibility"]["admissible"])
        self.assertEqual(result["updated_coordinates"], 2)
        self.assertEqual(
            result["validity"]["minimum_slacks"],
            (Fraction(0),),
        )

    def test_baseline_is_not_yet_pointwise_admissible(self):
        report = verify_grid_admissibility(
            self.baseline, self.n, [self.alpha], self.losses
        )
        self.assertFalse(report["admissible"])
        self.assertTrue(report["baseline_valid"])
        self.assertIsNotNone(report["witness"])

    def test_invalid_baseline_fails_closed(self):
        invalid = {
            (2, 0): 0,
            (1, 1): 0,
            (0, 2): 0,
        }
        report = verify_grid_controller(
            invalid, self.n, [self.alpha], self.losses
        )
        self.assertFalse(report["valid"])
        with self.assertRaises(ValueError):
            coordinate_sweep_admissible(
                invalid, self.n, [self.alpha], self.losses
            )

    def test_multiple_risk_tables_are_checked_jointly(self):
        # The second loss is half the first, with half the risk budget.
        # It is redundant mathematically but exercises the joint verifier.
        second = [
            [value / 2 for value in self.losses[0][0]],
            [value / 2 for value in self.losses[0][1]],
        ]
        result = coordinate_sweep_admissible(
            self.baseline,
            self.n,
            [self.alpha, self.alpha / 2],
            [self.losses[0], second],
        )
        self.assertTrue(result["validity"]["valid"])
        self.assertEqual(len(result["validity"]["minimum_slacks"]), 2)
        self.assertTrue(result["admissibility"]["admissible"])

    def test_nonmonotone_loss_grid_rejected(self):
        invalid_losses = [[
            [Fraction(1), Fraction(1, 2), Fraction(3, 4)],
            [Fraction(1), Fraction(1, 2), Fraction(0)],
        ]]
        with self.assertRaises(ValueError):
            verify_grid_controller(
                {(2, 0): 1, (1, 1): 1, (0, 2): 1},
                self.n,
                [self.alpha],
                invalid_losses,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
