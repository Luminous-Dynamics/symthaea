import unittest
from fractions import Fraction as F

from exact_rational_common_depth_crc import (
    ExactCommonDepthCRC, count_vectors, error_loss,
    largest_piecewise_feasible_depth,
)


class ExactRationalCommonDepthCRCTests(unittest.TestCase):
    def test_count_vector_state_sizes(self):
        self.assertEqual(len(count_vectors(50, 3)), 1326)
        self.assertEqual(len(count_vectors(51, 3)), 1378)

    def test_piecewise_losses_are_exact(self):
        self.assertEqual(error_loss(0, F(7, 4)), 0)
        self.assertEqual(error_loss(1, F(9, 10)), 0)
        self.assertEqual(error_loss(1, F(9, 5)), F(2, 5))
        self.assertEqual(error_loss(2, F(9, 5)), F(9, 10))

    def test_baseline_depth_inverts_both_piecewise_segments(self):
        # Envelope with h1=h2=0: a/2 <= budget, clipped to [0,2].
        self.assertEqual(largest_piecewise_feasible_depth(0, 0, F(3, 4), envelope=True), F(3, 2))
        # CRC has no admissible positive depth when B<1.
        self.assertEqual(largest_piecewise_feasible_depth(4, 2, F(99, 100), envelope=False), 0)
        # A nontrivial point on the a>=1 segment.
        self.assertEqual(largest_piecewise_feasible_depth(1, 1, F(3), envelope=True), F(2))


    def test_type1_piecewise_slope_keeps_the_loss_factor_one_half(self):
        # At depth a>=1 the envelope constraint is
        # h1*(a-1)/2 + (h2+1)*a/2 <= B.
        # h1=4, h2=3, B=51/10 gives a=(2B+h1)/(h1+h2+1)=71/40.
        self.assertEqual(
            largest_piecewise_feasible_depth(4, 3, F(51, 10), envelope=True),
            F(71, 40)
        )
        # Standard CRC adds the constant unseen-point envelope 1.
        self.assertEqual(
            largest_piecewise_feasible_depth(4, 3, F(51, 10), envelope=False),
            F(61, 35)
        )

    def test_baseline_constraint_is_exact_at_the_returned_depth(self):
        budget=F(51,10)
        a=largest_piecewise_feasible_depth(4,3,budget,envelope=True)
        lhs=F(4,2)*max(a-1,F(0))+F(3,2)*a+a/2
        self.assertEqual(lhs,budget)

    def test_both_baselines_are_exactly_valid_at_all_paper_levels(self):
        for alpha in ("0.02", "0.05", "0.10", "0.15"):
            engine = ExactCommonDepthCRC(50, alpha)
            for kind in ("crc", "envelope"):
                report = engine.verify(engine.baseline(kind))
                self.assertTrue(report["valid"], (alpha, kind, report))
                self.assertLessEqual(F(report["worst_total"]), F(report["budget"]))
                self.assertEqual(report["checked_augmented_states"], 1378)

    def test_every_snapshot_after_fifty_rounds_is_exactly_valid(self):
        engine = ExactCommonDepthCRC(4, "0.15")
        result = engine.run(rounds=50, snapshot_rounds=(0, 1, 2, 10, 25, 50))
        self.assertTrue(result["all_rounds_valid"])
        self.assertEqual(result["verified_table_count"], 1 + 2 * 50 + 6)
        self.assertEqual(result["exact_augmented_inequality_checks"], (1 + 2 * 50 + 6) * len(engine.augmented_states))
        self.assertTrue(all(row["valid"] for row in result["snapshots"]))
        self.assertTrue(all(F(row["exact_excess"]) <= 0 for row in result["snapshots"]))

    def test_invalid_table_fails_closed_with_exact_counterexample(self):
        engine = ExactCommonDepthCRC(2, "0.5")
        bad = {h: F(2) for h in engine.states}
        report = engine.verify(bad)
        self.assertFalse(report["valid"])
        self.assertTrue(report["positive_excess"])
        self.assertEqual(report["worst_augmented_count"], [0, 0, 3])
        with self.assertRaises(ValueError):
            engine.recertify_round(bad)

    def test_rational_alpha_parsing_is_decimal_exact(self):
        self.assertEqual(ExactCommonDepthCRC(50, "0.10").alpha, F(1, 10))
        self.assertEqual(ExactCommonDepthCRC(50, 0.10).alpha, F(1, 10))


if __name__ == "__main__":
    unittest.main(verbosity=2)
