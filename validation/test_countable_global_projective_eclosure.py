import unittest
from fractions import Fraction

from countable_global_projective_eclosure import (
    coordinate_weight,
    finite_weight_mass,
    projective_certificate,
    verify_countable_global_projective_path,
    verify_weight_scheme_prefix,
)


class CountableGlobalProjectiveEclosureTests(unittest.TestCase):
    def test_geometric_weights_have_exact_prefix_mass(self):
        for n in (1, 2, 5, 10, 100):
            result = verify_weight_scheme_prefix("geometric", n)
            self.assertTrue(result["matches"])
            self.assertEqual(result["prefix_mass"], 1 - Fraction(1, 1 << n))
        self.assertEqual(coordinate_weight(1, "geometric"), Fraction(1, 2))

    def test_telescoping_weights_have_exact_prefix_mass(self):
        for n in (1, 2, 5, 10, 100):
            result = verify_weight_scheme_prefix("telescoping", n)
            self.assertTrue(result["matches"])
            self.assertEqual(result["prefix_mass"], 1 - Fraction(1, n + 1))
        self.assertEqual(coordinate_weight(1, "telescoping"), Fraction(1, 2))

    def test_global_weight_budget_is_not_renormalized_at_arrival(self):
        self.assertEqual(finite_weight_mass((1, 2), "geometric"), Fraction(3, 4))
        values = {1: Fraction(8), 2: Fraction(16)}
        # Fixed global weights yield 1 + .5*(8-1) + .25*(16-1) = 8.25.
        self.assertEqual(projective_certificate((1, 2), values), Fraction(33, 4))
        # Local renormalization would instead be (8+16)/2=12.
        self.assertNotEqual(projective_certificate((1, 2), values), Fraction(12))

    def setUp(self):
        self.alpha = Fraction(1, 2)
        self.truth = (1, 2, 4, 7)
        self.active = [(1,), (1, 3), (1, 2, 3), (1, 2, 3, 5)]
        self.rejections = [(), (1,), (1, 3), (1, 3, 5)]
        self.paths = {
            1: [Fraction(1), Fraction(8), Fraction(8), Fraction(16)],
            2: [Fraction(1), Fraction(1), Fraction(4), Fraction(8)],
            3: [Fraction(1), Fraction(4), Fraction(8), Fraction(16)],
            4: [Fraction(1), Fraction(1), Fraction(1), Fraction(1)],
            5: [Fraction(1), Fraction(1), Fraction(1), Fraction(2)],
            7: [Fraction(1), Fraction(1), Fraction(1), Fraction(1)],
        }

    def test_random_predictable_arrivals_preserve_full_true_set_identity(self):
        result = verify_countable_global_projective_path(
            self.active, self.rejections, self.truth, self.paths, self.alpha,
            scheme="geometric",
        )
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["all_projective_identities_verified"])
        self.assertTrue(result["all_true_null_losses_dominated"])
        self.assertEqual(result["times_checked"], 4)
        self.assertEqual(result["local_constraints_checked"], 1 + 3 + 7 + 15)

    def test_telescoping_scheme_has_same_projective_identity(self):
        result = verify_countable_global_projective_path(
            self.active, self.rejections, self.truth, self.paths, self.alpha,
            scheme="telescoping",
        )
        self.assertTrue(result["valid"], result)
        self.assertTrue(result["all_projective_identities_verified"])

    def test_neutral_dormant_padding_is_required(self):
        bad = dict(self.paths)
        bad[2] = [Fraction(2), Fraction(1), Fraction(4), Fraction(8)]
        with self.assertRaisesRegex(ValueError, "dormant at one"):
            verify_countable_global_projective_path(
                self.active, self.rejections, self.truth, bad, self.alpha,
            )

    def test_never_arriving_coordinate_must_stay_neutral(self):
        bad = dict(self.paths)
        bad[7] = [Fraction(1), Fraction(1), Fraction(1), Fraction(2)]
        with self.assertRaisesRegex(ValueError, "never-active coordinate"):
            verify_countable_global_projective_path(
                self.active, self.rejections, self.truth, bad, self.alpha,
            )

    def test_non_nested_arrival_schedule_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "nested"):
            verify_countable_global_projective_path(
                ((1, 2), (2, 3)), ((1,), (2,)), (1, 2, 3),
                {1: [1, 1], 2: [1, 2], 3: [1, 3]}, self.alpha,
            )

    def test_nonmonotone_coordinate_path_only_withholds_supfdr_condition(self):
        nonmonotone = dict(self.paths)
        nonmonotone[1] = [Fraction(1), Fraction(8), Fraction(3), Fraction(16)]
        result = verify_countable_global_projective_path(
            self.active, self.rejections, self.truth, nonmonotone, self.alpha,
        )
        self.assertTrue(result["valid"], result)
        self.assertFalse(result["coordinate_paths_monotone"])
        self.assertFalse(result["supfdr_pathwise_condition_satisfied"])

    def test_monotone_coordinate_paths_support_supfdr_pathwise_condition(self):
        result = verify_countable_global_projective_path(
            self.active, self.rejections, self.truth, self.paths, self.alpha,
        )
        self.assertTrue(result["coordinate_paths_monotone"])
        self.assertTrue(result["supfdr_pathwise_condition_satisfied"])

    def test_missing_or_negative_coordinate_data_fail_closed(self):
        missing = dict(self.paths)
        del missing[3]
        with self.assertRaisesRegex(ValueError, "missing coordinate path"):
            verify_countable_global_projective_path(
                self.active, self.rejections, self.truth, missing, self.alpha,
            )
        negative = dict(self.paths)
        negative[3] = [1, -1, 1, 1]
        with self.assertRaisesRegex(ValueError, "negative value"):
            verify_countable_global_projective_path(
                self.active, self.rejections, self.truth, negative, self.alpha,
            )

    def test_rejection_must_be_active(self):
        with self.assertRaisesRegex(ValueError, "subset of the active family"):
            verify_countable_global_projective_path(
                ((1,),), ((2,),), (1, 2), {1: [1], 2: [1]}, self.alpha,
            )

    def test_empty_true_set_still_has_unit_cash_certificate(self):
        result = verify_countable_global_projective_path(
            ((1,), (1, 2)), ((), ()), (), {1: [1, 2], 2: [1, 4]}, self.alpha,
        )
        self.assertTrue(result["valid"], result)
        self.assertEqual(
            [row["full_true_certificate"] for row in result["path"]],
            [Fraction(1), Fraction(1)],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
