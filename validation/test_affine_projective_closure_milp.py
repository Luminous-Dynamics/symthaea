import unittest
from fractions import Fraction
from itertools import combinations, product

from affine_projective_closure_milp import (
    build_affine_closure_milp,
    minimal_auxiliary_values,
    render_cplex_lp,
    verify_affine_closure_milp_solution,
)
from countable_global_projective_eclosure import projective_certificate


class AffineProjectiveClosureMilpTests(unittest.TestCase):
    def test_model_has_linear_size_and_encodes_persistence_capacity(self):
        ids = tuple(range(1, 11))
        model = build_affine_closure_milp(
            ids,
            {i: Fraction(2) for i in ids},
            Fraction(1, 2),
            values={i: Fraction(i) for i in ids},
            costs={i: Fraction(1) for i in ids},
            capacity=Fraction(5),
            previous_rejections=(1, 3),
        )
        meta = model["metadata"]
        self.assertEqual(meta["model_variables"], 21)
        self.assertEqual(meta["model_constraint_rows"], 10 + 2 + 2 + 1)
        self.assertEqual(meta["closure_constraints_covered"], 2**10 - 1)
        self.assertEqual(len(model["constraints"]["closure_auxiliary"]), 10)
        self.assertEqual(len(model["constraints"]["persistence"]), 2)

    def test_exact_milp_certificate_matches_original_closure_constraints(self):
        model = build_affine_closure_milp(
            (1, 2),
            {1: Fraction(8), 2: Fraction(0)},
            Fraction(1, 2),
        )
        rejection = {1: 1, 2: 0}
        auxiliary = minimal_auxiliary_values(model, rejection)
        self.assertEqual(auxiliary, {1: Fraction(0), 2: Fraction(1, 8)})
        result = verify_affine_closure_milp_solution(model, rejection, auxiliary)
        self.assertTrue(result["valid"], result)
        self.assertEqual(result["original_intersections_checked"], 3)

    def test_understated_auxiliary_variable_is_rejected(self):
        model = build_affine_closure_milp(
            (1, 2),
            {1: Fraction(8), 2: Fraction(0)},
            Fraction(1, 2),
        )
        result = verify_affine_closure_milp_solution(
            model, {1: 1, 2: 0}, {1: Fraction(0), 2: Fraction(0)}
        )
        self.assertFalse(result["valid"])
        self.assertIn("closure_auxiliary_understated", [row[0] for row in result["failures"]])

    def test_original_intersection_violation_fails_even_if_auxiliary_witness_is_forged(self):
        model = build_affine_closure_milp(
            (1, 2),
            {1: Fraction(0), 2: Fraction(0)},
            Fraction(1, 2),
        )
        # A forged, too-small y vector must fail both the linear row and
        # the independent original-intersection replay.
        result = verify_affine_closure_milp_solution(
            model, {1: 1, 2: 0}, {1: Fraction(0), 2: Fraction(0)}
        )
        self.assertFalse(result["valid"])
        self.assertTrue(any(
            row[0] in ("closure_auxiliary_understated", "original_intersection_constraint_violation")
            for row in result["failures"]
        ))

    def test_capacity_and_persistence_are_independently_replayed(self):
        model = build_affine_closure_milp(
            (1, 2, 3),
            {1: Fraction(8), 2: Fraction(8), 3: Fraction(8)},
            Fraction(1, 2),
            costs={1: 1, 2: 1, 3: 1},
            capacity=2,
            previous_rejections=(1,),
        )
        rejection = {1: 1, 2: 1, 3: 0}
        auxiliary = minimal_auxiliary_values(model, rejection)
        good = verify_affine_closure_milp_solution(model, rejection, auxiliary)
        self.assertTrue(good["valid"], good)

        bad_capacity_model = build_affine_closure_milp(
            (1, 2, 3),
            {1: Fraction(8), 2: Fraction(8), 3: Fraction(8)},
            Fraction(1, 2),
            costs={1: 1, 2: 1, 3: 1},
            capacity=1,
            previous_rejections=(1,),
        )
        bad_capacity = verify_affine_closure_milp_solution(
            bad_capacity_model, rejection,
            minimal_auxiliary_values(bad_capacity_model, rejection)
        )
        self.assertFalse(bad_capacity["valid"])
        self.assertIn("capacity_exceeded", [row[0] for row in bad_capacity["failures"]])

        bad_persistence_model = build_affine_closure_milp(
            (1, 2, 3),
            {1: Fraction(8), 2: Fraction(8), 3: Fraction(8)},
            Fraction(1, 2),
            previous_rejections=(1,),
        )
        bad_persistence = verify_affine_closure_milp_solution(
            bad_persistence_model, {1: 0, 2: 1, 3: 0},
            minimal_auxiliary_values(bad_persistence_model, {1: 0, 2: 1, 3: 0})
        )
        self.assertFalse(bad_persistence["valid"])
        self.assertIn("persistence_violation", [row[0] for row in bad_persistence["failures"]])

    def test_differential_equivalence_to_all_subsets_and_rejection_masks(self):
        alpha = Fraction(1, 2)
        grid = (Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(5))
        for n in range(1, 5):
            active = tuple(range(1, n + 1))
            subsets = tuple(
                subset
                for size in range(1, n + 1)
                for subset in combinations(active, size)
            )
            for evidence in product(grid, repeat=n):
                coordinate_values = dict(zip(active, evidence))
                model = build_affine_closure_milp(active, coordinate_values, alpha)
                for mask in range(1 << n):
                    rejection = {i: int((mask >> (i - 1)) & 1) for i in active}
                    auxiliary = minimal_auxiliary_values(model, rejection)
                    certificate = verify_affine_closure_milp_solution(
                        model, rejection, auxiliary
                    )
                    r = sum(rejection.values())
                    brute = True
                    for subset in subsets:
                        fdp = Fraction(sum(rejection[i] for i in subset), r) if r else Fraction(0)
                        cert = projective_certificate(subset, coordinate_values)
                        if fdp > alpha * cert:
                            brute = False
                            break
                    self.assertEqual(certificate["valid"], brute, (n, evidence, rejection))

    def test_lp_export_scales_rational_rows_to_exact_integers(self):
        model = build_affine_closure_milp(
            (1, 2),
            {1: Fraction(8), 2: Fraction(0)},
            Fraction(1, 2),
            values={1: Fraction(3, 2), 2: Fraction(2)},
        )
        lp = render_cplex_lp(model)
        self.assertIn("Maximize", lp)
        self.assertIn("obj: 3 x_1 + 4 x_2", lp)
        self.assertIn("aux_1: 4 y_1 - 4 x_1 + 7 r >= 0", lp)
        self.assertIn("aux_2: 8 y_2 - 8 x_2 - r >= 0", lp)
        self.assertIn("closure_sum: 2 y_1 + 2 y_2 - r <= 0", lp)
        self.assertIn("Binaries", lp)
        self.assertIn("Generals", lp)
        self.assertTrue(lp.endswith("End\\n"))

    def test_bad_model_inputs_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "alpha"):
            build_affine_closure_milp((1,), {1: 1}, 1)
        with self.assertRaisesRegex(ValueError, "missing coordinate"):
            build_affine_closure_milp((1, 2), {1: 2}, Fraction(1, 2))
        with self.assertRaisesRegex(ValueError, "previous rejections"):
            build_affine_closure_milp((1,), {1: 1}, Fraction(1, 2), previous_rejections=(2,))
        with self.assertRaisesRegex(ValueError, "capacity"):
            build_affine_closure_milp(
                (1,), {1: 1}, Fraction(1, 2), costs={1: 1}
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
