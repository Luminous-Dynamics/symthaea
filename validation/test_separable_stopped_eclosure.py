import copy
import json
import random
import unittest
from fractions import Fraction

from separable_stopped_eclosure import (
    exhaustive_intersection_certificate,
    select_raw_projective_capacity_actions,
    verify_raw_projective_rejection_set,
    verify_raw_projective_sequence,
)


class SeparableStoppedEclosureTests(unittest.TestCase):
    def test_positive_part_separator_matches_exhaustive_all_intersections(self):
        rng = random.Random(20261031)
        for n in range(1, 9):
            for _ in range(80):
                active = rng.randrange(1, 1 << n)
                weights_raw = [rng.randrange(0, 20) for _ in range(n)]
                total = sum(weights_raw)
                weights = [
                    Fraction(x, max(total, 1)) * Fraction(3, 4)
                    for x in weights_raw
                ]
                evalues = [
                    Fraction(rng.randrange(0, 81), 10)
                    for _ in range(n)
                ]
                for i in range(n):
                    if not (active >> i) & 1:
                        evalues[i] = Fraction(1)
                alpha = Fraction(1, rng.choice([10, 20, 4, 2]))
                rejection = rng.randrange(0, 1 << n) & active
                if rejection == 0:
                    continue
                fast = verify_raw_projective_rejection_set(
                    rejection, evalues, weights, active, alpha
                )
                slow = exhaustive_intersection_certificate(
                    rejection, evalues, weights, active, alpha
                )
                self.assertEqual(fast["certified"], slow["certified"])
                self.assertEqual(fast["minimum_slack"], slow["minimum_slack"])
                witness = fast["worst_intersection_mask"]
                rejection_count = rejection.bit_count()
                witness_fdp = Fraction((witness & rejection).bit_count(), rejection_count)
                witness_evalue = Fraction(1) + sum(
                    (
                        weights[i] * (evalues[i] - 1)
                        for i in range(n) if (witness >> i) & 1
                    ),
                    Fraction(0),
                )
                self.assertEqual(
                    alpha * witness_evalue - witness_fdp,
                    fast["minimum_slack"],
                )

    def test_high_value_but_uncertified_candidate_is_rejected(self):
        e = [Fraction(1, 10), 1]
        weights = [Fraction(1, 2), Fraction(1, 2)]
        check = verify_raw_projective_rejection_set(
            0b01, e, weights, 0b11, Fraction(1, 2)
        )
        self.assertFalse(check["certified"])
        self.assertLess(check["minimum_slack"], 0)

    def test_capacity_aware_nested_selection_and_optimality(self):
        paths = [
            [4, 4, 4],
            [1, 1, 4],
        ]
        weights = [Fraction(1, 2), Fraction(1, 4)]
        result = select_raw_projective_capacity_actions(
            paths,
            weights,
            [0b01, 0b01, 0b11],
            Fraction(1, 2),
            [10, 8],
            [1, 1],
            [1, 1, 2],
        )
        self.assertEqual(
            [row["rejections"] for row in result["decisions"]],
            [[1], [1], [1, 2]],
        )
        self.assertEqual(
            result["guarantee_scope"].split(";")[0],
            "simultaneous stopped-FDR via future-extension-coherent intersection e-processes",
        )
        self.assertIn("NOT SupFDR", result["guarantee_scope"])
        self.assertTrue(result["verification"]["valid"])
        self.assertTrue(result["verification"]["optimality_verified"])
        json.dumps(result, sort_keys=True, separators=(",", ":"))

    def test_future_evidence_cannot_rewrite_past_decisions(self):
        common = {
            "weights": [Fraction(1, 2), Fraction(1, 4)],
            "active_masks": [0b01, 0b01, 0b11],
            "alpha": Fraction(1, 2),
            "values": [10, 8],
            "costs": [1, 1],
            "capacities": [1, 1, 2],
        }
        first = select_raw_projective_capacity_actions(
            [[4, 4, 4], [1, 1, 4]],
            common["weights"], common["active_masks"], common["alpha"],
            common["values"], common["costs"], common["capacities"],
        )
        second = select_raw_projective_capacity_actions(
            [[4, 4, 4], [1, 1, 100]],
            common["weights"], common["active_masks"], common["alpha"],
            common["values"], common["costs"], common["capacities"],
        )
        self.assertEqual(
            [row["rejections"] for row in first["decisions"][:2]],
            [row["rejections"] for row in second["decisions"][:2]],
        )
        self.assertEqual(
            [row["prefix_sha256"] for row in first["decisions"][:2]],
            [row["prefix_sha256"] for row in second["decisions"][:2]],
        )

    def test_independent_verifier_rejects_valid_but_suboptimal_selection(self):
        result = select_raw_projective_capacity_actions(
            [[4, 4], [4, 4]],
            [Fraction(1, 2), Fraction(1, 4)],
            [0b11, 0b11],
            Fraction(1, 2),
            [10, 8],
            [1, 1],
            [1, 2],
        )
        tampered = copy.deepcopy(result["decisions"])
        # {1} is still certified, but {1,2} has greater utility at t=1.
        tampered[1]["rejection_mask"] = 0b01
        tampered[1]["rejections"] = [1]
        tampered[1]["utility"] = [10, 1]
        tampered[1]["cost"] = [1, 1]
        tampered[1]["minimum_stopped_fdr_slack"] = [1, 4]
        tampered[1]["worst_intersection_mask"] = 0b01
        check = verify_raw_projective_sequence(
            tampered,
            tuple(tuple(Fraction(x) for x in row) for row in [[4, 4], [4, 4]]),
            (Fraction(1, 2), Fraction(1, 4)),
            (0b11, 0b11),
            Fraction(1, 2),
            (Fraction(10), Fraction(8)),
            (Fraction(1), Fraction(1)),
            (Fraction(1), Fraction(2)),
        )
        self.assertFalse(check["valid"])
        self.assertFalse(check["optimality_verified"])
        self.assertTrue(
            any(
                isinstance(f, tuple) and f[0] == "utility_optimality_failure"
                for f in check["failures"]
            )
        )

    def test_inactive_coordinate_must_be_neutral(self):
        with self.assertRaises(ValueError):
            select_raw_projective_capacity_actions(
                [[4, 4], [2, 4]],
                [Fraction(1, 2), Fraction(1, 4)],
                [0b01, 0b11],
                Fraction(1, 2),
                [1, 1],
                [1, 1],
                [1, 2],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
