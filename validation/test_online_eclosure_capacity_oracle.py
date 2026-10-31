import json
import unittest
from fractions import Fraction

from online_eclosure_capacity_oracle import (
    select_capacity_aware_sequence,
    verify_certified_set,
    verify_decision_sequence,
)


class OnlineEclosureCapacityTests(unittest.TestCase):
    def test_capacity_aware_nested_selection_inside_certified_menu(self):
        e = [4, 4]
        gamma = [Fraction(1, 2), Fraction(1, 2)]
        delta = Fraction(1, 2)
        values = [10, 5]
        costs = [1, 1]
        capacity = [1, 2]

        result = select_capacity_aware_sequence(
            e, gamma, delta, values, costs, capacity
        )
        decisions = result["decisions"]
        self.assertEqual(decisions[0]["rejections"], [1])
        self.assertEqual(decisions[1]["rejections"], [1, 2])
        self.assertLessEqual(decisions[0]["cumulative_cost"], 1)
        self.assertLessEqual(decisions[1]["cumulative_cost"], 2)
        self.assertTrue(result["verification"]["valid"])
        # The receipt is canonical-JSON-friendly for immutable replay.
        json.dumps(result, sort_keys=True, separators=(",", ":"))

    def test_independent_verifier_rejects_valid_but_suboptimal_set(self):
        import copy

        result = select_capacity_aware_sequence(
            [4, 4],
            [Fraction(1, 2), Fraction(1, 2)],
            Fraction(1, 2),
            [10, 5],
            [1, 1],
            [1, 2],
        )
        tampered = copy.deepcopy(result["decisions"])
        # {1} remains a valid closed-eBH rejection set at t=2, but
        # {1,2} has strictly greater value under the same capacity.
        tampered[1]["rejection_mask"] = 0b01
        tampered[1]["rejections"] = [1]
        tampered[1]["utility"] = [10, 1]
        tampered[1]["cumulative_cost"] = [1, 1]
        tampered[1]["minimum_closure_slack"] = [0, 1]
        tampered[1]["worst_intersection_mask"] = 0
        tampered[1]["intersections_checked_for_selected_set"] = 4
        check = verify_decision_sequence(
            tampered,
            [4, 4],
            [Fraction(1, 2), Fraction(1, 2)],
            Fraction(1, 2),
            [10, 5],
            [1, 1],
            [1, 2],
        )
        self.assertFalse(check["valid"])
        self.assertFalse(check["optimality_verified"])
        self.assertTrue(
            any(
                isinstance(failure, tuple)
                and failure[0] == "utility_optimality_failure"
                for failure in check["failures"]
            )
        )

    def test_invalid_new_hypothesis_is_not_added_even_with_capacity(self):
        result = select_capacity_aware_sequence(
            [4, Fraction(1, 10)],
            [Fraction(1, 2), Fraction(1, 2)],
            Fraction(1, 2),
            [10, 100],
            [1, 1],
            [1, 2],
        )
        self.assertEqual(result["decisions"][0]["rejections"], [1])
        self.assertEqual(result["decisions"][1]["rejections"], [1])
        check = verify_certified_set(
            0b11, 2, [4, Fraction(1, 10)],
            [Fraction(1, 2), Fraction(1, 2)], Fraction(1, 2)
        )
        self.assertFalse(check["certified"])

    def test_future_evidence_cannot_change_past_decisions(self):
        common = {
            "gamma": [Fraction(1, 2), Fraction(1, 4), Fraction(1, 4)],
            "delta": Fraction(1, 2),
            "values": [8, 5, 3],
            "costs": [1, 1, 1],
            "cumulative_capacity": [1, 2, 3],
        }
        first = select_capacity_aware_sequence(
            [4, 4, 100],
            common["gamma"], common["delta"], common["values"],
            common["costs"], common["cumulative_capacity"]
        )
        second = select_capacity_aware_sequence(
            [4, 4, Fraction(1, 1000)],
            common["gamma"], common["delta"], common["values"],
            common["costs"], common["cumulative_capacity"]
        )
        self.assertEqual(
            [x["rejections"] for x in first["decisions"][:2]],
            [x["rejections"] for x in second["decisions"][:2]],
        )
        self.assertEqual(
            [x["prefix_sha256"] for x in first["decisions"][:2]],
            [x["prefix_sha256"] for x in second["decisions"][:2]],
        )

    def test_capacity_is_cumulative_and_previous_decisions_irrevocable(self):
        result = select_capacity_aware_sequence(
            [8, 8, 8],
            [Fraction(1, 3)] * 3,
            Fraction(1, 2),
            [10, 9, 8],
            [1, 1, 1],
            [0, 1, 2],
        )
        self.assertEqual(result["decisions"][0]["rejections"], [])
        self.assertEqual(result["decisions"][1]["rejections"], [1])
        self.assertEqual(result["decisions"][2]["rejections"], [1, 2])

    def test_rejects_nonmonotone_capacity_and_bad_gamma(self):
        args = ([4, 4], [Fraction(1, 2), Fraction(1, 2)],
                Fraction(1, 2), [1, 1], [1, 1])
        with self.assertRaises(ValueError):
            select_capacity_aware_sequence(*args, [2, 1])
        with self.assertRaises(ValueError):
            select_capacity_aware_sequence(
                [4, 4], [1, 1], Fraction(1, 2), [1, 1], [1, 1], [1, 2]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
