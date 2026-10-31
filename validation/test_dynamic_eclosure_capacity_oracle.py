import copy
import json
import unittest
from fractions import Fraction

from dynamic_eclosure_capacity_oracle import (
    adjust_projective_paths,
    build_projective_intersection_paths,
    select_dynamic_capacity_actions,
    verify_dynamic_collection,
    verify_dynamic_decision_sequence,
)


class DynamicEclosureCapacityTests(unittest.TestCase):
    def setUp(self):
        # Under each coordinate null, M_i jumps from 1 to b_i with
        # probability 1/b_i at its activation update, otherwise to zero,
        # and is then frozen. These are valid one-step test martingales in
        # the common filtration; the fixture selects the all-success path.
        self.coordinate_paths = [
            [1, 8, 8],
            [1, 1, 16],
        ]
        self.weights = [Fraction(1, 2), Fraction(1, 4)]
        self.active_masks = [0b01, 0b11, 0b11]
        self.alpha = Fraction(1, 2)
        self.values = [10, 8]
        self.costs = [1, 1]
        self.capacity = [0, 1, 2]
        self.thresholds = [2, 4, 8, 16]
        self.increments = [1, 1, 1, 1]

    def test_projective_dynamic_closure_and_capacity_actions(self):
        raw = build_projective_intersection_paths(
            self.coordinate_paths, self.weights, self.active_masks
        )
        self.assertEqual(raw[0b01][1], Fraction(9, 2))
        self.assertEqual(raw[0b01][2], Fraction(9, 2))
        # Adding a dormant hypothesis does not change the earlier certificate.
        self.assertEqual(raw[0b01][1], raw[0b11][1])

        adjusted, spec = adjust_projective_paths(
            raw, thresholds=self.thresholds, increments=self.increments
        )
        collection = verify_dynamic_collection(adjusted, self.active_masks)
        self.assertTrue(collection["valid"], collection)
        self.assertEqual(collection["future_extension_checks"], 7)
        self.assertGreater(collection["projective_equalities"], 0)
        self.assertEqual(spec["integral_budget"], Fraction(15, 16))

        result = select_dynamic_capacity_actions(
            self.coordinate_paths,
            self.weights,
            self.active_masks,
            self.alpha,
            self.values,
            self.costs,
            self.capacity,
            thresholds=self.thresholds,
            increments=self.increments,
        )
        decisions = result["decisions"]
        self.assertEqual(decisions[0]["rejections"], [])
        self.assertEqual(decisions[1]["rejections"], [1])
        self.assertEqual(decisions[2]["rejections"], [1, 2])
        self.assertTrue(result["verification"]["valid"])
        self.assertTrue(result["verification"]["optimality_verified"])
        json.dumps(result, sort_keys=True, separators=(",", ":"))

    def test_future_extension_coherence_negative_control(self):
        bad = {
            0b01: [0, 2],
            0b10: [0, 0],
            0b11: [0, 1],
        }
        result = verify_dynamic_collection(bad, [0b01, 0b11])
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                isinstance(failure, tuple)
                and failure[0] == "future_extension_coherence_failure"
                for failure in result["failures"]
            )
        )

    def test_time_monotonicity_negative_control(self):
        bad = {
            0b01: [0, 2, 1],
            0b10: [0, 0, 0],
            0b11: [0, 2, 1],
        }
        result = verify_dynamic_collection(bad, [0b01, 0b11, 0b11])
        self.assertFalse(result["valid"])
        self.assertTrue(
            any(
                isinstance(failure, tuple)
                and failure[0] == "time_monotonicity_failure"
                for failure in result["failures"]
            )
        )

    def test_dormant_coordinate_must_be_neutral_one(self):
        bad_paths = [
            [1, 8, 8],
            [Fraction(9, 10), 1, 16],
        ]
        with self.assertRaises(ValueError):
            select_dynamic_capacity_actions(
                bad_paths, self.weights, self.active_masks, self.alpha,
                self.values, self.costs, self.capacity,
                thresholds=self.thresholds, increments=self.increments,
            )

    def test_global_weight_budget_is_not_renormalized_at_arrival(self):
        with self.assertRaises(ValueError):
            select_dynamic_capacity_actions(
                self.coordinate_paths,
                [Fraction(1, 2), Fraction(3, 4)],
                self.active_masks, self.alpha, self.values, self.costs,
                self.capacity, thresholds=self.thresholds,
                increments=self.increments,
            )

    def test_adjuster_integral_budget_is_enforced(self):
        with self.assertRaises(ValueError):
            select_dynamic_capacity_actions(
                self.coordinate_paths, self.weights, self.active_masks,
                self.alpha, self.values, self.costs, self.capacity,
                thresholds=[2, 4], increments=[2, 2],
            )

    def test_verifier_rejects_valid_but_suboptimal_selection(self):
        result = select_dynamic_capacity_actions(
            self.coordinate_paths, self.weights, self.active_masks,
            self.alpha, self.values, self.costs, self.capacity,
            thresholds=self.thresholds, increments=self.increments,
        )
        tampered = copy.deepcopy(result["decisions"])
        # At the final time {1} remains certified, but {1,2} has greater
        # utility and fits the available cumulative capacity.
        tampered[2]["rejection_mask"] = 0b01
        tampered[2]["rejections"] = [1]
        tampered[2]["utility"] = [10, 1]
        tampered[2]["cumulative_cost"] = [1, 1]
        tampered[2]["minimum_closure_slack"] = [0, 1]
        tampered[2]["worst_intersection_mask"] = 0b01
        check = verify_dynamic_decision_sequence(
            tampered,
            self.coordinate_paths,
            self.weights,
            self.active_masks,
            self.alpha,
            self.values,
            self.costs,
            self.capacity,
            {
                "thresholds": self.thresholds,
                "increments": self.increments,
            },
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

    def test_future_evidence_does_not_rewrite_past_decisions(self):
        first = select_dynamic_capacity_actions(
            self.coordinate_paths, self.weights, self.active_masks,
            self.alpha, self.values, self.costs, self.capacity,
            thresholds=self.thresholds, increments=self.increments,
        )
        changed = copy.deepcopy(self.coordinate_paths)
        changed[1][2] = 32
        second = select_dynamic_capacity_actions(
            changed, self.weights, self.active_masks,
            self.alpha, self.values, self.costs, self.capacity,
            thresholds=self.thresholds, increments=self.increments,
        )
        self.assertEqual(
            [row["rejections"] for row in first["decisions"][:2]],
            [row["rejections"] for row in second["decisions"][:2]],
        )
        self.assertEqual(
            [row["prefix_sha256"] for row in first["decisions"][:2]],
            [row["prefix_sha256"] for row in second["decisions"][:2]],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
