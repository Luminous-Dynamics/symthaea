import copy
import unittest

import numpy as np

from utility_recertification import solve_utility_recertification
from utility_recertification_certificate import verify_primal_dual_certificate


class PrimalDualReceiptTests(unittest.TestCase):
    def setUp(self):
        self.e = np.array([8.0, 4.0, 1.5, 0.4, 0.1])
        self.u = np.array([5.0, 3.0, 2.0, 1.0, 0.25])
        self.alpha = 0.1
        self.solution = solve_utility_recertification(self.e, self.u, self.alpha)

    def test_solver_emits_independently_valid_primal_dual_receipt(self):
        result = verify_primal_dual_certificate(
            self.e, self.u, self.alpha, self.solution
        )
        self.assertTrue(result["valid"], result)
        self.assertLessEqual(result["max_subset_violation"], 1e-8)
        self.assertGreaterEqual(result["minimum_dual_slack"], -1e-8)
        self.assertLessEqual(abs(result["duality_gap"]), 1e-8)

    def test_input_mutation_invalidates_digest(self):
        changed = self.e.copy()
        changed[0] += 0.01
        result = verify_primal_dual_certificate(
            changed, self.u, self.alpha, self.solution
        )
        self.assertFalse(result["valid"])
        self.assertIn("input digest mismatch", result["failures"])

    def test_negative_dual_multiplier_is_rejected(self):
        receipt = copy.deepcopy(self.solution)
        receipt["dual_multipliers"][0] = -1.0
        result = verify_primal_dual_certificate(
            self.e, self.u, self.alpha, receipt
        )
        self.assertFalse(result["valid"])
        self.assertTrue(
            any("negative dual multiplier" in x for x in result["failures"])
        )

    def test_primal_tampering_is_rejected(self):
        receipt = copy.deepcopy(self.solution)
        receipt["primal_vector"][0] += 0.5
        result = verify_primal_dual_certificate(
            self.e, self.u, self.alpha, receipt
        )
        self.assertFalse(result["valid"])

    def test_intersection_witness_must_be_canonical(self):
        receipt = copy.deepcopy(self.solution)
        i = next(
            i for i, row in enumerate(receipt["dual_rows"])
            if row["kind"] == "intersection"
        )
        receipt["dual_rows"][i]["subset"] = [1, 0]
        with self.assertRaises(ValueError):
            verify_primal_dual_certificate(
                self.e, self.u, self.alpha, receipt
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
