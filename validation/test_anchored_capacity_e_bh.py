import json
import unittest
from fractions import Fraction

from anchored_capacity_e_bh import (
    anchored_local_evalue,
    e_bh_baseline,
    select_anchored_capacity_e_bh,
    verify_anchored_baseline,
    verify_anchored_capacity_e_bh,
)


class AnchoredCapacityEbhTests(unittest.TestCase):
    def test_e_bh_anchor_is_always_inside_anchored_closure(self):
        e = [30, 14, 6]
        alpha = Fraction(1, 10)
        baseline = e_bh_baseline(e, alpha)
        self.assertEqual(baseline["rejections"], [1])
        self.assertEqual(baseline["k_star"], 1)
        check = verify_anchored_baseline(e, alpha)
        self.assertTrue(check["valid"], check)
        self.assertEqual(check["intersections_checked"], 7)

    def test_capacity_aware_extension_retains_e_bh_and_uses_followup_slots(self):
        e = [30, 14, 6]
        alpha = Fraction(1, 10)
        values = [1, 100, 1]
        costs = [1, 1, 1]
        result = select_anchored_capacity_e_bh(
            e, alpha, values, costs, capacity=2
        )
        self.assertEqual(result["baseline"]["rejections"], [1])
        self.assertEqual(result["selected"]["decisions"][0]["rejections"], [1, 2])
        self.assertTrue(result["baseline_retained"])
        check = verify_anchored_capacity_e_bh(result, e, alpha, values, costs)
        self.assertTrue(check["valid"], check)
        self.assertTrue(check["pareto_optimality_verified"])
        json.dumps(result, sort_keys=True, separators=(",", ":"))

    def test_anchored_closure_can_discover_when_ordinary_e_bh_is_empty(self):
        # Ordinary e-BH has no k meeting e_(k) >= m/(alpha*k), but the
        # anchored local collection has enough global slack to certify {1}.
        e = [29, 1, 0]
        alpha = Fraction(1, 10)
        baseline = e_bh_baseline(e, alpha)
        self.assertEqual(baseline["rejections"], [])
        result = select_anchored_capacity_e_bh(
            e,
            alpha,
            values=[10, 2, 1],
            costs=[1, 1, 1],
            capacity=1,
        )
        self.assertEqual(result["selected"]["decisions"][0]["rejections"], [1])
        self.assertTrue(result["baseline_retained"])
        check = verify_anchored_capacity_e_bh(
            result, e, alpha, [10, 2, 1], [1, 1, 1]
        )
        self.assertTrue(check["valid"], check)

    def test_zero_capacity_must_not_drop_baseline(self):
        with self.assertRaisesRegex(ValueError, "below the cost of the e-BH anchor"):
            select_anchored_capacity_e_bh(
                [30, 14, 6],
                Fraction(1, 10),
                [1, 1, 1],
                [1, 1, 1],
                capacity=0,
            )

    def test_local_e_collection_identity_and_nonnegativity(self):
        e = [Fraction(29), Fraction(1), Fraction(0)]
        m = len(e)
        for subset in range(1, 1 << m):
            local = anchored_local_evalue(e, subset)
            expected = (
                Fraction(1)
                - Fraction(subset.bit_count(), m)
                + sum(
                    (e[i] for i in range(m) if (subset >> i) & 1),
                    Fraction(0),
                ) / m
            )
            self.assertEqual(local, expected)
            self.assertGreaterEqual(local, 0)

    def test_tied_e_values_keep_all_threshold_ties(self):
        baseline = e_bh_baseline(
            [30, 30, 1],
            Fraction(1, 10),
        )
        self.assertEqual(baseline["k_star"], 2)
        self.assertEqual(baseline["rejections"], [1, 2])
        self.assertEqual(baseline["rejection_count"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
