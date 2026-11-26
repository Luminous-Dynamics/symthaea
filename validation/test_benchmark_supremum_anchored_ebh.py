import unittest
from fractions import Fraction

from benchmark_supremum_anchored_ebh import (
    _exhaustive_closure,
    _linear_closure_verify,
    _make_values,
    run,
)
from supremum_anchored_eclosure import select_additive_anchored_e_bh


class SupremumAnchoredEbhBenchmarkTests(unittest.TestCase):
    def test_e_bh_matches_exhaustive_oracle_on_small_family(self):
        alpha = Fraction(1, 5)
        values, anchor = _make_values(10, 978101, alpha)
        selected = select_additive_anchored_e_bh(
            coordinate_e_values=values, alpha=alpha, anchor=anchor
        )
        exhaustive = _exhaustive_closure(values, anchor, alpha)
        self.assertEqual(selected, exhaustive)
        self.assertTrue(_linear_closure_verify(values, selected, alpha))

    def test_runner_checks_anchor_and_closure_for_all_sizes(self):
        result = run(sizes=(8, 12, 20, 64), repeats=2, seed=978102)
        self.assertEqual(result["benchmark"], "supremum_anchored_additive_ebh")
        for row in result["results"]:
            self.assertTrue(row["anchor_retained"])
            self.assertTrue(row["closure_feasible"])
            if row["n"] <= 16:
                self.assertTrue(row["matches_exhaustive"])
            else:
                self.assertIsNone(row["matches_exhaustive"])

    def test_additive_closure_cutoff_equivalence(self):
        alpha = Fraction(1, 4)
        values = {
            "a": Fraction(4),
            "b": Fraction(2),
            "c": Fraction(1),
            "d": Fraction(1, 4),
        }
        selected = select_additive_anchored_e_bh(
            coordinate_e_values=values, alpha=alpha
        )
        self.assertEqual(selected, frozenset({"a", "b"}))
        self.assertTrue(_linear_closure_verify(values, selected, alpha))


if __name__ == "__main__":
    unittest.main()
