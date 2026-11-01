from fractions import Fraction
import unittest

from supremum_anchored_eclosure import (
    build_supremum_anchored_certificates,
    fdp_for_intersection,
    pathwise_supremum_bound,
    select_largest_certified_superset,
    verify_baseline_retention,
    verify_closure_path,
)


def all_auxiliary(universe, horizon, value=Fraction(1)):
    from itertools import combinations
    ids = tuple(sorted(universe))
    return {
        frozenset(row): [value] * horizon
        for size in range(1, len(ids) + 1)
        for row in combinations(ids, size)
    }


class SupremumAnchoredEclosureTests(unittest.TestCase):
    def test_baseline_is_retained_for_every_intersection_and_time(self):
        universe = ("a", "b", "c")
        baseline = [("a",), ("a", "b"), ("b", "c")]
        q = {"a": Fraction(1, 3), "b": Fraction(1, 3), "c": Fraction(1, 3)}
        local = build_supremum_anchored_certificates(
            universe=universe, baseline_path=baseline, coordinate_budgets=q,
            alpha=Fraction(1, 10),
            adjusted_auxiliary=all_auxiliary(universe, len(baseline)),
        )
        self.assertTrue(verify_baseline_retention(
            universe=universe, baseline_path=baseline,
            local_e_paths=local, alpha=Fraction(1, 10),
        ))

    def test_ever_rejected_indicator_and_residual_budget(self):
        universe = ("a", "b")
        baseline = [("a",), ("a", "b"), ("b",)]
        q = {"a": Fraction(1, 4), "b": Fraction(1, 2)}
        local = build_supremum_anchored_certificates(
            universe=universe, baseline_path=baseline, coordinate_budgets=q,
            alpha=Fraction(1, 5),
            adjusted_auxiliary=all_auxiliary(universe, 3, Fraction(2)),
        )
        self.assertEqual(
            local[frozenset({"a"})][0],
            Fraction(1, 1) / Fraction(1, 5) + Fraction(3, 4) * 2,
        )

    def test_pathwise_supremum_certificate(self):
        universe = ("a", "b")
        baseline = [("a",), ("a", "b"), ("b",)]
        q = {"a": Fraction(1, 4), "b": Fraction(1, 2)}
        local = build_supremum_anchored_certificates(
            universe=universe, baseline_path=baseline, coordinate_budgets=q,
            alpha=Fraction(1, 5),
            adjusted_auxiliary=all_auxiliary(universe, 3, Fraction(3, 2)),
        )
        outputs = [("a",), ("a", "b"), ("b",)]
        valid, _ = verify_closure_path(
            universe=universe, rejection_path=outputs,
            local_e_paths=local, alpha=Fraction(1, 5),
        )
        self.assertTrue(valid)
        observed, certificate = pathwise_supremum_bound(
            truth=("a", "b"), rejection_path=outputs,
            local_e_paths=local, alpha=Fraction(1, 5),
        )
        self.assertLessEqual(observed, certificate)

    def test_largest_certified_superset_is_baseline_anchored(self):
        universe = ("a", "b", "c")
        q = {"a": Fraction(1, 3), "b": Fraction(1, 3), "c": Fraction(1, 3)}
        baseline = [("a", "b")]
        paths = build_supremum_anchored_certificates(
            universe=universe, baseline_path=baseline, coordinate_budgets=q,
            alpha=Fraction(1, 5),
            adjusted_auxiliary=all_auxiliary(universe, 1, Fraction(3)),
        )
        chosen = select_largest_certified_superset(
            universe=universe, anchor=("a", "b"),
            local_e_values={s: v[0] for s, v in paths.items()},
            alpha=Fraction(1, 5),
        )
        self.assertTrue(frozenset({"a", "b"}) <= chosen)

    def test_old_intersection_certificate_is_invariant_to_dormant_extension(self):
        old_universe = ("a", "b")
        extended_universe = ("a", "b", "new")
        old_baseline = [("a",), ("a", "b"), ("b",)]
        extended_baseline = [
            ("a", "new"), ("a", "b", "new"), ("b", "new")
        ]
        old_q = {"a": Fraction(1, 4), "b": Fraction(1, 4)}
        extended_q = {
            "a": Fraction(1, 4), "b": Fraction(1, 4), "new": Fraction(1, 4)
        }
        old_local = build_supremum_anchored_certificates(
            universe=old_universe,
            baseline_path=old_baseline,
            coordinate_budgets=old_q,
            alpha=Fraction(1, 5),
            adjusted_auxiliary=all_auxiliary(old_universe, 3, Fraction(2)),
        )
        extended_local = build_supremum_anchored_certificates(
            universe=extended_universe,
            baseline_path=extended_baseline,
            coordinate_budgets=extended_q,
            alpha=Fraction(1, 5),
            adjusted_auxiliary=all_auxiliary(
                extended_universe, 3, Fraction(2)
            ),
        )
        for old_intersection, old_values in old_local.items():
            self.assertEqual(old_values, extended_local[old_intersection])

    def test_rejects_overallocated_or_negative_coordinate_budgets(self):
        for q in (
            {"a": Fraction(3, 4), "b": Fraction(1, 2)},
            {"a": Fraction(-1, 2), "b": Fraction(1, 2)},
        ):
            with self.assertRaises(ValueError):
                build_supremum_anchored_certificates(
                    universe=("a", "b"), baseline_path=[("a",)],
                    coordinate_budgets=q, alpha=Fraction(1, 10),
                    adjusted_auxiliary=all_auxiliary(("a", "b"), 1),
                )

    def test_rejects_missing_intersection_and_auxiliary_values(self):
        with self.assertRaises(ValueError):
            build_supremum_anchored_certificates(
                universe=("a", "b"), baseline_path=[("a",)],
                coordinate_budgets={"a": Fraction(1, 2), "b": Fraction(1, 2)},
                alpha=Fraction(1, 10),
                adjusted_auxiliary={frozenset({"a"}): [Fraction(1)]},
            )

    def test_invalid_auxiliary_path_values_fail_closed(self):
        aux = all_auxiliary(("a",), 2)
        aux[frozenset({"a"})] = [Fraction(1), Fraction(-1)]
        with self.assertRaises(ValueError):
            build_supremum_anchored_certificates(
                universe=("a",), baseline_path=[("a",), tuple()],
                coordinate_budgets={"a": Fraction(1)},
                alpha=Fraction(1, 10), adjusted_auxiliary=aux,
            )

    def test_duplicate_hypothesis_ids_fail_closed(self):
        with self.assertRaises(ValueError):
            build_supremum_anchored_certificates(
                universe=("a", "a"), baseline_path=[("a",)],
                coordinate_budgets={"a": Fraction(1)},
                alpha=Fraction(1, 10),
                adjusted_auxiliary={frozenset({"a"}): [Fraction(1)]},
            )

    def test_empty_rejection_and_empty_truth(self):
        self.assertEqual(fdp_for_intersection((), ("a",)), Fraction(0))
        self.assertEqual(
            pathwise_supremum_bound(
                truth=(), rejection_path=[("a",)],
                local_e_paths={}, alpha=Fraction(1, 10),
            ),
            (Fraction(0), Fraction(0)),
        )


if __name__ == "__main__":
    unittest.main()
