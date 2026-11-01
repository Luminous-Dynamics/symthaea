from fractions import Fraction
from itertools import combinations
import unittest

from supremum_anchored_eclosure import (
    build_supremum_anchored_certificates,
    build_additive_anchored_auxiliary,
    build_additive_anchored_coordinate_e_paths,
    select_additive_anchored_e_bh,
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

    def test_additive_auxiliary_matches_anchored_intersection_paths(self):
        universe = ("a", "b", "c")
        baseline = [("a",), ("a", "b")]
        q = {
            "a": Fraction(1, 10),
            "b": Fraction(1, 10),
            "c": Fraction(1, 10),
        }
        v = {
            "a": Fraction(1, 5),
            "b": Fraction(1, 5),
            "c": Fraction(1, 5),
        }
        coordinate_aux = {
            "a": [Fraction(2), Fraction(3)],
            "b": [Fraction(1), Fraction(2)],
            "c": [Fraction(4), Fraction(1)],
        }
        alpha = Fraction(1, 5)

        local_aux = build_additive_anchored_auxiliary(
            universe=universe,
            coordinate_budgets=q,
            auxiliary_budgets=v,
            coordinate_auxiliary_paths=coordinate_aux,
        )
        local_paths = build_supremum_anchored_certificates(
            universe=universe,
            baseline_path=baseline,
            coordinate_budgets=q,
            alpha=alpha,
            adjusted_auxiliary=local_aux,
        )
        coordinate_paths = build_additive_anchored_coordinate_e_paths(
            universe=universe,
            baseline_path=baseline,
            coordinate_budgets=q,
            auxiliary_budgets=v,
            coordinate_auxiliary_paths=coordinate_aux,
            alpha=alpha,
        )
        for intersection, path in local_paths.items():
            for t, value in enumerate(path):
                self.assertEqual(
                    value,
                    sum(
                        (coordinate_paths[i][t] for i in intersection),
                        Fraction(0),
                    ),
                )

    def test_additive_auxiliary_rejects_empty_horizon(self):
        with self.assertRaises(ValueError):
            build_additive_anchored_auxiliary(
                universe=("a",),
                coordinate_budgets={"a": Fraction(1, 2)},
                auxiliary_budgets={"a": Fraction(1, 2)},
                coordinate_auxiliary_paths={"a": []},
            )

    def test_additive_budget_residual_zero_and_fail_closed(self):
        universe = ("a", "b")
        zero_aux = {"a": [Fraction(0)], "b": [Fraction(0)]}
        local = build_additive_anchored_auxiliary(
            universe=universe,
            coordinate_budgets={"a": Fraction(1), "b": Fraction(0)},
            auxiliary_budgets={"a": Fraction(0), "b": Fraction(0)},
            coordinate_auxiliary_paths=zero_aux,
        )
        self.assertTrue(all(path == (Fraction(0),) for path in local.values()))
        with self.assertRaises(ValueError):
            build_additive_anchored_auxiliary(
                universe=universe,
                coordinate_budgets={"a": Fraction(3, 4), "b": Fraction(0)},
                auxiliary_budgets={"a": Fraction(1, 2), "b": Fraction(0)},
                coordinate_auxiliary_paths=zero_aux,
            )
        with self.assertRaises(ValueError):
            build_additive_anchored_coordinate_e_paths(
                universe=universe,
                baseline_path=[("a",)],
                coordinate_budgets={"a": Fraction(1, 2), "b": Fraction(0)},
                auxiliary_budgets={"a": Fraction(1, 2), "b": Fraction(0)},
                coordinate_auxiliary_paths={
                    "a": [Fraction(1)], "b": [Fraction(-1)]
                },
                alpha=Fraction(1, 10),
            )

    def test_additive_anchored_ebh_matches_exhaustive_closure(self):
        universe = ("a", "b", "c", "d")
        baseline = [("a",), ("a", "b")]
        q = {i: Fraction(1, 20) for i in universe}
        v = {i: Fraction(1, 10) for i in universe}
        coordinate_aux = {
            "a": [Fraction(6), Fraction(2)],
            "b": [Fraction(5), Fraction(7)],
            "c": [Fraction(7), Fraction(1)],
            "d": [Fraction(0), Fraction(6)],
        }
        alpha = Fraction(1, 5)
        paths = build_additive_anchored_coordinate_e_paths(
            universe=universe,
            baseline_path=baseline,
            coordinate_budgets=q,
            auxiliary_budgets=v,
            coordinate_auxiliary_paths=coordinate_aux,
            alpha=alpha,
        )
        for t in range(len(baseline)):
            coordinate_values = {i: paths[i][t] for i in universe}
            ebh = select_additive_anchored_e_bh(
                coordinate_e_values=coordinate_values,
                alpha=alpha,
                anchor=baseline[t],
            )
            local_values = {
                intersection: sum(
                    (coordinate_values[i] for i in intersection),
                    Fraction(0),
                )
                for intersection in (
                    frozenset(group)
                    for size in range(1, len(universe) + 1)
                    for group in combinations(universe, size)
                )
            }
            exhaustive = select_largest_certified_superset(
                universe=universe,
                anchor=baseline[t],
                local_e_values=local_values,
                alpha=alpha,
            )
            self.assertEqual(ebh, exhaustive)
            valid, _ = verify_closure_path(
                universe=universe,
                rejection_path=[ebh],
                local_e_paths={
                    key: (value,) for key, value in local_values.items()
                },
                alpha=alpha,
            )
            self.assertTrue(valid)

    def test_additive_ebh_always_retains_ever_rejected_baseline(self):
        universe = ("a", "b", "c")
        baseline = [("a",), ("a", "b"), ("b", "c")]
        q = {i: Fraction(1, 10) for i in universe}
        v = {i: Fraction(1, 10) for i in universe}
        coordinate_aux = {
            "a": [Fraction(0), Fraction(0), Fraction(0)],
            "b": [Fraction(0), Fraction(0), Fraction(0)],
            "c": [Fraction(0), Fraction(0), Fraction(0)],
        }
        paths = build_additive_anchored_coordinate_e_paths(
            universe=universe,
            baseline_path=baseline,
            coordinate_budgets=q,
            auxiliary_budgets=v,
            coordinate_auxiliary_paths=coordinate_aux,
            alpha=Fraction(1, 10),
        )
        for t, baseline_set in enumerate(baseline):
            chosen = select_additive_anchored_e_bh(
                coordinate_e_values={i: paths[i][t] for i in universe},
                alpha=Fraction(1, 10),
                anchor=baseline_set,
            )
            self.assertTrue(frozenset(baseline_set) <= chosen)

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
