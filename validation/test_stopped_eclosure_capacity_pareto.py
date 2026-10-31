import copy
import json
import random
import unittest
from fractions import Fraction

from stopped_eclosure_capacity_pareto import (
    _charges_for_cardinality,
    _verify_separator,
    select_capacity_aware_pareto,
    verify_capacity_aware_pareto,
)


def exhaustive_optimum(evalues, weights, active, alpha, values, costs, capacity, previous):
    n = len(evalues)
    best = None
    for mask in range(1 << n):
        if mask & active != mask or mask & previous != previous:
            continue
        selected = tuple(i for i in range(n) if (mask >> i) & 1)
        total_cost = sum((costs[i] for i in selected), Fraction(0))
        if total_cost > capacity:
            continue
        check = _verify_separator(evalues, weights, active, alpha, mask)
        if not check["certified"]:
            continue
        total_value = sum((values[i] for i in selected), Fraction(0))
        load = alpha - check["minimum_slack"] if selected else Fraction(0)
        candidate = (selected, total_value, total_cost, load)
        if best is None or (
            -candidate[1], candidate[2], candidate[3], candidate[0]
        ) < (
            -best[1], best[2], best[3], best[0]
        ):
            best = candidate
    return best


class StoppedEclosureParetoTests(unittest.TestCase):
    def test_charge_decomposition_equals_positive_part_separator(self):
        rng = random.Random(20261031)
        for n in range(1, 9):
            for _ in range(50):
                e = tuple(Fraction(rng.randrange(0, 100), 10) for _ in range(n))
                raw = [rng.randrange(1, 20) for _ in range(n)]
                total = sum(raw)
                w = tuple(Fraction(x, 3 * total) for x in raw)
                active = rng.randrange(1, 1 << n)
                for i in range(n):
                    if not ((active >> i) & 1):
                        e = e[:i] + (Fraction(1),) + e[i + 1:]
                alpha = Fraction(1, 10)
                k = rng.randrange(1, active.bit_count() + 1)
                baseline, charges = _charges_for_cardinality(e, w, active, alpha, k)
                mask = 0
                active_indices = [i for i in range(n) if (active >> i) & 1]
                for i in active_indices:
                    if rng.random() < 0.5:
                        mask |= 1 << i
                if mask.bit_count() != k:
                    chosen = active_indices[:k]
                    mask = sum(1 << i for i in chosen)
                load = baseline + sum(
                    (charges[i] for i in range(n) if (mask >> i) & 1),
                    Fraction(0),
                )
                check = _verify_separator(e, w, active, alpha, mask)
                self.assertTrue(check["certified"] == (alpha - load >= 0))
                self.assertEqual(check["minimum_slack"], alpha - load)

    def test_pareto_dp_matches_exhaustive_random_small_families(self):
        rng = random.Random(20261032)
        for n in range(1, 9):
            for _ in range(60):
                e = tuple(Fraction(rng.randrange(0, 81), 10) for _ in range(n))
                raw = [rng.randrange(1, 12) for _ in range(n)]
                total = sum(raw)
                w = tuple(Fraction(x, 4 * total) for x in raw)
                v = tuple(Fraction(rng.randrange(0, 50), 5) for _ in range(n))
                c = tuple(Fraction(rng.randrange(1, 12), 2) for _ in range(n))
                active = rng.randrange(1, 1 << n)
                previous = 0
                for i in range(n):
                    if ((active >> i) & 1) and rng.random() < 0.12:
                        previous |= 1 << i
                capacity = Fraction(rng.randrange(0, 50), 2)
                alpha = Fraction(1, 10)
                result = select_capacity_aware_pareto(
                    [[e[i]] for i in range(n)],
                    w,
                    [active],
                    alpha,
                    v,
                    c,
                    [capacity],
                    verify_independent=True,
                    frontier_limit=100_000,
                    independent_frontier_limit=100_000,
                )
                expected = exhaustive_optimum(
                    e, w, active, alpha, v, c, capacity, previous
                )
                # This one-update test has no mandatory prior commitments.
                self.assertEqual(
                    tuple(result["decisions"][0]["rejections"]),
                    tuple(i + 1 for i in expected[0]) if expected else (),
                )

    def test_nested_capacity_aware_sequence_and_independent_replay(self):
        paths = [
            [4, 4, 4],
            [1, 1, 4],
            [1, 1, 1],
        ]
        weights = [Fraction(1, 2), Fraction(1, 4), Fraction(1, 8)]
        values = [10, 8, 1]
        costs = [1, 1, 1]
        result = select_capacity_aware_pareto(
            paths,
            weights,
            [0b001, 0b011, 0b111],
            Fraction(1, 2),
            values,
            costs,
            [1, 1, 2],
        )
        self.assertEqual(
            [row["rejections"] for row in result["decisions"]],
            [[1], [1], [1, 2]],
        )
        self.assertTrue(result["verification"]["valid"])
        self.assertTrue(result["verification"]["independent_optimality_verified"])
        json.dumps(result, sort_keys=True, separators=(",", ":"))

    def test_future_strong_evidence_can_restore_a_nested_certificate(self):
        # At t=1, {1} is certified. At t=2, its raw certificate fails, but
        # adding a sufficiently strong new hypothesis makes the superset {1,2}
        # certified. This shows raw closure is not setwise persistent, yet a
        # nested extension can sometimes recover feasibility.
        result = select_capacity_aware_pareto(
            [[4, 1], [1, 10]],
            [Fraction(1, 2), Fraction(1, 2)],
            [0b01, 0b11],
            Fraction(3, 5),
            [10, 1],
            [1, 1],
            [1, 2],
        )
        self.assertEqual(
            [row["rejections"] for row in result["decisions"]],
            [[1], [1, 2]],
        )
        failed_single = _verify_separator(
            (Fraction(1), Fraction(10)),
            (Fraction(1, 2), Fraction(1, 2)),
            0b11,
            Fraction(3, 5),
            0b01,
        )
        self.assertFalse(failed_single["certified"])
        self.assertTrue(result["verification"]["valid"])

    def test_raw_menu_infeasibility_with_irrevocable_set_fails_closed(self):
        # One hypothesis loses evidence (a valid e-process may decrease).
        # Its old singleton is no longer in the raw closure menu and no
        # superset exists to restore certification.
        with self.assertRaisesRegex(RuntimeError, "not setwise persistent"):
            select_capacity_aware_pareto(
                [[4, 1]],
                [Fraction(1, 2)],
                [0b1, 0b1],
                Fraction(1, 2),
                [10],
                [1],
                [1, 1],
            )

    def test_independent_verifier_rejects_valid_but_suboptimal_decision(self):
        result = select_capacity_aware_pareto(
            [[20, 20], [20, 20]],
            [Fraction(1, 2), Fraction(1, 4)],
            [0b11, 0b11],
            Fraction(1, 2),
            [10, 8],
            [1, 1],
            [1, 2],
        )
        tampered = copy.deepcopy(result["decisions"])
        tampered[1]["rejection_mask"] = 0b01
        tampered[1]["rejections"] = [1]
        tampered[1]["utility"] = [10, 1]
        tampered[1]["cost"] = [1, 1]
        check = verify_capacity_aware_pareto(
            tampered,
            ((Fraction(20), Fraction(20)), (Fraction(20), Fraction(20))),
            (Fraction(1, 2), Fraction(1, 4)),
            (0b11, 0b11),
            Fraction(1, 2),
            (Fraction(10), Fraction(8)),
            (Fraction(1), Fraction(1)),
            (Fraction(1), Fraction(2)),
        )
        self.assertFalse(check["valid"])
        self.assertFalse(check["independent_optimality_verified"])
        self.assertTrue(
            any(
                isinstance(x, tuple) and x[0] == "utility_optimality_failure"
                for x in check["failures"]
            )
        )

    def test_one_hundred_hypothesis_frontier_smoke(self):
        n = 100
        e = [Fraction(1) if i < 70 else Fraction(2000) for i in range(n)]
        weights = [Fraction(1, n)] * n
        values = [Fraction(i + 1) for i in range(n)]
        costs = [Fraction(1)] * n
        active = (1 << n) - 1
        result = select_capacity_aware_pareto(
            [[x] for x in e],
            weights,
            [active],
            Fraction(1, 10),
            values,
            costs,
            [30],
            frontier_limit=20_000,
            independent_frontier_limit=20_000,
        )
        self.assertEqual(len(result["decisions"][0]["rejections"]), 30)
        self.assertEqual(
            result["decisions"][0]["rejections"],
            list(range(71, 101)),
        )
        self.assertTrue(result["verification"]["independent_optimality_verified"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
