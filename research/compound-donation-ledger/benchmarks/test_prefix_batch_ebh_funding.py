import random
import unittest
from fractions import Fraction as F

from prefix_batch_ebh_funding import (
    assign_selected,
    select_max_cardinality,
    verify_prefix_feasibility,
)


def brute_max_cardinality(deficits, bank):
    n = len(deficits)
    best = -1
    solutions = []
    for mask in range(1 << n):
        chosen = {
            j for j in range(n)
            if (mask >> j) & 1
            and deficits[j] is not None
            and deficits[j] > 0
        }
        feasible = all(
            sum((deficits[k] for k in chosen if k <= j), F(0))
            <= sum(bank[:j], F(0))
            for j in range(n)
        )
        if feasible:
            if len(chosen) > best:
                best = len(chosen)
                solutions = [chosen]
            elif len(chosen) == best:
                solutions.append(chosen)
    return best, solutions


class PrefixBatchFundingTests(unittest.TestCase):
    def test_selector_matches_all_subsets_on_random_exact_instances(self):
        rng = random.Random(10142)
        checks = 0
        for n in range(1, 10):
            for _ in range(1000):
                bank = [F(rng.randint(0, 10), 10) for _ in range(n)]
                deficits = [
                    None if rng.random() < 0.25
                    else F(rng.randint(1, 20), 10)
                    for _ in range(n)
                ]
                selected = select_max_cardinality(deficits, bank)
                optimum, _ = brute_max_cardinality(deficits, bank)
                self.assertEqual(len(selected), optimum)
                self.assertTrue(
                    verify_prefix_feasibility(deficits, bank, selected)
                )
                checks += 1
        self.assertEqual(checks, 9000)

    def test_ties_evict_later_destination_first(self):
        deficits = [None, F(1), F(1)]
        bank = [F(3, 2), F(0), F(0)]
        self.assertEqual(select_max_cardinality(deficits, bank), (1,))

    def test_source_assignment_is_forward_exact_and_oldest_first(self):
        deficits = [None, None, F(1, 3), F(1, 4)]
        token_bank = [
            [((0, 1), F(1, 4)), ((0, 2), F(1, 8))],
            [((1, 1), F(1, 2))],
            [],
            [],
        ]
        bank = [sum((amount for _, amount in rows), F(0))
                for rows in token_bank]
        selected = select_max_cardinality(deficits, bank)
        self.assertEqual(selected, (2, 3))
        fragments, remaining = assign_selected(deficits, token_bank, selected)
        self.assertEqual(
            sum((row.amount for row in fragments), F(0)),
            F(1, 3) + F(1, 4),
        )
        self.assertTrue(all(row.source_id < row.destination_id
                            for row in fragments))
        self.assertEqual(
            sum((amount for _, amount in remaining), F(0)),
            sum(bank, F(0)) - F(1, 3) - F(1, 4),
        )
        self.assertEqual(fragments[0].token_id, (0, 1))

    def test_assignment_rejects_unfundable_selected_set(self):
        deficits = [None, F(1)]
        token_bank = [[((0, 1), F(1, 2))], []]
        with self.assertRaises(ValueError):
            assign_selected(deficits, token_bank, (1,))

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            select_max_cardinality([F(1)], [])
        with self.assertRaises(ValueError):
            select_max_cardinality([F(-1)], [F(1)])
        with self.assertRaises(TypeError):
            select_max_cardinality([0.1], [F(1)])
        with self.assertRaises(ValueError):
            assign_selected(
                [None, F(1)],
                [[((1, 1), F(1))], []],
                (1,),
            )


if __name__ == "__main__":
    unittest.main()
