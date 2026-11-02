import random
import unittest
from fractions import Fraction as F

from compound_intersection_bound import (
    compound_scores,
    intersection_flow_sides,
    verify_every_intersection,
)


class CompoundIntersectionBoundTests(unittest.TestCase):
    def test_internal_edges_cancel_and_external_inflows_are_budgeted(self):
        n = 5
        outgoing = [F(1, 10), F(1, 5), F(1, 10), F(0), F(0)]
        edges = [
            (0, 1, F(1, 10)),  # internal if S={0,1,2}
            (1, 2, F(1, 10)),  # internal
            (0, 3, F(0)),      # zero amount still forward
            (2, 4, F(1, 20)),  # external outflow from S
        ]
        lhs, rhs = intersection_flow_sides(
            n, (0, 1, 2), outgoing, edges
        )
        self.assertLessEqual(lhs, rhs)

    def test_random_forward_flow_graphs_satisfy_every_intersection(self):
        rng = random.Random(10150)
        checked = 0
        for n in range(1, 11):
            for _ in range(100):
                outgoing = [F(rng.randint(0, 100), 1000) for _ in range(n)]
                edges = []
                for source in range(n):
                    remaining = outgoing[source]
                    destinations = list(range(source + 1, n))
                    rng.shuffle(destinations)
                    for destination in destinations:
                        if remaining <= 0 or rng.random() > 0.25:
                            continue
                        amount = remaining * F(rng.randint(1, 4), 5)
                        edges.append((source, destination, amount))
                        remaining -= amount
                checked += verify_every_intersection(n, outgoing, edges)
        self.assertGreater(checked, 100_000)

    def test_arbitrary_intersections_not_only_global_null_prefixes(self):
        n = 6
        outgoing = [F(1, 10) for _ in range(n)]
        edges = [
            (0, 2, F(1, 20)),
            (1, 3, F(1, 20)),
            (2, 5, F(1, 20)),
            (0, 4, F(1, 20)),
        ]
        # A non-contiguous null intersection with gaps before max(S).
        lhs, rhs = intersection_flow_sides(n, (1, 3, 5), outgoing, edges)
        self.assertLessEqual(lhs, rhs)
        # Enumerate every one of the 63 nonempty intersections.
        self.assertEqual(verify_every_intersection(n, outgoing, edges), 63)

    def test_compound_score_is_exact_and_nonnegative_under_donor_cap(self):
        gamma = [F(1, 10), F(1, 5), F(1, 10)]
        x = [F(3), F(2), F(5)]
        outgoing = [F(1, 10), F(1, 5), F(1, 10)]
        incoming = [F(0), F(1, 10), F(1, 10)]
        scores = compound_scores(gamma, x, outgoing, incoming)
        self.assertTrue(all(score >= 0 for score in scores))
        self.assertEqual(scores, (F(1, 5), F(3, 10), F(1, 2)))

    def test_backward_edges_overdraw_and_prefix_leak_are_rejected(self):
        with self.assertRaises(ValueError):
            intersection_flow_sides(3, (1,), [F(0)] * 3, [(2, 1, F(1))])
        with self.assertRaises(ValueError):
            intersection_flow_sides(3, (2,), [F(0)] * 3, [(0, 2, F(1))])
        with self.assertRaises(ValueError):
            intersection_flow_sides(
                3, (2,), [F(0), F(0), F(0)], [(0, 2, F(1))]
            )
        with self.assertRaises(ValueError):
            intersection_flow_sides(
                3, (0, 1), [F(0), F(1), F(0)], [(0, 1, F(1))]
            )

    def test_float_inputs_fail_closed(self):
        with self.assertRaises(TypeError):
            compound_scores([0.1], [1], [0], [0])


if __name__ == "__main__":
    unittest.main()
