import itertools
import random
import unittest
from fractions import Fraction as F

from ebh_minimum_credit import (
    allocate_from_source_bank,
    all_minimum_credits,
    brute_minimum_credit,
    choose_fundable_candidate,
    ebh_rejections,
    feasible_ranks,
    minimum_credit_to_enter_ebh,
)


class MinimumEBHCreditTests(unittest.TestCase):
    def test_exact_boundary_tie_and_inclusive_e_bh(self):
        scores = [F(10), F(3), F(0)]
        alpha = F(1, 10)
        self.assertIn(0, ebh_rejections(scores, alpha))
        self.assertEqual(minimum_credit_to_enter_ebh(scores, 1, alpha), F(2))
        updated = scores[:]
        updated[1] += F(2)
        self.assertIn(1, ebh_rejections(updated, alpha))

    def test_exhaustive_small_rational_score_grids(self):
        values = [F(0), F(1), F(2), F(3), F(4)]
        alphas = [F(1, 2), F(1, 3), F(1, 5), F(1, 10)]
        checks = 0
        for n in range(1, 5):
            for scores in itertools.product(values, repeat=n):
                for alpha in alphas:
                    batch = all_minimum_credits(scores, alpha)
                    for candidate in range(n):
                        expected = brute_minimum_credit(scores, candidate, alpha)
                        self.assertEqual(batch[candidate], expected)
                        amended = list(scores)
                        amended[candidate] += batch[candidate]
                        self.assertIn(candidate, ebh_rejections(amended, alpha))
                        checks += 1
        self.assertEqual(checks, 11720)

    def test_random_rational_vectors_and_candidate_breakpoints(self):
        rng = random.Random(10120)
        alphas = [F(1, 2), F(1, 3), F(1, 5), F(1, 10)]
        checks = 0
        for n in range(5, 8):
            for _ in range(1000):
                scores = [
                    F(rng.randrange(0, 40), rng.randrange(1, 9))
                    for _ in range(n)
                ]
                for alpha in alphas:
                    batch = all_minimum_credits(scores, alpha)
                    for candidate, credit in enumerate(batch):
                        self.assertEqual(
                            credit,
                            brute_minimum_credit(scores, candidate, alpha)
                        )
                        amended = list(scores)
                        amended[candidate] += credit
                        self.assertIn(candidate, ebh_rejections(amended, alpha))
                        if credit > 0:
                            for k in range(1, n + 1):
                                smaller = max(F(0), F(1, 1) / (alpha * k) - scores[candidate])
                                if smaller < credit:
                                    probe = list(scores)
                                    probe[candidate] += smaller
                                    self.assertNotIn(
                                        candidate, ebh_rejections(probe, alpha)
                                    )
                        checks += 1
        self.assertEqual(checks, 72000)

    def test_feasible_rank_characterization_for_every_candidate(self):
        rng = random.Random(10121)
        for n in range(1, 25):
            scores = [F(rng.randrange(0, 100), 7) for _ in range(n)]
            alpha = F(1, 7)
            feasible = feasible_ranks(scores, alpha)
            for candidate in range(n):
                expected = []
                for k in range(1, n + 1):
                    threshold = F(1, 1) / (alpha * k)
                    count = sum(
                        1 for other in range(n)
                        if other != candidate and scores[other] >= threshold
                    )
                    if count == k - 1:
                        expected.append(k)
                # Candidate-specific feasible ranks always exist.
                self.assertTrue(expected)
                self.assertTrue(
                    minimum_credit_to_enter_ebh(scores, candidate, alpha) >= 0
                )
                for k in expected:
                    threshold = F(1, 1) / (alpha * k)
                    if scores[candidate] < threshold:
                        self.assertGreaterEqual(
                            minimum_credit_to_enter_ebh(scores, candidate, alpha),
                            0
                        )

    def test_source_bank_assignment_is_forward_and_exact(self):
        bank = [F(1, 4), F(1, 8), F(1, 2), F(0)]
        result = allocate_from_source_bank(bank, destination=3, amount=F(1, 2))
        self.assertIsNotNone(result)
        assignments, remaining = result
        self.assertEqual(assignments, ((0, F(1, 4)), (1, F(1, 8)), (2, F(1, 8))))
        self.assertEqual(remaining, (F(0), F(0), F(3, 8), F(0)))
        self.assertTrue(all(source < 3 for source, _ in assignments))
        self.assertEqual(sum((v for _, v in assignments), F(0)), F(1, 2))

    def test_insufficient_bank_is_not_partially_spent(self):
        bank = [F(1, 10), F(1, 10), F(0)]
        self.assertIsNone(
            allocate_from_source_bank(bank, destination=2, amount=F(1, 3))
        )
        self.assertEqual(bank, [F(1, 10), F(1, 10), F(0)])

    def test_allocator_chooses_smallest_fundable_deficit(self):
        scores = [F(12), F(2), F(1), F(0)]
        bank = [F(2), F(1), F(0), F(0)]
        result = choose_fundable_candidate(scores, bank, F(1, 10))
        if result is not None:
            destination, deficit, assignments, remaining = result
            self.assertEqual(
                sum((amount for _, amount in assignments), F(0)), deficit
            )
            self.assertLess(destination, len(scores))
            self.assertTrue(all(source < destination for source, _ in assignments))
            self.assertEqual(
                sum(bank, F(0)) - sum(remaining, F(0)), deficit
            )


if __name__ == "__main__":
    unittest.main()
