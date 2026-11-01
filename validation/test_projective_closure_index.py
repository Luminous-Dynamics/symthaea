from fractions import Fraction
import random
import unittest

from projective_closure_index import ProjectiveClosureIndex, raw_separator_score


class ProjectiveClosureIndexTests(unittest.TestCase):
    def test_exact_formula_randomized_subsets(self):
        rng = random.Random(97146)
        for _ in range(2000):
            n = rng.randint(1, 10)
            values = [Fraction(rng.randint(-20, 20), rng.randint(1, 20)) for _ in range(n)]
            selected = [rng.random() < 0.5 for _ in values]
            index = ProjectiveClosureIndex(values, selected)
            self.assertEqual(index.score(), raw_separator_score(values, selected))
            index.assert_valid_avl()

    def test_differential_randomized_mutations(self):
        rng = random.Random(97147)
        values = [Fraction(rng.randint(-9, 9), rng.randint(1, 9)) for _ in range(64)]
        selected = [rng.random() < 0.4 for _ in values]
        index = ProjectiveClosureIndex(values, selected)
        for step in range(5000):
            i = rng.randrange(len(values))
            if rng.randrange(2) == 0:
                values[i] = Fraction(rng.randint(-15, 15), rng.randint(1, 15))
                index.update_evidence(i, values[i])
            else:
                selected[i] = not selected[i]
                index.set_selected(i, selected[i])
            self.assertEqual(index.score(), raw_separator_score(values, selected), f"step={step}")
            if step % 31 == 0:
                index.assert_valid_avl()

    def test_boundary_values_and_threshold_ties(self):
        values = [Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1, 3), Fraction(2)]
        selected = [True, True, True, True, False]
        index = ProjectiveClosureIndex(values, selected)
        self.assertEqual(index.score(), raw_separator_score(values, selected))
        values[2] = Fraction(1, 4)
        index.update_evidence(2, values[2])
        self.assertEqual(index.score(), raw_separator_score(values, selected))
        selected[0] = False
        index.set_selected(0, False)
        self.assertEqual(index.score(), raw_separator_score(values, selected))

    def test_empty_rejection_set(self):
        index = ProjectiveClosureIndex([Fraction(-1), Fraction(0), Fraction(3)], [False] * 3)
        self.assertEqual(index.score(), Fraction(0))
        self.assertTrue(index.verify())

    def test_append_dormant_zero_then_update(self):
        index = ProjectiveClosureIndex([Fraction(1, 4)], [True])
        new_index = index.append(Fraction(0), selected=False)
        self.assertEqual(new_index, 1)
        self.assertTrue(index.verify())
        index.update_evidence(new_index, Fraction(-1, 7))
        self.assertTrue(index.verify())
        index.set_selected(new_index, True)
        self.assertTrue(index.verify())


if __name__ == "__main__":
    unittest.main()
