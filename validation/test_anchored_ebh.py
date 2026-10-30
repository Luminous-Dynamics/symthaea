import unittest
import numpy as np
from anchored_ebh import (
    anchored_ebh, anchored_ebh_batch, anchored_closure_feasible,
    ebh, exact_eclosure_feasible, generate_likelihood_ratio_evalues,
)


class AnchoredEBHTests(unittest.TestCase):
    def test_paper_strict_example(self):
        result = anchored_ebh(np.array([39.5, 1.0]), 0.05)
        self.assertEqual(result["baseline_size"], 0)
        self.assertEqual(result["anchored_size"], 1)
        self.assertTrue(anchored_closure_feasible(
            np.array([39.5, 1.0]), .05, result["anchored_indices"]))

    def test_baseline_is_always_retained(self):
        rng = np.random.default_rng(12)
        for m in range(1, 40):
            for _ in range(30):
                e = np.exp(rng.normal(0, 2, m))
                result = anchored_ebh(e, .05)
                self.assertTrue(set(result["baseline_indices"]).issubset(
                    set(result["anchored_indices"])))

    def test_formula_equals_exhaustive_closure_for_random_small_families(self):
        rng = np.random.default_rng(13)
        for m in range(1, 9):
            for _ in range(80):
                e = np.exp(rng.normal(0, 1.8, m))
                alpha = float(rng.uniform(.01, .5))
                result = anchored_ebh(e, alpha)
                self.assertEqual(
                    anchored_closure_feasible(e, alpha, result["anchored_indices"]),
                    exact_eclosure_feasible(e, alpha, result["anchored_indices"]))
                self.assertTrue(exact_eclosure_feasible(
                    e, alpha, result["anchored_indices"]))

    def test_selected_output_is_largest_cardinality_certified_superset(self):
        rng = np.random.default_rng(14)
        for m in range(1, 9):
            for _ in range(20):
                e = np.exp(rng.normal(0, 1.5, m))
                alpha = float(rng.uniform(.02, .5))
                result = anchored_ebh(e, alpha)
                best = len(result["anchored_indices"])
                order = np.argsort(-e, kind="stable")
                for r in range(best + 1, m + 1):
                    self.assertFalse(exact_eclosure_feasible(e, alpha, order[:r]))

    def test_ties_are_deterministic(self):
        e = np.array([3.0, 3.0, 3.0, 0.0])
        np.testing.assert_array_equal(
            anchored_ebh(e, .2)["anchored_indices"],
            anchored_ebh(e, .2)["anchored_indices"])

    def test_invalid_evalues_rejected(self):
        for e in (np.array([]), np.array([-1.]), np.array([np.nan]),
                  np.array([np.inf]), np.array([[1., 2.]])):
            with self.assertRaises(ValueError):
                anchored_ebh(e, .05)

    def test_invalid_alpha_rejected(self):
        for alpha in (0., 1., -0.2, np.nan):
            with self.assertRaises(ValueError):
                anchored_ebh(np.array([1., 2.]), alpha)

    def test_anchored_feasibility_matches_direct_inequality(self):
        e = np.array([5., .8, 1.2, 10.])
        for rejected in ([0], [0, 3], [0, 1, 2, 3], []):
            self.assertEqual(
                anchored_closure_feasible(e, .1, np.array(rejected)),
                exact_eclosure_feasible(e, .1, np.array(rejected)))

    def test_null_lr_evalues_have_correct_marginal_mean_in_large_sample(self):
        e, null = generate_likelihood_ratio_evalues(
            repetitions=100000, m=4, nonnull_count=0, effect=1.0,
            dependence="equicorrelated", rho=.5, seed=99)
        self.assertTrue(np.all(null))
        self.assertTrue(np.allclose(e.mean(axis=0), 1.0, atol=.025))

    def test_valid_alternative_and_null_masks(self):
        e, null = generate_likelihood_ratio_evalues(
            repetitions=100, m=10, nonnull_count=3, effect=1.5,
            dependence="independent", rho=0., seed=90)
        self.assertEqual(int((~null).sum()), 3)
        self.assertTrue(np.all(e >= 0.0))
        self.assertTrue(np.all(np.isfinite(e)))

    def test_batched_and_scalar_outputs_agree(self):
        rng = np.random.default_rng(15)
        e = np.exp(rng.normal(0, 2, (50, 20)))
        batch = anchored_ebh_batch(e, .05)
        for i in range(e.shape[0]):
            scalar = anchored_ebh(e[i], .05)
            self.assertEqual(int(batch["baseline_size"][i]), scalar["baseline_size"])
            self.assertEqual(int(batch["anchored_size"][i]), scalar["anchored_size"])
            np.testing.assert_array_equal(
                batch["order"][i][:scalar["anchored_size"]],
                scalar["anchored_indices"])

    def test_e_bh_is_subset_of_anchored(self):
        e = np.array([8., 4., 2., 1., .5])
        a = anchored_ebh(e, .2)
        self.assertTrue(set(ebh(e, .2)).issubset(set(a["anchored_indices"])))


if __name__ == "__main__":
    unittest.main(verbosity=2)
