import unittest
import numpy as np
from recertified_eclosure import (
    METHODS, ebh_size, ma_ebh_size, closed_ebh_size, anchored_ebh_size,
    uebh_tail_probabilities, ru_ebh_tail_probabilities,
    rc_ebh_tail_probabilities, greatest_convex_minorant,
    draw_rejection_count, closed_mean_feasible_exhaustive,
    anchored_feasible, verify_ru_menu, verify_rc_menu_exhaustive,
    charges_from_tail_probabilities, simulate_cell,
)


class RecertifiedEClosureTests(unittest.TestCase):
    def test_gcm_is_below_input_and_convex(self):
        rng = np.random.default_rng(1)
        for n in range(2, 30):
            for _ in range(20):
                y = rng.random(n)
                g = greatest_convex_minorant(y)
                self.assertTrue(np.all(g <= y + 1e-10))
                self.assertTrue(np.all(np.diff(g, n=2) >= -1e-10))

    def test_closed_ebh_matches_exhaustive_mean_closure(self):
        rng = np.random.default_rng(2)
        for k in range(1, 10):
            for _ in range(50):
                e = np.exp(rng.normal(0, 1.8, k))
                alpha = float(rng.uniform(.01, .6))
                r = closed_ebh_size(e, alpha)
                order = np.argsort(-e, kind="stable")
                self.assertTrue(closed_mean_feasible_exhaustive(e, alpha, order[:r]))
                for larger in range(r+1, k+1):
                    self.assertFalse(closed_mean_feasible_exhaustive(e, alpha, order[:larger]))

    def test_anchored_and_closed_preserve_ebh(self):
        rng = np.random.default_rng(3)
        for k in range(2, 30):
            for _ in range(20):
                e = np.exp(rng.normal(0, 2, k))
                alpha = .1
                base = ebh_size(e, alpha)
                self.assertGreaterEqual(anchored_ebh_size(e, alpha), base)
                self.assertGreaterEqual(closed_ebh_size(e, alpha), base)
                order = np.argsort(-e, kind="stable")
                self.assertTrue(anchored_feasible(e, alpha, order[:anchored_ebh_size(e, alpha)]))

    def test_ru_and_rc_tails_are_nested_and_monotone(self):
        rng = np.random.default_rng(4)
        for k in range(2, 30):
            for _ in range(20):
                e = np.exp(rng.normal(0, 2, k))
                alpha = .1
                q0 = uebh_tail_probabilities(e, alpha)
                q1 = ru_ebh_tail_probabilities(e, alpha)
                q2 = rc_ebh_tail_probabilities(e, alpha)
                for q in (q0, q1, q2):
                    self.assertTrue(np.all((q >= 0) & (q <= 1)))
                    self.assertTrue(np.all(np.diff(q) <= 1e-9))
                self.assertTrue(np.all(q1 >= q0 - 1e-9))
                self.assertTrue(np.all(q2 >= q1 - 1e-9))

    def test_ru_menu_rank_charges(self):
        rng = np.random.default_rng(5)
        for k in range(2, 11):
            for _ in range(30):
                e = np.exp(rng.normal(0, 1.5, k))
                self.assertTrue(verify_ru_menu(e, .1))

    def test_rc_menu_passes_all_intersections_exhaustively(self):
        rng = np.random.default_rng(6)
        for k in range(2, 11):
            for _ in range(30):
                e = np.exp(rng.normal(0, 1.5, k))
                self.assertTrue(verify_rc_menu_exhaustive(e, .1))

    def test_rc_expected_rejections_at_least_closed_ebh(self):
        rng = np.random.default_rng(7)
        for k in range(2, 20):
            for _ in range(25):
                e = np.exp(rng.normal(0, 1.7, k))
                q = rc_ebh_tail_probabilities(e, .1)
                self.assertGreaterEqual(float(q.sum()) + 1e-9, float(closed_ebh_size(e, .1)))

    def test_shared_uniform_procedure_nesting(self):
        rng = np.random.default_rng(8)
        for k in range(2, 30):
            for _ in range(30):
                e = np.exp(rng.normal(0, 2, k))
                u = float(rng.random())
                sizes = [
                    ebh_size(e, .1),
                    draw_rejection_count(uebh_tail_probabilities(e, .1), u),
                    draw_rejection_count(ru_ebh_tail_probabilities(e, .1), u),
                    draw_rejection_count(rc_ebh_tail_probabilities(e, .1), u),
                ]
                self.assertEqual(sizes, sorted(sizes))

    def test_ru_recertifies_shared_u_ebh_decision_pointwise(self):
        e = np.array([39.5, 1.0])
        u = 0.99
        size_u = draw_rejection_count(uebh_tail_probabilities(e, .05), u)
        size_ru = draw_rejection_count(ru_ebh_tail_probabilities(e, .05), u)
        size_rc = draw_rejection_count(rc_ebh_tail_probabilities(e, .05), u)
        self.assertEqual(size_u, 0)
        self.assertEqual(size_ru, 1)
        self.assertGreaterEqual(size_rc, size_ru)

    def test_probability_distribution_from_tails(self):
        q = np.array([.9, .6, .2])
        p = np.concatenate(([1-q[0]], q[:-1]-q[1:], [q[-1]]))
        self.assertAlmostEqual(float(p.sum()), 1.0)
        self.assertTrue(np.all(p >= 0))

    def test_small_simulation_is_reproducible_and_reports_all_methods(self):
        kw = dict(repetitions=50,k=12,nonnull_count=4,signal_mean=2.0,rho=.3,alpha=.1,seed=99)
        a = simulate_cell(**kw)
        b = simulate_cell(**kw)
        self.assertEqual(set(a["metrics"]), set(METHODS))
        for m in METHODS:
            self.assertEqual(a["metrics"][m]["power"]["mean"], b["metrics"][m]["power"]["mean"])
            self.assertEqual(a["metrics"][m]["fdr"]["mean"], b["metrics"][m]["fdr"]["mean"])
        self.assertEqual(a["nesting"]["eBH_subset_U"], 1.0)
        self.assertEqual(a["nesting"]["U_subset_RU"], 1.0)
        self.assertEqual(a["nesting"]["RU_subset_RC"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
