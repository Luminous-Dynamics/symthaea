import math
import unittest
import numpy as np

from multistream_group_aggregators import (
    group_log_scores, log_adjuster_a1, log_adjuster_a2,
    mixed_power_group_log_score, power_group_log_scores,
    simulate_group_case, wilson_interval,
)


class GroupAggregatorTests(unittest.TestCase):
    def test_adjuster_boundaries(self):
        x = np.array([[0.0, 1e-8, 1e-4, 0.5]])
        a1 = log_adjuster_a1(x)
        a2 = log_adjuster_a2(x)
        self.assertAlmostEqual(float(np.exp(a1[0, 0])), 0.5, places=12)
        self.assertTrue(np.isneginf(a2[0, 0]))
        self.assertTrue(np.all(np.isfinite(a1)))
        self.assertTrue(np.all(np.isfinite(a2[0, 1:])))

    def test_adjusters_reject_log_v_below_zero(self):
        with self.assertRaises(ValueError):
            log_adjuster_a1(np.array([-1e-6]))
        with self.assertRaises(ValueError):
            log_adjuster_a2(np.array([-1e-6]))

    def test_power_score_matches_closed_form_for_s1_s2_s4(self):
        rng = np.random.default_rng(81)
        log_v = rng.uniform(0.0, 8.0, size=(30, 17))
        actual = power_group_log_scores(log_v)
        k = log_v.shape[-1]
        ranks = np.arange(1, k + 1, dtype=np.float64)
        for s in (1.0, 2.0, 4.0):
            h = np.sum(ranks ** (-s))
            expected = (
                np.logaddexp.reduce(s * log_v, axis=-1) / s
                - math.log((s + 1.0) / s) - math.log(k) - math.log(h) / s
            )
            np.testing.assert_allclose(actual[s], expected, rtol=1e-12, atol=1e-12)

    def test_mixed_f_reduces_to_single_power(self):
        rng = np.random.default_rng(82)
        log_v = rng.uniform(0.0, 7.0, size=(12, 9))
        for s in (1.0, 2.0, 4.0):
            general = mixed_power_group_log_score(log_v, (s,), (1.0,))
            special = power_group_log_scores(log_v, (s,))[s]
            np.testing.assert_allclose(general, special, rtol=1e-10, atol=1e-10)

    def test_mixed_f_weight_scale_is_irrelevant_after_normalization(self):
        rng = np.random.default_rng(83)
        log_v = rng.uniform(0.0, 4.0, size=(8, 15))
        a = np.array([0.9, 0.09, 0.01])
        b = 17.0 * a
        b = b / b.sum()
        a = a / a.sum()
        left = mixed_power_group_log_score(log_v, (1.0, 2.0, 4.0), tuple(a))
        right = mixed_power_group_log_score(log_v, (1.0, 2.0, 4.0), tuple(b))
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-10)

    def test_mixed_f_score_is_monotone_under_componentwise_evidence_increase(self):
        low = np.array([[0.0, 0.1, 0.3, 0.5]])
        high = low + 0.7
        self.assertTrue(np.all(mixed_power_group_log_score(high) >= mixed_power_group_log_score(low)))

    def test_group_score_names_and_finite_power_scores(self):
        log_v = np.array([[0.0, 0.1, 0.5, 2.0]])
        scores = group_log_scores(log_v)
        expected = {
            "bonferroni_max", "adjuster_a1_mean", "adjuster_a2_mean",
            "direct_power_s1", "direct_power_s2", "direct_power_s4",
            "power_portfolio_s1_s2_s4", "mixed_power_f_s1_s2_s4",
        }
        self.assertEqual(set(scores), expected)
        for name, value in scores.items():
            self.assertEqual(value.shape, (1,), name)
            self.assertFalse(np.any(np.isnan(value)), name)

    def test_direct_power_score_increases_with_group_evidence(self):
        low = power_group_log_scores(np.zeros((1, 20)))
        high = power_group_log_scores(np.ones((1, 20)))
        for s in (1.0, 2.0, 4.0):
            self.assertGreater(float(high[s][0]), float(low[s][0]))

    def test_simulation_reproducible_for_fixed_seed(self):
        kwargs = dict(k=12, timepoints=30, repetitions=40, signal_count=4,
                      rho=0.2, seed=20261030, batch_size=10)
        a = simulate_group_case(**kwargs)
        b = simulate_group_case(**kwargs)
        self.assertEqual(a["results"], b["results"])

    def test_all_null_simulation_emits_binomial_intervals(self):
        result = simulate_group_case(k=8, timepoints=20, repetitions=30,
            signal_count=0, rho=0.0, seed=12, batch_size=10)
        for row in result["results"].values():
            self.assertEqual(len(row["wilson95"]), 2)
            self.assertGreaterEqual(row["wilson95"][0], 0.0)
            self.assertLessEqual(row["wilson95"][1], 1.0)

    def test_wilson_edges(self):
        lo, hi = wilson_interval(0, 100)
        self.assertEqual(lo, 0.0)
        self.assertGreater(hi, 0.0)
        lo, hi = wilson_interval(100, 100)
        self.assertLess(lo, 1.0)
        self.assertEqual(hi, 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
