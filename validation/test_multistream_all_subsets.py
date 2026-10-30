import itertools
import math
import unittest
import numpy as np
from multistream_all_subsets import (
    all_subset_log_score, bonferroni_max_log_score, direct_power_log_score,
    simulate_case, wilson_interval,
)


class AllSubsetCertificateTests(unittest.TestCase):
    def test_top_prefix_equals_exhaustive_subset_max(self):
        rng = np.random.default_rng(7)
        log_v = rng.uniform(0, 4, size=7)
        score, _ = all_subset_log_score(log_v)
        h = np.cumsum(1 / np.arange(1, 8))
        vals = []
        for m in range(1, 8):
            best = max(sum(math.exp(log_v[i]) for i in subset) / math.sqrt(h[m-1])
                       for subset in itertools.combinations(range(7), m))
            vals.append(best)
        expected = math.log(max(vals)) - math.log(2 * 7 * math.sqrt(h[-1]))
        self.assertAlmostEqual(float(score), expected, places=12)

    def test_order_invariance(self):
        x = np.array([[0.1, 0.7, 1.2, 3.0, 0.0]])
        np.testing.assert_allclose(all_subset_log_score(x)[0],
                                   all_subset_log_score(x[:, ::-1])[0], atol=1e-12)

    def test_returns_valid_best_subset_size(self):
        x = np.array([[0., .2, 1., 4., 2.], [0., 0., 0., 0., 0.]])
        score, m = all_subset_log_score(x)
        self.assertEqual(m.shape, (2,))
        self.assertTrue(np.all((m >= 1) & (m <= 5)))
        self.assertTrue(np.all(np.isfinite(score)))

    def test_rejects_empty_or_invalid_input(self):
        for x in (np.array([]), np.array([-1e-6]), np.array([np.nan]), np.array([np.inf])):
            with self.assertRaises(ValueError):
                all_subset_log_score(x)

    def test_score_monotone_under_componentwise_evidence(self):
        x = np.array([[0., .1, .4, 1.]])
        y = x + .5
        self.assertGreaterEqual(float(all_subset_log_score(y)[0][0]),
                                float(all_subset_log_score(x)[0][0]))

    def test_score_includes_singleton_candidate(self):
        x = np.array([[0., .3, 1.2, 2.8]])
        score, _ = all_subset_log_score(x)
        k = x.shape[-1]
        singleton = np.max(x) - math.log(2*k*math.sqrt(sum(1/i for i in range(1,k+1))))
        self.assertGreaterEqual(float(score[0]), float(singleton)-1e-12)

    def test_bonferroni_max_formula(self):
        x = np.array([[0., 1., 2.]])
        self.assertAlmostEqual(float(bonferroni_max_log_score(x)[0]), 2-math.log(3))

    def test_direct_power_rejects_invalid_power(self):
        with self.assertRaises(ValueError):
            direct_power_log_score(np.zeros((1,3)), 0)
        with self.assertRaises(ValueError):
            direct_power_log_score(np.zeros((1,3)), float("nan"))

    def test_wilson_edges(self):
        self.assertEqual(wilson_interval(0,100)[0], 0)
        self.assertEqual(wilson_interval(100,100)[1], 1)

    def test_small_simulation_reproducible(self):
        kw = dict(k=10,timepoints=20,repetitions=30,signal_count=3,
                  rho=.2,seed=77,batch_size=10)
        a = simulate_case(**kw)
        b = simulate_case(**kw)
        self.assertEqual(a["results"], b["results"])
        self.assertEqual(a["selected_subset_size"], b["selected_subset_size"])

    def test_null_pilot_case_schema(self):
        r = simulate_case(k=8,timepoints=10,repetitions=25,signal_count=0,
                          rho=0,seed=5,batch_size=5)
        self.assertEqual(r["protocol"]["tested_claim"],
                         "simultaneous nonempty subset/group intersection-null certificate")
        self.assertEqual(set(r["results"]),
                         {"all_subsets","bonferroni_max","direct_power_s1",
                          "direct_power_s2","direct_power_s4"})


if __name__ == "__main__":
    unittest.main(verbosity=2)
