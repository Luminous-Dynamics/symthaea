import unittest
from action_subset_kernel_benchmark import (
    KERNELS, KERNEL_COUNTS, MODES, MODELS, SUBSETS, TABLES, WORLD,
    expected_information, kernel_catalog, make_dataset, observation_likelihoods,
    posterior_fit, run_benchmark, self_test, true_kernels,
)

class ActionSubsetKernelBenchmarkTests(unittest.TestCase):
    def test_kernel_catalog_is_complete_quarter_grid(self):
        kernels, pairs = kernel_catalog()
        self.assertEqual(len(kernels), 330)
        self.assertEqual(len(set(pairs)), 330)
        self.assertTrue(all(abs(sum(d) - 1.0) < 1e-12 for d in kernels))
        self.assertTrue(all(sum(p > 0 for p in d) <= 4 for d in kernels))

    def test_all_subset_outcome_distributions_normalize(self):
        for g in range(len(MODELS)):
            for s in SUBSETS:
                self.assertAlmostEqual(sum(WORLD[g][s]), 1.0, places=12)

    def test_observation_channels_are_normalized(self):
        for action in range(3):
            for mode in MODES:
                for g in range(len(MODELS)):
                    for d in KERNELS:
                        self.assertAlmostEqual(sum(observation_likelihoods(action, g, d, mode)), 1.0, places=12)

    def test_exact_target_subset_telemetry_dominates_weaker_channels(self):
        for action in range(3):
            values = [expected_information(action, mode) for mode in MODES]
            for metric in ("graph_information_bits", "action_kernel_information_bits", "joint_graph_kernel_information_bits"):
                self.assertLessEqual(values[0][metric], values[1][metric] + 1e-10)
                self.assertLessEqual(values[1][metric], values[2][metric] + 1e-10)

    def test_truth_regimes_explicitly_separate_in_and_out_of_grid(self):
        _, in_ids = true_kernels("in_support")
        _, out_ids = true_kernels("out_of_support")
        self.assertTrue(all(idx is not None for idx in in_ids))
        self.assertTrue(all(idx is None for idx in out_ids))

    def test_posterior_normalizes_in_all_channels(self):
        dists, _ = true_kernels("in_support")
        data = make_dataset(1, dists, 7, 4)
        for mode in MODES:
            p = posterior_fit(data, mode)
            self.assertAlmostEqual(sum(p["graph_posterior"]), 1.0, places=12)
            for marginal in p["kernel_marginals"]:
                self.assertAlmostEqual(sum(marginal), 1.0, places=12)
            for mean_dist in p["posterior_mean_kernels"]:
                self.assertAlmostEqual(sum(mean_dist), 1.0, places=12)

    def test_deterministic_paired_smoke_benchmark(self):
        a = run_benchmark(replicates=1, samples_per_action=2)
        b = run_benchmark(replicates=1, samples_per_action=2)
        self.assertEqual(a, b)
        self.assertEqual(len(a["rows"]), 24)
        self.assertEqual(len(a["exact_one_probe_expected_information"]), 9)

    def test_embedded_self_test(self):
        self.assertTrue(self_test())

if __name__ == "__main__":
    unittest.main()
