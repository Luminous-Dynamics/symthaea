import unittest
from groupwise_calibrated_prior_ambiguity_benchmark import run_benchmark


class GroupwiseCalibrationTests(unittest.TestCase):
    def test_methods_include_global_oracle_and_worst_group(self):
        r = run_benchmark(n_calibration=60, n_test=18, seed=29, n_observations=40)
        methods = r["summary"]["methods"]
        self.assertEqual(set(methods), {"global", "groupwise_oracle", "worst_group"})
        self.assertAlmostEqual(r["summary"]["worst_group_radius"], max(r["summary"]["group_radii"]))
        for method in methods.values():
            self.assertGreaterEqual(method["empirical_coverage"], 0.0)
            self.assertLessEqual(method["empirical_coverage"], 1.0)
            for objective in ("graph", "kernel", "joint"):
                self.assertIn(objective, method["regret_mean_bits"])

    def test_worst_group_radius_is_max_group_quantile(self):
        r = run_benchmark(n_calibration=60, n_test=12, seed=30, n_observations=40)
        summary = r["summary"]
        self.assertAlmostEqual(summary["worst_group_radius"], max(summary["group_radii"]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
