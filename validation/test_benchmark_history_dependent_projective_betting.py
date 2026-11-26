import unittest

from benchmark_history_dependent_projective_betting import (
    _wilson,
    run_benchmark,
)


class HistoryDependentProjectiveBettingBenchmarkTests(unittest.TestCase):
    def test_wilson_interval_contains_point_estimate(self):
        lo, hi = _wilson(25, 100)
        self.assertLessEqual(lo, 0.25)
        self.assertGreaterEqual(hi, 0.25)
        self.assertLess(lo, hi)

    def test_benchmark_is_reproducible_for_fixed_seed(self):
        first = run_benchmark(
            replicates=100, horizon=12, n_hypotheses=3,
            correlations=(0.2, 0.5), alternative_mean=0.2, seed=20261101,
        )
        second = run_benchmark(
            replicates=100, horizon=12, n_hypotheses=3,
            correlations=(0.2, 0.5), alternative_mean=0.2, seed=20261101,
        )
        self.assertEqual(first["configuration"], second["configuration"])
        for left, right in zip(first["scenarios"], second["scenarios"]):
            self.assertEqual(left["seed"], right["seed"])
            self.assertEqual(left["mean_final_active_count"], right["mean_final_active_count"])
            for method in ("joint_nominal", "joint_robust", "affine_global"):
                self.assertEqual(
                    left["methods"][method]["crossings"],
                    right["methods"][method]["crossings"],
                )

    def test_covariance_misspecification_is_explicit_in_manifest(self):
        result = run_benchmark(
            replicates=100, horizon=10, n_hypotheses=3,
            correlations=(0.8,), alternative_mean=0.2, seed=91,
        )
        null_scenario = result["scenarios"][0]
        self.assertFalse(null_scenario["nominal_joint_covariance_bound_valid"])
        self.assertTrue(null_scenario["robust_joint_covariance_bound_valid"])
        self.assertEqual(
            set(null_scenario["methods"]),
            {"joint_nominal", "joint_robust", "affine_global"},
        )

    def test_component_benchmark_disclaims_closed_fdr_claim(self):
        result = run_benchmark(
            replicates=50, horizon=8, n_hypotheses=2,
            correlations=(0.5,), alternative_mean=0.2, seed=7,
        )
        self.assertIn("not a closed-FDR", result["purpose"])
        self.assertEqual(len(result["scenarios"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
