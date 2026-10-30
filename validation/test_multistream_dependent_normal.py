import math
import unittest
import numpy as np

from multistream_dependent_normal import (
    e_bh_fdp_power,
    generate_correlation_matrix,
    log_adjuster_a1_from_log_m,
    log_adjuster_a2_from_log_m,
    run_dependent_normal_benchmark,
)


class DependentNormalBenchmarkTests(unittest.TestCase):
    def test_adjuster_domain_and_boundary_values(self):
        x = np.array([0.0, 1e-8, 1e-4, 0.1, 2.0])
        a1 = log_adjuster_a1_from_log_m(x)
        a2 = log_adjuster_a2_from_log_m(x)
        self.assertAlmostEqual(float(np.exp(a1[0])), 0.5, places=12)
        self.assertTrue(np.isneginf(a2[0]))
        self.assertTrue(np.all(np.isfinite(a1)))
        self.assertTrue(np.all(np.isfinite(a2[1:])))

    def test_adjusters_reject_values_outside_m_ge_one(self):
        with self.assertRaises(ValueError):
            log_adjuster_a1_from_log_m(np.array([-1e-4]))
        with self.assertRaises(ValueError):
            log_adjuster_a2_from_log_m(np.array([-1e-4]))

    def test_adjusters_are_non_decreasing_on_log_domain(self):
        x = np.array([0.0, 1e-5, 1e-3, 0.1, 1.0, 3.0, 8.0])
        a1 = log_adjuster_a1_from_log_m(x)
        a2 = log_adjuster_a2_from_log_m(x)
        self.assertTrue(np.all(np.diff(a1) >= -1e-12))
        self.assertTrue(np.all(np.diff(a2) >= -1e-12))

    def test_correlation_matrix_is_reproducible_and_psd(self):
        a, _ = generate_correlation_matrix(12, 77)
        b, _ = generate_correlation_matrix(12, 77)
        np.testing.assert_array_equal(a, b)
        self.assertTrue(np.allclose(np.diag(a), 1.0))
        self.assertGreaterEqual(np.linalg.eigvalsh(a).min(), -1e-8)

    def test_e_bh_null_and_alternative_counts(self):
        null = np.array([True, True, False, False])
        logs = np.array([[np.log(.01), np.log(.01), np.log(100), np.log(80)]])
        fdp, power, rejections = e_bh_fdp_power(logs, null, .05)
        self.assertEqual(int(rejections[0]), 2)
        self.assertEqual(float(fdp[0]), 0.0)
        self.assertEqual(float(power[0]), 1.0)

    def test_all_null_has_zero_power_and_valid_fdp(self):
        null = np.array([True, True, True])
        logs = np.array([[0.0, 0.0, 0.0], [np.log(100), 0.0, 0.0]])
        fdp, power, rejections = e_bh_fdp_power(logs, null, .05)
        self.assertTrue(np.all(power == 0.0))
        self.assertTrue(np.all((fdp >= 0.0) & (fdp <= 1.0)))

    def test_single_step_and_current_cumulative_are_distinct(self):
        null = np.array([True, True, False, False])
        increments = np.array([[np.log(4), np.log(4), np.log(4), np.log(4)]])
        cumulative = np.cumsum(increments, axis=-1)
        fdp_one, power_one, _ = e_bh_fdp_power(increments, null, .05)
        fdp_cum, power_cum, _ = e_bh_fdp_power(cumulative, null, .05)
        self.assertGreaterEqual(float(power_cum[0]), float(power_one[0]))

    def test_benchmark_reports_explicit_method_and_error_metrics(self):
        result = run_dependent_normal_benchmark(
            repetitions=4, timepoints=20, n_hypotheses=10, n_true=5,
            batch_size=2, correlation_seed=3, data_seed=4
        )
        self.assertIn("reference_script_single_step_ebh", result["methods"])
        self.assertIn("current_cumulative_ebh", result["methods"])
        for values in result["methods"].values():
            self.assertGreaterEqual(values["final_sup_fdr"], values["final_fdr"] - 1e-12)
            self.assertIn("final_fdr_mc", values)
            self.assertIn("final_power_mc", values)

    def test_small_benchmark_is_reproducible(self):
        kwargs = dict(repetitions=4, timepoints=20, n_hypotheses=10,
                      n_true=5, batch_size=2, correlation_seed=3, data_seed=4)
        a = run_dependent_normal_benchmark(**kwargs)
        b = run_dependent_normal_benchmark(**kwargs)
        for method in a["methods"]:
            for metric in ("final_fdr", "final_sup_fdr", "final_power", "final_mean_rejections"):
                self.assertEqual(a["methods"][method][metric], b["methods"][method][metric])


if __name__ == "__main__":
    unittest.main(verbosity=2)
