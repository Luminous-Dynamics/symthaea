
import unittest

from benchmark_covariance_preconditioned_betting import (
    covariance_envelope,
    simulate_cell,
)


class CovariancePreconditionedBenchmarkTests(unittest.TestCase):
    def test_envelope_is_symmetric_psd(self):
        import numpy as np

        proxy = covariance_envelope(5)
        self.assertGreaterEqual(float(np.linalg.eigvalsh(proxy).min()), 0.0)

    def test_gaussian_and_rademacher_scenarios_emit_matched_rows(self):
        for noise in ("gaussian", "rademacher_factor"):
            rows = simulate_cell(
                master_seed=975601,
                streams=128,
                dimension=3,
                horizon=10,
                noise_model=noise,
                rho_mode="switch_up",
                scenario="all_null",
            )
            self.assertEqual(len(rows), 6)
            self.assertEqual({row["noise_model"] for row in rows}, {noise})
            self.assertTrue(all(row["guarantee_scope"] ==
                                "single_intersection_crossing_diagnostic_only"
                                for row in rows))
            for row in rows:
                self.assertGreaterEqual(row["rate"], 0.0)
                self.assertLessEqual(row["rate"], 1.0)
                self.assertLessEqual(row["wilson_95"][0], row["rate"])
                self.assertGreaterEqual(row["wilson_95"][1], row["rate"])

    def test_fixed_prior_learning_rate_mixture_is_reported(self):
        rows = simulate_cell(
            master_seed=975602,
            streams=128,
            dimension=3,
            horizon=10,
            noise_model="gaussian",
            rho_mode=0.5,
            scenario="sparse_last",
        )
        methods = {row["method"] for row in rows}
        self.assertIn("fixed_prior_eta_mixture", methods)
        self.assertIn("preconditioned_eta_0.05", methods)

    def test_rejects_dependence_outside_registered_envelope(self):
        with self.assertRaises(ValueError):
            simulate_cell(
                master_seed=975603,
                streams=128,
                dimension=3,
                horizon=10,
                noise_model="gaussian",
                rho_mode=0.9,
                scenario="all_null",
            )


if __name__ == "__main__":
    unittest.main()
