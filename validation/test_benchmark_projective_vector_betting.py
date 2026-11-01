import unittest

from benchmark_projective_vector_betting import (
    covariance,
    covariance_envelope,
    run_fixed_intersection,
    run_predictable_arrival_null,
)


class ProjectiveVectorBettingBenchmarkTests(unittest.TestCase):
    def test_declared_envelope_covers_rho_grid(self):
        import numpy as np

        envelope = covariance_envelope(5, 0.2, 0.8, 0.5)
        for rho in np.linspace(0.2, 0.8, 21):
            self.assertGreaterEqual(
                float(np.linalg.eigvalsh(envelope - covariance(5, float(rho))).min()),
                -1e-9,
            )

    def test_fixed_intersection_result_schema(self):
        rows = run_fixed_intersection(
            seed=975101,
            streams=256,
            dimension=3,
            horizon=12,
            warmup=4,
            rho_values=(0.2, 0.8),
        )
        self.assertEqual(len(rows), 24)
        for row in rows:
            self.assertEqual(row["streams"], 256)
            self.assertGreaterEqual(row["rate"], 0.0)
            self.assertLessEqual(row["rate"], 1.0)
            self.assertEqual(row["guarantee_scope"],
                             "single_intersection_anytime_crossing_diagnostic_only")
            self.assertLessEqual(row["wilson_95"][0], row["rate"])
            self.assertGreaterEqual(row["wilson_95"][1], row["rate"])

    def test_fixed_seed_is_reproducible(self):
        kwargs = dict(
            seed=975102,
            streams=128,
            dimension=3,
            horizon=10,
            warmup=3,
            rho_values=(0.5,),
        )
        self.assertEqual(run_fixed_intersection(**kwargs),
                         run_fixed_intersection(**kwargs))

    def test_predictable_arrival_output(self):
        rows = run_predictable_arrival_null(
            seed=975103,
            streams=256,
            dimension=3,
            horizon=12,
            warmup=3,
            rho_values=(0.2, 0.8),
        )
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["scenario"] ==
                            "all_null_predictable_nested_arrivals" for row in rows))


if __name__ == "__main__":
    unittest.main()
