import unittest

from benchmark_supremum_anchored_portfolios import simulate_scenario


class SupremumAnchoredPortfolioBenchmarkTests(unittest.TestCase):
    def test_all_null_portfolio_retains_baseline_and_reports_supfdr(self):
        result = simulate_scenario(
            seed=978201,
            n=20,
            streams=128,
            horizon=8,
            alpha=0.05,
            p0=0.01,
            p_alt=0.0,
            nonnull_fraction=0.0,
            q_total=0.1,
            v_total=0.9,
            baseline_p1=0.2,
            expert_p1=(0.05, 0.2, 0.7),
            betas=(0.0, 0.25, 0.5),
        )
        for method in result["methods"].values():
            self.assertEqual(method["baseline_retention_rate"], 1.0)
            self.assertGreaterEqual(method["estimated_supfdr"], 0.0)
            self.assertLessEqual(method["estimated_supfdr"], 1.0)

    def test_fixed_prior_is_valid_comparator_not_selected_best(self):
        result = simulate_scenario(
            seed=978202,
            n=20,
            streams=128,
            horizon=8,
            alpha=0.05,
            p0=0.01,
            p_alt=0.3,
            nonnull_fraction=0.2,
            q_total=0.1,
            v_total=0.9,
            baseline_p1=0.2,
            expert_p1=(0.05, 0.2, 0.7),
            betas=(0.0, 0.25, 0.5),
        )
        self.assertIn("fixed_prior_expert_portfolio", result["methods"])
        self.assertIn(result["posthoc_best_single_expert"], result["methods"])
        self.assertEqual(result["methods"]["fixed_prior_expert_portfolio"]["baseline_retention_rate"], 1.0)
        self.assertIn("unattainable", result["interpretation"])


if __name__ == "__main__":
    unittest.main()
