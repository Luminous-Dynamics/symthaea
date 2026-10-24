"""Tests for running existing proposal policies across the adversarial scenario family."""
from __future__ import annotations

import unittest

from multifidelity_stress import POLICIES, required_policy_costs
from multifidelity_scenario_policy_bakeoff import (
    run_policy_on_scenario,
    run_policy_sweep,
)


class MultiFidelityScenarioPolicyBakeoffTests(unittest.TestCase):
    def test_one_policy_run_reconciles_objective_budget(self) -> None:
        result = run_policy_on_scenario(
            policy_name="adaptive_budget",
            scenario_id="local_bias_island",
            dimension=2,
            severity=1.0,
            seed=17,
            cost_budget=200,
            design_count=128,
        )
        self.assertTrue(result["synthetic_fixture_only"])
        self.assertEqual(result["cost_spent"], 200)
        self.assertEqual(result["unused_cost"], 0)
        self.assertEqual(sum(result["fidelity_cost_units_by_level"].values()), 200)
        self.assertEqual(sum(result["fidelity_evaluations"].values()), result["evaluation_count"])
        self.assertGreater(result["true_finite_pool_hypervolume"], 0.0)
        self.assertGreaterEqual(result["observed_high_fidelity_hypervolume"], 0.0)
        self.assertLessEqual(
            result["observed_high_fidelity_hypervolume"],
            result["true_finite_pool_hypervolume"] + 1e-9,
        )

    def test_all_existing_policies_execute_through_scenario_adapter(self) -> None:
        for policy in POLICIES:
            with self.subTest(policy=policy):
                result = run_policy_on_scenario(
                    policy_name=policy,
                    scenario_id="smooth_stationary",
                    dimension=5,
                    severity=0.5,
                    seed=23,
                    cost_budget=200,
                    design_count=128,
                )
                self.assertEqual(result["cost_spent"], 200)
                self.assertEqual(
                    sum(result["fidelity_cost_units_by_level"].values()),
                    result["cost_spent"],
                )
                self.assertEqual(
                    sum(result["fidelity_evaluations"].values()),
                    result["evaluation_count"],
                )
                self.assertGreaterEqual(result["false_promotion_count"], 0)

    def test_fixed_policy_calls_reconcile_to_frozen_default_costs(self) -> None:
        expected = {
            "high_only": {"low": 0, "medium": 0, "high": 8},
            "low_screen": {"low": 75, "medium": 0, "high": 5},
            "fixed_top_k": {"low": 50, "medium": 10, "high": 4},
            "discrepancy_aware": {"low": 55, "medium": 4, "high": 5},
            "adaptive_budget": {"low": 60, "medium": 8, "high": 4},
        }
        for policy, counts in expected.items():
            with self.subTest(policy=policy):
                result = run_policy_on_scenario(
                    policy, "warped_coordinates", 2, 1.0, 17, 200, design_count=128
                )
                self.assertEqual(result["fidelity_evaluations"], counts)
                expected_cost = sum(counts[f] * {"low": 1, "medium": 5, "high": 25}[f] for f in counts)
                self.assertEqual(result["cost_spent"], expected_cost)
                self.assertEqual(expected_cost, required_policy_costs()[policy])

    def test_policy_sweep_keeps_seed_paired_runs_and_per_policy_summaries(self) -> None:
        result = run_policy_sweep(
            scenario_ids=("smooth_stationary",),
            dimensions=(2,),
            severities=(0.0,),
            seeds=(17, 23),
            design_count=75,
            bootstrap_samples=100,
        )
        self.assertEqual(result["run_count"], 2 * len(POLICIES))
        self.assertEqual(len(result["paired_runs"]), result["run_count"])
        self.assertEqual(result["summary_count"], len(POLICIES))
        self.assertEqual(result["paired_comparison_count"], len(POLICIES) - 1)
        self.assertTrue(all(row["cost_spent"] == 200 for row in result["paired_runs"]))
        self.assertTrue(all(row["paired_seed_count"] == 2 for row in result["summaries"]))
        for comparison in result["paired_comparisons_vs_high_only"]:
            self.assertEqual(comparison["paired_seed_count"], 2)
            self.assertLessEqual(
                comparison["bootstrap_95_ci_mean_delta"]["lower"],
                comparison["bootstrap_95_ci_mean_delta"]["upper"],
            )

    def test_cost_change_is_charged_and_unused_budget_is_reported(self) -> None:
        costs = {"low": 1, "medium": 10, "high": 50}
        result = run_policy_on_scenario(
            "adaptive_budget", "constraint_boundary", 2, 1.5, 17,
            cost_budget=400, fidelity_costs=costs, design_count=128,
        )
        self.assertLessEqual(result["cost_spent"], 400)
        self.assertEqual(result["unused_cost"], 400 - result["cost_spent"])
        self.assertEqual(
            result["fidelity_cost_units_by_level"],
            {
                fidelity: result["fidelity_evaluations"][fidelity] * costs[fidelity]
                for fidelity in costs
            },
        )

    def test_design_count_below_policy_pool_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 75"):
            run_policy_on_scenario(
                "high_only", "smooth_stationary", 2, 1.0, 17,
                cost_budget=200, design_count=74,
            )

    def test_budget_below_policy_template_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "below the frozen template minimum"):
            run_policy_on_scenario(
                "high_only", "smooth_stationary", 2, 1.0, 17,
                cost_budget=199, design_count=128,
            )

    def test_seeded_policy_run_is_reproducible(self) -> None:
        args = dict(
            policy_name="discrepancy_aware",
            scenario_id="heteroscedastic_noise",
            dimension=5,
            severity=1.0,
            seed=29,
            cost_budget=200,
            design_count=128,
        )
        self.assertEqual(run_policy_on_scenario(**args), run_policy_on_scenario(**args))


if __name__ == "__main__":
    unittest.main()
