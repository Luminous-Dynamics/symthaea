"""Tests for controlled low/medium/high fidelity casting-like stress fixture."""
from __future__ import annotations

import itertools
import unittest

from multifidelity_stress import (
    BudgetExceeded,
    FIDELITY_COSTS,
    POLICIES,
    PolicyEvaluator,
    SyntheticCastingWorld,
    deterministic_sample,
    hypervolume_2d,
    required_policy_costs,
    run_cost_sweep,
    run_policy,
    run_suite,
    run_sweep,
)


class MultiFidelityStressTests(unittest.TestCase):
    def test_fidelity_costs_and_repeated_responses_are_deterministic(self) -> None:
        world = SyntheticCastingWorld(grid_size=9, severity=1.0)
        candidate_id = 37
        for fidelity, expected_cost in FIDELITY_COSTS.items():
            first = world._response(candidate_id, fidelity)
            second = world._response(candidate_id, fidelity)
            self.assertEqual(first, second)
            self.assertEqual(first.cost_units, expected_cost)
            self.assertEqual(first.fidelity, fidelity)

    def test_low_fidelity_has_objective_rank_and_boundary_feasibility_inversions(self) -> None:
        world = SyntheticCastingWorld(grid_size=21, severity=1.0)
        observations = []
        feasibility_inversions = 0
        for candidate_id in world.candidate_ids:
            low = world._response(candidate_id, "low")
            high = world.truth_for_audit_only(candidate_id)
            observations.append((low, high))
            feasibility_inversions += low.predicted_feasible != high.predicted_feasible

        rank_inversions = 0
        for (low_a, high_a), (low_b, high_b) in itertools.combinations(observations, 2):
            low_a_score = sum(low_a.objectives)
            low_b_score = sum(low_b.objectives)
            high_a_score = sum(high_a.objectives)
            high_b_score = sum(high_b.objectives)
            if (low_a_score < low_b_score and high_a_score > high_b_score) or (
                low_a_score > low_b_score and high_a_score < high_b_score
            ):
                rank_inversions += 1

        self.assertGreater(feasibility_inversions, 0)
        self.assertGreater(rank_inversions, 0)

    def test_policy_interface_does_not_expose_truth_oracle(self) -> None:
        world = SyntheticCastingWorld(grid_size=9)
        view = PolicyEvaluator(world, cost_budget=200)
        self.assertTrue(hasattr(view, "evaluate"))
        self.assertFalse(hasattr(view, "truth_for_audit_only"))
        self.assertFalse(hasattr(view, "all_truth"))
        self.assertFalse(hasattr(view, "true_finite_pool_hypervolume"))

    def test_budget_failure_is_fail_closed_without_mutating_ledger(self) -> None:
        world = SyntheticCastingWorld(grid_size=9)
        view = PolicyEvaluator(world, cost_budget=10)
        with self.assertRaises(BudgetExceeded):
            view.evaluate(0, "high")
        self.assertEqual(view.cost_spent, 0)
        self.assertEqual(view.high_calls, 0)
        self.assertEqual(view.records, ())

    def test_sample_is_unique_reproducible_and_seeded(self) -> None:
        world = SyntheticCastingWorld(grid_size=9)
        first = deterministic_sample(world.candidate_ids, 30, 11)
        second = deterministic_sample(world.candidate_ids, 30, 11)
        other = deterministic_sample(world.candidate_ids, 30, 12)
        self.assertEqual(first, second)
        self.assertEqual(len(first), len(set(first)))
        self.assertNotEqual(first, other)

    def test_exact_hypervolume_known_answer_and_dominated_point_invariance(self) -> None:
        front = [(2.0, 8.0), (4.0, 4.0), (7.0, 2.0)]
        self.assertAlmostEqual(hypervolume_2d(front, (10.0, 10.0)), 46.0, places=12)
        self.assertAlmostEqual(
            hypervolume_2d(front + [(5.0, 8.0), (4.5, 8.5)], (10.0, 10.0)),
            46.0,
            places=12,
        )

    def test_all_policies_reconcile_counts_costs_and_budget(self) -> None:
        for result in run_suite(seed=17, cost_budget=200, grid_size=21, severity=1.0)["policy_results"]:
            with self.subTest(policy=result["policy"]):
                self.assertTrue(result["synthetic_fixture_only"])
                self.assertEqual(result["cost_spent"], 200)
                self.assertEqual(result["unused_cost"], 0)
                self.assertEqual(sum(result["fidelity_cost_units_by_level"].values()), 200)
                self.assertEqual(sum(result["fidelity_evaluations"].values()), result["evaluation_count"])
                self.assertEqual(
                    result["evaluation_count"],
                    result["fidelity_evaluations"]["low"]
                    + result["fidelity_evaluations"]["medium"]
                    + result["fidelity_evaluations"]["high"],
                )
                self.assertGreater(result["true_finite_pool_hypervolume"], 0.0)
                self.assertGreaterEqual(result["observed_high_fidelity_hypervolume"], 0.0)
                self.assertLessEqual(
                    result["observed_high_fidelity_hypervolume"],
                    result["true_finite_pool_hypervolume"] + 1e-9,
                )

    def test_equal_cost_templates_have_expected_fidelity_mix(self) -> None:
        expected = {
            "high_only": {"low": 0, "medium": 0, "high": 8},
            "low_screen": {"low": 75, "medium": 0, "high": 5},
            "fixed_top_k": {"low": 50, "medium": 10, "high": 4},
            "discrepancy_aware": {"low": 55, "medium": 4, "high": 5},
        }
        for policy, counts in expected.items():
            with self.subTest(policy=policy):
                result = run_policy(policy, seed=23, cost_budget=200, grid_size=21)
                self.assertEqual(result["fidelity_evaluations"], counts)
                self.assertEqual(result["cost_spent"], 200)

    def test_run_is_exactly_reproducible_for_fixed_seed_and_severity(self) -> None:
        first = run_policy("discrepancy_aware", seed=29, cost_budget=200, grid_size=13, severity=0.8)
        second = run_policy("discrepancy_aware", seed=29, cost_budget=200, grid_size=13, severity=0.8)
        self.assertEqual(first, second)

    def test_paired_severity_sweep_reports_distributions_without_collapsing_runs(self) -> None:
        first = run_sweep(seeds=(17, 23), severities=(0.5, 1.0), cost_budget=200, grid_size=9)
        second = run_sweep(seeds=(17, 23), severities=(0.5, 1.0), cost_budget=200, grid_size=9)
        self.assertEqual(first, second)
        self.assertTrue(first["synthetic_fixture_only"])
        self.assertEqual(len(first["paired_runs"]), 2 * 2 * len(POLICIES))
        self.assertEqual(len(first["summary_by_policy_and_severity"]), 2 * len(POLICIES))
        self.assertEqual(len(first["paired_comparisons_vs_high_only"]), 2 * (len(POLICIES) - 1))
        for comparison in first["paired_comparisons_vs_high_only"]:
            self.assertEqual(comparison["paired_seed_count"], 2)
            self.assertLessEqual(
                comparison["bootstrap_95_ci_mean_delta"]["lower"],
                comparison["bootstrap_95_ci_mean_delta"]["upper"],
            )
        for summary in first["summary_by_policy_and_severity"]:
            self.assertEqual(summary["seed_count"], 2)
            self.assertLessEqual(
                summary["q25_normalized_hypervolume_fraction"],
                summary["median_normalized_hypervolume_fraction"],
            )
            self.assertLessEqual(
                summary["median_normalized_hypervolume_fraction"],
                summary["q75_normalized_hypervolume_fraction"],
            )

    def test_adaptive_budget_policy_allocates_by_cost_and_reconciles(self) -> None:
        default = run_policy(
            "adaptive_budget", seed=17, cost_budget=200, grid_size=9, severity=1.0
        )
        self.assertEqual(default["cost_spent"], 200)
        self.assertEqual(default["fidelity_evaluations"], {"low": 60, "medium": 8, "high": 4})

        minimum_budget = required_policy_costs()["adaptive_budget"]
        minimum = run_policy(
            "adaptive_budget", seed=17, cost_budget=minimum_budget, grid_size=9,
            severity=1.0,
        )
        self.assertEqual(minimum_budget, 65)
        self.assertEqual(minimum["cost_spent"], minimum_budget)
        self.assertEqual(minimum["fidelity_evaluations"], {"low": 5, "medium": 2, "high": 2})

        costs = {"low": 1, "medium": 10, "high": 50}
        budget = max(required_policy_costs(costs).values())
        adaptive = run_policy(
            "adaptive_budget", seed=17, cost_budget=budget, grid_size=9,
            severity=1.0, fidelity_costs=costs,
        )
        self.assertEqual(budget, 400)
        self.assertLessEqual(adaptive["cost_spent"], budget)
        self.assertEqual(
            adaptive["cost_spent"],
            sum(adaptive["fidelity_cost_units_by_level"].values()),
        )
        self.assertEqual(
            adaptive["fidelity_cost_units_by_level"],
            {
                fidelity: adaptive["fidelity_evaluations"][fidelity] * costs[fidelity]
                for fidelity in costs
            },
        )
        self.assertGreater(adaptive["high_fidelity_call_count"], 0)
        self.assertGreater(adaptive["fidelity_evaluations"]["medium"], 0)

    def test_cost_schedule_sweep_uses_one_common_absolute_budget(self) -> None:
        schedules = (
            {"low": 1, "medium": 5, "high": 25},
            {"low": 1, "medium": 10, "high": 50},
        )
        result = run_cost_sweep(
            seeds=(17, 23), severities=(0.5,), cost_schedules=schedules,
            grid_size=9, bootstrap_samples=100,
        )
        self.assertEqual(result["cost_budget"], 400)
        self.assertEqual(len(result["cost_schedules"]), 2)
        self.assertTrue(all(item["cost_budget"] == 400 for item in result["cost_schedules"]))
        self.assertEqual(len(result["paired_runs"]), 2 * 1 * 2 * len(POLICIES))
        self.assertEqual(
            len(result["paired_comparisons_vs_high_only"]),
            2 * (len(POLICIES) - 1),
        )
        adaptive_rows = [
            row for row in result["paired_runs"] if row["policy"] == "adaptive_budget"
        ]
        self.assertEqual(len(adaptive_rows), 4)
        for row in adaptive_rows:
            self.assertLessEqual(row["cost_spent"], result["cost_budget"])
            self.assertEqual(
                row["cost_spent"],
                sum(row["fidelity_cost_units_by_level"].values()),
            )

    def test_custom_fidelity_cost_ratios_are_reported_and_reconciled(self) -> None:
        costs = {"low": 1, "medium": 10, "high": 50}
        result = run_suite(seed=31, grid_size=9, severity=1.0, fidelity_costs=costs)
        self.assertEqual(result["cost_budget"], 400)
        for row in result["policy_results"]:
            with self.subTest(policy=row["policy"]):
                self.assertEqual(row["fidelity_costs"], costs)
                self.assertEqual(sum(row["fidelity_cost_units_by_level"].values()), row["cost_spent"])
                self.assertLessEqual(row["cost_spent"], result["cost_budget"])
                self.assertEqual(row["fidelity_cost_units_by_level"], {
                    fidelity: row["fidelity_evaluations"][fidelity] * costs[fidelity]
                    for fidelity in costs
                })

    def test_invalid_fidelity_cost_maps_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly low, medium, and high"):
            run_suite(seed=1, grid_size=9, fidelity_costs={"low": 1, "high": 25})
        with self.assertRaisesRegex(ValueError, "positive integer"):
            run_suite(seed=1, grid_size=9, fidelity_costs={"low": 1, "medium": 0, "high": 25})

    def test_policy_templates_reject_incompatible_budget_or_grid(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 200"):
            run_suite(seed=1, cost_budget=199, grid_size=21)
        with self.assertRaisesRegex(ValueError, "at least 75"):
            run_suite(seed=1, cost_budget=200, grid_size=8)

    def test_invalid_grid_severity_and_policy_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SyntheticCastingWorld(grid_size=4)
        with self.assertRaises(ValueError):
            SyntheticCastingWorld(grid_size=9, severity=float("nan"))
        with self.assertRaises(ValueError):
            run_policy("magic_policy", seed=1, cost_budget=200, grid_size=9)


if __name__ == "__main__":
    unittest.main()
