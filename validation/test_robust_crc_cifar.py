import unittest
import numpy as np

from robust_crc_cifar import (
    DistributionallyRobustRandomizedCRC,
    evaluate_equal_share_for_scenario,
    run_comparison,
)


SCENARIOS = {
    "nominal": (7300, 969, 1731),
    "less_superclass_correct": (7300, 500, 2200),
    "more_fine_errors": (6500, 1200, 2300),
    "higher_accuracy": (8000, 800, 1200),
    "mixed_shift": (7000, 1500, 1500),
}


class DistributionallyRobustCRCTests(unittest.TestCase):
    def test_scenario_weights_and_test_probabilities_are_normalized(self):
        robust = DistributionallyRobustRandomizedCRC(5, .2, SCENARIOS)
        for name, table in robust.scenario_tables.items():
            self.assertAlmostEqual(float(table["weights"].sum()), 1.0, places=12)
            self.assertTrue(np.allclose(table["conditional_test_probabilities"].sum(axis=1), 1.0))

    def test_robust_policy_satisfies_all_augmented_state_constraints(self):
        robust = DistributionallyRobustRandomizedCRC(5, .2, SCENARIOS)
        result = robust.solve()
        self.assertTrue(result["verification"]["valid"], result["verification"])
        self.assertEqual(
            result["verification"]["augmented_states_checked"],
            len(robust.augmented),
        )

    def test_worst_case_epigraph_matches_minimum_scenario_metric(self):
        robust = DistributionallyRobustRandomizedCRC(5, .2, SCENARIOS)
        result = robust.solve()
        actual = min(
            row["correct_specificity"]
            for row in result["scenario_metrics"].values()
        )
        self.assertAlmostEqual(actual, result["worst_case_specificity"], places=8)

    def test_robust_objective_is_not_worse_than_equal_share_for_scenario_set(self):
        report = run_comparison(
            n=5, alpha=.2, scenario_pools=SCENARIOS, equal_share_rounds=30
        )
        equal_worst = min(
            row["equal_share_policy"]["correct_specificity"]
            for row in report["scenario_comparison"].values()
        )
        self.assertGreaterEqual(
            report["worst_case_specificity"] + 1e-8,
            equal_worst,
        )

    def test_every_scenario_uses_same_policy_but_different_weights(self):
        robust = DistributionallyRobustRandomizedCRC(5, .2, SCENARIOS)
        result = robust.solve()
        P = np.asarray(result["probabilities"])
        self.assertEqual(P.shape, (len(robust.states), len(robust.actions)))
        values = [
            robust.evaluate_scenario(P, name)["correct_specificity"]
            for name in robust.names
        ]
        self.assertGreater(max(values)-min(values), 1e-4)

    def test_scenario_order_does_not_change_the_robust_optimum(self):
        reversed_scenarios = dict(reversed(list(SCENARIOS.items())))
        a = DistributionallyRobustRandomizedCRC(5, .2, SCENARIOS).solve()
        b = DistributionallyRobustRandomizedCRC(5, .2, reversed_scenarios).solve()
        self.assertAlmostEqual(
            a["worst_case_specificity"],
            b["worst_case_specificity"],
            places=8,
        )


    def test_regular_finite_population_grid_cardinality_and_boundaries(self):
        from robust_crc_cifar import make_composition_grid_scenarios
        scenarios=make_composition_grid_scenarios(10000,500,500)
        self.assertEqual(len(scenarios),171)
        self.assertTrue(all(sum(v)==10000 for v in scenarios.values()))
        self.assertTrue(all(min(v)>=500 for v in scenarios.values()))
        self.assertIn("mix_500_500_9000",scenarios)
        self.assertIn("mix_9000_500_500",scenarios)

    def test_small_grid_robust_policy_remains_valid(self):
        from robust_crc_cifar import run_grid_robustness_stress
        result=run_grid_robustness_stress(
            n=5,alpha=.2,population_total=21,
            min_count_per_type=6,step=3,equal_share_rounds=20
        )
        self.assertEqual(result["protocol"]["scenario_count"],3)
        self.assertTrue(result["robust_verification"]["valid"])
        self.assertLessEqual(result["robust_verification"]["max_budget_excess"],1e-8)
        self.assertGreaterEqual(
            result["worst_case"]["robust_specificity"]+1e-7,
            result["worst_case"]["equal_share_specificity"]
        )

    def test_rejects_invalid_scenario_sets(self):
        with self.assertRaises(ValueError):
            DistributionallyRobustRandomizedCRC(5, .2, {"only_one": (70,20,10)})
        with self.assertRaises(ValueError):
            DistributionallyRobustRandomizedCRC(5, .2, {
                "one": (70,20,10), "different_total": (70,20,11)
            })
        with self.assertRaises(ValueError):
            DistributionallyRobustRandomizedCRC(5, .2, {
                "too_small": (4,20,10), "other": (70,20,10)
            })

    def test_maximin_policy_returns_dual_optimality_receipt(self):
        robust = DistributionallyRobustRandomizedCRC(5, .2, SCENARIOS)
        result = robust.solve()
        self.assertTrue(result["solver"]["kkt"]["valid"])
        self.assertLessEqual(result["solver"]["kkt"]["primal_inequality_violation"], 1e-8)
        self.assertLessEqual(result["solver"]["kkt"]["stationarity_max_abs"], 1e-7)
        self.assertLessEqual(result["solver"]["kkt"]["duality_gap_abs"], 1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=2)
