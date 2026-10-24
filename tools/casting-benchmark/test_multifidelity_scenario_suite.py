"""Tests for the adversarial multi-fidelity scenario diagnostic suite."""
from __future__ import annotations

import unittest

from multifidelity_scenario_suite import (
    DEFAULT_DESIGN_COUNT,
    FIDELITY_COSTS,
    SCENARIO_CONTRACTS,
    SUPPORTED_SCENARIOS,
    MultiFidelityScenario,
    audit_scenario,
    halton_designs,
    run_scenario_sweep,
)


class MultiFidelityScenarioSuiteTests(unittest.TestCase):
    def test_scenario_registry_is_complete_and_has_distinct_failure_mechanisms(self) -> None:
        self.assertEqual(set(SCENARIO_CONTRACTS), set(SUPPORTED_SCENARIOS))
        mechanisms = {item["failure_mechanism"] for item in SCENARIO_CONTRACTS.values()}
        self.assertEqual(len(mechanisms), len(SUPPORTED_SCENARIOS))
        for scenario_id, contract in SCENARIO_CONTRACTS.items():
            with self.subTest(scenario=scenario_id):
                self.assertTrue(contract["failure_mechanism"])
                self.assertTrue(contract["expected_diagnostic"])

    def test_halton_designs_are_repeatable_and_seed_paired(self) -> None:
        first = halton_designs(32, 5, 17)
        repeated = halton_designs(32, 5, 17)
        other_seed = halton_designs(32, 5, 23)
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(len(first), 32)
        self.assertEqual(len({design.candidate_id for design in first}), 32)
        self.assertTrue(all(0.0 <= value < 1.0 for d in first for value in d.x))

    def test_all_fidelities_match_truth_when_severity_is_zero(self) -> None:
        for scenario_id in SUPPORTED_SCENARIOS:
            with self.subTest(scenario=scenario_id):
                world = MultiFidelityScenario(
                    scenario_id, dimension=2, severity=0.0, seed=17, design_count=16
                )
                for candidate_id in world.candidate_ids:
                    truth = world.truth_for_audit_only(candidate_id)
                    for fidelity in ("low", "medium"):
                        observed = world.evaluate(candidate_id, fidelity)
                        self.assertEqual(observed.objectives, truth.objectives)
                        self.assertEqual(observed.constraint_margin, truth.constraint_margin)

    def test_responses_are_deterministic_for_same_seed_design_and_fidelity(self) -> None:
        for scenario_id in SUPPORTED_SCENARIOS:
            with self.subTest(scenario=scenario_id):
                world = MultiFidelityScenario(scenario_id, 5, 1.25, 29, design_count=24)
                for fidelity in ("low", "medium", "high"):
                    self.assertEqual(
                        world.evaluate(7, fidelity), world.evaluate(7, fidelity)
                    )

    def test_scenario_audit_honors_non_default_design_count(self) -> None:
        result = audit_scenario(
            "constraint_boundary", dimension=5, severity=1.0, seed=17, design_count=12
        )
        self.assertEqual(result["design_count"], 12)
        self.assertTrue(result["synthetic_fixture_only"])
        self.assertEqual(result["fidelity_costs"], FIDELITY_COSTS)
        for fidelity in ("low", "medium"):
            self.assertEqual(
                result["diagnostics_by_fidelity"][fidelity]["objective_pair_comparisons"],
                12 * 11,
            )
            self.assertGreaterEqual(
                result["diagnostics_by_fidelity"][fidelity]["objective_rank_inversion_fraction"],
                0.0,
            )
            self.assertLessEqual(
                result["diagnostics_by_fidelity"][fidelity]["objective_rank_inversion_fraction"],
                1.0,
            )

    def test_local_bias_and_boundary_scenarios_expose_auditable_failures(self) -> None:
        local = audit_scenario(
            "local_bias_island", dimension=2, severity=2.0, seed=23, design_count=128
        )
        boundary = audit_scenario(
            "constraint_boundary", dimension=2, severity=2.0, seed=23, design_count=128
        )
        self.assertGreater(
            local["diagnostics_by_fidelity"]["low"]["mean_absolute_objective_error"], 0.0
        )
        self.assertGreater(
            boundary["diagnostics_by_fidelity"]["low"]["feasibility_flip_count"], 0
        )

    def test_zero_severity_has_zero_rank_and_feasibility_mismatch(self) -> None:
        result = audit_scenario(
            "warped_coordinates", dimension=10, severity=0.0, seed=31, design_count=32
        )
        for fidelity in ("low", "medium"):
            row = result["diagnostics_by_fidelity"][fidelity]
            self.assertEqual(row["objective_rank_inversion_count"], 0)
            self.assertEqual(row["feasibility_flip_count"], 0)
            self.assertAlmostEqual(row["mean_absolute_objective_error"], 0.0, places=14)

    def test_scenario_sweep_retains_raw_runs_and_pairing(self) -> None:
        result = run_scenario_sweep(
            scenario_ids=("smooth_stationary", "constraint_boundary"),
            dimensions=(2, 5),
            severities=(0.0, 1.0),
            seeds=(17, 23),
            design_count=12,
        )
        self.assertEqual(result["run_count"], 2 * 2 * 2 * 2)
        self.assertEqual(len(result["paired_runs"]), result["run_count"])
        self.assertEqual(
            len(result["summary_by_scenario_dimension_severity_fidelity"]),
            2 * 2 * 2 * 2,
        )
        self.assertTrue(result["synthetic_fixture_only"])
        self.assertTrue(all(run["scope_note"].startswith("Fidelity-diagnostic") for run in result["paired_runs"]))

    def test_nonfinite_severity_and_unknown_scenario_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            MultiFidelityScenario("smooth_stationary", severity=float("nan"))
        with self.assertRaisesRegex(ValueError, "unknown scenario_id"):
            MultiFidelityScenario("optimism_only")

    def test_design_count_below_minimum_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 8"):
            MultiFidelityScenario("smooth_stationary", design_count=7)
        with self.assertRaisesRegex(ValueError, "at least 8"):
            audit_scenario("smooth_stationary", 2, 1.0, 17, design_count=7)


if __name__ == "__main__":
    unittest.main()
