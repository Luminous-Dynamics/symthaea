#!/usr/bin/env python3
"""Tests for the compositional mismatch runner and independent trace scorer."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from multifidelity_scenario_suite import MultiFidelityScenario, SUPPORTED_SCENARIOS
from multifidelity_compositional_bakeoff import CompositionalScenarioWorld, run_sweep
from score_compositional_bakeoff import score_document, loads_no_duplicate_keys, family_response, design_point


ROOT = Path(__file__).resolve().parents[2]


class CompositionalBakeoffTests(unittest.TestCase):
    def test_diagonal_compositions_match_single_family_for_all_fidelities(self) -> None:
        costs = {"low": 1, "medium": 5, "high": 25}
        for family in SUPPORTED_SCENARIOS:
            composed = CompositionalScenarioWorld(family, family, 2, 1.25, 23, costs, 75)
            baseline = MultiFidelityScenario(
                family, dimension=2, severity=1.25, seed=23,
                fidelity_costs=costs, design_count=75,
            )
            for candidate_id in (0, 7, 31, 74):
                for fidelity in ("low", "medium", "high"):
                    actual = composed._response(candidate_id, fidelity)
                    expected = baseline.evaluate(candidate_id, fidelity)
                    self.assertEqual(actual.objectives, expected.objectives, (family, candidate_id, fidelity))
                    self.assertEqual(actual.constraint_margin, expected.constraint_margin, (family, candidate_id, fidelity))
                    self.assertEqual(actual.cost_units, expected.cost_units)

    def test_independent_equation_replay_matches_every_family(self) -> None:
        for family in SUPPORTED_SCENARIOS:
            for dimension in (2, 5):
                for candidate_id in (0, 11, 37):
                    x = design_point(candidate_id, dimension, 29)
                    source = MultiFidelityScenario(
                        family, dimension, 1.25, 29,
                        {"low": 1, "medium": 5, "high": 25}, design_count=75,
                    )
                    source_x = source.design(candidate_id).x
                    self.assertEqual(x, source_x)
                    for fidelity in ("low", "medium", "high"):
                        actual = family_response(family, x, fidelity, 1.25, 29)
                        expected = source.evaluate(candidate_id, fidelity)
                        self.assertEqual(actual[0], expected.objectives, (family, dimension, candidate_id, fidelity))
                        self.assertEqual(actual[1], expected.constraint_margin, (family, dimension, candidate_id, fidelity))

    def test_off_diagonal_components_are_separable(self) -> None:
        costs = {"low": 1, "medium": 5, "high": 25}
        composed = CompositionalScenarioWorld(
            "local_bias_island", "constraint_boundary", 2, 1.0, 17, costs, 75
        )
        objective_world = MultiFidelityScenario(
            "local_bias_island", 2, 1.0, 17, costs, design_count=75
        )
        constraint_world = MultiFidelityScenario(
            "constraint_boundary", 2, 1.0, 17, costs, design_count=75
        )
        for candidate_id in (3, 19, 63):
            actual = composed._response(candidate_id, "low")
            objective = objective_world.evaluate(candidate_id, "low")
            constraint = constraint_world.evaluate(candidate_id, "low")
            self.assertEqual(actual.objectives, objective.objectives)
            self.assertEqual(actual.constraint_margin, constraint.constraint_margin)

    def test_small_sweep_replays_and_preserves_pairing(self) -> None:
        document = run_sweep(
            objective_families=("local_bias_island", "warped_coordinates"),
            constraint_families=("constraint_boundary", "discontinuous_step"),
            dimensions=(2,),
            severities=(0.5, 1.0),
            seeds=(17, 23),
            design_count=75,
        )
        self.assertEqual(document["run_count"], 2 * 2 * 1 * 2 * 2 * 5)
        self.assertEqual(len(document["paired_runs"]), document["run_count"])
        by_key = {
            (r["objective_family"], r["constraint_family"], r["dimension"], r["severity"], r["seed"], r["policy"]): r
            for r in document["paired_runs"]
        }
        self.assertEqual(len(by_key), document["run_count"])
        receipt = score_document(document, ROOT)
        self.assertTrue(receipt["valid"])
        self.assertEqual(receipt["run_count"], document["run_count"])
        self.assertEqual(receipt["cell_summary_count"], 2 * 2 * 1 * 2 * 5)
        self.assertEqual(receipt["paired_comparison_count"], 2 * 2 * 1 * 2 * 4)

    def test_scorer_rejects_modified_objective_value(self) -> None:
        document = run_sweep(
            objective_families=("local_bias_island",),
            constraint_families=("constraint_boundary",),
            dimensions=(2,),
            severities=(0.5,),
            seeds=(17,),
            policies=("high_only", "low_screen"),
            design_count=75,
        )
        document["paired_runs"][0]["evaluations"][0]["objectives"][0] += 0.01
        with self.assertRaisesRegex(ValueError, "replay rejected"):
            score_document(document, ROOT)

    def test_scorer_rejects_modified_budget_ledger(self) -> None:
        document = run_sweep(
            objective_families=("smooth_stationary",),
            constraint_families=("constraint_boundary",),
            dimensions=(2,),
            severities=(1.0,),
            seeds=(17,),
            policies=("high_only",),
            design_count=75,
        )
        document["paired_runs"][0]["cost_spent"] -= 1
        with self.assertRaisesRegex(ValueError, "replay rejected"):
            score_document(document, ROOT)

    def test_duplicate_json_keys_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            loads_no_duplicate_keys('{"seed":17,"seed":23}')


if __name__ == "__main__":
    unittest.main()
