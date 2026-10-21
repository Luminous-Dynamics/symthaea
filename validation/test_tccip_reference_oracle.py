"""Unit tests for the independent TCCIP standard-library oracle."""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from validation.tccip_reference_oracle import (
    DEFAULT_MANIFEST,
    Intervention,
    evaluate_fixture,
    parse_manifest,
    render_artifact,
    task_utility,
    validate_fixture,
)


class TccipReferenceOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixtures, cls.manifest_bytes = parse_manifest(DEFAULT_MANIFEST)
        cls.by_name = {fixture.name: fixture for fixture in cls.fixtures}

    def test_manifest_is_versioned_and_covers_three_cases(self) -> None:
        self.assertEqual(len(self.fixtures), 5)
        self.assertEqual(
            set(self.by_name),
            {
                "useful_xor_coordination",
                "irrelevant_cross_component_connectivity",
                "harmful_cross_component_coupling",
                "task_relevant_high_x1_prior",
                "task_relevant_high_x0_prior",
            },
        )
        for fixture in self.fixtures:
            validate_fixture(fixture)

    def test_xor_fixture_matches_analytic_task_and_intervention_values(self) -> None:
        fixture = self.by_name["useful_xor_coordination"]
        self.assertEqual(task_utility(fixture), 1.0)
        _, _, summary = evaluate_fixture(fixture)
        self.assertEqual(summary["cross_loss"], 0.5)
        self.assertEqual(summary["control_loss"], 0.0)
        self.assertEqual(summary["tccip"], 0.5)

    def test_irrelevant_connectivity_has_zero_contrast_and_zero_partition_range(self) -> None:
        fixture = self.by_name["irrelevant_cross_component_connectivity"]
        _, partitions, summary = evaluate_fixture(fixture)
        self.assertEqual(summary["tccip"], 0.0)
        self.assertTrue(all(point.tccip == 0.0 for point in partitions))

    def test_harmful_coupling_keeps_negative_value(self) -> None:
        fixture = self.by_name["harmful_cross_component_coupling"]
        _, partitions, summary = evaluate_fixture(fixture)
        self.assertEqual(summary["intact"], 0.0)
        self.assertEqual(summary["cross_loss"], -0.5)
        self.assertEqual(summary["tccip"], -0.5)
        self.assertEqual(min(point.tccip for point in partitions), -0.5)
        self.assertEqual(max(point.tccip for point in partitions), 0.5)

    def test_partition_sweep_counts_match_expected_oracle(self) -> None:
        expected = {
            "useful_xor_coordination": (127, 112, 15, -0.5, 0.5),
            "irrelevant_cross_component_connectivity": (15, 8, 7, 0.0, 0.0),
            "harmful_cross_component_coupling": (511, 448, 63, -0.5, 0.5),
            "task_relevant_high_x1_prior": (15, 8, 7, -0.3, 0.3),
            "task_relevant_high_x0_prior": (15, 8, 7, -0.3, 0.3),
        }
        for fixture in self.fixtures:
            _, partitions, summary = evaluate_fixture(fixture)
            total, usable, skipped, lower, upper = expected[fixture.name]
            self.assertEqual(summary["total"], total)
            self.assertEqual(len(partitions), usable)
            self.assertEqual(summary["skipped"], skipped)
            self.assertEqual(min(point.tccip for point in partitions), lower)
            self.assertEqual(max(point.tccip for point in partitions), upper)

    def test_same_graph_reverses_tccip_sign_when_input_prior_is_reversed(self) -> None:
        high_x1 = self.by_name["task_relevant_high_x1_prior"]
        high_x0 = self.by_name["task_relevant_high_x0_prior"]
        self.assertEqual(high_x1.rules, high_x0.rules)
        self.assertEqual(high_x1.components, high_x0.components)
        self.assertEqual(high_x1.interventions, high_x0.interventions)
        _, _, high_summary = evaluate_fixture(high_x1)
        _, _, low_summary = evaluate_fixture(high_x0)
        self.assertAlmostEqual(high_summary["cross_loss"], 0.4, places=12)
        self.assertAlmostEqual(high_summary["control_loss"], 0.1, places=12)
        self.assertAlmostEqual(high_summary["tccip"], 0.3, places=12)
        self.assertAlmostEqual(low_summary["cross_loss"], 0.1, places=12)
        self.assertAlmostEqual(low_summary["control_loss"], 0.4, places=12)
        self.assertAlmostEqual(low_summary["tccip"], -0.3, places=12)

    def test_nonexistent_intervention_edge_is_rejected(self) -> None:
        fixture = self.by_name["useful_xor_coordination"]
        bad = replace(
            fixture,
            interventions=(
                Intervention("invalid", source=7, target=4, cost=1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "not on a topological edge|not a real edge"):
            validate_fixture(bad)

    def test_oracle_fails_closed_when_expected_value_is_corrupted(self) -> None:
        fixture = self.by_name["useful_xor_coordination"]
        corrupted = replace(fixture, expected=(1.0, 0.25, 0.0, 0.25))
        with self.assertRaisesRegex(AssertionError, "cross-cut loss"):
            evaluate_fixture(corrupted)

    def test_artifact_is_deterministic_and_binds_manifest_hash(self) -> None:
        first = render_artifact(self.fixtures, self.manifest_bytes, "source-test-sha")
        second = render_artifact(self.fixtures, self.manifest_bytes, "source-test-sha")
        self.assertEqual(first, second)
        self.assertIn("# source_sha=source-test-sha", first)
        self.assertIn("manifest_sha256=", first)
        self.assertIn("not Rust execution output", first)
        self.assertIn("partition\tuseful_xor_coordination", first)

    def test_malformed_manifest_row_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.tsv"
            path.write_text(
                "# schema=tccip-toy-expectations-v3\n"
                "only\ttwo\tfields\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "expected 19 fields"):
                parse_manifest(path)


if __name__ == "__main__":
    unittest.main()
