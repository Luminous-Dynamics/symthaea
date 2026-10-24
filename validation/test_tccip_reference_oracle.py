"""Unit tests for the independent TCCIP standard-library oracle."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from validation.tccip_reference_oracle import (
    DEFAULT_MANIFEST,
    Intervention,
    evaluate_fixture,
    input_information_baseline,
    pairwise_utility_interactions,
    task_utility_for_cuts,
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

    def test_manifest_is_versioned_and_covers_all_cases(self) -> None:
        self.assertEqual(len(self.fixtures), 6)
        self.assertEqual(
            set(self.by_name),
            {
                "useful_xor_coordination",
                "irrelevant_cross_component_connectivity",
                "harmful_cross_component_coupling",
                "task_relevant_high_x1_prior",
                "task_relevant_high_x0_prior",
                "role_matched_xor_synergy_uniform_prior",
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
            "role_matched_xor_synergy_uniform_prior": (15, 8, 7, 0.0, 0.0),
        }
        for fixture in self.fixtures:
            _, partitions, summary = evaluate_fixture(fixture)
            total, usable, skipped, lower, upper = expected[fixture.name]
            self.assertEqual(summary["total"], total)
            self.assertEqual(len(partitions), usable)
            self.assertEqual(summary["skipped"], skipped)
            self.assertAlmostEqual(min(point.tccip for point in partitions), lower, places=12)
            self.assertAlmostEqual(max(point.tccip for point in partitions), upper, places=12)

    def test_role_matched_xor_edges_are_both_essential_but_tccip_is_zero(self) -> None:
        fixture = self.by_name["role_matched_xor_synergy_uniform_prior"]
        scores, _, summary = evaluate_fixture(fixture)
        info = input_information_baseline(fixture)
        self.assertEqual(summary["intact"], 1.0)
        self.assertEqual(summary["cross_loss"], 0.5)
        self.assertEqual(summary["control_loss"], 0.5)
        self.assertEqual(summary["tccip"], 0.0)
        # Both individual edges matter equally to performance.
        self.assertEqual([score.loss for score in scores], [0.5, 0.5])
        # Yet the task has one bit of joint input information, and zero MI
        # between either individual input and the XOR target.
        self.assertAlmostEqual(info["target_entropy_bits"], 1.0, places=12)
        self.assertEqual(info["individual_input_mi_bits"], [0.0, 0.0])
        self.assertAlmostEqual(info["joint_input_mi_bits"], 1.0, places=12)
        interactions = pairwise_utility_interactions(fixture)
        pair = next(
            row for row in interactions
            if row["first_intervention_id"] == "cross_cut_xor_left"
            and row["second_intervention_id"] == "matched_internal_xor_right"
        )
        self.assertEqual(pair["first_cut_utility"], 0.5)
        self.assertEqual(pair["second_cut_utility"], 0.5)
        self.assertEqual(pair["joint_cut_utility"], 0.5)
        self.assertEqual(pair["utility_interaction_contrast"], 0.5)

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

    def test_pairwise_utility_interaction_detects_xor_complementarity(self) -> None:
        fixture = self.by_name["role_matched_xor_synergy_uniform_prior"]
        pairs = pairwise_utility_interactions(fixture)
        pair = next(
            row for row in pairs
            if row["first_intervention_id"] == "cross_cut_xor_left"
            and row["second_intervention_id"] == "matched_internal_xor_right"
        )
        self.assertEqual(pair["intact_utility"], 1.0)
        self.assertEqual(pair["first_cut_utility"], 0.5)
        self.assertEqual(pair["second_cut_utility"], 0.5)
        self.assertEqual(pair["joint_cut_utility"], 0.5)
        self.assertEqual(pair["utility_interaction_contrast"], 0.5)

    def test_pairwise_utility_interaction_is_zero_for_irrelevant_branch(self) -> None:
        fixture = self.by_name["irrelevant_cross_component_connectivity"]
        pairs = pairwise_utility_interactions(fixture)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["utility_interaction_contrast"], 0.0)

    def test_pairwise_interaction_manifest_matches_all_exact_joint_cut_values(self) -> None:
        path = Path(__file__).resolve().parents[1] / "validation/fixtures/tccip_pairwise_utility_interactions.tsv"
        lines = path.read_text(encoding="utf-8").splitlines()
        self.assertIn(
            "# schema=tccip-pairwise-utility-interaction-expectations-v1",
            lines,
        )
        seen = set()
        for line in lines:
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t")
            self.assertEqual(len(fields), 8, line)
            key = (fields[0], fields[1], fields[2])
            self.assertNotIn(key, seen)
            seen.add(key)
            fixture = self.by_name[fields[0]]
            first = next(item for item in fixture.interventions if item.identifier == fields[1])
            second = next(item for item in fixture.interventions if item.identifier == fields[2])
            actual = (
                task_utility_for_cuts(fixture, ()),
                task_utility_for_cuts(fixture, (first,)),
                task_utility_for_cuts(fixture, (second,)),
                task_utility_for_cuts(fixture, (first, second)),
            )
            expected = tuple(map(float, fields[3:7]))
            for got, wanted in zip(actual, expected):
                self.assertAlmostEqual(got, wanted, places=12, msg=str(key))
            contrast = actual[0] - actual[1] - actual[2] + actual[3]
            self.assertAlmostEqual(contrast, float(fields[7]), places=12, msg=str(key))
        self.assertEqual(len(seen), 16)

    def test_duplicate_edge_interventions_are_rejected(self) -> None:
        fixture = self.by_name["role_matched_xor_synergy_uniform_prior"]
        duplicate = replace(
            fixture,
            interventions=fixture.interventions + (
                Intervention("duplicate-edge", source=2, target=4, cost=1),
            ),
        )
        with self.assertRaisesRegex(ValueError, "duplicate intervention edge"):
            validate_fixture(duplicate)

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
        self.assertIn("information_baseline\trole_matched_xor_synergy_uniform_prior", first)
        self.assertIn("\t1\t0,0\t1", first)
        data_lines = [line for line in first.splitlines() if line and not line.startswith("#")]
        table = list(csv.reader(io.StringIO("\n".join(data_lines)), delimiter="\t"))
        self.assertEqual(len(table[0]), 28)
        self.assertTrue(all(len(row) == 28 for row in table[1:]))
        xor_info = next(
            row for row in table
            if row[0] == "information_baseline" and row[1] == "role_matched_xor_synergy_uniform_prior"
        )
        self.assertEqual(xor_info[19:22], ["1", "0,0", "1"])
        xor_interaction = next(
            row for row in table
            if row[0] == "utility_interaction"
            and row[1] == "role_matched_xor_synergy_uniform_prior"
            and row[22] == "cross_cut_xor_left"
            and row[23] == "matched_internal_xor_right"
        )
        self.assertEqual(xor_interaction[24:28], ["0.5", "0.5", "0.5", "0.5"])

    def test_declared_partition_mask_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-mask.tsv"
            manifest = self.manifest_bytes.decode("utf-8")
            self.assertIn("\t240\t1.0\t", manifest)
            path.write_text(manifest.replace("\t240\t1.0\t", "\t241\t1.0\t", 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "partition mask"):
                parse_manifest(path)

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
