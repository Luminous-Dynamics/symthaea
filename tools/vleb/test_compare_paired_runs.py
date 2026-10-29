#!/usr/bin/env python3
"""Adversarial tests for VLEB's paired task-cluster comparison analyzer."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

MODULE_PATH = Path(__file__).with_name("compare_paired_runs.py")
SPEC = importlib.util.spec_from_file_location("vleb_compare_paired_runs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot import paired comparison analyzer")
analyzer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = analyzer
SPEC.loader.exec_module(analyzer)


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def jbytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


class PairedComparisonFixture:
    PROTOCOL = "a" * 64
    REVISION = "b" * 40
    IMAGE = "sha256:" + "c" * 64

    def __init__(self) -> None:
        self.task_ids = [f"task-{index:03d}" for index in range(1, 9)]
        self.seeds = [101, 202]
        self.arms = [
            {
                "arm_id": "reference",
                "model_id": "baseline-model",
                "model_version": "v1",
                "scaffold_sha256": sha(b"reference-scaffold"),
                "execution_profile_sha256": sha(b"reference-execution-profile"),
            },
            {
                "arm_id": "candidate",
                "model_id": "candidate-model",
                "model_version": "v2",
                "scaffold_sha256": sha(b"candidate-scaffold"),
                "execution_profile_sha256": sha(b"candidate-execution-profile"),
            },
        ]
        task_pins = {task_id: sha(("task-contract:" + task_id).encode()) for task_id in self.task_ids}
        rows = []
        sequence = 0
        for task_index, task_id in enumerate(self.task_ids):
            for seed_index, seed in enumerate(self.seeds):
                for arm_index, arm in enumerate(self.arms):
                    sequence += 1
                    if arm["arm_id"] == "reference":
                        outcome = "pass" if (task_index + seed_index) % 3 == 0 else "fail"
                    else:
                        outcome = "pass" if (task_index + seed_index) % 2 == 0 else "fail"
                        if task_index == 0 and seed_index == 1:
                            outcome = "unresolved"
                    run_id = f"run-{sequence:03d}"
                    rows.append({
                        "task_id": task_id,
                        "seed": seed,
                        "arm_id": arm["arm_id"],
                        "outcome": outcome,
                        "run_id": run_id,
                        "attempt_number": 1,
                        "task_contract_sha256": task_pins[task_id],
                        "protocol_sha256": self.PROTOCOL,
                        "candidate_patch_sha256": sha(f"patch:{arm['arm_id']}:{task_id}".encode()),
                        "receipt_sha256": sha(("receipt:" + run_id).encode()),
                        "audit_report_sha256": sha(("audit:" + run_id).encode()),
                        "expected_revision": self.REVISION,
                        "image_id": self.IMAGE,
                        "network_mode": "none",
                        "cost_usd": 0.12 if arm_index == 0 else 0.08,
                        "duration_ms": 1000 + task_index * 10 + seed_index * 20 + arm_index * 75,
                    })
        self.manifest = {
            "schema_version": "vleb-paired-comparison-v1",
            "protocol_sha256": self.PROTOCOL,
            "cohort_id": "vleb-paired-pilot-001",
            "task_ids": self.task_ids,
            "task_contract_sha256_by_task": task_pins,
            "seeds": self.seeds,
            "reference_arm": "reference",
            "arms": self.arms,
            "bootstrap": {"replicates": 1000, "seed": 20261029, "confidence_level": 0.95},
            "results": rows,
        }

    def analyze(self, manifest: dict[str, Any] | None = None) -> dict[str, Any]:
        value = self.manifest if manifest is None else manifest
        return analyzer.analyze_manifest(
            value,
            input_sha256=sha(jbytes(value)),
            expected_protocol_sha256=self.PROTOCOL,
            expected_analyzer_sha256=sha(MODULE_PATH.read_bytes()),
        )


class PairedComparisonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fixture = PairedComparisonFixture()

    def test_valid_paired_cohort_reports_clustered_comparison_and_stays_analysis_only(self) -> None:
        result = self.fixture.analyze()
        self.assertTrue(result["valid"])
        self.assertEqual(result["arm_profiles"], self.fixture.arms)
        self.assertEqual(
            result["arm_profiles"][1]["execution_profile_sha256"],
            self.fixture.arms[1]["execution_profile_sha256"],
        )
        self.assertEqual(result["task_cluster_count"], 8)
        self.assertEqual(result["seed_count_per_task"], 2)
        self.assertEqual(result["expected_trial_count_per_arm"], 16)
        self.assertEqual(result["observed_trial_count"], 32)
        self.assertEqual(result["arm_summaries"]["candidate"]["scheduled_trial_count"], 16)
        candidate = result["arm_summaries"]["candidate"]
        self.assertAlmostEqual(candidate["cost_per_scheduled_trial_usd"], candidate["total_cost_usd"] / 16)
        self.assertEqual(candidate["duration_percentile_method"], "nearest_rank")
        for arm in result["arm_summaries"].values():
            self.assertIn("two_way_task_seed_bootstrap_sensitivity_95ci_pp", arm)
        self.assertTrue(result["pairwise_comparisons"])
        self.assertEqual(result["pairwise_comparisons"][0]["paired_task_cluster_count"], 8)
        self.assertEqual(result["pairwise_comparisons"][0]["seeds_per_task"], 2)
        task_rows = result["pairwise_comparisons"][0]["task_level_comparisons"]
        self.assertEqual([row["task_cluster_ordinal"] for row in task_rows], list(range(1, 9)))
        self.assertTrue(all("task_id" not in row for row in task_rows))
        self.assertEqual(
            sum(row["direction"] == "comparison_win" for row in task_rows),
            result["pairwise_comparisons"][0]["task_clusters_won"],
        )
        self.assertEqual(
            sum(row["direction"] == "reference_win" for row in task_rows),
            result["pairwise_comparisons"][0]["task_clusters_lost"],
        )
        self.assertEqual(
            sum(row["direction"] == "tie" for row in task_rows),
            result["pairwise_comparisons"][0]["task_clusters_tied"],
        )
        self.assertEqual(result["bootstrap"]["cluster_unit"], "task_id; each primary draw carries all paired seed outcomes and all comparison arms")
        self.assertIn("seed labels", result["bootstrap"]["sensitivity_cluster_unit"])
        self.assertTrue(result["analysis_only"])
        self.assertFalse(result["score_eligible"])
        self.assertTrue(result["warnings"])

    def test_unresolved_trials_stay_in_the_locked_denominator(self) -> None:
        result = self.fixture.analyze()
        candidate = result["arm_summaries"]["candidate"]
        self.assertEqual(candidate["unresolved_count"], 1)
        self.assertEqual(candidate["scheduled_trial_count"], len(self.fixture.task_ids) * len(self.fixture.seeds))
        self.assertEqual(
            candidate["success_rate_pct"],
            round(100.0 * candidate["success_count"] / candidate["scheduled_trial_count"], 6),
        )

    def test_pairwise_report_separates_resolved_disagreements_from_unresolved_cells(self) -> None:
        result = self.fixture.analyze()
        pair = result["pairwise_comparisons"][0]
        rows = {
            (row["task_id"], row["seed"], row["arm_id"]): row
            for row in self.fixture.manifest["results"]
        }
        reference = self.fixture.manifest["reference_arm"]
        comparison = pair["comparison_arm"]
        cells = [
            (rows[(task, seed, reference)]["outcome"], rows[(task, seed, comparison)]["outcome"])
            for task in self.fixture.task_ids for seed in self.fixture.seeds
        ]
        self.assertEqual(pair["paired_trial_cell_count"], len(cells))
        self.assertEqual(
            pair["both_success_cells"] + pair["reference_only_success_cells"]
            + pair["comparison_only_success_cells"] + pair["both_non_success_cells"],
            len(cells),
        )
        self.assertEqual(
            pair["success_disagreement_cells"],
            pair["reference_only_success_cells"] + pair["comparison_only_success_cells"],
        )
        self.assertEqual(
            pair["resolved_disagreement_cells"],
            sum(left != "unresolved" and right != "unresolved" and left != right for left, right in cells),
        )
        self.assertEqual(
            pair["unresolved_pair_cells"],
            sum(left == "unresolved" or right == "unresolved" for left, right in cells),
        )
        self.assertEqual(
            pair["effective_comparison_task_cluster_count"],
            pair["task_clusters_won"] + pair["task_clusters_lost"],
        )
        self.assertEqual(
            pair["effective_comparison_task_cluster_fraction_pct"],
            round(100.0 * pair["effective_comparison_task_cluster_count"] / len(self.fixture.task_ids), 6),
        )
        self.assertEqual(pair["task_win_sign_test"]["non_tied_task_clusters"], pair["task_clusters_won"] + pair["task_clusters_lost"])
        self.assertEqual(pair["task_win_sign_test"]["p_value_two_sided"], 0.25)
        self.assertEqual(pair["task_win_sign_test"]["holm_adjusted_p_value"], 0.25)
        self.assertTrue(pair["task_win_sign_test"]["estimable"])
        self.assertIn("two_way_task_seed_bootstrap_sensitivity_95ci_pp", pair)
        self.assertFalse(pair["intervals_are_simultaneous"])

    def test_deterministic_bootstrap_repeats_exactly(self) -> None:
        first = self.fixture.analyze()
        second = self.fixture.analyze()
        self.assertEqual(first["arm_summaries"], second["arm_summaries"])
        self.assertEqual(first["pairwise_comparisons"], second["pairwise_comparisons"])

    def test_single_seed_report_warns_that_seed_variability_is_not_estimable(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["seeds"] = [101]
        manifest["results"] = [row for row in manifest["results"] if row["seed"] == 101]
        result = self.fixture.analyze(manifest)
        self.assertTrue(result["valid"])
        self.assertIn(
            "single declared seed; crossed bootstrap cannot reveal between-seed variability",
            result["warnings"],
        )

    def test_crossed_bootstrap_exposes_seed_sensitivity_hidden_by_task_only_interval(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        for row in manifest["results"]:
            if row["arm_id"] == "reference":
                row["outcome"] = "fail"
            else:
                row["outcome"] = "pass" if row["seed"] == 101 else "fail"
        result = self.fixture.analyze(manifest)
        candidate = result["arm_summaries"]["candidate"]
        task_ci = candidate["task_cluster_bootstrap_95ci_pp"]
        crossed_ci = candidate["two_way_task_seed_bootstrap_sensitivity_95ci_pp"]
        # Each task averages to 50% over both fixed seeds, so a task-only
        # bootstrap is degenerate despite strong dependence on the chosen seeds.
        self.assertEqual(task_ci, {"lower_pp": 50.0, "upper_pp": 50.0})
        self.assertLess(crossed_ci["lower_pp"], 50.0)
        self.assertGreater(crossed_ci["upper_pp"], 50.0)
        pair = result["pairwise_comparisons"][0]
        self.assertEqual(pair["paired_cluster_bootstrap_95ci_pp"], {"lower_pp": 50.0, "upper_pp": 50.0})
        self.assertLess(pair["two_way_task_seed_bootstrap_sensitivity_95ci_pp"]["lower_pp"], 50.0)
        self.assertGreater(pair["two_way_task_seed_bootstrap_sensitivity_95ci_pp"]["upper_pp"], 50.0)

    def test_exact_task_win_sign_test_is_directional_and_handles_ties(self) -> None:
        self.assertAlmostEqual(analyzer.exact_two_sided_sign_test_pvalue(5, 0), 0.0625)
        self.assertEqual(analyzer.exact_two_sided_sign_test_pvalue(4, 4), 1.0)
        self.assertIsNone(analyzer.exact_two_sided_sign_test_pvalue(0, 0))

    def test_holm_adjustment_is_monotone_and_caps_at_one(self) -> None:
        adjusted = analyzer.holm_adjust_pvalues({"arm-a": 0.01, "arm-b": 0.04, "arm-c": 0.03})
        self.assertEqual(adjusted, {"arm-a": 0.03, "arm-c": 0.06, "arm-b": 0.06})
        capped = analyzer.holm_adjust_pvalues({"arm-a": 0.8, "arm-b": 0.9})
        self.assertEqual(capped, {"arm-a": 1.0, "arm-b": 1.0})

    def test_missing_cartesian_cell_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"].pop()
        with self.assertRaisesRegex(ValueError, "complete locked Cartesian cohort"):
            self.fixture.analyze(manifest)

    def test_duplicate_cell_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][-1]["task_id"] = manifest["results"][0]["task_id"]
        manifest["results"][-1]["seed"] = manifest["results"][0]["seed"]
        manifest["results"][-1]["arm_id"] = manifest["results"][0]["arm_id"]
        with self.assertRaisesRegex(ValueError, "duplicate result cell"):
            self.fixture.analyze(manifest)

    def test_paired_arms_must_use_same_immutable_environment(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        row = next(r for r in manifest["results"] if r["task_id"] == "task-001" and r["seed"] == 101 and r["arm_id"] == "candidate")
        row["image_id"] = "sha256:" + "d" * 64
        with self.assertRaisesRegex(ValueError, "paired arms disagree"):
            self.fixture.analyze(manifest)

    def test_each_arm_must_pin_the_complete_execution_profile(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        del manifest["arms"][0]["execution_profile_sha256"]
        with self.assertRaisesRegex(ValueError, "execution_profile_sha256"):
            self.fixture.analyze(manifest)

    def test_task_contract_must_match_locked_digest(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][0]["task_contract_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "locked per-task digest"):
            self.fixture.analyze(manifest)

    def test_protocol_mismatch_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][0]["protocol_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "externally pinned protocol"):
            self.fixture.analyze(manifest)

    def test_primary_trial_cannot_be_replaced_by_retry(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][0]["attempt_number"] = 2
        with self.assertRaisesRegex(ValueError, "attempt_number must be 1"):
            self.fixture.analyze(manifest)

    def test_replayed_receipt_or_audit_digest_is_rejected(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][1]["receipt_sha256"] = manifest["results"][0]["receipt_sha256"]
        with self.assertRaisesRegex(ValueError, "receipt digest replay"):
            self.fixture.analyze(manifest)

        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][1]["audit_report_sha256"] = manifest["results"][0]["audit_report_sha256"]
        with self.assertRaisesRegex(ValueError, "audit-report digest replay"):
            self.fixture.analyze(manifest)

    def test_no_success_has_null_cost_per_success_not_zero(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        for row in manifest["results"]:
            if row["arm_id"] == "candidate":
                row["outcome"] = "fail"
        result = self.fixture.analyze(manifest)
        self.assertIsNone(result["arm_summaries"]["candidate"]["cost_per_successful_trial_usd"])

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            analyzer.strict_json_loads(b'{"cohort_id":"one","cohort_id":"two"}')

    def test_nonfinite_json_numbers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-finite JSON constant"):
            analyzer.strict_json_loads(b'{"cost_usd":NaN}')

    def test_bootstrap_controls_are_frozen_and_bounded(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["bootstrap"]["replicates"] = 999
        with self.assertRaisesRegex(ValueError, "bootstrap replicates"):
            self.fixture.analyze(manifest)

        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["bootstrap"]["confidence_level"] = 0.90
        with self.assertRaisesRegex(ValueError, "requires a preregistered 0.95"):
            self.fixture.analyze(manifest)

    def test_network_mode_must_be_a_valid_string_and_matched(self) -> None:
        manifest = copy.deepcopy(self.fixture.manifest)
        manifest["results"][0]["network_mode"] = ["none"]
        with self.assertRaises(ValueError):
            self.fixture.analyze(manifest)


    def test_cli_accepts_exact_pins_and_writes_a_valid_analysis_only_report(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-paired-cli-") as temporary:
            root = Path(temporary)
            source = root / "manifest.json"
            report = root / "report.json"
            source.write_bytes(jbytes(self.fixture.manifest))
            completed = subprocess.run(
                [
                    sys.executable, str(MODULE_PATH),
                    "--input", str(source),
                    "--expected-input-sha256", sha(source.read_bytes()),
                    "--expected-protocol-sha256", self.fixture.PROTOCOL,
                    "--expected-analyzer-sha256", sha(MODULE_PATH.read_bytes()),
                    "--output", str(report),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output = json.loads(completed.stdout)
            saved = json.loads(report.read_text(encoding="utf-8"))
            self.assertEqual(output, saved)
            self.assertTrue(output["valid"])
            self.assertTrue(output["analysis_only"])
            self.assertFalse(output["score_eligible"])

    def test_cli_rejects_wrong_external_manifest_pin_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory(prefix="vleb-paired-cli-reject-") as temporary:
            source = Path(temporary) / "manifest.json"
            source.write_bytes(jbytes(self.fixture.manifest))
            completed = subprocess.run(
                [
                    sys.executable, str(MODULE_PATH),
                    "--input", str(source),
                    "--expected-input-sha256", "f" * 64,
                    "--expected-protocol-sha256", self.fixture.PROTOCOL,
                    "--expected-analyzer-sha256", sha(MODULE_PATH.read_bytes()),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
            )
            self.assertEqual(completed.returncode, 1)
            output = json.loads(completed.stdout)
            self.assertFalse(output["valid"])
            self.assertTrue(output["analysis_only"])
            self.assertFalse(output["score_eligible"])
            self.assertEqual(output["issues"][0]["code"], "paired_comparison_rejected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
