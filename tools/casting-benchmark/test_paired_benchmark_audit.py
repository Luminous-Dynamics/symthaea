#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from paired_benchmark_audit import analyze_manifest, main

SHA_A = "a" * 64
SHA_B = "b" * 64


def _curve(points):
    return [{"cost_units": cost, "quality": quality} for cost, quality in points]


def _run(unit, policy, curve, *, completed=True, spent=10, shifts=None, decision="recommend"):
    if shifts is None:
        shifts = [
            {"shift_id": "nominal", "quality": 0.8 if policy == "candidate" else 0.7,
             "false_feasible_count": 0, "decision_count": 1},
            {"shift_id": "boundary_drift", "quality": 0.6 if policy == "candidate" else 0.4,
             "false_feasible_count": 0 if policy == "candidate" else 1, "decision_count": 1},
        ]
    if decision == "abstain":
        shifts = [
            {"shift_id": item["shift_id"], "quality": 0.0,
             "false_feasible_count": 0, "decision_count": 0}
            for item in shifts
        ]
    return {
        "unit_id": unit,
        "task_family": "local_bias_island" if unit != "u3" else "structural_discrepancy",
        "constraint_family": "constraint_boundary",
        "dimension": 2,
        "severity": 0.5,
        "seed": {"u1": 17, "u2": 23, "u3": 41}[unit],
        "policy_id": policy,
        "budget_units": 10,
        "cost_spent_units": spent,
        "completed": completed,
        "decision": decision,
        "curve": _curve(curve),
        "shift_results": shifts,
        "run_receipt_sha256": "e" * 64 if policy == "baseline" else "f" * 64,
        "policy_source_sha256": SHA_A if policy == "baseline" else SHA_B,
        "policy_config_sha256": "d" * 64 if policy == "baseline" else "e" * 64,
        "environment_sha256": "1" * 64,
        "suite_manifest_sha256": "2" * 64,
    }


def manifest():
    runs = []
    baseline_curves = {
        "u1": [(0, 0.0), (5, 0.4), (10, 0.6)],
        "u2": [(0, 0.0), (5, 0.3), (10, 0.5)],
        "u3": [(0, 0.0), (5, 0.4), (10, 0.55)],
    }
    candidate_curves = {
        "u1": [(0, 0.0), (5, 0.6), (10, 0.8)],
        "u2": [(0, 0.0), (5, 0.5), (10, 0.7)],
        "u3": [(0, 0.0), (5, 0.6), (8, 0.75)],
    }
    for unit in ("u1", "u2", "u3"):
        runs.append(_run(unit, "baseline", baseline_curves[unit]))
        runs.append(_run(
            unit, "candidate", candidate_curves[unit],
            completed=(unit != "u3"),
            spent=(12 if unit == "u2" else (8 if unit == "u3" else 10)),
        ))
    return {
        "schema_version": "1.0.0",
        "benchmark_id": "CASTING-PAIRED-COST-001",
        "protocol_sha256": "c" * 64,
        "normalization_id": "finite_pool_hv_fixed_reference_v1",
        "baseline_policy": "baseline",
        "quality_target": 0.55,
        "sampling_unit": "unit_id",
        "independence_note": "Each task-family and seed combination is declared an independent generated unit; this declaration is not externally verified.",
        "bootstrap_replicates": 100,
        "bootstrap_seed": 1701,
        "runs": runs,
    }


class PairedBenchmarkAuditTests(unittest.TestCase):
    def test_known_answer_paired_effects_and_bootstrap_are_deterministic(self):
        doc = manifest()
        first = analyze_manifest(doc)
        second = analyze_manifest(doc)
        self.assertEqual(first, second)
        self.assertEqual(first["unit_count"], 3)
        effect = first["comparisons_vs_baseline"]["candidate"]["effects"]["cost_normalized_quality_auc_delta"]
        self.assertEqual(effect["n_paired_units"], 3)
        self.assertEqual(effect["wins"], 3)
        self.assertEqual(effect["ties"], 0)
        self.assertEqual(effect["losses"], 0)
        self.assertAlmostEqual(effect["median_delta"], 0.15)
        self.assertAlmostEqual(effect["mean_delta"], (0.15 + 0.15 + 0.165) / 3)
        self.assertEqual(effect["bootstrap_95pct_ci_mean_delta"], [effect["mean_delta"], effect["mean_delta"]])
        self.assertEqual(effect["bootstrap_95pct_ci_win_rate"], [1.0, 1.0])

    def test_target_cost_and_hit_rates_include_all_units(self):
        receipt = analyze_manifest(manifest())
        policy = receipt["per_policy"]
        self.assertAlmostEqual(policy["baseline"]["target_hit_rate"], 2 / 3)
        self.assertEqual(policy["candidate"]["target_hit_rate"], 1.0)
        self.assertEqual(policy["baseline"]["target_hit_count"], 2)
        self.assertEqual(policy["candidate"]["target_hit_count"], 3)
        costs = receipt["comparisons_vs_baseline"]["candidate"]["effects"]["cost_to_target_reduction_when_both_reach"]
        self.assertEqual(costs["n_paired_units"], 2)
        self.assertGreater(costs["median_delta"], 0.0)

    def test_worst_shift_results_and_false_feasible_rate_are_reported(self):
        receipt = analyze_manifest(manifest())
        effects = receipt["comparisons_vs_baseline"]["candidate"]["effects"]
        self.assertAlmostEqual(effects["mean_shift_quality_delta"]["median_delta"], 0.15)
        self.assertAlmostEqual(effects["worst_shift_quality_delta"]["median_delta"], 0.2)
        self.assertAlmostEqual(effects["false_feasible_rate_reduction"]["mean_delta"], 0.5)
        boundary = receipt["comparisons_vs_baseline"]["candidate"]["by_shift"]["boundary_drift"]
        self.assertEqual(boundary["wins"], 3)

    def test_incomplete_and_over_budget_runs_are_not_dropped(self):
        receipt = analyze_manifest(manifest())
        candidate = receipt["per_policy"]["candidate"]
        self.assertEqual(candidate["n_independent_units"], 3)
        self.assertEqual(candidate["incomplete_count"], 1)
        self.assertEqual(candidate["over_budget_count"], 1)
        self.assertEqual(
            receipt["comparisons_vs_baseline"]["candidate"]["effects"]["cost_normalized_quality_auc_delta"]["n_paired_units"],
            3,
        )

    def test_reports_family_stratification_not_only_pooled_average(self):
        receipt = analyze_manifest(manifest())
        by_family = receipt["comparisons_vs_baseline"]["candidate"]["by_task_family"]
        self.assertEqual(set(by_family), {"local_bias_island", "structural_discrepancy"})
        self.assertEqual(by_family["local_bias_island"]["auc_delta"]["n_paired_units"], 2)
        self.assertEqual(by_family["structural_discrepancy"]["auc_delta"]["n_paired_units"], 1)

    def test_rejects_candidate_or_shift_rows_as_sampling_units(self):
        doc = manifest()
        doc["sampling_unit"] = "candidate_rows"
        with self.assertRaisesRegex(ValueError, "sampling_unit must be unit_id"):
            analyze_manifest(doc)

    def test_rejects_missing_policy_pair_instead_of_silently_dropping_unit(self):
        doc = manifest()
        doc["runs"] = [row for row in doc["runs"] if not (row["unit_id"] == "u3" and row["policy_id"] == "candidate")]
        with self.assertRaisesRegex(ValueError, "policy coverage mismatch"):
            analyze_manifest(doc)

    def test_rejects_duplicate_unit_policy_and_shift_ids(self):
        doc = manifest()
        doc["runs"].append(dict(doc["runs"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate run"):
            analyze_manifest(doc)
        doc = manifest()
        doc["runs"][0]["shift_results"].append(dict(doc["runs"][0]["shift_results"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate shift_id"):
            analyze_manifest(doc)

    def test_rejects_unmatched_budget_and_shift_set(self):
        doc = manifest()
        doc["runs"][1]["budget_units"] = 11
        with self.assertRaisesRegex(ValueError, "metadata"):
            analyze_manifest(doc)
        doc = manifest()
        doc["runs"][1]["shift_results"].pop()
        with self.assertRaisesRegex(ValueError, "shift coverage mismatch"):
            analyze_manifest(doc)

    def test_rejects_nonmonotonic_cost_points_or_curve_beyond_budget(self):
        doc = manifest()
        doc["runs"][0]["curve"] = _curve([(0, 0), (5, 0.4), (4, 0.6)])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            analyze_manifest(doc)
        doc = manifest()
        doc["runs"][0]["curve"] = _curve([(0, 0), (11, 0.6)])
        with self.assertRaisesRegex(ValueError, "exceeds the common budget"):
            analyze_manifest(doc)

    def test_rejects_invalid_false_feasible_denominator_and_abstention(self):
        doc = manifest()
        doc["runs"][0]["shift_results"][0]["false_feasible_count"] = 2
        with self.assertRaisesRegex(ValueError, "exceeds decision_count"):
            analyze_manifest(doc)
        doc = manifest()
        row = doc["runs"][0]
        row["decision"] = "abstain"
        with self.assertRaisesRegex(ValueError, "abstaining run"):
            analyze_manifest(doc)

    def test_rejects_invalid_sha_and_normalization(self):
        doc = manifest()
        doc["runs"][0]["run_receipt_sha256"] = "not-a-sha"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            analyze_manifest(doc)
        doc = manifest()
        doc["runs"][1]["task_family"] = "different-normalization-does-not-belong"
        with self.assertRaisesRegex(ValueError, "metadata"):
            analyze_manifest(doc)

    def test_cli_receipt_binds_raw_manifest_and_evaluator(self):
        doc = manifest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runs.json"
            output = root / "receipt.json"
            source.write_text(json.dumps(doc, separators=(",", ":")), encoding="utf-8")
            source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([str(source), "--output", str(output)])
            self.assertEqual(code, 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["source_artifacts"]["raw_manifest_sha256"], source_digest)
            self.assertEqual(len(receipt["evaluator_sha256"]), 64)
            self.assertEqual(receipt["receipt_label"], "PAIRED_MATCHED_COST_COMPARISON_NOT_PHYSICAL_CERTIFICATION")
            self.assertEqual(receipt["protocol_sha256"], "c" * 64)

    def test_cli_rejects_duplicate_json_keys_and_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "runs.json"
            output = root / "receipt.json"
            source.write_text('{"benchmark_id":"wrong","benchmark_id":"other"}', encoding="utf-8")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(source), "--output", str(output)])
            self.assertEqual(code, 2)
            self.assertIn("duplicate JSON key", stderr.getvalue())
            self.assertFalse(output.exists())
            doc = manifest()
            source.write_text(json.dumps(doc), encoding="utf-8")
            original = source.read_bytes()
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = main([str(source), "--output", str(source)])
            self.assertEqual(code, 2)
            self.assertEqual(source.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
