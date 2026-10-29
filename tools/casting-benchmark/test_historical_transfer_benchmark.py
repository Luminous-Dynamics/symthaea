#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

from historical_transfer_benchmark import (
    BENCHMARK_ID,
    HIGH_COST_UNITS,
    LOW_COST_UNITS,
    POLICY_IDS,
    SOURCE_TASKS,
    TARGET_TASKS,
    TASK_SUITE_SHA256,
    _descriptor,
    _f1,
    _metrics,
    design_point,
    high_fidelity,
    low_fidelity,
    main,
    run_benchmark,
)


class HistoricalTransferBenchmarkTests(unittest.TestCase):
    def test_suite_has_disjoint_task_families_and_global_ids(self):
        source_families = {task["family"] for task in SOURCE_TASKS}
        target_families = {task["family"] for task in TARGET_TASKS}
        self.assertFalse(source_families & target_families)
        all_ids = [task["task_id"] for task in (*SOURCE_TASKS, *TARGET_TASKS)]
        self.assertEqual(len(all_ids), len(set(all_ids)))
        self.assertEqual(len(TASK_SUITE_SHA256), 64)

    def test_design_points_are_deterministic_and_in_unit_box(self):
        points = [design_point(i) for i in range(1, 500)]
        self.assertEqual(points, [design_point(i) for i in range(1, 500)])
        self.assertEqual(len(points), len(set(points)))
        for x in points:
            self.assertEqual(len(x), 2)
            self.assertTrue(all(0.0 <= value < 1.0 for value in x))

    def test_low_and_high_fidelity_have_nonzero_task_dependent_discrepancy(self):
        x = (0.63, 0.29)
        differences = []
        for task in SOURCE_TASKS:
            high = high_fidelity(task, x)
            low = low_fidelity(task, x)
            self.assertEqual(len(high), 3)
            self.assertEqual(len(low), 3)
            self.assertTrue(all(math.isfinite(value) for value in (*high, *low)))
            differences.append(tuple(high[j] - low[j] for j in range(3)))
        self.assertGreater(max(abs(value) for row in differences for value in row), 0.0)
        self.assertGreater(
            max(differences[i][0] for i in range(len(differences)))
            - min(differences[i][0] for i in range(len(differences))),
            0.0,
        )

    def test_known_answer_cost_accounting_and_policy_roster(self):
        receipt = run_benchmark(history_points=3, test_points=12, nearest_k=4, reuse_count=10)
        self.assertEqual(receipt["benchmark_id"], BENCHMARK_ID)
        self.assertEqual(receipt["evidence_class"], "synthetic")
        self.assertEqual(receipt["receipt_label"], "HISTORICAL_TRANSFER_SYNTHETIC_PREDICTION_AUDIT_NOT_OPTIMIZER_SOTA_EVIDENCE")
        self.assertFalse(receipt["target_high_fidelity_used_by_predictor"])
        self.assertEqual(receipt["source_task_count"], 8)
        self.assertEqual(receipt["target_task_count"], 4)
        self.assertEqual(receipt["history_pair_count"], 24)
        costs = receipt["cost_accounting"]
        self.assertEqual(costs["archive_build_cost_units"], 24 * (LOW_COST_UNITS + HIGH_COST_UNITS))
        self.assertEqual(costs["amortized_archive_cost_per_target_task_units"], 24 * 26 / 10)
        self.assertEqual(costs["target_low_evaluation_cost_per_task_units"], 12 * LOW_COST_UNITS)
        self.assertEqual(costs["target_high_only_evaluation_cost_per_task_units"], 12 * HIGH_COST_UNITS)
        self.assertEqual(set(receipt["aggregate_metrics_macro_averaged_over_target_tasks"]), set(POLICY_IDS))
        self.assertEqual(set(receipt["comparisons_vs_low_fidelity_only"]), set(POLICY_IDS) - {"low_fidelity_only"})
        self.assertIn("too few", receipt["claim_ceiling"])

    def test_target_points_are_nested_within_only_four_independent_units(self):
        receipt = run_benchmark(history_points=2, test_points=20, nearest_k=4, reuse_count=10)
        self.assertEqual(receipt["target_task_count"], 4)
        for policy in POLICY_IDS:
            aggregation = receipt["aggregate_metrics_macro_averaged_over_target_tasks"][policy]
            self.assertEqual(aggregation["target_task_units"], 4)
            self.assertTrue(aggregation["target_points_nested_within_tasks"])
            self.assertEqual(
                sum(row["policy_metrics"][policy]["audit_sample_count"] for row in receipt["target_task_results"]),
                4 * 20,
            )

    def test_source_archive_cannot_contain_target_task_ids(self):
        receipt = run_benchmark(history_points=2, test_points=8, nearest_k=3, reuse_count=1)
        self.assertEqual(len(receipt["historical_archive_sha256"]), 64)
        self.assertEqual(len(receipt["source_sha256"]), 64)
        self.assertEqual(len(receipt["task_suite_sha256"]), 64)
        # Input manifest output exposes source task count only; target rows expose no archive records.
        self.assertTrue(all("target_task_id" in row for row in receipt["target_task_results"]))
        self.assertTrue(all("source_task_id" not in row for row in receipt["target_task_results"]))

    def test_metrics_use_predicted_feasible_denominator(self):
        truth = [(0.2, 0.3, -0.1), (0.4, 0.2, 0.2), (0.3, 0.4, -0.2), (0.5, 0.5, 0.0)]
        pred = [(0.3, 0.3, 0.1), (0.4, 0.3, 0.1), (0.4, 0.4, -0.1), (0.5, 0.5, 0.2)]
        result = _metrics(truth, pred)
        self.assertEqual(result["predicted_feasible_count"], 3)
        self.assertEqual(result["true_feasible_count"], 2)
        self.assertEqual(result["false_feasible_count"], 2)
        self.assertAlmostEqual(result["false_feasible_rate_among_predicted_feasible"], 2 / 3)
        self.assertEqual(result["false_infeasible_count"], 0)

    def test_pareto_f1_edge_cases(self):
        self.assertEqual(_f1(set(), set()), 1.0)
        self.assertEqual(_f1({1}, set()), 0.0)
        self.assertEqual(_f1({1, 2}, {2, 3}), 0.5)

    def test_repeat_run_is_byte_stable_except_source_digest_not_affected_by_output(self):
        first = run_benchmark(history_points=2, test_points=8, nearest_k=3, reuse_count=10)
        second = run_benchmark(history_points=2, test_points=8, nearest_k=3, reuse_count=10)
        self.assertEqual(first, second)

    def test_rejects_bad_parameters(self):
        for kwargs, message in (
            ({"history_points": 0}, "history_points"),
            ({"test_points": -1}, "test_points"),
            ({"nearest_k": 0}, "nearest_k"),
            ({"reuse_count": 0}, "reuse_count"),
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaisesRegex(ValueError, message):
                    run_benchmark(**kwargs)

    def test_rejects_unreasonably_large_distance_sweep(self):
        with self.assertRaisesRegex(ValueError, "must be <= 25000000"):
            run_benchmark(history_points=1000, test_points=1000, nearest_k=16, reuse_count=10)

    def test_cli_writes_auditable_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "receipt.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main([
                    "--history-points", "2", "--test-points", "8",
                    "--nearest-k", "3", "--reuse-count", "10",
                    "--output", str(out),
                ])
            self.assertEqual(rc, 0)
            receipt = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(receipt["receipt_label"], "HISTORICAL_TRANSFER_SYNTHETIC_PREDICTION_AUDIT_NOT_OPTIMIZER_SOTA_EVIDENCE")
            self.assertEqual(receipt["history_pair_count"], 16)
            self.assertIn("suite_sha256=", stdout.getvalue())

    def test_cli_refuses_to_overwrite_its_own_source(self):
        from historical_transfer_benchmark import __file__ as module_path
        source = Path(module_path).resolve()
        original = source.read_bytes()
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            rc = main([
                "--history-points", "1", "--test-points", "1", "--nearest-k", "1",
                "--reuse-count", "1", "--output", str(source),
            ])
        self.assertEqual(rc, 2)
        self.assertIn("source module", stderr.getvalue())
        self.assertEqual(source.read_bytes(), original)

    def test_invalid_descriptor_rejected(self):
        invalid = {"task_id": "bad", "family": "bad", "descriptor": [0.2, float("nan"), 0.4]}
        with self.assertRaisesRegex(ValueError, "finite"):
            _descriptor(invalid)


if __name__ == "__main__":
    unittest.main()
