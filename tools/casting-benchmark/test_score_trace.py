"""Adversarial tests for independent scoring of hashed evaluation traces."""
from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any

from score_trace import TraceScoreError, hypervolume_2d_minimize, score_trace

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = Path(__file__).parent / "fixtures" / "success_receipt.fixture.json"
TRACE_SCHEMA_PATH = ROOT / "docs" / "research" / "CASTING_BENCHMARK_TRACE.schema.json"
SCORER_PATH = Path(__file__).parent / "score_trace.py"
SCHEMA_BYTES = TRACE_SCHEMA_PATH.read_bytes()
SCORER_BYTES = SCORER_PATH.read_bytes()
BASE_RECEIPT = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def make_row(index: int, status: str, objectives: list[float] | None, candidate: str | None = None) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "evaluation_index": index,
        "candidate_id": candidate or f"candidate-{index:03}",
        "status": status,
        "objectives": objectives,
    }


def encode_rows(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)).encode("utf-8")


def make_receipt(
    trace_bytes: bytes,
    statuses: list[str],
    budget: int,
    metrics: dict[str, float],
) -> dict[str, Any]:
    receipt = copy.deepcopy(BASE_RECEIPT)
    n = len(statuses)
    receipt["status"] = "success"
    receipt["synthetic_fixture_only"] = True
    receipt["evaluation_mode"] = "single_fidelity"
    receipt["suite"].update({
        "dimension": 30,
        "objective_count": 2,
        "objective_direction": "minimize_all",
        "ideal_point": [0.0, 0.0],
        "reference_point": [10.0, 10.0],
    })
    receipt["budget"] = {"objective_evaluations": budget, "initial_design_evaluations": min(2, n)}
    receipt["accounting"].update({
        "objective_evaluations": n,
        "feasible_evaluations": statuses.count("feasible"),
        "infeasible_evaluations": statuses.count("infeasible"),
        "invalid_evaluations": statuses.count("invalid"),
        "failed_evaluations": statuses.count("failed"),
        "wall_clock_ms": max(1, n),
    })
    receipt["termination_reason"] = "budget_exhausted" if n == budget else "predeclared_target_reached"
    receipt["metrics"] = {
        "final_hypervolume": metrics["final_hypervolume"],
        "normalized_hypervolume_auc": metrics["normalized_hypervolume_auc"],
        "feasible_fraction": metrics["feasible_fraction"],
        "invalid_or_failed_fraction": metrics["invalid_or_failed_fraction"],
        "igd_plus": None,
        "spacing": None,
        "evaluations_to_target": None,
        "post_selection_coverage": None,
        "regional_min_coverage": None,
        "weighted_ess": None,
        "mean_interval_width_proxy": None,
        "abstention_fraction": 0.0,
    }
    receipt["artifacts"]["raw_trace_sha256"] = hashlib.sha256(trace_bytes).hexdigest()
    receipt["artifacts"]["trace_schema_sha256"] = hashlib.sha256(SCHEMA_BYTES).hexdigest()
    receipt["artifacts"]["scorer_sha256"] = hashlib.sha256(SCORER_BYTES).hexdigest()
    return receipt


def score_known_answer(rows: list[dict[str, Any]], planned_budget: int, expected: dict[str, float] | None = None):
    trace = encode_rows(rows)
    statuses = [row["status"] for row in rows]
    receipt = make_receipt(trace, statuses, planned_budget, expected or {
        "final_hypervolume": 46.0,
        "normalized_hypervolume_auc": 0.4,
        "feasible_fraction": 1.0,
        "invalid_or_failed_fraction": 0.0,
    })
    return score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)


class IndependentTraceScorerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.known_rows = [
            make_row(1, "feasible", [2.0, 8.0]),
            make_row(2, "feasible", [4.0, 4.0]),
            make_row(3, "feasible", [7.0, 2.0]),
            make_row(4, "feasible", [5.0, 8.0]),  # Dominated by evaluation 2.
        ]
        self.known_trace = encode_rows(self.known_rows)
        self.known_metrics = {
            "final_hypervolume": 46.0,
            "normalized_hypervolume_auc": 0.4,
            "feasible_fraction": 1.0,
            "invalid_or_failed_fraction": 0.0,
        }

    def test_known_answer_hypervolume_and_early_stop_auc(self) -> None:
        result = score_known_answer(self.known_rows, 6, self.known_metrics)
        self.assertAlmostEqual(result["metrics_recomputed"]["final_hypervolume"], 46.0, places=12)
        self.assertAlmostEqual(result["metrics_recomputed"]["normalized_hypervolume_auc"], 0.4, places=12)
        self.assertEqual(result["planned_budget"], 6)
        self.assertEqual(result["objective_evaluations"], 4)
        self.assertTrue(result["reported_metrics_match"])

    def test_dominated_point_does_not_change_hypervolume(self) -> None:
        front = [[2.0, 8.0], [4.0, 4.0], [7.0, 2.0]]
        self.assertAlmostEqual(hypervolume_2d_minimize(front, [10.0, 10.0]), 46.0)
        self.assertAlmostEqual(hypervolume_2d_minimize(front + [[5.0, 8.0]], [10.0, 10.0]), 46.0)

    def test_infeasible_design_does_not_enter_archive_but_consumes_budget(self) -> None:
        rows = [
            make_row(1, "feasible", [2.0, 8.0]),
            make_row(2, "infeasible", [1.0, 1.0]),  # Would dominate if incorrectly included.
            make_row(3, "feasible", [4.0, 4.0]),
            make_row(4, "feasible", [7.0, 2.0]),
        ]
        expected = {
            "final_hypervolume": 46.0,
            "normalized_hypervolume_auc": 0.295,
            "feasible_fraction": 0.75,
            "invalid_or_failed_fraction": 0.0,
        }
        result = score_known_answer(rows, 4, expected)
        self.assertEqual(result["status_counts"]["infeasible"], 1)
        self.assertAlmostEqual(result["metrics_recomputed"]["normalized_hypervolume_auc"], 0.295)

    def test_no_feasible_points_has_zero_hypervolume(self) -> None:
        rows = [
            make_row(1, "infeasible", [3.0, 4.0]),
            make_row(2, "failed", None),
        ]
        expected = {
            "final_hypervolume": 0.0,
            "normalized_hypervolume_auc": 0.0,
            "feasible_fraction": 0.0,
            "invalid_or_failed_fraction": 0.5,
        }
        result = score_known_answer(rows, 2, expected)
        self.assertEqual(result["metrics_recomputed"]["final_hypervolume"], 0.0)

    def _receipt_for(self, trace: bytes, rows: list[dict[str, Any]], budget: int = 4):
        return make_receipt(trace, [row["status"] for row in rows], budget, self.known_metrics)

    def test_raw_trace_hash_mismatch_is_rejected(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        receipt["artifacts"]["raw_trace_sha256"] = "0" * 64
        with self.assertRaisesRegex(TraceScoreError, "raw_trace_sha256"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_schema_hash_mismatch_is_rejected(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        with self.assertRaisesRegex(TraceScoreError, "trace_schema_sha256"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES + b" ", SCORER_BYTES)

    def test_scorer_hash_mismatch_is_rejected(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        with self.assertRaisesRegex(TraceScoreError, "scorer_sha256"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES + b" ")

    def test_duplicate_json_keys_are_rejected(self) -> None:
        trace = b'{"schema_version":"1.0.0","evaluation_index":1,"evaluation_index":1,"candidate_id":"x","status":"feasible","objectives":[2,8]}\n'
        receipt = self._receipt_for(trace, [make_row(1, "feasible", [2.0, 8.0])], 1)
        with self.assertRaisesRegex(TraceScoreError, "duplicate JSON object key"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_nonfinite_json_number_is_rejected(self) -> None:
        trace = b'{"schema_version":"1.0.0","evaluation_index":1,"candidate_id":"x","status":"feasible","objectives":[NaN,8]}\n'
        receipt = self._receipt_for(trace, [make_row(1, "feasible", [2.0, 8.0])], 1)
        with self.assertRaisesRegex(TraceScoreError, "non-standard JSON numeric constant"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_index_gap_is_rejected(self) -> None:
        rows = [make_row(1, "feasible", [2.0, 8.0]), make_row(3, "feasible", [4.0, 4.0])]
        trace = encode_rows(rows)
        receipt = self._receipt_for(trace, rows, 2)
        with self.assertRaisesRegex(TraceScoreError, "contiguous and 1-based"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_trace_accounting_status_mismatch_is_rejected(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        receipt["accounting"]["feasible_evaluations"] = 3
        receipt["accounting"]["infeasible_evaluations"] = 1
        with self.assertRaisesRegex(TraceScoreError, "trace status count feasible"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_reported_metric_mismatch_is_rejected(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        receipt["metrics"]["final_hypervolume"] = 45.0
        with self.assertRaisesRegex(TraceScoreError, "reported metric mismatch: final_hypervolume"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_unsupported_dimension_and_fidelity_fail_closed(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        receipt["suite"]["objective_count"] = 3
        receipt["suite"]["ideal_point"] = [0.0, 0.0, 0.0]
        receipt["suite"]["reference_point"] = [10.0, 10.0, 10.0]
        with self.assertRaisesRegex(TraceScoreError, "exactly two"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES)

        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        receipt["evaluation_mode"] = "multi_fidelity"
        receipt["accounting"]["fidelity_evaluations"] = {"low": 2, "high": 2}
        receipt["accounting"]["fidelity_cost_units_by_level"] = {"low": 2.0, "high": 20.0}
        with self.assertRaisesRegex(TraceScoreError, "single_fidelity"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_below_ideal_objective_is_rejected(self) -> None:
        rows = [make_row(1, "feasible", [-0.1, 8.0])]
        trace = encode_rows(rows)
        receipt = self._receipt_for(trace, rows, 1)
        with self.assertRaisesRegex(TraceScoreError, "below frozen ideal"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_invalid_and_failed_status_require_null_objectives(self) -> None:
        rows = [make_row(1, "failed", [1.0, 2.0])]
        trace = encode_rows(rows)
        receipt = self._receipt_for(trace, rows, 1)
        with self.assertRaisesRegex(TraceScoreError, "invalid/failed evaluations require"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_unknown_trace_fields_and_blank_lines_are_rejected(self) -> None:
        row = make_row(1, "feasible", [2.0, 8.0])
        row["algorithm_claimed_success"] = True
        trace = encode_rows([row])
        receipt = self._receipt_for(trace, [make_row(1, "feasible", [2.0, 8.0])], 1)
        with self.assertRaisesRegex(TraceScoreError, "trace fields mismatch"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

        trace = encode_rows([make_row(1, "feasible", [2.0, 8.0])]) + b"\n"
        receipt = self._receipt_for(trace, [make_row(1, "feasible", [2.0, 8.0])], 1)
        with self.assertRaisesRegex(TraceScoreError, "blank JSONL rows"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_budget_exhausted_requires_actual_count_equal_budget(self) -> None:
        rows = self.known_rows
        trace = encode_rows(rows)
        receipt = self._receipt_for(trace, rows, 6)
        receipt["termination_reason"] = "budget_exhausted"
        with self.assertRaisesRegex(TraceScoreError, "requires actual evaluation count"):
            score_trace(receipt, trace, SCHEMA_BYTES, SCORER_BYTES)

    def test_synthetic_receipt_is_rejected_in_publication_mode(self) -> None:
        receipt = self._receipt_for(self.known_trace, self.known_rows, 6)
        with self.assertRaisesRegex(TraceScoreError, "fixture receipts cannot be published"):
            score_trace(receipt, self.known_trace, SCHEMA_BYTES, SCORER_BYTES, for_publication=True)


if __name__ == "__main__":
    unittest.main()
