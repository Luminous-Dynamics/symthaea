#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import unittest

from decision_robustness_eval import BENCHMARK_ID, score_document, validate_document


def fixture() -> dict:
    return {
        "schema_version": "1.0.0",
        "benchmark_id": BENCHMARK_ID,
        "task_id": "known-answer-1",
        "run_id": "seed-17",
        "split": {
            "selection_frozen_before_holdout": True,
            "holdout_role": "independent_post_selection",
            "selection_freeze_sha256": "a" * 64,
            "holdout_manifest_sha256": "b" * 64,
        },
        "reference_point": [1.2, 1.2],
        "candidate_set": [
            {"candidate_id": "selected-a", "predicted_objectives": [0.2, 0.7], "predicted_constraint_margin": 0.1, "selected": True},
            {"candidate_id": "pool-better", "predicted_objectives": [0.4, 0.4], "predicted_constraint_margin": 0.2, "selected": False},
        ],
        "holdout_shifts": [
            {"shift_id": "shift-a", "shift_family": "boundary_drift", "evaluations": [
                {"candidate_id": "selected-a", "objectives": [0.2, 0.7], "constraint_margin": -0.2},
                {"candidate_id": "pool-better", "objectives": [0.4, 0.4], "constraint_margin": 0.2},
            ]},
            {"shift_id": "shift-b", "shift_family": "parameter_drift", "evaluations": [
                {"candidate_id": "selected-a", "objectives": [0.2, 0.7], "constraint_margin": 0.1},
                {"candidate_id": "pool-better", "objectives": [0.4, 0.4], "constraint_margin": 0.2},
            ]},
        ],
    }


class DecisionRobustnessTests(unittest.TestCase):
    def score(self, doc: dict | None = None) -> dict:
        data = fixture() if doc is None else doc
        raw = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
        return score_document(data, raw)

    def test_detects_selection_exploitation_as_false_feasible(self) -> None:
        metrics = self.score()["metrics"]
        self.assertEqual(metrics["selected_false_feasible_count"], 1)
        self.assertEqual(metrics["selected_predicted_feasible_count"], 2)
        self.assertAlmostEqual(metrics["selected_false_feasible_rate"], 0.5)
        self.assertAlmostEqual(metrics["selected_feasibility_rate"], 0.5)

    def test_finite_pool_regret_detects_better_unselected_candidate(self) -> None:
        row = self.score()["per_shift"][0]
        self.assertGreater(row["candidate_pool_feasible_hypervolume"], row["selected_feasible_hypervolume"])
        self.assertGreater(row["finite_pool_normalized_regret"], 0.0)
        self.assertLessEqual(row["finite_pool_normalized_regret"], 1.0)

    def test_selection_of_every_candidate_has_zero_finite_pool_regret(self) -> None:
        doc = fixture()
        for item in doc["candidate_set"]:
            item["selected"] = True
        receipt = self.score(doc)
        self.assertAlmostEqual(receipt["metrics"]["finite_pool_normalized_regret_mean"], 0.0)
        self.assertAlmostEqual(receipt["metrics"]["finite_pool_hypervolume_retention_mean"], 1.0)

    def test_reports_false_infeasible(self) -> None:
        doc = fixture()
        doc["candidate_set"][0]["predicted_constraint_margin"] = -0.1
        metrics = self.score(doc)["metrics"]
        self.assertEqual(metrics["selected_false_infeasible_count"], 1)
        self.assertEqual(metrics["selected_predicted_infeasible_count"], 2)

    def test_rejects_incomplete_candidate_by_shift_matrix(self) -> None:
        doc = fixture()
        doc["holdout_shifts"][0]["evaluations"].pop()
        with self.assertRaisesRegex(ValueError, "evaluate every candidate"):
            validate_document(doc)

    def test_rejects_duplicate_candidate(self) -> None:
        doc = fixture()
        doc["candidate_set"].append(dict(doc["candidate_set"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate candidate_id"):
            validate_document(doc)

    def test_rejects_non_independent_split_label(self) -> None:
        doc = fixture()
        doc["split"]["holdout_role"] = "optimizer_trace"
        with self.assertRaisesRegex(ValueError, "holdout_role"):
            validate_document(doc)

    def test_receipt_never_claims_to_authenticate_holdout(self) -> None:
        receipt = self.score()
        self.assertFalse(receipt["split_declarations"]["independence_verified_by_evaluator"])
        self.assertEqual(receipt["receipt_label"], "POST_SELECTION_SHIFT_AUDIT_NOT_PHYSICAL_CERTIFICATION")

    def test_receipt_binds_exact_input_bytes(self) -> None:
        doc = fixture()
        raw = json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()
        receipt = score_document(doc, raw)
        self.assertEqual(receipt["input_sha256"], hashlib.sha256(raw).hexdigest())

    def test_rejects_objective_outside_reference_box(self) -> None:
        doc = fixture()
        doc["holdout_shifts"][0]["evaluations"][0]["objectives"] = [1.2, 0.5]
        with self.assertRaisesRegex(ValueError, "reference_point"):
            validate_document(doc)

    def test_no_feasible_pool_reference_is_not_misreported_as_zero_regret(self) -> None:
        doc = fixture()
        for shift in doc["holdout_shifts"]:
            for entry in shift["evaluations"]:
                entry["constraint_margin"] = -1.0
        receipt = self.score(doc)
        self.assertIsNone(receipt["per_shift"][0]["finite_pool_normalized_regret"])
        self.assertEqual(receipt["metrics"]["shifts_without_feasible_reference_hypervolume"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
