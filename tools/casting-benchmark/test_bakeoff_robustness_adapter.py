#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import adapt_bakeoff_to_robustness as adapter
from adapt_bakeoff_to_robustness import (
    ADAPTER_BENCHMARK_ID,
    ADAPTER_SCHEMA_VERSION,
    adapt_documents,
)
from decision_robustness_eval import validate_document, score_document


def _canonical(document: dict) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def fixture_documents() -> tuple[dict, bytes, dict, bytes, dict, bytes]:
    selector = {
        "objective_family": "local_bias_island",
        "constraint_family": "constraint_boundary",
        "dimension": 2,
        "severity": 0.5,
        "seed": 17,
        "policy": "low_screen",
    }
    candidate_pool_sha = "e" * 64
    design_count = 75
    sweep = {
        "benchmark_id": "CASTING-MF-COMPOSITION-001",
        "synthetic_fixture_only": True,
        "objective_families": ["local_bias_island"],
        "constraint_families": ["constraint_boundary"],
        "dimensions": [2],
        "severities": [0.5],
        "seeds": [17],
        "policies": ["low_screen"],
        "design_count": design_count,
        "cost_budget": 200,
        "fidelity_costs": {"low": 1, "medium": 5, "high": 25},
        "source_sha256": {"tools/casting-benchmark/example.py": "f" * 64},
        "run_count": 1,
        "paired_runs": [{
            **selector,
            "design_count": design_count,
            "candidate_pool_sha256": candidate_pool_sha,
            "fidelity_costs": {"low": 1, "medium": 5, "high": 25},
            "cost_budget": 200,
            "cost_spent": 26,
            "evaluations": [
                {"candidate_id": 0, "fidelity": "high", "objectives": [0.2, 0.7], "constraint_margin": 0.1, "cost_units": 25},
                {"candidate_id": 1, "fidelity": "low", "objectives": [0.3, 0.6], "constraint_margin": 0.2, "cost_units": 1},
            ],
        }],
        "warning": "test fixture only",
    }
    sweep_raw = _canonical(sweep)
    selection = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "benchmark_id": ADAPTER_BENCHMARK_ID,
        "source_sweep_sha256": hashlib.sha256(sweep_raw).hexdigest(),
        "selector": selector,
        "candidate_pool_sha256": candidate_pool_sha,
        "manifest_generator_sha256": "9" * 64,
        "nominal_prediction_source": "test_low_fidelity_reference",
        "reference_point": [1.2, 1.2],
        "selected_candidate_ids": [0],
        "selection_frozen_before_holdout": True,
        "candidate_predictions": [
            {
                "candidate_id": cid,
                "objectives": [0.1 + cid * 0.001, 0.7 - cid * 0.003],
                "constraint_margin": 0.1,
            }
            for cid in range(design_count)
        ],
    }
    selection_raw = _canonical(selection)
    holdout = {
        "schema_version": ADAPTER_SCHEMA_VERSION,
        "benchmark_id": ADAPTER_BENCHMARK_ID,
        "source_sweep_sha256": hashlib.sha256(sweep_raw).hexdigest(),
        "candidate_pool_sha256": candidate_pool_sha,
        "manifest_generator_sha256": "9" * 64,
        "selection_manifest_sha256": hashlib.sha256(selection_raw).hexdigest(),
        "selector": selector,
        "holdout_role": "synthetic_post_selection_diagnostic",
        "evidence_class": "synthetic",
        "shift_suite_id": "test-synthetic-shift-suite",
        "shifts": [
            {
                "shift_id": "boundary-shift",
                "shift_family": "constraint_boundary_drift",
                "evaluations": [
                    {
                        "candidate_id": cid,
                        "objectives": [0.12 + cid * 0.001, 0.72 - cid * 0.003],
                        "constraint_margin": -0.2 if cid == 0 else 0.1,
                    }
                    for cid in range(design_count)
                ],
            },
            {
                "shift_id": "parameter-shift",
                "shift_family": "parameter_drift",
                "evaluations": [
                    {
                        "candidate_id": cid,
                        "objectives": [0.12 + cid * 0.001, 0.72 - cid * 0.003],
                        "constraint_margin": 0.1,
                    }
                    for cid in range(design_count)
                ],
            },
        ],
    }
    holdout_raw = _canonical(holdout)
    return sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw


def adapted():
    sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
    return adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw)


class BakeoffRobustnessAdapterTests(unittest.TestCase):
    def test_adapter_creates_valid_evaluator_input(self) -> None:
        document = adapted()
        parsed = validate_document(document)
        self.assertEqual(len(parsed["candidates"]), 75)
        self.assertEqual(parsed["selected_ids"], {"0"})
        self.assertEqual(len(parsed["shifts"]), 2)

    def test_adapter_binds_exact_sweep_and_manifest_bytes(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        converted = adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw)
        self.assertEqual(converted["source_run"]["sweep_sha256"], hashlib.sha256(sweep_raw).hexdigest())
        self.assertEqual(converted["source_run"]["manifest_generator_sha256"], "9" * 64)
        adapter_bytes = Path(adapter.__file__).resolve().read_bytes()
        self.assertEqual(converted["source_run"]["adapter_sha256"], hashlib.sha256(adapter_bytes).hexdigest())
        self.assertEqual(converted["split"]["selection_freeze_sha256"], hashlib.sha256(selection_raw).hexdigest())
        self.assertEqual(converted["split"]["holdout_manifest_sha256"], hashlib.sha256(holdout_raw).hexdigest())

    def test_audit_detects_selected_design_failing_one_of_two_shifts(self) -> None:
        document = adapted()
        raw = _canonical(document)
        receipt = score_document(document, raw)
        self.assertEqual(receipt["metrics"]["selected_false_feasible_count"], 1)
        self.assertEqual(receipt["metrics"]["selected_predicted_feasible_count"], 2)
        self.assertAlmostEqual(receipt["metrics"]["selected_feasibility_rate"], 0.5)

    def test_rejects_selection_bound_to_different_sweep_bytes(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        selection["source_sweep_sha256"] = "1" * 64
        with self.assertRaisesRegex(ValueError, "not bound to these exact sweep bytes"):
            adapt_documents(sweep, sweep_raw, selection, _canonical(selection), holdout, holdout_raw)

    def test_rejects_incomplete_candidate_prediction_pool(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        selection["candidate_predictions"].pop()
        with self.assertRaisesRegex(ValueError, "cover every candidate exactly once"):
            adapt_documents(sweep, sweep_raw, selection, _canonical(selection), holdout, holdout_raw)

    def test_rejects_missing_shift_observation(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        holdout["shifts"][0]["evaluations"].pop()
        with self.assertRaisesRegex(ValueError, "cover every candidate exactly once"):
            adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, _canonical(holdout))

    def test_rejects_mismatched_selector(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        holdout["selector"]["policy"] = "joint_action_cost_aware"
        with self.assertRaisesRegex(ValueError, "selector does not match"):
            adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, _canonical(holdout))

    def test_rejects_duplicate_shift_id(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        holdout["shifts"][1]["shift_id"] = holdout["shifts"][0]["shift_id"]
        with self.assertRaisesRegex(ValueError, "duplicate shift_id"):
            adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, _canonical(holdout))

    def test_rejects_non_synthetic_or_unknown_sweep(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        sweep["synthetic_fixture_only"] = False
        updated_raw = _canonical(sweep)
        selection["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        holdout["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "synthetic_fixture_only"):
            adapt_documents(sweep, updated_raw, selection, _canonical(selection), holdout, _canonical(holdout))

    def test_rejects_evaluation_cost_ledger_mismatch(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        sweep["paired_runs"][0]["cost_spent"] = 99
        updated_raw = _canonical(sweep)
        selection["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        holdout["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "cost_spent disagrees"):
            adapt_documents(sweep, updated_raw, selection, _canonical(selection), holdout, _canonical(holdout))

    def test_rejects_malformed_source_digest(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        sweep["source_sha256"]["tools/casting-benchmark/example.py"] = "bad-digest"
        updated_raw = _canonical(sweep)
        selection["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        holdout["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "64-character SHA-256"):
            adapt_documents(sweep, updated_raw, selection, _canonical(selection), holdout, _canonical(holdout))

    def test_rejects_source_lock_path_traversal(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        sweep["source_sha256"] = {"../unsafe.py": "f" * 64}
        updated_raw = _canonical(sweep)
        selection["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        holdout["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "invalid source-lock path"):
            adapt_documents(sweep, updated_raw, selection, _canonical(selection), holdout, _canonical(holdout))

    def test_rejects_manifest_with_mismatched_reference_point(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        selection["reference_point"] = [1.3, 1.2]
        with self.assertRaisesRegex(ValueError, "frozen benchmark point"):
            adapt_documents(sweep, sweep_raw, selection, _canonical(selection), holdout, holdout_raw)

    def test_rejects_selection_not_declared_frozen_before_holdout(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        selection["selection_frozen_before_holdout"] = False
        with self.assertRaisesRegex(ValueError, "selection_frozen_before_holdout=true"):
            adapt_documents(sweep, sweep_raw, selection, _canonical(selection), holdout, holdout_raw)

    def test_rejects_holdout_manifest_with_wrong_role(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        holdout["holdout_role"] = "optimizer_trace"
        with self.assertRaisesRegex(ValueError, "not a supported frozen role"):
            adapt_documents(sweep, sweep_raw, selection, selection_raw, holdout, _canonical(holdout))

    def test_rejects_selection_id_outside_candidate_pool(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        selection["selected_candidate_ids"] = [75]
        with self.assertRaisesRegex(ValueError, "valid pool candidate IDs"):
            adapt_documents(sweep, sweep_raw, selection, _canonical(selection), holdout, holdout_raw)

    def test_rejects_ambiguous_run_selector(self) -> None:
        sweep, sweep_raw, selection, selection_raw, holdout, holdout_raw = fixture_documents()
        sweep["paired_runs"].append(dict(sweep["paired_runs"][0]))
        sweep["run_count"] = 2
        updated_raw = _canonical(sweep)
        selection["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        holdout["source_sweep_sha256"] = hashlib.sha256(updated_raw).hexdigest()
        with self.assertRaisesRegex(ValueError, "match exactly one"):
            adapt_documents(sweep, updated_raw, selection, _canonical(selection), holdout, _canonical(holdout))


if __name__ == "__main__":
    unittest.main(verbosity=2)
