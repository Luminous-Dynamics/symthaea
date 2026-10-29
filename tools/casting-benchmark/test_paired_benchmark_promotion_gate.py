#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from paired_benchmark_audit import analyze_manifest, canonical_json
from paired_benchmark_promotion_gate import assess_promotion, main

HASH = "a" * 64
BASE_SOURCE = "b" * 64
CAND_SOURCE = "c" * 64
BASE_CONFIG = "d" * 64
CAND_CONFIG = "e" * 64
ENV = "1" * 64
SUITE = "2" * 64
BENCH_PROTOCOL = "3" * 64
VALID_PROTOCOL = "4" * 64


def _sha(label):
    return hashlib.sha256(label.encode()).hexdigest()


def paired_manifest(unit_count=30, evidence_class="synthetic", *, incomplete=False, over_budget=False):
    runs = []
    for index in range(unit_count):
        unit = f"unit-{index:03d}"
        family = "alloy_a" if index < unit_count // 2 else "alloy_b"
        seed = 1000 + index
        for policy, curve in (
            ("baseline", [{"cost_units": 0, "quality": 0.0},
                          {"cost_units": 5, "quality": 0.4},
                          {"cost_units": 10, "quality": 0.6}]),
            ("candidate", [{"cost_units": 0, "quality": 0.0},
                           {"cost_units": 5, "quality": 0.6},
                           {"cost_units": 10, "quality": 0.8}]),
        ):
            candidate = policy == "candidate"
            runs.append({
                "unit_id": unit,
                "task_family": family,
                "constraint_family": "constraint_boundary",
                "dimension": 2,
                "severity": 0.5,
                "seed": seed,
                "policy_id": policy,
                "budget_units": 10,
                "cost_spent_units": 12 if over_budget and candidate and index == 0 else 10,
                "completed": not (incomplete and candidate and index == 0),
                "decision": "recommend",
                "curve": curve,
                "shift_results": [
                    {"shift_id": "nominal", "quality": 0.8 if candidate else 0.7,
                     "false_feasible_count": 0, "decision_count": 1},
                    {"shift_id": "boundary_drift", "quality": 0.6 if candidate else 0.4,
                     "false_feasible_count": 0 if candidate else 1, "decision_count": 1},
                ],
                "run_receipt_sha256": _sha(f"receipt:{unit}:{policy}"),
                "policy_source_sha256": CAND_SOURCE if candidate else BASE_SOURCE,
                "policy_config_sha256": CAND_CONFIG if candidate else BASE_CONFIG,
                "environment_sha256": ENV,
                "suite_manifest_sha256": SUITE,
            })
    return {
        "schema_version": "1.0.0",
        "benchmark_id": "CASTING-PAIRED-COST-001",
        "evidence_class": evidence_class,
        "protocol_sha256": BENCH_PROTOCOL,
        "normalization_id": "fixed_normalized_quality_v1",
        "baseline_policy": "baseline",
        "quality_target": 0.55,
        "sampling_unit": "unit_id",
        "independence_note": "Each unit represents a separately generated task-seed instance; independence remains a declared assumption.",
        "bootstrap_replicates": 200,
        "bootstrap_seed": 17,
        "runs": runs,
    }


def validation_manifest(count=100, evidence_class="synthetic", *, false_count=0, predicted_count=None):
    if predicted_count is None:
        predicted_count = count
    records = []
    for index in range(count):
        predicted = index < predicted_count
        observed = not (predicted and index < false_count)
        records.append({
            "unit_id": f"validation-unit-{index:04d}",
            "task_family": "alloy_a" if index % 2 == 0 else "alloy_b",
            "shift_id": "physical_boundary" if index % 3 == 0 else "nominal",
            "predicted_feasible": predicted,
            "observed_feasible": observed,
            "decision_receipt_sha256": _sha(f"decision:{index}"),
            "outcome_receipt_sha256": _sha(f"outcome:{index}"),
        })
    return {
        "schema_version": "1.0.0",
        "validation_id": "validation-001",
        "candidate_policy": "candidate",
        "evidence_class": evidence_class,
        "policy_source_sha256": CAND_SOURCE,
        "policy_config_sha256": CAND_CONFIG,
        "validation_protocol_sha256": VALID_PROTOCOL,
        "environment_sha256": "5" * 64,
        "split": "independent_held_out",
        "policy_frozen_before_unseal": True,
        "outcomes_unsealed_after_freeze": True,
        "independence_unit": "unit_id",
        "independence_note": "One distinct physical or simulated unit per record, as declared by the experiment protocol.",
        "records": records,
    }


def contract(required_evidence="synthetic", minimum_units=30):
    return {
        "schema_version": "1.0.0",
        "contract_id": "casting-promotion-gate-v1",
        "candidate_policy": "candidate",
        "baseline_policy": "baseline",
        "required_evidence_class": required_evidence,
        "benchmark_protocol_sha256": BENCH_PROTOCOL,
        "validation_protocol_sha256": VALID_PROTOCOL,
        "minimum_independent_units": minimum_units,
        "minimum_units_per_critical_family": 5,
        "minimum_validation_units": 60,
        "minimum_cost_target_pairs": 20,
        "critical_task_families": ["alloy_a", "alloy_b"],
        "minimum_median_relative_auc_improvement": 0.05,
        "minimum_all_unit_win_rate": 0.70,
        "maximum_critical_family_regression": 0.05,
        "minimum_cost_to_target_reduction": 0.20,
        "maximum_false_feasible_rate": 0.05,
        "confidence": 0.95,
        "require_complete_runs": True,
        "require_in_budget_runs": True,
    }


def scored_receipt(doc, raw_manifest_sha256=None):
    receipt = analyze_manifest(doc)
    raw = canonical_json(doc) if raw_manifest_sha256 is None else raw_manifest_sha256
    receipt["source_artifacts"]["raw_manifest_sha256"] = (
        hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else raw
    )
    audit_source = Path(__file__).with_name("paired_benchmark_audit.py").read_bytes()
    receipt["evaluator_sha256"] = hashlib.sha256(audit_source).hexdigest()
    return receipt


def assess(doc=None, ct=None, val=None):
    doc = paired_manifest() if doc is None else doc
    ct = contract() if ct is None else ct
    val = validation_manifest() if val is None else val
    doc_raw = canonical_json(doc)
    receipt = scored_receipt(doc, doc_raw)
    receipt_raw = canonical_json(receipt)
    return assess_promotion(
        receipt, doc, ct, val,
        paired_receipt_file_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        paired_run_manifest_sha256=hashlib.sha256(doc_raw).hexdigest(),
        contract_manifest_sha256=_sha("contract raw bytes"),
        validation_manifest_sha256=_sha("validation raw bytes"),
        evaluator_sha256=_sha("promotion evaluator source"),
    )


class PairedPromotionGateTests(unittest.TestCase):
    def test_known_answer_ready_for_next_stage(self):
        receipt = assess()
        self.assertEqual(receipt["decision"], "ready_for_next_stage")
        self.assertEqual(receipt["failed_checks"], [])
        self.assertEqual(receipt["inconclusive_checks"], [])
        self.assertEqual(receipt["independent_task_units"], 30)
        self.assertEqual(receipt["false_feasible_validation_events"], 0)
        self.assertLessEqual(receipt["false_feasible_one_sided_upper_bound"], 0.05)

    def test_ties_count_as_nonwins_against_all_unit_win_target(self):
        doc = paired_manifest()
        base = next(run for run in doc["runs"] if run["unit_id"] == "unit-000" and run["policy_id"] == "baseline")
        candidate = next(run for run in doc["runs"] if run["unit_id"] == "unit-000" and run["policy_id"] == "candidate")
        candidate["curve"] = [dict(point) for point in base["curve"]]
        # Use a separate valid paired set where the candidate ties on every unit.
        tied = paired_manifest()
        for row in tied["runs"]:
            if row["policy_id"] == "candidate":
                baseline_row = next(x for x in tied["runs"] if x["unit_id"] == row["unit_id"] and x["policy_id"] == "baseline")
                row["curve"] = [dict(point) for point in baseline_row["curve"]]
        result = assess(tied)
        effect = result["checks"]
        win_check = next(check for check in effect if check["check"] == "minimum_all_unit_win_rate")
        self.assertFalse(win_check["passed"])
        self.assertIn("ties count as non-wins", win_check["detail"])

    def test_recomputes_paired_metrics_instead_of_trusting_receipt(self):
        doc = paired_manifest()
        receipt = scored_receipt(doc)
        receipt["per_policy"]["candidate"]["mean_cost_normalized_quality_auc"] = 0.001
        doc_raw = canonical_json(doc)
        with self.assertRaisesRegex(ValueError, "does not match recomputation"):
            assess_promotion(
                receipt, doc, contract(), validation_manifest(),
                paired_receipt_file_sha256=_sha("tampered receipt"),
                paired_run_manifest_sha256=hashlib.sha256(doc_raw).hexdigest(),
                contract_manifest_sha256=_sha("contract"),
                validation_manifest_sha256=_sha("validation"),
                evaluator_sha256=_sha("source"),
            )

    def test_paired_receipt_must_match_exact_run_manifest_digest(self):
        doc = paired_manifest()
        doc_raw = canonical_json(doc)
        receipt = scored_receipt(doc, doc_raw)
        with self.assertRaisesRegex(ValueError, "does not match recomputation"):
            assess_promotion(
                receipt, doc, contract(), validation_manifest(),
                paired_receipt_file_sha256=_sha("receipt"),
                paired_run_manifest_sha256=_sha("different raw run bytes"),
                contract_manifest_sha256=_sha("contract"),
                validation_manifest_sha256=_sha("validation"),
                evaluator_sha256=_sha("source"),
            )

    def test_wrong_evidence_class_cannot_pass(self):
        result = assess(
            doc=paired_manifest(evidence_class="synthetic"),
            ct=contract(required_evidence="physical"),
            val=validation_manifest(evidence_class="synthetic"),
        )
        self.assertEqual(result["decision"], "wrong_evidence_class")

    def test_small_independent_sample_is_inconclusive(self):
        result = assess(
            doc=paired_manifest(unit_count=10),
            ct=contract(minimum_units=30),
        )
        self.assertEqual(result["decision"], "inconclusive")
        self.assertTrue(any(check["check"] == "minimum_independent_units" for check in result["inconclusive_checks"]))

    def test_incomplete_or_over_budget_runs_fail_closed(self):
        self.assertEqual(assess(doc=paired_manifest(incomplete=True))["decision"], "not_promoted")
        self.assertEqual(assess(doc=paired_manifest(over_budget=True))["decision"], "not_promoted")

    def test_risk_upper_bound_rejects_too_many_false_feasible_outcomes(self):
        result = assess(val=validation_manifest(count=100, false_count=12))
        self.assertEqual(result["decision"], "not_promoted")
        risk = next(check for check in result["checks"] if check["check"] == "false_feasible_risk")
        self.assertFalse(risk["passed"])

    def test_no_predicted_feasible_validation_trials_is_inconclusive(self):
        result = assess(val=validation_manifest(count=100, predicted_count=0))
        self.assertEqual(result["decision"], "inconclusive")
        self.assertIsNone(result["false_feasible_one_sided_upper_bound"])

    def test_rejects_validation_policy_source_or_config_mismatch(self):
        val = validation_manifest()
        val["policy_source_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "source digest"):
            assess(val=val)
        val = validation_manifest()
        val["policy_config_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "configuration digest"):
            assess(val=val)

    def test_rejects_wrong_protocol_or_non_held_out_split(self):
        val = validation_manifest()
        val["split"] = "training"
        with self.assertRaisesRegex(ValueError, "independent_held_out"):
            assess(val=val)
        val = validation_manifest()
        val["validation_protocol_sha256"] = "9" * 64
        with self.assertRaisesRegex(ValueError, "protocol digest"):
            assess(val=val)

    def test_missing_critical_family_is_inconclusive(self):
        doc = paired_manifest()
        for run in doc["runs"]:
            run["task_family"] = "alloy_b"
        receipt = scored_receipt(doc)
        doc_raw = canonical_json(doc)
        receipt = scored_receipt(doc, doc_raw)
        result = assess_promotion(
            receipt, doc, contract(), validation_manifest(),
            paired_receipt_file_sha256=_sha("paired"),
            paired_run_manifest_sha256=hashlib.sha256(doc_raw).hexdigest(),
            contract_manifest_sha256=_sha("contract"),
            validation_manifest_sha256=_sha("validation"), evaluator_sha256=_sha("source"),
        )
        self.assertEqual(result["decision"], "inconclusive")

    def test_duplicate_validation_units_and_invalid_digests_are_rejected(self):
        val = validation_manifest()
        val["records"][1]["unit_id"] = val["records"][0]["unit_id"]
        with self.assertRaisesRegex(ValueError, "must be unique"):
            assess(val=val)
        val = validation_manifest()
        val["records"][0]["outcome_receipt_sha256"] = "not-a-digest"
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            assess(val=val)

    def test_contract_validation_rejects_duplicate_families_and_invalid_targets(self):
        ct = contract()
        ct["critical_task_families"].append("alloy_a")
        with self.assertRaisesRegex(ValueError, "must be unique"):
            assess(ct=ct)
        ct = contract()
        ct["minimum_all_unit_win_rate"] = 1.2
        with self.assertRaisesRegex(ValueError, "must be in"):
            assess(ct=ct)

    def test_cli_binds_input_manifest_bytes_and_evaluator(self):
        doc = paired_manifest()
        val = validation_manifest()
        ct = contract()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path, run_path, contract_path, validation_path, output_path = [
                root / name for name in (
                    "paired.json", "runs.json", "contract.json", "validation.json", "receipt.json"
                )
            ]
            run_bytes = json.dumps(doc, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
            run_path.write_bytes(run_bytes)
            scored = scored_receipt(doc, run_bytes)
            paired_path.write_text(json.dumps(scored, separators=(",", ":")), encoding="utf-8")
            contract_path.write_text(json.dumps(ct, separators=(",", ":")), encoding="utf-8")
            validation_path.write_text(json.dumps(val, separators=(",", ":")), encoding="utf-8")
            raw_hashes = [
                hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (paired_path, run_path, contract_path, validation_path)
            ]
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                code = main([
                    str(paired_path), str(run_path), str(contract_path), str(validation_path),
                    "--output", str(output_path),
                ])
            self.assertEqual(code, 0)
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(result["decision"], "ready_for_next_stage")
            self.assertEqual(result["source_artifacts"]["paired_receipt_file_sha256"], raw_hashes[0])
            self.assertEqual(result["source_artifacts"]["paired_run_manifest_sha256"], raw_hashes[1])
            self.assertEqual(result["source_artifacts"]["promotion_contract_sha256"], raw_hashes[2])
            self.assertEqual(result["source_artifacts"]["validation_manifest_sha256"], raw_hashes[3])
            self.assertEqual(len(result["evaluator_sha256"]), 64)

    def test_cli_rejects_duplicate_json_keys_and_output_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paired_path, run_path, contract_path, validation_path = [
                root / name for name in ("paired.json", "runs.json", "contract.json", "validation.json")
            ]
            paired_path.write_text('{"schema_version":"1.0.0","schema_version":"1.0.0"}', encoding="utf-8")
            run_path.write_text(json.dumps(paired_manifest()), encoding="utf-8")
            contract_path.write_text(json.dumps(contract()), encoding="utf-8")
            validation_path.write_text(json.dumps(validation_manifest()), encoding="utf-8")
            paths = [paired_path, run_path, contract_path, validation_path]
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                rc = main([*(str(p) for p in paths), "--output", str(root / "out.json")])
            self.assertEqual(rc, 2)
            self.assertIn("duplicate JSON key", stderr.getvalue())
            raw_runs = json.dumps(paired_manifest(), separators=(",", ":")).encode()
            run_path.write_bytes(raw_runs)
            paired_path.write_text(
                json.dumps(scored_receipt(paired_manifest(), raw_runs), separators=(",", ":")),
                encoding="utf-8",
            )
            original = paired_path.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                rc = main([*(str(p) for p in paths), "--output", str(paired_path)])
            self.assertEqual(rc, 2)
            self.assertEqual(paired_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
