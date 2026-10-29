#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from promotion_risk_gate import clopper_pearson_upper_bound
from task_conformal_feasibility_gate import evaluate as screen_evaluate
from task_conformal_validation_audit import (
    AUDIT_LABEL,
    evaluate as audit_evaluate,
    main,
)

SOURCE = "a" * 64
POLICY = "b" * 64


def pool_points():
    return [
        {"candidate_id": "candidate-000", "design": [0.0, 0.5]},
        {"candidate_id": "candidate-001", "design": [1.0, 0.5]},
    ]


def pool_digest():
    raw = json.dumps(pool_points(), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def make_case(counts=None, *, false_a=0, false_b=0, no_recommendations=False,
              minimum_validation_tasks=60):
    if counts is None:
        counts = {"alloy_a": 60}
    strata = list(counts)
    config = {
        "schema_version": "1.0.0",
        "config_id": "conformal-screen-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "candidate_pool_sha256": pool_digest(),
        "alpha": 0.05,
        "minimum_calibration_tasks_per_stratum": 19,
        "maximum_descriptor_distance": 0.20,
        "frozen_before_outcome_unseal": True,
    }
    calibration_tasks = []
    for stratum in strata:
        for i in range(20):
            score = i / 100.0
            calibration_tasks.append({
                "task_id": f"cal-{stratum}-{i:03d}",
                "task_family": "calibration-family",
                "stratum": stratum,
                "descriptor": [0.5 + (i % 5) / 100, 0.5],
                "rows": [
                    {"candidate_id": "candidate-000", "design": [0.0, 0.5],
                     "predicted_margin": score, "observed_margin": 0.0},
                    {"candidate_id": "candidate-001", "design": [1.0, 0.5],
                     "predicted_margin": score, "observed_margin": 0.0},
                ],
            })
    calibration = {
        "schema_version": "1.0.0",
        "calibration_id": "calibration-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "split": "independent_calibration",
        "policy_frozen_before_outcomes": True,
        "candidate_pool_frozen_before_outcomes": True,
        "outcomes_unsealed_after_freeze": True,
        "independence_unit": "task_id",
        "independence_note": "One separately generated task per unit; candidate points are nested.",
        "tasks": calibration_tasks,
    }
    target_tasks = []
    validation_tasks = []
    serial = 0
    for stratum, count in counts.items():
        for j in range(count):
            task_id = f"validation-task-{serial:04d}"
            serial += 1
            target_tasks.append({
                "task_id": task_id,
                "task_family": f"heldout-{stratum}",
                "stratum": stratum,
                "descriptor": [0.5, 0.5],
                "predictions": [
                    {"candidate_id": "candidate-000", "design": [0.0, 0.5],
                     "predicted_margin": 0.18 if no_recommendations else 0.25},
                    {"candidate_id": "candidate-001", "design": [1.0, 0.5],
                     "predicted_margin": 0.18},
                ],
            })
            failure = (stratum == "alloy_a" and j < false_a) or (stratum == "alloy_b" and j < false_b)
            validation_tasks.append({
                "task_id": task_id,
                "stratum": stratum,
                "outcomes": [
                    {"candidate_id": "candidate-000", "observed_margin": -0.1 if failure else 0.1},
                    {"candidate_id": "candidate-001", "observed_margin": 0.1},
                ],
            })
    target_manifest = {
        "schema_version": "1.0.0",
        "target_manifest_id": "independent-validation-predictions-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "split": "independent_held_out_predictions",
        "candidate_pool_frozen_before_outcomes": True,
        "tasks": target_tasks,
    }
    config_bytes = encode(config)
    calibration_bytes = encode(calibration)
    target_bytes = encode(target_manifest)
    target_hash = hashlib.sha256(target_bytes).hexdigest()
    decision = screen_evaluate(config, calibration, target_manifest)
    decision["source_artifacts"] = {
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "calibration_manifest_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "target_prediction_manifest_sha256": target_hash,
        "evaluator_sha256": "e" * 64,
    }
    decision_bytes = encode(decision)
    decision_hash = hashlib.sha256(decision_bytes).hexdigest()
    contract = {
        "schema_version": "1.0.0",
        "audit_id": "false-feasible-audit-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "decision_receipt_sha256": decision_hash,
        "target_prediction_manifest_sha256": target_hash,
        "risk_target": 0.05,
        "confidence": 0.95,
        "minimum_validation_tasks_per_critical_stratum": minimum_validation_tasks,
        "minimum_recommended_tasks_per_critical_stratum": 59,
        "critical_strata": strata,
        "frozen_before_outcomes": True,
    }
    contract_bytes = encode(contract)
    validation = {
        "schema_version": "1.0.0",
        "validation_id": "validation-outcomes-v1",
        "evidence_class": "synthetic",
        "policy_source_sha256": SOURCE,
        "policy_config_sha256": POLICY,
        "split": "independent_held_out_validation",
        "policy_frozen_before_outcomes": True,
        "outcomes_unsealed_after_decisions": True,
        "independence_unit": "task_id",
        "independence_note": "Each task is a separately generated validation unit.",
        "audit_contract_sha256": hashlib.sha256(contract_bytes).hexdigest(),
        "decision_receipt_sha256": decision_hash,
        "target_prediction_manifest_sha256": target_hash,
        "tasks": validation_tasks,
    }
    result = audit_evaluate(
        contract, decision, target_manifest, validation,
        contract_manifest_sha256=hashlib.sha256(contract_bytes).hexdigest(),
        contract_manifest_sha256=hashlib.sha256(case["bytes"]["contract"]).hexdigest(),

        decision_receipt_sha256=decision_hash,
        target_prediction_manifest_sha256=target_hash,
    )
    return {
        "config": config, "calibration": calibration, "target": target_manifest,
        "decision": decision, "validation": validation, "contract": contract,
        "result": result, "bytes": {
            "config": config_bytes, "calibration": calibration_bytes,
            "target": target_bytes, "decision": decision_bytes,
            "validation": encode(validation), "contract": contract_bytes,
        },
    }


class TaskConformalValidationAuditTests(unittest.TestCase):
    def test_zero_false_feasible_tasks_pass_only_with_adequate_independent_sample(self):
        case = make_case()
        result = case["result"]
        self.assertEqual(result["receipt_label"], AUDIT_LABEL)
        self.assertEqual(result["decision"], "risk_gate_passed_not_physical_safety_certification")
        stratum = result["critical_strata"]["alloy_a"]
        self.assertEqual(stratum["validation_task_count"], 60)
        self.assertEqual(stratum["tasks_with_recommendations"], 60)
        self.assertEqual(stratum["tasks_with_any_false_feasible_recommendation"], 0)
        self.assertLessEqual(stratum["one_sided_upper_confidence_bound"], 0.05)

    def test_stratified_gate_catches_a_bad_family_hidden_by_aggregate_rate(self):
        case = make_case({"alloy_a": 300, "alloy_b": 300}, false_a=20)
        result = case["result"]
        pooled_upper = clopper_pearson_upper_bound(20, 600, 0.95)
        self.assertLessEqual(pooled_upper, 0.05)
        self.assertEqual(result["decision"], "not_promoted")
        self.assertEqual(result["critical_strata"]["alloy_a"]["decision"], "fail")
        self.assertGreater(
            result["critical_strata"]["alloy_a"]["one_sided_upper_confidence_bound"], 0.05
        )
        self.assertEqual(result["critical_strata"]["alloy_b"]["decision"], "pass")

    def test_known_risk_failure_is_not_masked_by_small_total_sample(self):
        case = make_case({"alloy_a": 60}, false_a=8, minimum_validation_tasks=100)
        stratum = case["result"]["critical_strata"]["alloy_a"]
        self.assertEqual(stratum["decision"], "fail")
        self.assertIn("insufficient_independent_validation_tasks", stratum["inconclusive_reasons"])
        self.assertEqual(case["result"]["decision"], "not_promoted")

    def test_missing_recommendations_are_inconclusive_not_safe(self):
        case = make_case({"alloy_a": 60}, no_recommendations=True)
        stratum = case["result"]["critical_strata"]["alloy_a"]
        self.assertEqual(stratum["decision"], "inconclusive")
        self.assertIsNone(stratum["one_sided_upper_confidence_bound"])
        self.assertEqual(stratum["tasks_with_recommendations"], 0)
        self.assertEqual(case["result"]["decision"], "inconclusive")

    def test_underpowered_validation_cannot_be_promoted_even_when_zero_failure_bound_is_low(self):
        case = make_case({"alloy_a": 59}, minimum_validation_tasks=60)
        result = case["result"]
        stratum = result["critical_strata"]["alloy_a"]
        self.assertLessEqual(stratum["one_sided_upper_confidence_bound"], 0.05)
        self.assertEqual(stratum["decision"], "inconclusive")
        self.assertEqual(result["decision"], "inconclusive")

    def test_candidate_rows_do_not_inflate_independent_task_denominator(self):
        case = make_case({"alloy_a": 60})
        result = case["result"]
        self.assertEqual(result["independent_validation_task_count"], 60)
        self.assertEqual(result["recommended_task_denominator"], 60)
        self.assertEqual(result["recommended_candidate_count"], 60)
        self.assertEqual(result["false_feasible_candidate_count"], 0)

    def test_rejects_tampered_receipt_or_target_manifest_digest(self):
        case = make_case()
        with self.assertRaisesRegex(ValueError, "does not match audit contract"):
            audit_evaluate(
                case["contract"], case["decision"], case["target"], case["validation"],
                contract_manifest_sha256=hashlib.sha256(case["bytes"]["contract"]).hexdigest(),

                decision_receipt_sha256="f" * 64,
                target_prediction_manifest_sha256=hashlib.sha256(case["bytes"]["target"]).hexdigest(),
            )
        with self.assertRaisesRegex(ValueError, "target prediction raw-byte digest"):
            audit_evaluate(
                case["contract"], case["decision"], case["target"], case["validation"],
                contract_manifest_sha256=hashlib.sha256(case["bytes"]["contract"]).hexdigest(),

                decision_receipt_sha256=case["contract"]["decision_receipt_sha256"],
                target_prediction_manifest_sha256="f" * 64,
            )

    def test_validation_candidate_roster_must_match_decision(self):
        case = make_case()
        case["validation"]["tasks"][0]["outcomes"].pop()
        with self.assertRaisesRegex(ValueError, "candidate roster mismatch"):
            audit_evaluate(
                case["contract"], case["decision"], case["target"], case["validation"],
                contract_manifest_sha256=hashlib.sha256(case["bytes"]["contract"]).hexdigest(),

                decision_receipt_sha256=case["contract"]["decision_receipt_sha256"],
                target_prediction_manifest_sha256=case["contract"]["target_prediction_manifest_sha256"],
            )

    def test_unknown_critical_stratum_is_rejected(self):
        case = make_case()
        case["validation"]["tasks"][0]["stratum"] = "undeclared"
        with self.assertRaisesRegex(ValueError, "undeclared critical stratum"):
            audit_evaluate(
                case["contract"], case["decision"], case["target"], case["validation"],
                contract_manifest_sha256=hashlib.sha256(case["bytes"]["contract"]).hexdigest(),

                decision_receipt_sha256=case["contract"]["decision_receipt_sha256"],
                target_prediction_manifest_sha256=case["contract"]["target_prediction_manifest_sha256"],
            )

    def test_contract_requires_the_zero_failure_sample_floor(self):
        case = make_case()
        case["contract"]["minimum_recommended_tasks_per_critical_stratum"] = 58
        case["validation"]["audit_contract_sha256"] = hashlib.sha256(encode(case["contract"])).hexdigest()
        with self.assertRaisesRegex(ValueError, "planning floor 59"):
            audit_evaluate(
                case["contract"], case["decision"], case["target"], case["validation"],
                contract_manifest_sha256=hashlib.sha256(case["bytes"]["contract"]).hexdigest(),

                decision_receipt_sha256=case["contract"]["decision_receipt_sha256"],
                target_prediction_manifest_sha256=case["contract"]["target_prediction_manifest_sha256"],
            )

    def test_cli_emits_digest_bound_audit_and_refuses_overwrite(self):
        case = make_case()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = {name: root / f"{name}.json" for name in case["bytes"]}
            for name, raw in case["bytes"].items():
                paths[name].write_bytes(raw)
            output = root / "audit-receipt.json"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                rc = main([
                    str(paths["contract"]), str(paths["decision"]), str(paths["target"]),
                    str(paths["validation"]), "--output", str(output),
                ])
            self.assertEqual(rc, 0)
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["decision"], "risk_gate_passed_not_physical_safety_certification")
            self.assertEqual(
                receipt["source_artifacts"]["decision_receipt_sha256"],
                hashlib.sha256(paths["decision"].read_bytes()).hexdigest(),
            )
            self.assertEqual(
                receipt["source_artifacts"]["target_prediction_manifest_sha256"],
                hashlib.sha256(paths["target"].read_bytes()).hexdigest(),
            )
            original = output.read_bytes()
            with contextlib.redirect_stderr(io.StringIO()):
                rc = main([
                    str(paths["contract"]), str(paths["decision"]), str(paths["target"]),
                    str(paths["validation"]), "--output", str(output),
                ])
            self.assertEqual(rc, 2)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
