#!/usr/bin/env python3
"""Independent post-decision audit for task-level conformal feasibility recommendations.

This scorer consumes target outcomes only after the decision receipt and all input
digests have been frozen. It counts any false-feasible recommendation per task as
one failure; multiple candidates in one casting/task do not inflate the independent
validation denominator.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from promotion_risk_gate import clopper_pearson_upper_bound

SCHEMA_VERSION = "1.0.0"
SCREEN_LABEL = "TASK_LEVEL_SPLIT_CONFORMAL_FEASIBILITY_SCREEN_NOT_PHYSICAL_SAFETY_CERTIFICATE"
AUDIT_LABEL = "INDEPENDENT_TASK_FALSE_FEASIBLE_AUDIT_NOT_PHYSICAL_SAFETY_CERTIFICATION"
EVIDENCE_CLASSES = {"synthetic", "public_analytic", "simulator_backed", "physical"}
CONTRACT_FIELDS = frozenset({
    "schema_version", "audit_id", "evidence_class", "policy_source_sha256",
    "policy_config_sha256", "decision_receipt_sha256", "target_prediction_manifest_sha256",
    "risk_target", "confidence", "minimum_validation_tasks_per_critical_stratum",
    "minimum_recommended_tasks_per_critical_stratum", "critical_strata",
    "frozen_before_outcomes",
})
VALIDATION_FIELDS = frozenset({
    "schema_version", "validation_id", "evidence_class", "policy_source_sha256",
    "policy_config_sha256", "split", "policy_frozen_before_outcomes",
    "outcomes_unsealed_after_decisions", "independence_unit", "independence_note",
    "audit_contract_sha256", "decision_receipt_sha256", "target_prediction_manifest_sha256", "tasks",
})
VALIDATION_TASK_FIELDS = frozenset({"task_id", "stratum", "outcomes"})
OUTCOME_FIELDS = frozenset({"candidate_id", "observed_margin"})


def _strict_object(value: Any, expected: frozenset[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise ValueError(
            f"{label} keys mismatch; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    return value


def _nonblank(value: object, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty trimmed text")
    return value


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must be a 64-character SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must be hexadecimal") from exc
    return value.lower()


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be an integer >= 1")
    return value


def validate_contract(raw: Any) -> dict[str, Any]:
    contract = _strict_object(raw, CONTRACT_FIELDS, "audit contract")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"audit contract schema_version must be {SCHEMA_VERSION}")
    _nonblank(contract["audit_id"], "audit_id")
    if contract["evidence_class"] not in EVIDENCE_CLASSES:
        raise ValueError("audit contract evidence_class is unsupported")
    for field in (
        "policy_source_sha256", "policy_config_sha256", "decision_receipt_sha256",
        "target_prediction_manifest_sha256",
    ):
        _sha256(contract[field], field)
    risk_target = _finite(contract["risk_target"], "risk_target")
    confidence = _finite(contract["confidence"], "confidence")
    if not 0.0 < risk_target < 1.0:
        raise ValueError("risk_target must be in (0, 1)")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    min_units = _positive_integer(
        contract["minimum_validation_tasks_per_critical_stratum"],
        "minimum_validation_tasks_per_critical_stratum",
    )
    min_recommended = _positive_integer(
        contract["minimum_recommended_tasks_per_critical_stratum"],
        "minimum_recommended_tasks_per_critical_stratum",
    )
    zero_failure_floor = math.ceil(math.log(1.0 - confidence) / math.log(1.0 - risk_target))
    if min_recommended < zero_failure_floor:
        raise ValueError(
            "minimum_recommended_tasks_per_critical_stratum is below the zero-failure "
            f"planning floor {zero_failure_floor} for the configured confidence/risk target"
        )
    strata = contract["critical_strata"]
    if not isinstance(strata, list) or not strata:
        raise ValueError("critical_strata must be a non-empty array")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in strata):
        raise ValueError("critical_strata must contain non-empty trimmed strings")
    if len(set(strata)) != len(strata):
        raise ValueError("critical_strata must be unique")
    if contract["frozen_before_outcomes"] is not True:
        raise ValueError("audit contract must be frozen before validation outcomes")
    return {
        **contract, "risk_target": risk_target, "confidence": confidence,
        "minimum_validation_tasks_per_critical_stratum": min_units,
        "minimum_recommended_tasks_per_critical_stratum": min_recommended,
    }


def _check_receipt_bindings(
    contract: dict[str, Any],
    decision: Any,
    target_manifest: Any,
    validation: Any,
    *,
    contract_manifest_sha256: str,
    decision_receipt_sha256: str,
    target_prediction_manifest_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    contract_manifest_sha256 = _sha256(contract_manifest_sha256, "contract_manifest_sha256")
    if not isinstance(decision, dict) or decision.get("receipt_label") != SCREEN_LABEL:
        raise ValueError("decision receipt label is not the task-conformal feasibility screen")
    if decision.get("evidence_class") != contract["evidence_class"]:
        raise ValueError("decision receipt evidence class does not match audit contract")
    if decision.get("policy_source_sha256") != contract["policy_source_sha256"].lower():
        raise ValueError("decision receipt policy source digest mismatch")
    if decision.get("policy_config_sha256") != contract["policy_config_sha256"].lower():
        raise ValueError("decision receipt policy config digest mismatch")
    if decision_receipt_sha256 != contract["decision_receipt_sha256"].lower():
        raise ValueError("decision receipt raw-byte digest does not match audit contract")
    if target_prediction_manifest_sha256 != contract["target_prediction_manifest_sha256"].lower():
        raise ValueError("target prediction raw-byte digest does not match audit contract")
    sources = decision.get("source_artifacts")
    if not isinstance(sources, dict) or sources.get("target_prediction_manifest_sha256") != target_prediction_manifest_sha256:
        raise ValueError("decision receipt does not bind the supplied raw target prediction manifest")

    if not isinstance(target_manifest, dict):
        raise ValueError("target prediction manifest must be an object")
    if target_manifest.get("target_manifest_id") != decision.get("target_manifest_id"):
        raise ValueError("target manifest ID does not match decision receipt")
    if target_manifest.get("evidence_class") != contract["evidence_class"]:
        raise ValueError("target manifest evidence class mismatch")
    target_source = _sha256(target_manifest.get("policy_source_sha256"), "target policy_source_sha256")
    target_config = _sha256(target_manifest.get("policy_config_sha256"), "target policy_config_sha256")
    if target_source != contract["policy_source_sha256"].lower():
        raise ValueError("target manifest source digest mismatch")
    if target_config != contract["policy_config_sha256"].lower():
        raise ValueError("target manifest config digest mismatch")

    if not isinstance(validation, dict):
        raise ValueError("validation outcome manifest must be an object")
    _strict_object(validation, VALIDATION_FIELDS, "validation outcome manifest")
    if validation["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"validation outcome schema_version must be {SCHEMA_VERSION}")
    _nonblank(validation["validation_id"], "validation_id")
    if validation["evidence_class"] != contract["evidence_class"]:
        raise ValueError("validation outcome evidence class mismatch")
    for field in ("policy_source_sha256", "policy_config_sha256"):
        if _sha256(validation[field], f"validation.{field}") != contract[field].lower():
            raise ValueError(f"validation {field} mismatch")
    if validation["split"] != "independent_held_out_validation":
        raise ValueError("validation split must be independent_held_out_validation")
    if validation["policy_frozen_before_outcomes"] is not True:
        raise ValueError("validation policy must be frozen before outcomes")
    if validation["outcomes_unsealed_after_decisions"] is not True:
        raise ValueError("validation outcomes must be unsealed after the decisions were frozen")
    if validation["independence_unit"] != "task_id":
        raise ValueError("validation independence_unit must be task_id")
    _nonblank(validation["independence_note"], "validation.independence_note")
    if _sha256(validation["audit_contract_sha256"], "validation.audit_contract_sha256") != contract_manifest_sha256.lower():
        raise ValueError("validation outcomes are not bound to this exact frozen audit contract")
    if _sha256(validation["decision_receipt_sha256"], "validation.decision_receipt_sha256") != decision_receipt_sha256:
        raise ValueError("validation outcomes are not bound to this exact decision receipt")
    if _sha256(validation["target_prediction_manifest_sha256"], "validation.target_prediction_manifest_sha256") != target_prediction_manifest_sha256:
        raise ValueError("validation outcomes are not bound to this exact target prediction manifest")

    decisions = decision.get("target_task_results")
    target_tasks = target_manifest.get("tasks")
    validation_tasks = validation["tasks"]
    if not isinstance(decisions, list) or not isinstance(target_tasks, list) or not isinstance(validation_tasks, list):
        raise ValueError("decision, target, and validation task arrays are required")
    decision_ids = [task.get("task_id") for task in decisions if isinstance(task, dict)]
    target_ids = [task.get("task_id") for task in target_tasks if isinstance(task, dict)]
    validation_ids = [task.get("task_id") for task in validation_tasks if isinstance(task, dict)]
    if not decision_ids or len(decision_ids) != len(target_tasks) or len(target_tasks) != len(validation_tasks):
        raise ValueError("decision, target, and outcome task counts must be equal and non-zero")
    if len(set(decision_ids)) != len(decision_ids) or len(set(target_ids)) != len(target_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("task IDs must be unique in every manifest")
    if decision_ids != target_ids or decision_ids != validation_ids:
        raise ValueError("decision, target, and validation task IDs/order do not match")

    # Verify the decision receipt actually reports the target predictions it
    # claims to screen; a digest binds bytes but does not make their semantics true.
    for index, (decision_task, target_task) in enumerate(zip(decisions, target_tasks)):
        if decision_task.get("stratum") != target_task.get("stratum"):
            raise ValueError(f"decision/target stratum mismatch at task index {index}")
        if decision_task.get("task_family") != target_task.get("task_family"):
            raise ValueError(f"decision/target task family mismatch at task index {index}")
        selected = decision_task.get("candidates")
        predictions = target_task.get("predictions")
        if not isinstance(selected, list) or not isinstance(predictions, list) or len(selected) != len(predictions):
            raise ValueError(f"decision/target candidate count mismatch at task index {index}")
        for candidate_index, (decision_candidate, prediction) in enumerate(zip(selected, predictions)):
            if not isinstance(decision_candidate, dict) or not isinstance(prediction, dict):
                raise ValueError(f"decision/target candidate must be an object at task index {index}")
            if decision_candidate.get("candidate_id") != prediction.get("candidate_id"):
                raise ValueError(f"decision/target candidate ID mismatch at task index {index}")
            decision_margin = _finite(decision_candidate.get("predicted_margin"), "decision predicted_margin")
            target_margin = _finite(prediction.get("predicted_margin"), "target predicted_margin")
            if decision_margin != target_margin:
                raise ValueError(f"decision/target predicted margin mismatch at task index {index}")
            if decision_candidate.get("action") not in {"recommend_feasible", "abstain"}:
                raise ValueError(f"unrecognized decision action at task index {index}")
    return decisions, validation_tasks


def evaluate(
    contract_raw: Any,
    decision: Any,
    target_manifest: Any,
    validation_raw: Any,
    *,
    contract_manifest_sha256: str,
    decision_receipt_sha256: str,
    target_prediction_manifest_sha256: str,
) -> dict[str, Any]:
    contract = validate_contract(contract_raw)
    validation = validation_raw
    decisions, validation_tasks = _check_receipt_bindings(
        contract, decision, target_manifest, validation,
        contract_manifest_sha256=contract_manifest_sha256,
        decision_receipt_sha256=decision_receipt_sha256,
        target_prediction_manifest_sha256=target_prediction_manifest_sha256,
    )
    decision_by_id = {task["task_id"]: task for task in decisions}
    validation_by_id: dict[str, dict[str, Any]] = {}
    seen_strata: set[str] = set()
    for ti, raw_task in enumerate(validation_tasks):
        task = _strict_object(raw_task, VALIDATION_TASK_FIELDS, f"validation.tasks[{ti}]")
        task_id = _nonblank(task["task_id"], f"validation.tasks[{ti}].task_id")
        stratum = _nonblank(task["stratum"], f"validation.tasks[{ti}].stratum")
        if stratum not in contract["critical_strata"]:
            raise ValueError(f"validation contains undeclared critical stratum: {stratum}")
        seen_strata.add(stratum)
        rows = task["outcomes"]
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"validation.tasks[{ti}].outcomes must be non-empty")
        row_ids: set[str] = set()
        for ri, raw_row in enumerate(rows):
            row = _strict_object(raw_row, OUTCOME_FIELDS, f"validation.tasks[{ti}].outcomes[{ri}]")
            cid = _nonblank(row["candidate_id"], f"validation.tasks[{ti}].outcomes[{ri}].candidate_id")
            if cid in row_ids:
                raise ValueError(f"duplicate outcome candidate_id in task {task_id}: {cid}")
            row_ids.add(cid)
            _finite(row["observed_margin"], f"validation.tasks[{ti}].outcomes[{ri}].observed_margin")
        selected = decision_by_id[task_id]
        if selected.get("stratum") != stratum:
            raise ValueError(f"validation stratum does not match decision for task {task_id}")
        decision_candidates = selected.get("candidates")
        if not isinstance(decision_candidates, list):
            raise ValueError(f"decision candidates missing for task {task_id}")
        decisions_by_candidate = {row.get("candidate_id"): row for row in decision_candidates}
        if list(row_ids) == [] or set(decisions_by_candidate) != row_ids:
            raise ValueError(f"validation candidate roster mismatch for task {task_id}")
        # Preserve actual order separately: each validation task must cover every
        # candidate from the exact decision pool, once, and no other candidates.
        if len(decisions_by_candidate) != len(decision_candidates):
            raise ValueError(f"duplicate decision candidates in task {task_id}")
        outcomes_by_candidate = {row["candidate_id"]: row for row in rows}
        if len(outcomes_by_candidate) != len(rows):
            raise ValueError(f"duplicate outcome candidates in task {task_id}")
        false_candidates = 0
        recommended_candidates = 0
        for candidate in decision_candidates:
            cid = candidate.get("candidate_id")
            if candidate.get("action") == "recommend_feasible":
                recommended_candidates += 1
                if float(outcomes_by_candidate[cid]["observed_margin"]) < 0.0:
                    false_candidates += 1
        validation_by_id[task_id] = {
            "stratum": stratum,
            "task_id": task_id,
            "recommended_candidate_count": recommended_candidates,
            "false_feasible_candidate_count": false_candidates,
            "recommended_task": recommended_candidates > 0,
            "false_feasible_task": recommended_candidates > 0 and false_candidates > 0,
        }

    per_stratum: dict[str, dict[str, Any]] = {}
    hard_failure = False
    inconclusive = False
    for stratum in contract["critical_strata"]:
        tasks = [row for row in validation_by_id.values() if row["stratum"] == stratum]
        units = len(tasks)
        recommended_tasks = [row for row in tasks if row["recommended_task"]]
        failures = sum(row["false_feasible_task"] for row in recommended_tasks)
        n = len(recommended_tasks)
        upper = clopper_pearson_upper_bound(failures, n, contract["confidence"]) if n else None
        reasons = []
        if units < contract["minimum_validation_tasks_per_critical_stratum"]:
            reasons.append("insufficient_independent_validation_tasks")
        if n < contract["minimum_recommended_tasks_per_critical_stratum"]:
            reasons.append("insufficient_tasks_with_recommendations")
        if not n:
            decision_state = "inconclusive"
        elif upper is not None and upper > contract["risk_target"]:
            # A risk-bound failure is a known promotion failure. Do not mask it
            # with an unrelated small total-validation sample.
            decision_state = "fail"
        elif reasons:
            decision_state = "inconclusive"
        else:
            decision_state = "pass"
        if decision_state == "fail":
            hard_failure = True
        elif decision_state == "inconclusive":
            inconclusive = True
        per_stratum[stratum] = {
            "validation_task_count": units,
            "tasks_with_recommendations": n,
            "tasks_with_any_false_feasible_recommendation": failures,
            "observed_false_feasible_task_rate": failures / n if n else None,
            "one_sided_upper_confidence_bound": upper,
            "risk_target": contract["risk_target"],
            "confidence": contract["confidence"],
            "decision": decision_state,
            "inconclusive_reasons": reasons,
        }

    if hard_failure:
        overall = "not_promoted"
    elif inconclusive:
        overall = "inconclusive"
    else:
        overall = "risk_gate_passed_not_physical_safety_certification"

    return {
        "schema_version": SCHEMA_VERSION,
        "receipt_label": AUDIT_LABEL,
        "decision": overall,
        "audit_id": contract["audit_id"],
        "validation_id": validation["validation_id"],
        "evidence_class": contract["evidence_class"],
        "policy_source_sha256": contract["policy_source_sha256"].lower(),
        "policy_config_sha256": contract["policy_config_sha256"].lower(),
        "risk_target": contract["risk_target"],
        "confidence": contract["confidence"],
        "independent_validation_task_count": len(validation_by_id),
        "recommended_task_denominator": sum(row["recommended_task"] for row in validation_by_id.values()),
        "false_feasible_task_count": sum(row["false_feasible_task"] for row in validation_by_id.values()),
        "recommended_candidate_count": sum(row["recommended_candidate_count"] for row in validation_by_id.values()),
        "false_feasible_candidate_count": sum(row["false_feasible_candidate_count"] for row in validation_by_id.values()),
        "critical_strata": per_stratum,
        "claim_ceiling": (
            "Exact post-decision audit of a frozen candidate recommendation receipt on asserted independent "
            "held-out tasks. Per-stratum bounds rely on representative independent tasks and do not establish "
            "physical safety, process qualification, holdout secrecy, or performance under arbitrary distribution shift."
        ),
    }


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant forbidden: {value}")


def _load_json(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        return raw, json.loads(raw, object_pairs_hook=_reject_duplicate_keys, parse_constant=_reject_constant)
    except (ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON in {path}: {exc}") from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("contract", type=Path)
    parser.add_argument("decision_receipt", type=Path)
    parser.add_argument("target_predictions", type=Path)
    parser.add_argument("validation_outcomes", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        paths = [args.contract.resolve(), args.decision_receipt.resolve(),
                 args.target_predictions.resolve(), args.validation_outcomes.resolve()]
        if len(set(paths)) != len(paths):
            raise ValueError("all audit inputs must be different files")
        contract_bytes, contract = _load_json(args.contract)
        decision_bytes, decision = _load_json(args.decision_receipt)
        target_bytes, target_manifest = _load_json(args.target_predictions)
        validation_bytes, validation = _load_json(args.validation_outcomes)
        contract_hash = hashlib.sha256(contract_bytes).hexdigest()
        decision_hash = hashlib.sha256(decision_bytes).hexdigest()
        target_hash = hashlib.sha256(target_bytes).hexdigest()
        result = evaluate(
            contract, decision, target_manifest, validation,
            contract_manifest_sha256=contract_hash,
            decision_receipt_sha256=decision_hash,
            target_prediction_manifest_sha256=target_hash,
        )
        result["source_artifacts"] = {
            "audit_contract_sha256": contract_hash,
            "decision_receipt_sha256": decision_hash,
            "target_prediction_manifest_sha256": target_hash,
            "validation_outcome_manifest_sha256": hashlib.sha256(validation_bytes).hexdigest(),
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        output = args.output.resolve()
        if output in paths or output.exists():
            raise ValueError("output must be a new path and must not overwrite any input")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        print(f"task-conformal validation audit failed: {exc}", file=sys.stderr)
        return 2
    print(f"task-conformal validation audit written: {args.output}")
    print(f"decision={result['decision']} tasks={result['independent_validation_task_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
