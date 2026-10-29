#!/usr/bin/env python3
"""Fail-closed promotion readiness assessment for paired casting benchmarks.

This is a research-stage admission gate, not physical safety certification.
It recomputes false-feasible risk bounds from a supplied independent-validation
manifest, but cannot itself establish the truth of the holdout or independence claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

from paired_benchmark_audit import canonical_json
from promotion_risk_gate import clopper_pearson_upper_bound

SCHEMA_VERSION = "1.0.0"
CONTRACT_FIELDS = frozenset({
    "schema_version", "contract_id", "candidate_policy", "baseline_policy",
    "required_evidence_class", "benchmark_protocol_sha256", "validation_protocol_sha256",
    "minimum_independent_units", "minimum_units_per_critical_family",
    "minimum_validation_units", "minimum_cost_target_pairs", "critical_task_families",
    "minimum_median_relative_auc_improvement", "minimum_all_unit_win_rate",
    "maximum_critical_family_regression", "minimum_cost_to_target_reduction",
    "maximum_false_feasible_rate", "confidence", "require_complete_runs",
    "require_in_budget_runs",
})
VALIDATION_FIELDS = frozenset({
    "schema_version", "validation_id", "candidate_policy", "evidence_class",
    "policy_source_sha256", "policy_config_sha256", "validation_protocol_sha256",
    "environment_sha256", "split", "policy_frozen_before_unseal",
    "outcomes_unsealed_after_freeze", "independence_unit", "independence_note", "records",
})
RECORD_FIELDS = frozenset({
    "unit_id", "task_family", "shift_id", "predicted_feasible", "observed_feasible",
    "decision_receipt_sha256", "outcome_receipt_sha256",
})
ALLOWED_EVIDENCE = {"synthetic", "public_analytic", "simulator_backed", "physical"}
PAIRED_RECEIPT_LABEL = "PAIRED_MATCHED_COST_COMPARISON_NOT_PHYSICAL_CERTIFICATION"
RISK_RECEIPT_LABEL = "PROMOTION_READINESS_NOT_PHYSICAL_CERTIFICATION"


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
        raise ValueError(f"{label} must be a non-empty trimmed string")
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
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _integer(value: object, label: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")
    return value


def _probability(value: object, label: str, *, strictly_positive: bool = False) -> float:
    number = _finite(value, label)
    lower_ok = number > 0.0 if strictly_positive else number >= 0.0
    if not lower_ok or number >= 1.0:
        low = "(0" if strictly_positive else "[0"
        raise ValueError(f"{label} must be in {low}, 1)")
    return number


def _quantile_interval(effect: dict[str, Any], key: str, label: str) -> tuple[float, float]:
    interval = effect.get(key)
    if not isinstance(interval, list) or len(interval) != 2:
        raise ValueError(f"{label}.{key} must be a two-value interval")
    lower, upper = _finite(interval[0], f"{label}.{key}[0]"), _finite(interval[1], f"{label}.{key}[1]")
    if lower > upper:
        raise ValueError(f"{label}.{key} lower endpoint exceeds upper endpoint")
    return lower, upper


def validate_contract(raw: Any) -> dict[str, Any]:
    contract = _strict_object(raw, CONTRACT_FIELDS, "contract")
    if contract["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"contract schema_version must be {SCHEMA_VERSION}")
    _nonblank(contract["contract_id"], "contract_id")
    _nonblank(contract["candidate_policy"], "candidate_policy")
    _nonblank(contract["baseline_policy"], "baseline_policy")
    if contract["candidate_policy"] == contract["baseline_policy"]:
        raise ValueError("candidate_policy and baseline_policy must differ")
    if contract["required_evidence_class"] not in ALLOWED_EVIDENCE:
        raise ValueError("required_evidence_class is not a supported evidence class")
    _sha256(contract["benchmark_protocol_sha256"], "benchmark_protocol_sha256")
    _sha256(contract["validation_protocol_sha256"], "validation_protocol_sha256")
    for field in (
        "minimum_independent_units", "minimum_units_per_critical_family",
        "minimum_validation_units", "minimum_cost_target_pairs",
    ):
        _integer(contract[field], field)
    families = contract["critical_task_families"]
    if not isinstance(families, list) or not families:
        raise ValueError("critical_task_families must be a non-empty array")
    if any(not isinstance(item, str) or not item or item != item.strip() for item in families):
        raise ValueError("critical_task_families must contain non-empty trimmed strings")
    if len(set(families)) != len(families):
        raise ValueError("critical_task_families must be unique")
    _probability(contract["minimum_median_relative_auc_improvement"], "minimum_median_relative_auc_improvement")
    _probability(contract["minimum_all_unit_win_rate"], "minimum_all_unit_win_rate")
    _probability(contract["maximum_critical_family_regression"], "maximum_critical_family_regression")
    _probability(contract["minimum_cost_to_target_reduction"], "minimum_cost_to_target_reduction")
    _probability(contract["maximum_false_feasible_rate"], "maximum_false_feasible_rate", strictly_positive=True)
    _probability(contract["confidence"], "confidence", strictly_positive=True)
    if contract["confidence"] <= 0.5:
        raise ValueError("confidence must exceed 0.5")
    if not isinstance(contract["require_complete_runs"], bool):
        raise ValueError("require_complete_runs must be boolean")
    if not isinstance(contract["require_in_budget_runs"], bool):
        raise ValueError("require_in_budget_runs must be boolean")
    return contract


def validate_validation_manifest(raw: Any) -> dict[str, Any]:
    validation = _strict_object(raw, VALIDATION_FIELDS, "validation manifest")
    if validation["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"validation manifest schema_version must be {SCHEMA_VERSION}")
    _nonblank(validation["validation_id"], "validation_id")
    _nonblank(validation["candidate_policy"], "validation candidate_policy")
    if validation["evidence_class"] not in ALLOWED_EVIDENCE:
        raise ValueError("validation evidence_class is unsupported")
    for field in ("policy_source_sha256", "policy_config_sha256", "validation_protocol_sha256", "environment_sha256"):
        _sha256(validation[field], f"validation.{field}")
    if validation["split"] != "independent_held_out":
        raise ValueError("validation split must be independent_held_out")
    for field in ("policy_frozen_before_unseal", "outcomes_unsealed_after_freeze"):
        if validation[field] is not True:
            raise ValueError(f"validation.{field} must be true")
    _nonblank(validation["independence_unit"], "validation.independence_unit")
    if validation["independence_unit"] != "unit_id":
        raise ValueError("validation.independence_unit must be unit_id; nested outcomes cannot be counted as independent trials")
    _nonblank(validation["independence_note"], "validation.independence_note")
    records = validation["records"]
    if not isinstance(records, list) or not records:
        raise ValueError("validation records must be a non-empty array")
    seen_units: set[str] = set()
    seen_decision_digests: set[str] = set()
    seen_outcome_digests: set[str] = set()
    for index, raw_record in enumerate(records):
        record = _strict_object(raw_record, RECORD_FIELDS, f"validation.records[{index}]")
        unit_id = _nonblank(record["unit_id"], f"validation.records[{index}].unit_id")
        if unit_id in seen_units:
            raise ValueError(f"validation unit_id must be unique; duplicate {unit_id}")
        seen_units.add(unit_id)
        _nonblank(record["task_family"], f"validation.records[{index}].task_family")
        _nonblank(record["shift_id"], f"validation.records[{index}].shift_id")
        for field in ("predicted_feasible", "observed_feasible"):
            if not isinstance(record[field], bool):
                raise ValueError(f"validation.records[{index}].{field} must be boolean")
        decision_digest = _sha256(record["decision_receipt_sha256"], f"validation.records[{index}].decision_receipt_sha256")
        outcome_digest = _sha256(record["outcome_receipt_sha256"], f"validation.records[{index}].outcome_receipt_sha256")
        if decision_digest in seen_decision_digests:
            raise ValueError("validation decision receipt digest must be unique per unit")
        if outcome_digest in seen_outcome_digests:
            raise ValueError("validation outcome receipt digest must be unique per unit")
        seen_decision_digests.add(decision_digest)
        seen_outcome_digests.add(outcome_digest)
    return validation


def _check(label: str, passed: bool, detail: str, failures: list[dict[str, str]]) -> dict[str, Any]:
    if not passed:
        failures.append({"check": label, "detail": detail})
    return {"check": label, "passed": bool(passed), "detail": detail}


def assess_promotion(
    paired_receipt: Any,
    contract_raw: Any,
    validation_raw: Any,
    *,
    paired_manifest_sha256: str,
    contract_manifest_sha256: str,
    validation_manifest_sha256: str,
    evaluator_sha256: str,
) -> dict[str, Any]:
    contract = validate_contract(contract_raw)
    validation = validate_validation_manifest(validation_raw)
    for name, digest in (
        ("paired_manifest_sha256", paired_manifest_sha256),
        ("contract_manifest_sha256", contract_manifest_sha256),
        ("validation_manifest_sha256", validation_manifest_sha256),
        ("evaluator_sha256", evaluator_sha256),
    ):
        _sha256(digest, name)

    if not isinstance(paired_receipt, dict):
        raise ValueError("paired receipt must be an object")
    if paired_receipt.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("paired receipt schema_version mismatch")
    if paired_receipt.get("benchmark_id") != "CASTING-PAIRED-COST-001":
        raise ValueError("paired receipt benchmark_id mismatch")
    if paired_receipt.get("receipt_label") != PAIRED_RECEIPT_LABEL:
        raise ValueError("paired receipt label is invalid")
    paired_sources = paired_receipt.get("source_artifacts")
    if not isinstance(paired_sources, dict):
        raise ValueError("paired receipt source_artifacts missing")
    _sha256(paired_sources.get("raw_manifest_sha256"), "paired receipt raw_manifest_sha256")
    _sha256(paired_receipt.get("evaluator_sha256"), "paired receipt evaluator_sha256")
    if paired_receipt.get("baseline_policy") != contract["baseline_policy"]:
        raise ValueError("paired receipt baseline does not match contract")
    if paired_receipt.get("protocol_sha256") != contract["benchmark_protocol_sha256"]:
        raise ValueError("paired receipt protocol digest does not match frozen contract")
    candidate = contract["candidate_policy"]
    baseline = contract["baseline_policy"]
    policies = paired_receipt.get("policy_ids")
    if not isinstance(policies, list) or candidate not in policies or baseline not in policies:
        raise ValueError("paired receipt does not contain candidate and baseline policies")
    comparison = paired_receipt.get("comparisons_vs_baseline", {}).get(candidate)
    if not isinstance(comparison, dict) or comparison.get("baseline_policy") != baseline:
        raise ValueError("candidate comparison against the declared baseline is missing")
    if paired_receipt.get("evidence_class") not in ALLOWED_EVIDENCE:
        raise ValueError("paired receipt lacks a supported evidence_class")
    if validation["candidate_policy"] != candidate:
        raise ValueError("validation manifest candidate does not match contract")
    if validation["validation_protocol_sha256"] != contract["validation_protocol_sha256"]:
        raise ValueError("validation protocol digest does not match contract")

    status = "ready_for_next_stage"
    failures: list[dict[str, str]] = []
    inconclusive: list[dict[str, str]] = []
    checks: list[dict[str, Any]] = []

    paired_evidence = paired_receipt["evidence_class"]
    validation_evidence = validation["evidence_class"]
    required_evidence = contract["required_evidence_class"]
    if paired_evidence != required_evidence or validation_evidence != required_evidence:
        status = "wrong_evidence_class"
        checks.append({
            "check": "evidence_class",
            "passed": False,
            "detail": f"required={required_evidence}; paired={paired_evidence}; independent_validation={validation_evidence}",
        })
    else:
        checks.append({
            "check": "evidence_class", "passed": True,
            "detail": f"paired and validation evidence both match required {required_evidence}",
        })

    provenance = paired_receipt.get("policy_provenance", {}).get(candidate)
    if not isinstance(provenance, dict):
        raise ValueError("paired receipt lacks candidate policy provenance")
    if validation["policy_source_sha256"].lower() != provenance.get("policy_source_sha256"):
        raise ValueError("validation source digest does not match the paired candidate policy source")
    if validation["policy_config_sha256"].lower() != provenance.get("policy_config_sha256"):
        raise ValueError("validation configuration digest does not match the paired candidate configuration")

    per_policy = paired_receipt.get("per_policy", {})
    candidate_summary = per_policy.get(candidate)
    if not isinstance(candidate_summary, dict):
        raise ValueError("candidate per-policy summary missing")
    independent_count = paired_receipt.get("unit_count")
    if isinstance(independent_count, bool) or not isinstance(independent_count, int) or independent_count < 0:
        raise ValueError("paired receipt unit_count is invalid")
    comparisons = comparison.get("effects", {})

    checks.append(_check(
        "minimum_independent_units",
        independent_count >= contract["minimum_independent_units"],
        f"observed={independent_count}; required={contract['minimum_independent_units']}",
        inconclusive,
    ))
    checks.append(_check(
        "complete_runs",
        not contract["require_complete_runs"] or candidate_summary.get("incomplete_count") == 0,
        f"incomplete_count={candidate_summary.get('incomplete_count')}",
        failures,
    ))
    checks.append(_check(
        "in_budget_runs",
        not contract["require_in_budget_runs"] or candidate_summary.get("over_budget_count") == 0,
        f"over_budget_count={candidate_summary.get('over_budget_count')}",
        failures,
    ))

    relative_auc = comparisons.get("relative_auc_improvement")
    if not isinstance(relative_auc, dict):
        raise ValueError("relative AUC effect missing")
    rel_n = relative_auc.get("n_paired_units")
    rel_median = relative_auc.get("median_delta")
    if not isinstance(rel_n, int) or rel_n != independent_count or not isinstance(rel_median, (int, float)):
        checks.append(_check(
            "relative_auc_denominator_complete", False,
            f"relative_auc_pairs={rel_n}; independent_units={independent_count}",
            inconclusive,
        ))
    else:
        checks.append(_check(
            "minimum_median_relative_auc_improvement",
            float(rel_median) >= contract["minimum_median_relative_auc_improvement"],
            f"observed_median={float(rel_median):.8g}; required={contract['minimum_median_relative_auc_improvement']}",
            failures,
        ))
        _, relative_ci_high = _quantile_interval(relative_auc, "bootstrap_95pct_ci_median_delta", "relative_auc_improvement")
        relative_ci_low, _ = _quantile_interval(relative_auc, "bootstrap_95pct_ci_median_delta", "relative_auc_improvement")
        checks.append(_check(
            "relative_auc_effect_interval",
            relative_ci_low >= contract["minimum_median_relative_auc_improvement"],
            f"95% percentile bootstrap median-effect interval=[{relative_ci_low:.8g}, {relative_ci_high:.8g}]",
            failures,
        ))

    win = comparisons.get("cost_normalized_quality_auc_delta")
    if not isinstance(win, dict):
        raise ValueError("AUC win-rate effect missing")
    win_rate = win.get("win_rate_all_units")
    win_interval_low, win_interval_high = _quantile_interval(
        win, "bootstrap_95pct_ci_win_rate_all_units", "cost_normalized_quality_auc_delta"
    )
    checks.append(_check(
        "minimum_all_unit_win_rate",
        isinstance(win_rate, (int, float)) and float(win_rate) >= contract["minimum_all_unit_win_rate"],
        f"observed_all_unit_win_rate={win_rate}; required={contract['minimum_all_unit_win_rate']}; ties count as non-wins",
        failures,
    ))
    checks.append(_check(
        "win_rate_interval",
        win_interval_low >= contract["minimum_all_unit_win_rate"],
        f"bootstrap interval=[{win_interval_low:.8g}, {win_interval_high:.8g}]",
        failures,
    ))

    family_results = comparison.get("by_task_family", {})
    for family in contract["critical_task_families"]:
        family_result = family_results.get(family)
        if not isinstance(family_result, dict):
            inconclusive.append({"check": f"critical_family:{family}", "detail": "critical family absent from receipt"})
            checks.append({"check": f"critical_family:{family}", "passed": False, "detail": "critical family absent from receipt"})
            continue
        effect = family_result.get("relative_auc_improvement")
        if not isinstance(effect, dict) or effect.get("n_paired_units", 0) < contract["minimum_units_per_critical_family"] or family_result.get("relative_auc_excluded_zero_baseline_units", 0) > 0:
            detail = (
                f"relative_pairs={effect.get('n_paired_units') if isinstance(effect, dict) else None}; "
                f"family_units={family_result.get('unit_count')}; "
                f"zero_baseline_exclusions={family_result.get('relative_auc_excluded_zero_baseline_units')}"
            )
            inconclusive.append({"check": f"critical_family:{family}", "detail": detail})
            checks.append({"check": f"critical_family:{family}", "passed": False, "detail": detail})
            continue
        median_delta = effect.get("median_delta")
        ci_low, ci_high = _quantile_interval(effect, "bootstrap_95pct_ci_median_delta", f"critical_family:{family}")
        threshold = -contract["maximum_critical_family_regression"]
        family_pass = isinstance(median_delta, (int, float)) and float(median_delta) >= threshold and ci_low >= threshold
        detail = f"median_relative_delta={median_delta}; lower_95pct_bootstrap_bound={ci_low:.8g}; minimum={threshold}"
        checks.append({"check": f"critical_family:{family}", "passed": bool(family_pass), "detail": detail})
        if not family_pass:
            failures.append({"check": f"critical_family:{family}", "detail": detail})

    cost_effect = comparisons.get("cost_to_target_reduction_when_both_reach")
    if not isinstance(cost_effect, dict):
        raise ValueError("cost-to-target effect missing")
    cost_n = cost_effect.get("n_paired_units")
    if not isinstance(cost_n, int) or cost_n < contract["minimum_cost_target_pairs"]:
        detail = f"paired_target_reaches={cost_n}; required={contract['minimum_cost_target_pairs']}"
        inconclusive.append({"check": "cost_to_target_pairs", "detail": detail})
        checks.append({"check": "cost_to_target_pairs", "passed": False, "detail": detail})
    else:
        cost_median = cost_effect.get("median_delta")
        cost_ci_low, cost_ci_high = _quantile_interval(
            cost_effect, "bootstrap_95pct_ci_median_delta", "cost_to_target_reduction_when_both_reach"
        )
        pass_cost = (
            isinstance(cost_median, (int, float))
            and float(cost_median) >= contract["minimum_cost_to_target_reduction"]
            and cost_ci_low >= contract["minimum_cost_to_target_reduction"]
        )
        detail = f"median_reduction={cost_median}; lower_95pct_bootstrap_bound={cost_ci_low:.8g}; required={contract['minimum_cost_to_target_reduction']}"
        checks.append({"check": "minimum_cost_to_target_reduction", "passed": bool(pass_cost), "detail": detail})
        if not pass_cost:
            failures.append({"check": "minimum_cost_to_target_reduction", "detail": detail})

    records = validation["records"]
    validation_units = len(records)
    predicted_feasible = sum(record["predicted_feasible"] for record in records)
    false_feasible = sum(record["predicted_feasible"] and not record["observed_feasible"] for record in records)
    if validation_units < contract["minimum_validation_units"]:
        inconclusive.append({
            "check": "minimum_validation_units",
            "detail": f"observed={validation_units}; required={contract['minimum_validation_units']}",
        })
        checks.append({
            "check": "minimum_validation_units", "passed": False,
            "detail": f"observed={validation_units}; required={contract['minimum_validation_units']}",
        })
    else:
        checks.append({
            "check": "minimum_validation_units", "passed": True,
            "detail": f"observed={validation_units}; required={contract['minimum_validation_units']}",
        })

    if predicted_feasible == 0:
        risk_upper = None
        inconclusive.append({"check": "false_feasible_risk", "detail": "no predicted-feasible decisions in independent validation"})
        checks.append({
            "check": "false_feasible_risk", "passed": False,
            "detail": "no predicted-feasible decisions in independent validation",
        })
    else:
        risk_upper = clopper_pearson_upper_bound(false_feasible, predicted_feasible, contract["confidence"])
        risk_pass = risk_upper <= contract["maximum_false_feasible_rate"]
        detail = (
            f"false_feasible={false_feasible}; predicted_feasible_trials={predicted_feasible}; "
            f"one_sided_{contract['confidence']:.4g}_upper_bound={risk_upper:.8g}; "
            f"maximum={contract['maximum_false_feasible_rate']}"
        )
        checks.append({"check": "false_feasible_risk", "passed": bool(risk_pass), "detail": detail})
        if not risk_pass:
            failures.append({"check": "false_feasible_risk", "detail": detail})

    if status != "wrong_evidence_class":
        if inconclusive:
            status = "inconclusive"
        elif failures:
            status = "not_promoted"
        else:
            status = "ready_for_next_stage"

    return {
        "schema_version": SCHEMA_VERSION,
        "gate_id": contract["contract_id"],
        "receipt_label": RISK_RECEIPT_LABEL,
        "decision": status,
        "candidate_policy": candidate,
        "baseline_policy": baseline,
        "required_evidence_class": required_evidence,
        "observed_evidence_classes": {
            "paired_benchmark": paired_evidence,
            "independent_validation": validation_evidence,
        },
        "independent_task_units": independent_count,
        "validation_units": validation_units,
        "predicted_feasible_validation_trials": predicted_feasible,
        "false_feasible_validation_events": false_feasible,
        "false_feasible_one_sided_upper_bound": risk_upper,
        "maximum_false_feasible_rate": contract["maximum_false_feasible_rate"],
        "confidence": contract["confidence"],
        "checks": checks,
        "failed_checks": failures,
        "inconclusive_checks": inconclusive,
        "source_artifacts": {
            "paired_receipt_manifest_sha256": paired_manifest_sha256,
            "promotion_contract_sha256": contract_manifest_sha256,
            "validation_manifest_sha256": validation_manifest_sha256,
        },
        "evaluator_sha256": evaluator_sha256,
        "claim_ceiling": (
            "Promotion-readiness decision from supplied, digest-bound receipts and an asserted "
            "independent validation set. Does not verify truth of source artifacts, statistical "
            "independence, holdout secrecy, generalization to other evidence classes, process "
            "qualification, or physical safety. 'ready_for_next_stage' is not deployment approval."
        ),
    }


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant is forbidden: {value}")


def _load_json(path: Path, label: str) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    try:
        doc = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be strict UTF-8 JSON") from exc
    return raw, doc


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paired_receipt", type=Path)
    parser.add_argument("contract", type=Path)
    parser.add_argument("validation_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        input_paths = {args.paired_receipt.resolve(), args.contract.resolve(), args.validation_manifest.resolve()}
        if len(input_paths) != 3:
            raise ValueError("paired receipt, contract, and validation manifest must be distinct paths")
        output_path = args.output.resolve()
        if output_path in input_paths:
            raise ValueError("output must not overwrite any input")
        paired_raw, paired = _load_json(args.paired_receipt, "paired receipt")
        contract_raw, contract = _load_json(args.contract, "promotion contract")
        validation_raw, validation = _load_json(args.validation_manifest, "validation manifest")
        receipt = assess_promotion(
            paired, contract, validation,
            paired_manifest_sha256=hashlib.sha256(paired_raw).hexdigest(),
            contract_manifest_sha256=hashlib.sha256(contract_raw).hexdigest(),
            validation_manifest_sha256=hashlib.sha256(validation_raw).hexdigest(),
            evaluator_sha256=hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (OSError, ValueError, TypeError, OverflowError, KeyError) as exc:
        print(f"promotion readiness gate failed: {exc}", file=sys.stderr)
        return 2
    print(f"promotion readiness receipt written: {output_path}")
    print(f"decision={receipt['decision']}")
    print(f"failed_checks={len(receipt['failed_checks'])}")
    print(f"inconclusive_checks={len(receipt['inconclusive_checks'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
