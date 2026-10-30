#!/usr/bin/env python3
"""Audit paired evaluator outcomes against preregistered metamorphic relations.

The tool validates evaluator behavior on a supplied fixed receipt. It does not
run a judge, authenticate receipt provenance, estimate population effects, or
identify causal effects. Relations are intervention contracts, not causal
inference claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

PLAN_SCHEMA = "symthaea.science_bench_refresh_metamorphic_plan.v1"
RECEIPT_SCHEMA = "symthaea.science_bench_refresh_metamorphic_receipt.v1"
REPORT_SCHEMA = "symthaea.science_bench_refresh_metamorphic_audit.v1"
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

PLAN_FIELDS = {
    "schema", "purpose", "plan_id", "evaluator_source_revision",
    "evaluator_prompt_sha256", "rubric_sha256", "criterion_ids",
    "score_tolerance_millionths", "pairs", "plan_commitment_sha256",
}
PAIR_FIELDS = {
    "pair_id", "relation_family", "relation_rule",
    "baseline_input_sha256", "variant_input_sha256",
    "affected_criterion_ids", "unaffected_criterion_ids",
    "expected_variant_decision", "expected_selected_candidate_id",
}
RECEIPT_FIELDS = {
    "schema", "purpose", "run_id", "plan_sha256",
    "evaluator_source_revision", "evaluator_prompt_sha256", "rubric_sha256",
    "observations",
}
OBSERVATION_FIELDS = {"pair_id", "baseline", "variant"}
OUTCOME_FIELDS = {
    "decision", "criterion_scores", "hard_veto_active",
    "selected_candidate_id", "selected_position", "unique_source_group_count",
    "pointwise_candidate_scores", "pairwise_preferred_candidate_id",
}
CRITERION_SCORE_FIELDS = {"criterion_id", "score_millionths"}
CANDIDATE_SCORE_FIELDS = {"candidate_id", "score_millionths"}
MAX_PAIRS = 100_000
MAX_CRITERIA = 128

DECISIONS = {
    "supported_within_scope", "contradicted_within_scope",
    "mixed_or_contested", "insufficient_evidence", "scope_ambiguous",
    "temporally_stale", "not_applicable", "not_evaluable", "abstain",
}
SUPPORT_ORDER = {
    "contradicted_within_scope": 0,
    "temporally_stale": 1,
    "insufficient_evidence": 1,
    "scope_ambiguous": 1,
    "not_evaluable": 1,
    "not_applicable": 1,
    "abstain": 1,
    "mixed_or_contested": 2,
    "supported_within_scope": 3,
}
RULES_BY_FAMILY = {
    "invariance": {"all_outputs_invariant"},
    "directional_sensitivity": {"support_non_decreasing", "support_non_increasing"},
    "locality": {"only_unaffected_scores_invariant"},
    "order_equivariance": {"candidate_identity_invariant_position_changes"},
    "provenance_non_duplication": {"unique_origin_count_not_increased"},
    "cross_mode_consistency": {"pointwise_pairwise_choice_agree"},
    "counterfactual_consistency": {"variant_decision_equals_declared"},
}


class MetamorphicAuditError(ValueError):
    """Invalid plan or receipt structure."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def plan_commitment(plan: dict[str, Any]) -> str:
    fields = {key: value for key, value in plan.items() if key != "plan_commitment_sha256"}
    return sha256(b"symthaea.metamorphic.plan.v1\0" + canonical_bytes(fields))


def _object(value: Any, expected: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MetamorphicAuditError(f"{context} must be an object")
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing:
        raise MetamorphicAuditError(f"{context} missing fields: {','.join(missing)}")
    if unknown:
        raise MetamorphicAuditError(f"{context} unknown fields: {','.join(unknown)}")
    return value


def _digest(value: Any, context: str, allow_zero: bool) -> None:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise MetamorphicAuditError(f"{context} must be 64 lowercase hexadecimal characters")
    if not allow_zero and set(value) == {"0"}:
        raise MetamorphicAuditError(f"{context} cannot be a zero digest")


def _revision(value: Any, context: str, allow_placeholder: bool) -> None:
    if not isinstance(value, str) or not value.strip():
        raise MetamorphicAuditError(f"{context} must be non-empty")
    if not allow_placeholder and (HEX40.fullmatch(value) is None or set(value) == {"0"}):
        raise MetamorphicAuditError(f"{context} must be a nonzero lowercase 40-character revision")


def validate_plan(plan: Any) -> dict[str, Any]:
    value = _object(plan, PLAN_FIELDS, "plan")
    if value["schema"] != PLAN_SCHEMA:
        raise MetamorphicAuditError("unsupported metamorphic plan schema")
    purpose = value["purpose"]
    if not isinstance(purpose, str) or purpose not in {"example_only", "qualified_run"}:
        raise MetamorphicAuditError("purpose must be example_only or qualified_run")
    example = purpose == "example_only"
    _revision(value["evaluator_source_revision"], "evaluator_source_revision", example)
    _digest(value["evaluator_prompt_sha256"], "evaluator_prompt_sha256", example)
    _digest(value["rubric_sha256"], "rubric_sha256", example)
    if not isinstance(value["plan_id"], str) or not value["plan_id"].strip() or len(value["plan_id"]) > 128:
        raise MetamorphicAuditError("plan_id must be non-empty and at most 128 characters")
    tolerance = value["score_tolerance_millionths"]
    if not isinstance(tolerance, int) or isinstance(tolerance, bool) or not 0 <= tolerance <= 1_000_000:
        raise MetamorphicAuditError("score_tolerance_millionths must be an integer in 0..1,000,000")
    criteria = value["criterion_ids"]
    if not isinstance(criteria, list) or not 1 <= len(criteria) <= MAX_CRITERIA:
        raise MetamorphicAuditError(f"criterion_ids must contain 1..={MAX_CRITERIA} entries")
    if any(not isinstance(item, str) or not item.strip() or len(item) > 128 for item in criteria):
        raise MetamorphicAuditError("criterion_ids must be non-empty strings of at most 128 characters")
    if len(criteria) != len(set(criteria)):
        raise MetamorphicAuditError("criterion_ids contains duplicates")
    pairs = value["pairs"]
    if not isinstance(pairs, list) or not 1 <= len(pairs) <= MAX_PAIRS:
        raise MetamorphicAuditError(f"pairs must contain 1..={MAX_PAIRS} rows")
    pair_ids: set[str] = set()
    criteria_set = set(criteria)
    for index, pair in enumerate(pairs):
        row = _object(pair, PAIR_FIELDS, f"pairs[{index}]")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip() or len(pair_id) > 128:
            raise MetamorphicAuditError(f"pairs[{index}].pair_id must be non-empty")
        if pair_id in pair_ids:
            raise MetamorphicAuditError(f"duplicate pair_id {pair_id}")
        pair_ids.add(pair_id)
        family = row["relation_family"]
        rule = row["relation_rule"]
        if not isinstance(family, str) or family not in RULES_BY_FAMILY:
            raise MetamorphicAuditError(f"pair {pair_id} has unsupported relation_family")
        if not isinstance(rule, str) or rule not in RULES_BY_FAMILY[family]:
            raise MetamorphicAuditError(f"pair {pair_id} has a relation_rule incompatible with {family}")
        _digest(row["baseline_input_sha256"], f"pair {pair_id} baseline_input_sha256", example)
        _digest(row["variant_input_sha256"], f"pair {pair_id} variant_input_sha256", example)
        for field in ("affected_criterion_ids", "unaffected_criterion_ids"):
            values = row[field]
            if not isinstance(values, list) or any(not isinstance(item, str) for item in values):
                raise MetamorphicAuditError(f"pair {pair_id} {field} must be a string array")
            if len(values) != len(set(values)):
                raise MetamorphicAuditError(f"pair {pair_id} {field} contains duplicates")
            if not set(values).issubset(criteria_set):
                raise MetamorphicAuditError(f"pair {pair_id} {field} references an unknown criterion")
        affected = set(row["affected_criterion_ids"])
        unaffected = set(row["unaffected_criterion_ids"])
        if affected & unaffected:
            raise MetamorphicAuditError(f"pair {pair_id} affected/unaffected criteria overlap")
        if family == "locality" and (not affected or not unaffected):
            raise MetamorphicAuditError(f"locality pair {pair_id} requires affected and unaffected criteria")
        expected_decision = row["expected_variant_decision"]
        if expected_decision is not None and (
            not isinstance(expected_decision, str) or expected_decision not in DECISIONS
        ):
            raise MetamorphicAuditError(f"pair {pair_id} has unsupported expected_variant_decision")
        expected_candidate = row["expected_selected_candidate_id"]
        if expected_candidate is not None and (not isinstance(expected_candidate, str) or not expected_candidate.strip()):
            raise MetamorphicAuditError(f"pair {pair_id} expected_selected_candidate_id must be null or non-empty")
            if rule == "variant_decision_equals_declared" and expected_decision is None:
            raise MetamorphicAuditError(f"counterfactual pair {pair_id} requires expected_variant_decision")
        if rule == "candidate_identity_invariant_position_changes" and expected_candidate is None:
            raise MetamorphicAuditError(f"order-equivariance pair {pair_id} requires expected_selected_candidate_id")
    commitment = value["plan_commitment_sha256"]
    _digest(commitment, "plan_commitment_sha256", example)
    expected_commitment = plan_commitment(value)
    if expected_commitment != commitment:
        raise MetamorphicAuditError("plan commitment mismatch")
    return value


def _validate_outcome(outcome: Any, pair_id: str, criteria: list[str]) -> dict[str, Any]:
    row = _object(outcome, OUTCOME_FIELDS, f"outcome[{pair_id}]")
    if not isinstance(row["decision"], str) or row["decision"] not in DECISIONS:
        raise MetamorphicAuditError(f"outcome[{pair_id}] has unsupported decision")
    if not isinstance(row["hard_veto_active"], bool):
        raise MetamorphicAuditError(f"outcome[{pair_id}].hard_veto_active must be boolean")
    for field in ("selected_candidate_id", "pairwise_preferred_candidate_id"):
        item = row[field]
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise MetamorphicAuditError(f"outcome[{pair_id}].{field} must be null or non-empty")
    position = row["selected_position"]
    if position is not None and (not isinstance(position, int) or isinstance(position, bool) or position not in (0, 1)):
        raise MetamorphicAuditError(f"outcome[{pair_id}].selected_position must be null, 0 or 1")
    groups = row["unique_source_group_count"]
    if not isinstance(groups, int) or isinstance(groups, bool) or groups < 0:
        raise MetamorphicAuditError(f"outcome[{pair_id}].unique_source_group_count must be a nonnegative integer")
    scores = row["criterion_scores"]
    if not isinstance(scores, list) or len(scores) != len(criteria):
        raise MetamorphicAuditError(f"outcome[{pair_id}] criterion_scores must cover all criteria")
    score_map: dict[str, int] = {}
    for index, score in enumerate(scores):
        score = _object(score, CRITERION_SCORE_FIELDS, f"outcome[{pair_id}].criterion_scores[{index}]")
        criterion_id = score["criterion_id"]
        value = score["score_millionths"]
        if not isinstance(criterion_id, str) or criterion_id not in criteria or criterion_id in score_map:
            raise MetamorphicAuditError(f"outcome[{pair_id}] has duplicate or unknown criterion score")
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= 1_000_000:
            raise MetamorphicAuditError(f"outcome[{pair_id}] has out-of-range score")
        score_map[criterion_id] = value
    if set(score_map) != set(criteria):
        raise MetamorphicAuditError(f"outcome[{pair_id}] criterion score set differs from plan")
    candidates = row["pointwise_candidate_scores"]
    if not isinstance(candidates, list) or len(candidates) not in (0, 2):
        raise MetamorphicAuditError(f"outcome[{pair_id}] pointwise_candidate_scores must be empty or contain two candidates")
    candidate_map: dict[str, int] = {}
    for index, candidate in enumerate(candidates):
        candidate = _object(candidate, CANDIDATE_SCORE_FIELDS, f"outcome[{pair_id}].pointwise_candidate_scores[{index}]")
        candidate_id = candidate["candidate_id"]
        score = candidate["score_millionths"]
        if not isinstance(candidate_id, str) or not candidate_id.strip() or candidate_id in candidate_map:
            raise MetamorphicAuditError(f"outcome[{pair_id}] has duplicate/invalid candidate ID")
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 1_000_000:
            raise MetamorphicAuditError(f"outcome[{pair_id}] has invalid pointwise score")
        candidate_map[candidate_id] = score
    preferred = row["pairwise_preferred_candidate_id"]
    if preferred is not None and (
        not isinstance(preferred, str) or preferred not in candidate_map
    ):
        raise MetamorphicAuditError(f"outcome[{pair_id}] pairwise preference names unknown candidate")
    return {"raw": row, "criterion_scores": score_map, "candidate_scores": candidate_map}


def _decision_support(decision: str) -> int:
    return SUPPORT_ORDER[decision]


def _check_pair(pair: dict[str, Any], observation: dict[str, Any], tolerance: int, criteria: list[str]) -> dict[str, Any]:
    pair_id = pair["pair_id"]
    baseline = _validate_outcome(observation["baseline"], pair_id + ".baseline", criteria)
    variant = _validate_outcome(observation["variant"], pair_id + ".variant", criteria)
    before = baseline["raw"]
    after = variant["raw"]
    before_scores = baseline["criterion_scores"]
    after_scores = variant["criterion_scores"]
    deltas = {criterion: after_scores[criterion] - before_scores[criterion] for criterion in criteria}
    abs_deltas = {criterion: abs(value) for criterion, value in deltas.items()}
    failures: list[str] = []
    rule = pair["relation_rule"]

    if rule == "all_outputs_invariant":
        if before["decision"] != after["decision"]:
            failures.append("decision changed under a declared invariant intervention")
        if before["hard_veto_active"] != after["hard_veto_active"]:
            failures.append("hard-veto state changed under a declared invariant intervention")
        if any(value > tolerance for value in abs_deltas.values()):
            failures.append("criterion score drift exceeded tolerance under a declared invariant intervention")
        if before["selected_candidate_id"] != after["selected_candidate_id"]:
            failures.append("selected candidate changed under a declared invariant intervention")
        if before["unique_source_group_count"] != after["unique_source_group_count"]:
            failures.append("source-origin group count changed under a declared invariant intervention")

    elif rule == "support_non_decreasing":
        if _decision_support(after["decision"]) < _decision_support(before["decision"]):
            failures.append("support decreased after a positive evidence intervention")
        for criterion in pair["affected_criterion_ids"]:
            if deltas[criterion] < -tolerance:
                failures.append(f"affected criterion {criterion} score decreased after positive evidence")
    elif rule == "support_non_increasing":
        if _decision_support(after["decision"]) > _decision_support(before["decision"]):
            failures.append("support increased after evidence withdrawal or weakening")
        for criterion in pair["affected_criterion_ids"]:
            if deltas[criterion] > tolerance:
                failures.append(f"affected criterion {criterion} score increased after evidence withdrawal")
    elif rule == "only_unaffected_scores_invariant":
        affected = set(pair["affected_criterion_ids"])
        unaffected = set(pair["unaffected_criterion_ids"])
        if not affected or not unaffected:
            failures.append("locality relation has no affected/unaffected criterion partition")
        for criterion in sorted(unaffected):
            if abs_deltas[criterion] > tolerance:
                failures.append(f"unaffected criterion {criterion} drifted beyond tolerance")
        if not any(abs_deltas[criterion] > tolerance for criterion in affected):
            failures.append("intervention did not produce an observable change in any affected criterion")
    elif rule == "candidate_identity_invariant_position_changes":
        expected = pair["expected_selected_candidate_id"]
        if before["selected_candidate_id"] != expected or after["selected_candidate_id"] != expected:
            failures.append("selected candidate identity changed across position swap")
        if before["selected_position"] == after["selected_position"]:
            failures.append("selected candidate position did not change after position swap")
        if before["selected_position"] is None or after["selected_position"] is None:
            failures.append("position-swap outcome omitted selected position")
    elif rule == "unique_origin_count_not_increased":
        if after["unique_source_group_count"] > before["unique_source_group_count"]:
            failures.append("duplicate/mirror source increased independent source-origin count")
        if before["decision"] != after["decision"]:
            failures.append("duplicate/mirror source changed the reference decision")
        if any(value > tolerance for value in abs_deltas.values()):
            failures.append("duplicate/mirror source caused criterion score inflation beyond tolerance")
    elif rule == "pointwise_pairwise_choice_agree":
        for side, result in (("baseline", baseline), ("variant", variant)):
            scores = result["candidate_scores"]
            if len(scores) != 2:
                failures.append(f"{side} omitted the two pointwise candidate scores")
                continue
            maximum = max(scores.values())
            winners = {candidate for candidate, score in scores.items() if score == maximum}
            preferred = result["raw"]["pairwise_preferred_candidate_id"]
            if preferred not in winners:
                failures.append(f"{side} pairwise preference disagrees with pointwise score ordering")
    elif rule == "variant_decision_equals_declared":
        expected = pair["expected_variant_decision"]
        if after["decision"] != expected:
            failures.append(f"counterfactual variant decision differs from declared expectation {expected}")
        if after["decision"] == before["decision"]:
            failures.append("counterfactual intervention did not change the declared decision")

    return {
        "pair_id": pair_id,
        "relation_family": pair["relation_family"],
        "relation_rule": rule,
        "passed": not failures,
        "criterion_score_deltas_millionths": deltas,
        "maximum_absolute_criterion_drift_millionths": max(abs_deltas.values()) if abs_deltas else 0,
        "decision_before": before["decision"],
        "decision_after": after["decision"],
        "unique_source_groups_before": before["unique_source_group_count"],
        "unique_source_groups_after": after["unique_source_group_count"],
        "failures": failures,
    }


def audit_receipt(plan_bytes: bytes, receipt_bytes: bytes) -> dict[str, Any]:
    try:
        plan_raw = json.loads(plan_bytes)
        receipt_raw = json.loads(receipt_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise MetamorphicAuditError(f"plan/receipt JSON malformed: {error}") from error
    plan = validate_plan(plan_raw)
    receipt = _object(receipt_raw, RECEIPT_FIELDS, "receipt")
    if receipt["schema"] != RECEIPT_SCHEMA:
        raise MetamorphicAuditError("unsupported metamorphic receipt schema")
    if receipt["purpose"] != plan["purpose"]:
        raise MetamorphicAuditError("receipt purpose differs from plan purpose")
    if not isinstance(receipt["run_id"], str) or not receipt["run_id"].strip():
        raise MetamorphicAuditError("receipt run_id must be non-empty")
    _revision(receipt["evaluator_source_revision"], "receipt evaluator_source_revision", plan["purpose"] == "example_only")
    _digest(receipt["evaluator_prompt_sha256"], "receipt evaluator_prompt_sha256", plan["purpose"] == "example_only")
    _digest(receipt["rubric_sha256"], "receipt rubric_sha256", plan["purpose"] == "example_only")
    if receipt["evaluator_source_revision"] != plan["evaluator_source_revision"]:
        raise MetamorphicAuditError("receipt evaluator revision differs from plan")
    if receipt["evaluator_prompt_sha256"] != plan["evaluator_prompt_sha256"]:
        raise MetamorphicAuditError("receipt evaluator prompt digest differs from plan")
    if receipt["rubric_sha256"] != plan["rubric_sha256"]:
        raise MetamorphicAuditError("receipt rubric digest differs from plan")
    plan_sha256 = sha256(plan_bytes)
    if receipt["plan_sha256"] != plan_sha256:
        raise MetamorphicAuditError("receipt plan_sha256 does not bind exact plan bytes")
    observations = receipt["observations"]
    if not isinstance(observations, list):
        raise MetamorphicAuditError("receipt observations must be an array")
    by_id: dict[str, dict[str, Any]] = {}
    for index, observation in enumerate(observations):
        row = _object(observation, OBSERVATION_FIELDS, f"observations[{index}]")
        pair_id = row["pair_id"]
        if not isinstance(pair_id, str) or not pair_id.strip():
            raise MetamorphicAuditError(f"observations[{index}].pair_id must be non-empty")
        if pair_id in by_id:
            raise MetamorphicAuditError(f"duplicate observation pair_id {pair_id}")
        by_id[pair_id] = row

    planned_ids = {pair["pair_id"] for pair in plan["pairs"]}
    observed_ids = set(by_id)
    missing = sorted(planned_ids - observed_ids)
    extra = sorted(observed_ids - planned_ids)
    if missing or extra:
        details = []
        if missing:
            details.append("missing pairs: " + ",".join(missing))
        if extra:
            details.append("unexpected pairs: " + ",".join(extra))
        raise MetamorphicAuditError("; ".join(details))

    results = [
        _check_pair(pair, by_id[pair["pair_id"]], plan["score_tolerance_millionths"], plan["criterion_ids"])
        for pair in plan["pairs"]
    ]
    grouped: dict[str, dict[str, int]] = {}
    for result in results:
        row = grouped.setdefault(result["relation_family"], {"planned": 0, "passed": 0, "failed": 0})
        row["planned"] += 1
        if result["passed"]:
            row["passed"] += 1
        else:
            row["failed"] += 1
    max_drift = max((result["maximum_absolute_criterion_drift_millionths"] for result in results), default=0)
    return {
        "schema": REPORT_SCHEMA,
        "plan_id": plan["plan_id"],
        "plan_sha256": plan_sha256,
        "receipt_sha256": sha256(receipt_bytes),
        "valid": True,
        "all_relations_pass": all(result["passed"] for result in results),
        "planned_pair_count": len(plan["pairs"]),
        "observed_pair_count": len(observations),
        "passed_pair_count": sum(result["passed"] for result in results),
        "failed_pair_count": sum(not result["passed"] for result in results),
        "maximum_criterion_score_drift_millionths": max_drift,
        "results_by_relation_family": grouped,
        "pair_results": results,
        "fixed_receipt_only": True,
        "evaluator_execution_proven": False,
        "receipt_authenticity_proven": False,
        "causal_effect_identified": False,
        "generalized_accuracy_estimated": False,
        "construct_validity_proven": False,
        "external_anchor_required": True,
        "scoring_eligible_declared": False,
        "claim_boundary": "Checks declared metamorphic relations on a supplied fixed paired receipt only; no evaluator execution authenticity, causal effect identification, population generalization, construct validity, or scoring qualification is established.",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        plan_bytes = args.plan.read_bytes()
        receipt_bytes = args.receipt.read_bytes()
        report = audit_receipt(plan_bytes, receipt_bytes)
    except (OSError, MetamorphicAuditError) as error:
        payload = json.dumps({"valid": False, "issues": [str(error)]}, indent=2) + "\n"
        print(payload, end="")
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["valid"] and report["all_relations_pass"] else 11


if __name__ == "__main__":
    raise SystemExit(main())
