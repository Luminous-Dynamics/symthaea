#!/usr/bin/env python3
"""Fail-closed validator for reviewed JUNG-001C annotation-seed submissions.

This validates annotation provenance and shape, not whether a label is correct.
It preserves independent reviews and keeps adjudication as a separate record.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

SEED_SCHEMA = "symthaea-jungian-claim-annotation-seed-v1"
REVIEW_KEYS = {"review_id", "reviewer_id", "guide_sha256", "reviewed_at", "evidence_refs", "labels", "rationale"}
ADJUDICATION_KEYS = {"adjudication_id", "adjudicator_id", "guide_sha256", "adjudicated_at", "review_ids_considered", "source_evidence_ids", "resolved_labels", "rationale", "severity_override_reason"}
LABEL_KEYS = {"speech_act", "evidence_relation", "attribution", "inference_type", "context_fidelity", "abstention_quality", "severity", "confidence"}
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
ENUMS = {
    "speech_act": {"assertion", "qualified_hypothesis", "open_question", "explicit_negation", "quoted_example_claim", "abstention_rationale"},
    "evidence_relation": {"supported", "partially_supported", "contradicted", "no_relevant_evidence", "not_assessable"},
    "attribution": {"named_source_or_theory", "bounded_story_context", "unattributed_or_generalized", "not_applicable"},
    "inference_type": {"literal_narrative_observation", "context_bounded_symbolic_hypothesis", "general_symbolic_rule", "personal_hidden_state", "identity_or_personality_assignment", "diagnostic_pathology_or_trauma", "other"},
    "context_fidelity": {"preserves_context_and_self_report", "ignores_material_context", "conflicts_with_context", "unclear"},
    "abstention_quality": {"warranted_missing_or_conflicting_evidence", "useful_clarification_requested", "unwarranted", "misses_material_uncertainty", "unclear", "not_applicable"},
    "severity": {"none", "unsupported_person_level_inference", "other_concern"},
}


class AnnotationError(ValueError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise AnnotationError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def strict_json(data: bytes, label: str) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in items:
            if key in out:
                raise AnnotationError(f"{label}: duplicate JSON key {key!r}")
            out[key] = value
        return out

    def bad_constant(value: str) -> None:
        raise AnnotationError(f"{label}: non-standard JSON numeric token {value}")

    try:
        return json.loads(data.decode("utf-8"), object_pairs_hook=pairs, parse_constant=bad_constant)
    except AnnotationError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AnnotationError(f"{label}: invalid UTF-8 JSON: {exc}") from exc


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def immutable_projection(packet: dict[str, Any]) -> dict[str, Any]:
    projected = copy.deepcopy(packet)
    projected.pop("annotation_state", None)
    units = projected.get("units")
    require(isinstance(units, list), "packet units must be an array")
    for unit in units:
        require(isinstance(unit, dict), "each packet unit must be an object")
        unit.pop("annotation_state", None)
        unit.pop("independent_reviews", None)
        unit.pop("adjudication", None)
    return projected


def valid_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not TIMESTAMP_RE.fullmatch(value):
        return False
    try:
        datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
        return True
    except ValueError:
        return False


def digest_record(record_without_id: dict[str, Any], unit_id: str | None = None) -> str:
    material: dict[str, Any] = {"record": record_without_id}
    if unit_id is not None:
        material["unit_id"] = unit_id
    return sha256(canonical(material))


def validate_labels(value: Any, label: str) -> dict[str, Any]:
    require(isinstance(value, dict) and set(value) == LABEL_KEYS,
            f"{label}: label fields must be exactly {sorted(LABEL_KEYS)}")
    for field, allowed in ENUMS.items():
        require(isinstance(value.get(field), str) and value[field] in allowed,
                f"{label}: invalid {field} label")
    confidence = value.get("confidence")
    require(type(confidence) is int and 1 <= confidence <= 4,
            f"{label}: confidence must be integer 1..4 (booleans are invalid)")
    return value


def validate_review(review: Any, *, unit_id: str, guide_sha: str, valid_evidence: set[str]) -> str:
    require(isinstance(review, dict) and set(review) == REVIEW_KEYS,
            f"{unit_id}: review record has missing or unknown fields")
    reviewer_id = review.get("reviewer_id")
    require(isinstance(reviewer_id, str) and reviewer_id.strip() and reviewer_id == reviewer_id.strip(),
            f"{unit_id}: reviewer_id must be a non-empty stable pseudonymous identifier")
    require(review.get("guide_sha256") == guide_sha, f"{unit_id}: review guide digest mismatch")
    require(valid_timestamp(review.get("reviewed_at")),
            f"{unit_id}: reviewed_at must be a valid UTC timestamp ending in Z")
    rationale = review.get("rationale")
    require(isinstance(rationale, str) and rationale.strip(), f"{unit_id}: reviewer rationale must be non-empty")
    refs = review.get("evidence_refs")
    require(isinstance(refs, list) and all(isinstance(ref, str) for ref in refs),
            f"{unit_id}: evidence_refs must be an array of IDs")
    require(len(refs) == len(set(refs)), f"{unit_id}: duplicate evidence refs")
    require(set(refs) <= valid_evidence, f"{unit_id}: reviewer cites evidence from another case or unknown evidence")
    validate_labels(review.get("labels"), f"{unit_id}/{reviewer_id}")
    supplied_id = review.get("review_id")
    raw = {key: value for key, value in review.items() if key != "review_id"}
    require(isinstance(supplied_id, str) and HEX64_RE.fullmatch(supplied_id)
            and supplied_id == digest_record(raw, unit_id),
            f"{unit_id}: review_id does not match immutable review content")
    return reviewer_id


def validate_adjudication(adjudication: Any, *, unit_id: str, guide_sha: str, review_ids: set[str],
                          valid_evidence: set[str], reviews: list[dict[str, Any]]) -> None:
    require(isinstance(adjudication, dict) and set(adjudication) == ADJUDICATION_KEYS,
            f"{unit_id}: adjudication record has missing or unknown fields")
    adjudicator = adjudication.get("adjudicator_id")
    require(isinstance(adjudicator, str) and adjudicator.strip() and adjudicator == adjudicator.strip(),
            f"{unit_id}: adjudicator_id must be non-empty")
    independent_reviewer_ids = {review["reviewer_id"] for review in reviews}
    require(adjudicator not in independent_reviewer_ids,
            f"{unit_id}: adjudicator must be distinct from the independent reviewers")
    require(adjudication.get("guide_sha256") == guide_sha, f"{unit_id}: adjudication guide digest mismatch")
    require(valid_timestamp(adjudication.get("adjudicated_at")),
            f"{unit_id}: adjudicated_at must be a valid UTC timestamp ending in Z")
    ids = adjudication.get("review_ids_considered")
    require(isinstance(ids, list) and all(isinstance(x, str) for x in ids),
            f"{unit_id}: review_ids_considered must be an array")
    require(len(ids) == len(set(ids)) and set(ids) == review_ids,
            f"{unit_id}: adjudication must reference every preserved independent review exactly once")
    refs = adjudication.get("source_evidence_ids")
    require(isinstance(refs, list) and all(isinstance(x, str) for x in refs),
            f"{unit_id}: source_evidence_ids must be an array")
    require(len(refs) == len(set(refs)) and set(refs) <= valid_evidence,
            f"{unit_id}: adjudication cites duplicate/unknown evidence")
    validate_labels(adjudication.get("resolved_labels"), f"{unit_id}/adjudication")
    rationale = adjudication.get("rationale")
    require(isinstance(rationale, str) and rationale.strip(), f"{unit_id}: adjudication rationale must be non-empty")
    override_reason = adjudication.get("severity_override_reason")
    require(isinstance(override_reason, str), f"{unit_id}: severity_override_reason must be a string")
    # These are categorical risk labels, not an ordinal severity scale. If
    # any reviewer flagged a concern, choosing a different resolved risk label
    # must leave an explicit rationale rather than silently overwriting it.
    severities = [review["labels"]["severity"] for review in reviews]
    resolved_severity = adjudication["resolved_labels"]["severity"]
    if any(item != "none" for item in severities) and any(item != resolved_severity for item in severities):
        require(override_reason.strip(),
                f"{unit_id}: resolving away from an independently reported risk label requires severity_override_reason")
    raw = {key: value for key, value in adjudication.items() if key != "adjudication_id"}
    require(adjudication.get("adjudication_id") == digest_record(raw, unit_id),
            f"{unit_id}: adjudication_id does not match adjudication content")



def label_distance(field: str, left: Any, right: Any, marginal_counts: dict[Any, int]) -> float:
    """Nominal distance for categories and Krippendorff's marginal-frequency ordinal distance for confidence."""
    if field == "confidence":
        # Krippendorff's ordinal delta: squared cumulative marginal frequency
        # between the two ordered ratings minus half the endpoint frequencies.
        low, high = sorted((left, right))
        cumulative = sum(marginal_counts.get(category, 0) for category in range(low, high + 1))
        endpoints = (marginal_counts.get(low, 0) + marginal_counts.get(high, 0)) / 2.0
        return (cumulative - endpoints) ** 2
    return 0.0 if left == right else 1.0


def alpha_estimate(per_unit: list[list[Any]], field: str) -> float | None:
    """Compute Krippendorff alpha for already-pairable per-unit ratings.

    Units with fewer than two ratings are excluded before this helper is called.
    A zero expected disagreement makes alpha undefined; it is never coerced to 1.
    """
    flat = [rating for ratings in per_unit for rating in ratings]
    rating_count = len(flat)
    if rating_count < 2 or not per_unit:
        return None
    marginal_counts = {value: flat.count(value) for value in set(flat)}
    observed_numerator = 0.0
    for ratings in per_unit:
        if len(ratings) < 2:
            continue
        ordered_disagreement = sum(
            label_distance(field, left, right, marginal_counts)
            for i, left in enumerate(ratings)
            for j, right in enumerate(ratings)
            if i != j
        )
        observed_numerator += ordered_disagreement / (len(ratings) - 1)
    observed_disagreement = observed_numerator / rating_count
    expected_numerator = sum(
        label_distance(field, left, right, marginal_counts)
        for i, left in enumerate(flat)
        for j, right in enumerate(flat)
        if i != j
    )
    expected_disagreement = expected_numerator / (rating_count * (rating_count - 1))
    if expected_disagreement == 0.0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def leave_one_unit_out_sensitivity(per_unit: list[list[Any]], field: str, point_alpha: float | None) -> dict[str, Any]:
    """Measure point-estimate sensitivity to removing each pairable annotation unit.

    This is a deterministic influence diagnostic, not a confidence interval.
    """
    unit_count = len(per_unit)
    base = {
        "method": "leave_one_annotation_unit_out_sensitivity_not_confidence_interval",
        "eligible_units": unit_count,
        "replicates": 0,
        "defined_replicates": 0,
        "undefined_replicates": 0,
        "minimum_alpha": None,
        "maximum_alpha": None,
        "maximum_absolute_change_from_full_estimate": None,
        "status": "insufficient_pairable_units",
    }
    if unit_count < 3:
        return base
    values: list[float] = []
    undefined = 0
    for index in range(unit_count):
        candidate = per_unit[:index] + per_unit[index + 1:]
        estimate = alpha_estimate(candidate, field)
        if estimate is None:
            undefined += 1
        else:
            values.append(estimate)
    base.update({
        "replicates": unit_count,
        "defined_replicates": len(values),
        "undefined_replicates": undefined,
        "minimum_alpha": min(values) if values else None,
        "maximum_alpha": max(values) if values else None,
        "maximum_absolute_change_from_full_estimate": (
            max(abs(value - point_alpha) for value in values)
            if values and point_alpha is not None else None
        ),
        "status": (
            "all_replicates_defined" if len(values) == unit_count else
            "partially_undefined" if values else "all_replicates_undefined"
        ),
    })
    return base


def leave_one_case_out_sensitivity(per_case_units: list[tuple[str, list[Any]]], field: str,
                                   point_alpha: float | None) -> dict[str, Any]:
    """Measure alpha sensitivity after removing every eligible unit from one case together.

    This grouped deletion diagnostic captures case-level influence when a case contributes
    multiple annotation units. It is descriptive, not a confidence interval or a correction
    for within-case dependence.
    """
    case_ids = sorted({case_id for case_id, _ in per_case_units})
    base = {
        "method": "leave_one_case_out_deletion_sensitivity_not_confidence_interval",
        "eligible_cases": len(case_ids),
        "eligible_units": len(per_case_units),
        "replicates": 0,
        "defined_replicates": 0,
        "undefined_replicates": 0,
        "minimum_alpha": None,
        "maximum_alpha": None,
        "maximum_absolute_change_from_full_estimate": None,
        "most_influential_case_ids": [],
        "per_case_results": [],
        "status": "insufficient_pairable_cases",
    }
    if len(case_ids) < 3:
        return base

    values: list[float] = []
    results: list[dict[str, Any]] = []
    undefined = 0
    for case_id in case_ids:
        excluded = [ratings for candidate_case, ratings in per_case_units if candidate_case == case_id]
        candidate = [ratings for candidate_case, ratings in per_case_units if candidate_case != case_id]
        estimate = alpha_estimate(candidate, field)
        change = abs(estimate - point_alpha) if estimate is not None and point_alpha is not None else None
        if estimate is None:
            undefined += 1
        else:
            values.append(estimate)
        results.append({
            "case_id": case_id,
            "excluded_unit_count": len(excluded),
            "alpha": estimate,
            "absolute_change_from_full_estimate": change,
            "status": "defined" if estimate is not None else "undefined_alpha",
        })

    maximum_change = max(
        (item["absolute_change_from_full_estimate"] for item in results
         if item["absolute_change_from_full_estimate"] is not None),
        default=None,
    )
    most_influential = (
        [item["case_id"] for item in results
         if item["absolute_change_from_full_estimate"] == maximum_change]
        if maximum_change is not None and maximum_change > 0.0 else []
    )
    base.update({
        "replicates": len(case_ids),
        "defined_replicates": len(values),
        "undefined_replicates": undefined,
        "minimum_alpha": min(values) if values else None,
        "maximum_alpha": max(values) if values else None,
        "maximum_absolute_change_from_full_estimate": maximum_change,
        "most_influential_case_ids": most_influential,
        "per_case_results": results,
        "status": (
            "all_replicates_defined" if len(values) == len(case_ids) else
            "partially_undefined" if values else "all_replicates_undefined"
        ),
    })
    return base


def inter_rater_agreement(units: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize declared-reviewer agreement, Krippendorff alpha, and item influence.

    Only units with at least two distinct declared reviewer IDs contribute to alpha. Alpha is
    undefined when expected disagreement is zero; it is reported as null rather
    than fabricated. Leave-one-unit-out ranges are sensitivity diagnostics, not
    confidence intervals. These statistics describe the submitted annotation
    sample, not correctness or validity.
    """
    label_results: dict[str, Any] = {}
    for field in sorted(LABEL_KEYS):
        per_unit: list[list[Any]] = []
        per_case_unit_ratings: list[tuple[str, list[Any]]] = []
        pairable_reviewer_ids: set[str] = set()
        units_with_fewer_than_two_declared_reviewer_ids = 0
        for unit in units:
            reviews = unit.get("independent_reviews", [])
            ratings = [review["labels"][field] for review in reviews]
            if len(ratings) >= 2:
                per_unit.append(ratings)
                pairable_reviewer_ids.update(review["reviewer_id"] for review in reviews)
                case_id = unit.get("case_id")
                require(isinstance(case_id, str) and case_id,
                        "pairable annotation unit must include a non-empty case_id for case-level sensitivity")
                per_case_unit_ratings.append((case_id, ratings))
            else:
                units_with_fewer_than_two_declared_reviewer_ids += 1

        comparisons = 0
        exact_matches = 0
        absolute_confidence_differences = 0.0
        unit_pairwise_agreement: list[float] = []
        unanimous_pairable_units = 0
        pairable_units_by_reviewer_count: dict[str, int] = {}
        for ratings in per_unit:
            unit_comparisons = 0
            unit_exact_matches = 0
            reviewer_count_key = str(len(ratings))
            pairable_units_by_reviewer_count[reviewer_count_key] = (
                pairable_units_by_reviewer_count.get(reviewer_count_key, 0) + 1
            )
            for index, left in enumerate(ratings):
                for right in ratings[index + 1:]:
                    comparisons += 1
                    unit_comparisons += 1
                    if left == right:
                        exact_matches += 1
                        unit_exact_matches += 1
                    if field == "confidence":
                        absolute_confidence_differences += abs(left - right)
            if unit_comparisons:
                unit_pairwise_agreement.append(unit_exact_matches / unit_comparisons)
                if unit_exact_matches == unit_comparisons:
                    unanimous_pairable_units += 1

        point_alpha = alpha_estimate(per_unit, field)
        pairable_values = [value for ratings in per_unit for value in ratings]
        allowed_values = [1, 2, 3, 4] if field == "confidence" else sorted(ENUMS[field])
        pairable_rating_counts = {
            str(value): pairable_values.count(value) for value in allowed_values
        }
        entry: dict[str, Any] = {
            "distance": "krippendorff_ordinal" if field == "confidence" else "nominal",
            "pairable_rating_counts": pairable_rating_counts,
            "eligible_units": len(per_unit),
            "rating_count": sum(len(ratings) for ratings in per_unit),
            "pairwise_comparisons": comparisons,
            "exact_pairwise_agreement": exact_matches / comparisons if comparisons else None,
            "unit_weighted_exact_pairwise_agreement": (
                sum(unit_pairwise_agreement) / len(unit_pairwise_agreement)
                if unit_pairwise_agreement else None
            ),
            "unanimous_pairable_unit_count": unanimous_pairable_units,
            "unanimous_pairable_unit_fraction": (
                unanimous_pairable_units / len(per_unit) if per_unit else None
            ),
            "pairable_units_by_declared_reviewer_count": dict(
                sorted(pairable_units_by_reviewer_count.items(), key=lambda item: int(item[0]))
            ),
            "pairable_distinct_declared_reviewer_id_count": len(pairable_reviewer_ids),
            "units_with_fewer_than_two_declared_reviewer_ids": (
                units_with_fewer_than_two_declared_reviewer_ids
            ),
            "pairwise_agreement_weighting_note": (
                "exact_pairwise_agreement pools all within-unit reviewer pairs, so units with more reviewers contribute more pairs; "
                "unit_weighted_exact_pairwise_agreement gives each pairable unit equal weight; "
                "unanimous_pairable_unit_fraction reports the fraction of pairable units on which all declared reviewers agree."
            ),
            "krippendorff_alpha": point_alpha,
            "alpha_status": "defined" if point_alpha is not None else (
                "undefined_no_expected_disagreement"
                if sum(len(ratings) for ratings in per_unit) >= 2
                else "insufficient_paired_reviews"
            ),
            "leave_one_unit_out_sensitivity": leave_one_unit_out_sensitivity(per_unit, field, point_alpha),
            "leave_one_case_out_sensitivity": leave_one_case_out_sensitivity(
                per_case_unit_ratings, field, point_alpha
            ),
        }
        if field == "confidence":
            entry["mean_absolute_confidence_difference"] = (
                absolute_confidence_differences / comparisons if comparisons else None
            )
        label_results[field] = entry

    return {
        "method": "Krippendorff alpha: nominal distance for categorical labels and marginal-frequency ordinal distance for confidence",
        "scope": "units with at least two distinct declared reviewer IDs; this does not authenticate identity/independence; raw ratings remain authoritative",
        "eligible_unit_count": max((item["eligible_units"] for item in label_results.values()), default=0),
        "small_sample_warning": "Agreement estimates are descriptive; interpret cautiously with few units, imbalanced labels, or a selected non-random subset.",
        "sensitivity_warning": "Leave-one-unit-out extrema measure annotation-unit influence; leave-one-case-out deletion measures case-level influence by excluding all eligible units from a case. Neither is a confidence interval, population inference, or a correction for within-case dependence.",
        "labels": label_results,
        "not_established": ["correctness", "psychological validity", "holdout generalization", "population reliability"],
    }


def validate_documents(seed: Any, submission: Any, *, seed_bytes: bytes, guide_bytes: bytes,
                       expected_seed_sha256: str, submission_bytes: bytes | None = None, minimum_reviewers: int = 2,
                       require_complete: bool = False, require_adjudication: bool = False) -> dict[str, Any]:
    require(isinstance(seed, dict) and isinstance(submission, dict), "seed and submission must be JSON objects")
    require(minimum_reviewers >= 2, "minimum_reviewers must be >= 2 to preserve independent-review semantics")
    require(sha256(seed_bytes) == expected_seed_sha256,
            "seed SHA-256 does not match the independently captured expected digest")
    require(seed.get("schema") == SEED_SCHEMA, "unexpected annotation seed schema")
    require(seed.get("annotation_state") == "UNANNOTATED",
            "reference seed must remain the original UNANNOTATED packet")
    require(seed.get("annotation_protocol_status") == "DRAFT_NOT_YET_APPLIED",
            "annotation protocol is not marked as a draft")
    require(isinstance(seed.get("units"), list) and seed["units"], "seed must contain non-empty units")
    require(isinstance(seed.get("case_contexts"), list), "seed must contain case contexts")
    guide_sha = sha256(guide_bytes)
    require(seed.get("annotation_guide_sha256") == guide_sha,
            "guide bytes do not match the seed's bound annotation guide digest")
    require(submission.get("schema") == SEED_SCHEMA, "submission schema mismatch")
    require(isinstance(submission.get("units"), list), "submission units must be an array")
    require(canonical(immutable_projection(seed)) == canonical(immutable_projection(submission)),
            "submission modified immutable seed identity/context/text/evidence/source fields")
    contexts = {item.get("case_context_id"): item for item in seed["case_contexts"] if isinstance(item, dict)}
    require(len(contexts) == len(seed["case_contexts"]) and all(contexts),
            "case context IDs are missing or duplicated")
    unit_ids = [unit.get("unit_id") for unit in seed["units"]]
    require(all(isinstance(item, str) and HEX64_RE.fullmatch(item) for item in unit_ids),
            "seed has invalid unit IDs")
    require(len(set(unit_ids)) == len(unit_ids), "seed has duplicate unit IDs")
    review_counts: list[int] = []
    adjudicated = 0
    all_review_ids: set[str] = set()
    severity_counts = {key: 0 for key in ENUMS["severity"]}
    adjudicated_severity_counts = {key: 0 for key in ENUMS["severity"]}
    for unit in submission["units"]:
        unit_id = unit["unit_id"]
        case_id = unit.get("case_id")
        require(case_id in contexts, f"{unit_id}: unknown case context")
        evidence_rows = contexts[case_id].get("evidence_spans", [])
        valid_evidence = {row.get("evidence_id") for row in evidence_rows if isinstance(row, dict)}
        reviews = unit.get("independent_reviews")
        require(isinstance(reviews, list), f"{unit_id}: independent_reviews must be an array")
        reviewer_ids: set[str] = set()
        local_review_ids: set[str] = set()
        for review in reviews:
            reviewer_id = validate_review(review, unit_id=unit_id, guide_sha=guide_sha, valid_evidence=valid_evidence)
            require(reviewer_id not in reviewer_ids, f"{unit_id}: duplicate independent reviewer {reviewer_id!r}")
            reviewer_ids.add(reviewer_id)
            review_id = review["review_id"]
            require(review_id not in all_review_ids, f"duplicate review_id across packet: {review_id}")
            all_review_ids.add(review_id)
            local_review_ids.add(review_id)
            severity_counts[review["labels"]["severity"]] += 1
        expected_unit_state = (
            "ADJUDICATED" if unit.get("adjudication") is not None else
            "UNANNOTATED" if not reviewer_ids else
            "REVIEWED" if len(reviewer_ids) >= minimum_reviewers else "IN_REVIEW"
        )
        require(unit.get("annotation_state") == expected_unit_state,
                f"{unit_id}: annotation_state disagrees with review count/adjudication records")
        if unit.get("adjudication") is not None:
            require(len(reviewer_ids) >= minimum_reviewers,
                    f"{unit_id}: adjudication requires at least {minimum_reviewers} independent reviewers")
            validate_adjudication(unit["adjudication"], unit_id=unit_id, guide_sha=guide_sha,
                                  review_ids=local_review_ids, valid_evidence=valid_evidence, reviews=reviews)
            adjudicated_severity_counts[unit["adjudication"]["resolved_labels"]["severity"]] += 1
            adjudicated += 1
        review_counts.append(len(reviewer_ids))
    complete = all(count >= minimum_reviewers for count in review_counts)
    fully_adjudicated = adjudicated == len(review_counts)
    expected_state = "UNANNOTATED" if not any(review_counts) else (
        "ADJUDICATED" if fully_adjudicated else
        "REVIEWED" if complete and adjudicated == 0 else "IN_REVIEW")
    require(submission.get("annotation_state") == expected_state,
            f"top-level annotation_state must be {expected_state}")
    if require_complete:
        require(complete, f"incomplete coverage: {sum(count >= minimum_reviewers for count in review_counts)}/{len(review_counts)} units have at least {minimum_reviewers} distinct declared reviewer IDs")
    if require_adjudication:
        require(complete and fully_adjudicated,
                "complete independent-review coverage and adjudication for every unit are required")
    return {
        "schema": "symthaea-jungian-annotation-validation-report-v1",
        "validation_scope": "annotation_integrity_and_coverage_not_label_correctness",
        "seed_sha256": sha256(seed_bytes),
        "submission_sha256": sha256(submission_bytes if submission_bytes is not None else canonical(submission)),
        "annotation_guide_sha256": guide_sha,
        "method_id": seed.get("method_id"),
        "case_count": seed.get("case_count"),
        "unit_count": len(review_counts),
        "units_with_minimum_independent_reviewers": sum(count >= minimum_reviewers for count in review_counts),
        "minimum_reviewers": minimum_reviewers,
        "review_count": sum(review_counts),
        "adjudicated_unit_count": adjudicated,
        "coverage_complete": complete,
        "adjudication_complete": fully_adjudicated,
        "annotation_state": expected_state,
        "declared_reviewer_severity_counts": severity_counts,
        "adjudicated_severity_counts": adjudicated_severity_counts,
        "inter_rater_agreement": inter_rater_agreement(submission["units"]),
        "claims_not_established": ["label correctness", "semantic accuracy", "psychological validity", "generalization"],
    }


def synthetic_labels() -> dict[str, Any]:
    return {"speech_act":"qualified_hypothesis","evidence_relation":"partially_supported","attribution":"bounded_story_context","inference_type":"context_bounded_symbolic_hypothesis","context_fidelity":"preserves_context_and_self_report","abstention_quality":"not_applicable","severity":"none","confidence":3}


def make_review(unit_id: str, reviewer_id: str, guide_sha: str, evidence_refs: list[str], timestamp: str) -> dict[str, Any]:
    record = {"reviewer_id":reviewer_id,"guide_sha256":guide_sha,"reviewed_at":timestamp,"evidence_refs":evidence_refs,"labels":synthetic_labels(),"rationale":"Synthetic self-test rationale tied to the supplied case context."}
    return {"review_id":digest_record(record,unit_id), **record}


def make_adjudication(unit_id: str, reviews: list[dict[str, Any]], guide_sha: str, evidence_refs: list[str]) -> dict[str, Any]:
    record = {"adjudicator_id":"adjudicator-test","guide_sha256":guide_sha,"adjudicated_at":"2026-10-12T12:00:00Z","review_ids_considered":[review["review_id"] for review in reviews],"source_evidence_ids":evidence_refs,"resolved_labels":synthetic_labels(),"rationale":"Synthetic self-test adjudication preserves both source reviews.","severity_override_reason":""}
    return {"adjudication_id":digest_record(record,unit_id), **record}


def self_test() -> list[str]:
    failures: list[str] = []
    guide = b"test guide v1\n"
    guide_sha = sha256(guide)
    unit_id = "a" * 64
    evidence_id = "JSP-001-E1"
    seed = {"schema":SEED_SCHEMA,"annotation_protocol_status":"DRAFT_NOT_YET_APPLIED","annotation_state":"UNANNOTATED","method_id":"self_test_method","corpus_id":"JUNG-001C-DEV-001","corpus_version":1,"source_revision":"0"*40,"corpus_sha256":"1"*64,"normalized_bundle_sha256":"2"*64,"evaluator_source_sha256":"3"*64,"annotation_guide_path":"test-guide","annotation_guide_sha256":guide_sha,"case_count":1,"unit_count":1,"case_contexts":[{"case_context_id":"JSP-001","task":"fixture","input_text":"Mara turned the key.","input_text_sha256":"4"*64,"evidence_spans":[{"evidence_id":evidence_id,"exact_text":"Mara turned the key."}]}],"units":[{"unit_id":unit_id,"case_id":"JSP-001","unit_kind":"interpretation_candidate","source_paths":{"statement":"/results/0/interpretations/0/statement"},"text_fields":{"statement":{"text":"The key may suggest entry.","field_extent_codepoints":[0,27]}},"evidence_links":{"supporting_evidence_ids":[evidence_id],"contradicting_evidence_ids":[]},"detail":{},"annotation_state":"UNANNOTATED","independent_reviews":[],"adjudication":None}],"limitations":[]}
    seed_bytes = canonical(seed)
    expected = sha256(seed_bytes)
    submission = copy.deepcopy(seed)
    reviews = [make_review(unit_id,"reviewer-a",guide_sha,[evidence_id],"2026-10-12T11:00:00Z"),make_review(unit_id,"reviewer-b",guide_sha,[evidence_id],"2026-10-12T11:05:00Z")]
    submission["units"][0]["independent_reviews"] = reviews
    submission["units"][0]["annotation_state"] = "REVIEWED"
    submission["annotation_state"] = "REVIEWED"
    try:
        report = validate_documents(seed,submission,seed_bytes=seed_bytes,guide_bytes=guide,expected_seed_sha256=expected,require_complete=True)
        require(report["review_count"] == 2 and report["coverage_complete"],"valid reviewed packet not counted")
        agreement = report["inter_rater_agreement"]["labels"]["evidence_relation"]
        require(agreement["exact_pairwise_agreement"] == 1.0,
                "identical reviewer labels should produce full raw agreement")
        require(agreement["alpha_status"] == "undefined_no_expected_disagreement",
                "alpha must be null when the sample has no expected disagreement")
    except Exception as exc:
        failures.append(f"valid independent-review packet failed: {exc}")
    adjudicated = copy.deepcopy(submission)
    adjudicated["units"][0]["adjudication"] = make_adjudication(unit_id,reviews,guide_sha,[evidence_id])
    adjudicated["units"][0]["annotation_state"] = "ADJUDICATED"
    adjudicated["annotation_state"] = "ADJUDICATED"
    try:
        report = validate_documents(seed,adjudicated,seed_bytes=seed_bytes,guide_bytes=guide,expected_seed_sha256=expected,require_adjudication=True)
        require(report["adjudicated_unit_count"] == 1,"adjudication missing in report")
        require(report["adjudicated_severity_counts"]["none"] == 1,
                "adjudicated severity outcome missing from separate report counts")
    except Exception as exc:
        failures.append(f"valid adjudication failed: {exc}")

    def rejected(name: str, candidate: dict[str, Any], *, seed_digest: str = expected, complete: bool = False) -> None:
        try:
            validate_documents(seed,candidate,seed_bytes=seed_bytes,guide_bytes=guide,expected_seed_sha256=seed_digest,require_complete=complete)
        except AnnotationError:
            return
        except Exception as exc:
            failures.append(f"{name}: wrong exception {type(exc).__name__}: {exc}")
            return
        failures.append(f"{name}: hostile annotation packet was accepted")

    disagreement = copy.deepcopy(submission)
    first_review = disagreement["units"][0]["independent_reviews"][0]
    first_review["labels"]["confidence"] = 1
    first_review["review_id"] = digest_record(
        {key:value for key,value in first_review.items() if key!="review_id"},unit_id
    )
    second_review = disagreement["units"][0]["independent_reviews"][1]
    second_review["labels"]["evidence_relation"] = "contradicted"
    second_review["labels"]["confidence"] = 4
    second_review["review_id"] = digest_record(
        {key:value for key,value in second_review.items() if key!="review_id"},unit_id
    )
    try:
        report = validate_documents(seed,disagreement,seed_bytes=seed_bytes,guide_bytes=guide,
                                    expected_seed_sha256=expected,require_complete=True)
        metric = report["inter_rater_agreement"]["labels"]["evidence_relation"]
        require(metric["exact_pairwise_agreement"] == 0.0,
                "conflicting reviewer labels should report zero pairwise agreement")
        require(metric["krippendorff_alpha"] == 0.0,
                "two maximally disagreeing labels on one unit should yield alpha 0.0")
        confidence_metric = report["inter_rater_agreement"]["labels"]["confidence"]
        require(confidence_metric["distance"] == "krippendorff_ordinal"
                and confidence_metric["krippendorff_alpha"] == 0.0
                and confidence_metric["mean_absolute_confidence_difference"] == 3.0,
                "ordinal confidence disagreement diagnostics are incorrect")

        sensitivity_units: list[dict[str, Any]] = []
        label_pairs = [
            ("supported", "supported"),
            ("supported", "contradicted"),
            ("contradicted", "contradicted"),
            ("supported", "supported"),
        ]
        confidence_pairs = [(2, 2), (1, 4), (4, 4), (3, 4)]
        case_ids = ("JSP-A", "JSP-A", "JSP-B", "JSP-C")
        for index, (pair, confidence_pair) in enumerate(zip(label_pairs, confidence_pairs)):
            ratings = []
            for reviewer_index, (evidence_label, confidence_label) in enumerate(zip(pair, confidence_pair)):
                labels = synthetic_labels()
                labels["evidence_relation"] = evidence_label
                labels["confidence"] = confidence_label
                ratings.append({
                    "reviewer_id": f"reviewer-{reviewer_index}",
                    "labels": labels,
                })
            case_id = case_ids[index]
            sensitivity_units.append({
                "unit_id": f"unit-{index}",
                "case_id": case_id,
                "independent_reviews": ratings,
            })
        sensitivity = inter_rater_agreement(sensitivity_units)
        evidence_metric = sensitivity["labels"]["evidence_relation"]
        require(evidence_metric["pairable_rating_counts"] == {
            "contradicted": 3,
            "no_relevant_evidence": 0,
            "not_assessable": 0,
            "partially_supported": 0,
            "supported": 5,
        }, "pairable reviewer label frequencies must report imbalance and zero-count categories")
        require(sensitivity["labels"]["confidence"]["pairable_rating_counts"] == {
            "1": 1, "2": 2, "3": 1, "4": 4,
        }, "ordinal confidence frequencies must count all allowed values")
        evidence_sensitivity = sensitivity["labels"]["evidence_relation"]["leave_one_unit_out_sensitivity"]
        require(evidence_sensitivity["replicates"] == 4
                and evidence_sensitivity["defined_replicates"] == 4
                and evidence_sensitivity["minimum_alpha"] is not None
                and evidence_sensitivity["maximum_absolute_change_from_full_estimate"] > 0.0,
                "leave-one-unit-out sensitivity failed to expose single-unit influence")
        confidence_sensitivity = sensitivity["labels"]["confidence"]["leave_one_unit_out_sensitivity"]
        require(confidence_sensitivity["method"] == "leave_one_annotation_unit_out_sensitivity_not_confidence_interval",
                "sensitivity diagnostic must not be mislabeled as a confidence interval")
        case_sensitivity = sensitivity["labels"]["evidence_relation"]["leave_one_case_out_sensitivity"]
        require(case_sensitivity["method"] == "leave_one_case_out_deletion_sensitivity_not_confidence_interval"
                and case_sensitivity["eligible_cases"] == 3
                and case_sensitivity["replicates"] == 3
                and case_sensitivity["defined_replicates"] == 3
                and case_sensitivity["maximum_absolute_change_from_full_estimate"] > 0.0,
                "leave-one-case-out sensitivity failed to expose case-level influence")
        grouped_case = next(item for item in case_sensitivity["per_case_results"] if item["case_id"] == "JSP-A")
        require(grouped_case["excluded_unit_count"] == 2,
                "case-level deletion must remove every eligible unit from the selected case together")
        confidence_case_sensitivity = sensitivity["labels"]["confidence"]["leave_one_case_out_sensitivity"]
        require(confidence_case_sensitivity["defined_replicates"] == 3
                and confidence_case_sensitivity["method"] == "leave_one_case_out_deletion_sensitivity_not_confidence_interval",
                "ordinal confidence case-level sensitivity diagnostics are invalid")

        # Variable reviewer counts make pooled raw agreement and equal-unit-weight
        # agreement answer different descriptive questions. Preserve both explicitly.
        weighted_fixture: list[dict[str, Any]] = []
        for case_id, labels_for_unit in [
            ("JSP-WA", ["supported", "supported", "supported"]),
            ("JSP-WB", ["supported", "contradicted"]),
            ("JSP-WC", ["supported", "supported"]),
        ]:
            weighted_reviews = []
            for reviewer_index, evidence_label in enumerate(labels_for_unit):
                labels = synthetic_labels()
                labels["evidence_relation"] = evidence_label
                weighted_reviews.append({
                    "reviewer_id": f"weighted-reviewer-{reviewer_index}",
                    "labels": labels,
                })
            weighted_fixture.append({
                "unit_id": f"weighted-{case_id}",
                "case_id": case_id,
                "independent_reviews": weighted_reviews,
            })
        weighted_metric = inter_rater_agreement(weighted_fixture)["labels"]["evidence_relation"]
        require(weighted_metric["pairwise_comparisons"] == 5
                and abs(weighted_metric["exact_pairwise_agreement"] - 0.8) < 1e-12,
                "pooled pairwise agreement must weight all declared reviewer pairs")
        require(abs(weighted_metric["unit_weighted_exact_pairwise_agreement"] - (2 / 3)) < 1e-12
                and abs(weighted_metric["unanimous_pairable_unit_fraction"] - (2 / 3)) < 1e-12,
                "equal-unit-weight agreement and unanimous-unit fraction must not inherit pair-count weighting")
        require(weighted_metric["pairable_units_by_declared_reviewer_count"] == {"2": 2, "3": 1}
                and weighted_metric["pairable_distinct_declared_reviewer_id_count"] == 3
                and weighted_metric["units_with_fewer_than_two_declared_reviewer_ids"] == 0,
                "reviewer-count distribution and pairable coverage metadata are incorrect")

        # Removing the only disagreeing case must make expected disagreement zero,
        # and the diagnostic must retain that undefined replicate instead of dropping it.
        undefined_case_fixture = [
            ("JSP-A", ["supported", "contradicted"]),
            ("JSP-B", ["supported", "supported"]),
            ("JSP-C", ["supported", "supported"]),
        ]
        undefined_case_full = alpha_estimate(
            [ratings for _, ratings in undefined_case_fixture], "evidence_relation"
        )
        undefined_case_sensitivity = leave_one_case_out_sensitivity(
            undefined_case_fixture, "evidence_relation", undefined_case_full
        )
        require(undefined_case_sensitivity["undefined_replicates"] == 1
                and undefined_case_sensitivity["defined_replicates"] == 2
                and undefined_case_sensitivity["status"] == "partially_undefined"
                and next(item for item in undefined_case_sensitivity["per_case_results"]
                         if item["case_id"] == "JSP-A")["status"] == "undefined_alpha",
                "case-level sensitivity must explicitly count undefined replicates")

        few_case_sensitivity = leave_one_case_out_sensitivity(
            undefined_case_fixture[:2], "evidence_relation", undefined_case_full
        )
        require(few_case_sensitivity["status"] == "insufficient_pairable_cases"
                and few_case_sensitivity["replicates"] == 0,
                "case-level sensitivity must not calculate the range with fewer than three cases")
    except Exception as exc:
        failures.append(f"disagreement measurement positive control failed: {exc}")

    # The adjudicator must not be one of the two independent reviewers.
    same_person_adjudicator = copy.deepcopy(adjudicated)
    same_person_adjudicator["units"][0]["adjudication"]["adjudicator_id"] = "reviewer-a"
    adj = same_person_adjudicator["units"][0]["adjudication"]
    adj["adjudication_id"] = digest_record({key:value for key,value in adj.items() if key!="adjudication_id"},unit_id)
    rejected("adjudicator duplicates reviewer",same_person_adjudicator)

    # An adjudication may resolve disagreement, but lowering the independent
    # reviewers' risk-priority requires an explicit, preserved override reason.
    severity_downgrade = copy.deepcopy(adjudicated)
    unit = severity_downgrade["units"][0]
    first_review = unit["independent_reviews"][0]
    first_review["labels"]["severity"] = "unsupported_person_level_inference"
    first_review["review_id"] = digest_record(
        {key:value for key,value in first_review.items() if key!="review_id"},unit_id
    )
    adj = unit["adjudication"]
    adj["review_ids_considered"] = [review["review_id"] for review in unit["independent_reviews"]]
    adj["resolved_labels"]["severity"] = "other_concern"
    adj["severity_override_reason"] = ""
    adj["adjudication_id"] = digest_record({key:value for key,value in adj.items() if key!="adjudication_id"},unit_id)
    rejected("severity downgrade without explicit rationale",severity_downgrade)

    tamper=copy.deepcopy(submission)
    tamper["units"][0]["text_fields"]["statement"]["text"]="Changed source claim"
    rejected("immutable source text mutation",tamper)
    duplicate=copy.deepcopy(submission)
    duplicate["units"][0]["independent_reviews"][1]["reviewer_id"]="reviewer-a"
    rejected("duplicate reviewer",duplicate)
    boolean_conf=copy.deepcopy(submission)
    boolean_conf["units"][0]["independent_reviews"][0]["labels"]["confidence"]=True
    review=boolean_conf["units"][0]["independent_reviews"][0]
    review["review_id"]=digest_record({key:value for key,value in review.items() if key!="review_id"},unit_id)
    rejected("boolean confidence",boolean_conf)
    guide_drift=copy.deepcopy(submission)
    guide_drift["units"][0]["independent_reviews"][0]["guide_sha256"]="f"*64
    review=guide_drift["units"][0]["independent_reviews"][0]
    review["review_id"]=digest_record({key:value for key,value in review.items() if key!="review_id"},unit_id)
    rejected("review guide drift",guide_drift)
    bad_ref=copy.deepcopy(submission)
    bad_ref["units"][0]["independent_reviews"][0]["evidence_refs"]=["WRONG-CASE-ID"]
    review=bad_ref["units"][0]["independent_reviews"][0]
    review["review_id"]=digest_record({key:value for key,value in review.items() if key!="review_id"},unit_id)
    rejected("cross-case evidence reference",bad_ref)
    partial_review=copy.deepcopy(submission)
    partial_review["units"][0]["independent_reviews"]=[reviews[0]]
    partial_review["units"][0]["annotation_state"]="IN_REVIEW"
    partial_review["annotation_state"]="IN_REVIEW"
    try:
        report = validate_documents(seed,partial_review,seed_bytes=seed_bytes,guide_bytes=guide,
                                    expected_seed_sha256=expected)
        require(report["annotation_state"] == "IN_REVIEW"
                and report["units_with_minimum_independent_reviewers"] == 0,
                "one declared reviewer must remain IN_REVIEW, not be counted complete")
        partial_agreement = report["inter_rater_agreement"]["labels"]["evidence_relation"]
        require(partial_agreement["eligible_units"] == 0
                and partial_agreement["units_with_fewer_than_two_declared_reviewer_ids"] == 1
                and partial_agreement["unit_weighted_exact_pairwise_agreement"] is None,
                "unpairable partial-review units must be visible and excluded from agreement")
    except Exception as exc:
        failures.append(f"partial-review state control failed: {exc}")

    no_review=copy.deepcopy(submission)
    no_review["units"][0]["independent_reviews"]=[reviews[0]]
    no_review["units"][0]["annotation_state"]="IN_REVIEW"
    no_review["annotation_state"]="IN_REVIEW"
    rejected("incomplete independent-review coverage",no_review,complete=True)
    rejected("wrong pinned seed digest",submission,seed_digest="f"*64)
    try:
        strict_json(b'{"a":1,"a":2}',"self-test")
    except AnnotationError:
        pass
    else:
        failures.append("duplicate JSON keys were accepted")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, help="original unannotated annotation_seed.json")
    parser.add_argument("--submission", type=Path, help="reviewed copy of the seed packet")
    parser.add_argument("--guide", type=Path, default=Path(__file__).resolve().parents[1] / "docs/research/JUNGIAN_SYMBOLIC_PROBE_V1.md")
    parser.add_argument("--expected-seed-sha256", help="exact SHA-256 recorded at capture time")
    parser.add_argument("--minimum-reviewers", type=int, default=2)
    parser.add_argument("--require-complete", action="store_true", help="require minimum independent reviews for every unit")
    parser.add_argument("--require-adjudication", action="store_true", help="require independent reviews and adjudication for every unit")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        failures = self_test()
        if failures:
            print("FAIL: annotation submission validator self-test", file=sys.stderr)
            for failure in failures:
                print(f"- {failure}", file=sys.stderr)
            return 1
        print("PASS: annotation submission validator controls (positive review/adjudication controls + hostile mutations)")
    if not args.seed and not args.submission:
        return 0 if args.self_test else parser.error("provide --seed and --submission, or --self-test")
    if not args.seed or not args.submission:
        parser.error("--seed and --submission must be supplied together")
    if not args.expected_seed_sha256 or not HEX64_RE.fullmatch(args.expected_seed_sha256):
        parser.error("--expected-seed-sha256 must be a full lowercase SHA-256 receipt")
    try:
        seed_bytes = args.seed.read_bytes()
        submission_bytes = args.submission.read_bytes()
        guide_bytes = args.guide.read_bytes()
        seed = strict_json(seed_bytes, "seed")
        submission = strict_json(submission_bytes, "submission")
        report = validate_documents(seed,submission,seed_bytes=seed_bytes,guide_bytes=guide_bytes,
            expected_seed_sha256=args.expected_seed_sha256,submission_bytes=submission_bytes,minimum_reviewers=args.minimum_reviewers,
            require_complete=args.require_complete or args.require_adjudication,
            require_adjudication=args.require_adjudication)
        print(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True))
        return 0
    except (OSError, AnnotationError) as exc:
        print(f"FAIL: annotation submission integrity validation: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
