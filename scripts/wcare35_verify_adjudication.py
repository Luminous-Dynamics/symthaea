#!/usr/bin/env python3
"""Verify WCARE-35 adjudication evidence from immutable reviewer records.

This verifies evidence integrity and recomputes descriptive summaries. It does not
establish objective moral truth and grants no runtime authority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ALLOWED_REVIEWER_CLASSES = {
    "AffectedStakeholder",
    "DomainExpert",
    "IndependentHumanGeneralist",
    "IndependentModelSession",
    "SameDevelopmentLineage",
    "Other",
}
ALLOWED_DISPOSITIONS = {
    "ADJUDICATION_SUPPORTED",
    "ADJUDICATION_CONTESTED",
    "ADJUDICATION_INVALID",
    "INFRASTRUCTURE_INDETERMINATE",
}
BUILTIN_AGREEMENT_METRIC = "MEAN_PAIRWISE_ORDINAL_AGREEMENT"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def load_json(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return raw, value


def parse_time(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("timestamp must be a string")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must be timezone-aware")
    return parsed


def emit(payload: dict) -> int:
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 0 if payload.get("adjudication_integrity_verified") else 1


def invalid(detail: str, **extra: object) -> int:
    payload = {
        "authority": "MeasurementOnly",
        "disposition": "ADJUDICATION_INVALID",
        "detail": detail,
        "adjudication_integrity_verified": False,
        "objective_moral_truth_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }
    payload.update(extra)
    return emit(payload)


def normalized_rating_counts(values: list[int]) -> dict[str, int]:
    counts = Counter(values)
    return {str(score): counts.get(score, 0) for score in range(5)}


def numeric_equal(left: object, right: float | None, tol: float = 1e-12) -> bool:
    if right is None:
        return left is None
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isclose(float(left), right, rel_tol=tol, abs_tol=tol)
    )


def mean_pairwise_ordinal_agreement(rating_values: dict[str, list[int]]) -> float | None:
    total = 0.0
    pairs = 0
    for values in rating_values.values():
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total += 1.0 - abs(values[i] - values[j]) / 4.0
                pairs += 1
    return total / pairs if pairs else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("rubric", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("reviewer_records", nargs="+", type=Path)
    args = parser.parse_args()

    try:
        plan_bytes, plan = load_json(args.plan)
        rubric_bytes, rubric = load_json(args.rubric)
        result_bytes, result = load_json(args.result)
        reviewer_artifacts = [load_json(path) for path in args.reviewer_records]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return invalid(f"artifact_read_or_parse_failed:{type(exc).__name__}")

    protocol = "wcare35-adjudication-v1"
    if any(value.get("protocol_version") != protocol for value in (plan, rubric, result)):
        return invalid("protocol_version_mismatch")

    plan_sha = sha256_bytes(plan_bytes)
    rubric_sha = sha256_bytes(rubric_bytes)
    if plan.get("rubric_sha256") != rubric_sha or result.get("rubric_sha256") != rubric_sha:
        return invalid("rubric_digest_mismatch")
    if result.get("adjudication_plan_sha256") != plan_sha:
        return invalid("adjudication_plan_digest_mismatch")

    epoch = plan.get("adjudication_epoch")
    case_id = plan.get("case_id")
    if result.get("adjudication_epoch") != epoch or result.get("case_id") != case_id:
        return invalid("result_plan_identity_mismatch")
    if result.get("disposition") not in ALLOWED_DISPOSITIONS:
        return invalid("invalid_result_disposition")
    if result.get("raw_records_preserved") is not True:
        return invalid("raw_records_not_preserved")
    if result.get("consensus_overwrote_raw_ratings") is not False:
        return invalid("consensus_overwrote_raw_ratings")

    try:
        parse_time(plan.get("plan_created_utc"))
    except (TypeError, ValueError):
        return invalid("plan_timestamp_invalid")

    dimensions = rubric.get("dimensions")
    if not isinstance(dimensions, list) or not dimensions:
        return invalid("rubric_dimensions_missing")
    dimension_ids = [entry.get("id") for entry in dimensions if isinstance(entry, dict)]
    if (
        len(dimension_ids) != len(dimensions)
        or any(not isinstance(value, str) or not value for value in dimension_ids)
        or len(set(dimension_ids)) != len(dimension_ids)
    ):
        return invalid("rubric_dimension_ids_invalid")
    if rubric.get("rating_scale", {}).get("min") != 0 or rubric.get("rating_scale", {}).get("max") != 4:
        return invalid("unsupported_rating_scale")
    if (
        rubric.get("aggregate_score_is_moral_truth") is not False
        or rubric.get("dimension_failure_may_be_hidden_by_total") is not False
    ):
        return invalid("rubric_claim_boundary_missing")

    minimum_panel_size = plan.get("minimum_panel_size")
    if (
        not isinstance(minimum_panel_size, int)
        or isinstance(minimum_panel_size, bool)
        or minimum_panel_size < 1
    ):
        return invalid("invalid_minimum_panel_size")

    sought = plan.get("reviewer_classes_sought")
    if (
        not isinstance(sought, list)
        or not sought
        or any(value not in ALLOWED_REVIEWER_CLASSES for value in sought)
        or len(set(sought)) != len(sought)
    ):
        return invalid("invalid_reviewer_classes_sought")

    minimum_class_counts = plan.get("minimum_reviewer_class_counts")
    if not isinstance(minimum_class_counts, dict) or set(minimum_class_counts) != set(sought):
        return invalid("minimum_reviewer_class_counts_mismatch_sought_classes")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 1
        for value in minimum_class_counts.values()
    ):
        return invalid("minimum_reviewer_class_count_invalid")
    if sum(minimum_class_counts.values()) > minimum_panel_size:
        return invalid("minimum_reviewer_class_counts_exceed_panel_minimum")

    require_identity_blinding = plan.get("require_candidate_identity_blinding")
    require_condition_blinding = plan.get("require_condition_blinding")
    if not isinstance(require_identity_blinding, bool) or not isinstance(require_condition_blinding, bool):
        return invalid("blinding_requirement_not_boolean")

    minimum_rated_per_reviewer = plan.get("minimum_rated_dimensions_per_reviewer")
    minimum_ratings_per_dimension = plan.get("minimum_ratings_per_dimension")
    for field, value in (
        ("minimum_rated_dimensions_per_reviewer", minimum_rated_per_reviewer),
        ("minimum_ratings_per_dimension", minimum_ratings_per_dimension),
    ):
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 1
            or value > len(dimension_ids)
        ):
            return invalid(f"{field}_invalid")

    hard_dimensions = plan.get("hard_safety_dimensions")
    if (
        not isinstance(hard_dimensions, list)
        or len(set(hard_dimensions)) != len(hard_dimensions)
        or any(value not in dimension_ids for value in hard_dimensions)
    ):
        return invalid("invalid_hard_safety_dimensions")
    failure_max = plan.get("hard_safety_failure_rating_max")
    if (
        not isinstance(failure_max, int)
        or isinstance(failure_max, bool)
        or not 0 <= failure_max <= 4
    ):
        return invalid("invalid_hard_safety_failure_threshold")

    records_by_id: dict[str, dict] = {}
    raw_hash_by_id: dict[str, str] = {}
    submitted_by_id: dict[str, datetime] = {}
    identity_by_id: dict[str, str] = {}

    for raw, record in reviewer_artifacts:
        if record.get("protocol_version") != protocol:
            return invalid("reviewer_protocol_version_mismatch")
        if record.get("adjudication_epoch") != epoch or record.get("case_id") != case_id:
            return invalid("reviewer_plan_identity_mismatch")
        if (
            record.get("adjudication_plan_sha256") != plan_sha
            or record.get("rubric_sha256") != rubric_sha
        ):
            return invalid("reviewer_plan_or_rubric_digest_mismatch")
        if record.get("original_record_immutable") is not True:
            return invalid("reviewer_record_not_immutable")

        reviewer_id = record.get("reviewer_record_id")
        if not isinstance(reviewer_id, str) or not reviewer_id or reviewer_id in records_by_id:
            return invalid("duplicate_or_invalid_reviewer_record_id")
        identity = record.get("reviewer_identity_commitment_sha256")
        if not is_sha256(identity):
            return invalid("reviewer_identity_commitment_invalid", reviewer_record_id=reviewer_id)
        reviewer_class = record.get("reviewer_class")
        if reviewer_class not in ALLOWED_REVIEWER_CLASSES:
            return invalid("invalid_reviewer_class", reviewer_record_id=reviewer_id)
        if (
            not isinstance(record.get("blinded_to_candidate_identity"), bool)
            or not isinstance(record.get("blinded_to_condition"), bool)
        ):
            return invalid("reviewer_blinding_state_missing", reviewer_record_id=reviewer_id)
        if (
            not isinstance(record.get("unblinding_events"), list)
            or any(not isinstance(value, str) for value in record["unblinding_events"])
        ):
            return invalid("reviewer_unblinding_events_invalid", reviewer_record_id=reviewer_id)
        try:
            submitted = parse_time(record.get("submitted_utc"))
        except (TypeError, ValueError):
            return invalid("reviewer_timestamp_invalid", reviewer_record_id=reviewer_id)

        ratings = record.get("ratings")
        if not isinstance(ratings, list) or len(ratings) != len(dimension_ids):
            return invalid("reviewer_dimension_census_length_mismatch", reviewer_record_id=reviewer_id)
        seen: set[str] = set()
        for item in ratings:
            if not isinstance(item, dict):
                return invalid("reviewer_rating_not_object", reviewer_record_id=reviewer_id)
            dimension = item.get("dimension_id")
            if dimension not in dimension_ids or dimension in seen:
                return invalid("reviewer_dimension_census_invalid", reviewer_record_id=reviewer_id)
            seen.add(dimension)
            status = item.get("status")
            if status == "RATED":
                rating = item.get("rating")
                if (
                    not isinstance(rating, int)
                    or isinstance(rating, bool)
                    or not 0 <= rating <= 4
                ):
                    return invalid("reviewer_rating_invalid", reviewer_record_id=reviewer_id, dimension_id=dimension)
            elif status in {"MISSING", "NOT_APPLICABLE"}:
                if "rating" in item:
                    return invalid("nonrated_dimension_carries_value", reviewer_record_id=reviewer_id, dimension_id=dimension)
            else:
                return invalid("reviewer_rating_status_invalid", reviewer_record_id=reviewer_id, dimension_id=dimension)

        records_by_id[reviewer_id] = record
        raw_hash_by_id[reviewer_id] = sha256_bytes(raw)
        submitted_by_id[reviewer_id] = submitted
        identity_by_id[reviewer_id] = identity

    superseded_by: dict[str, str] = {}
    for reviewer_id, record in records_by_id.items():
        predecessor = record.get("supersedes_record_id")
        if predecessor is None:
            continue
        if not isinstance(predecessor, str) or not predecessor or predecessor == reviewer_id:
            return invalid("invalid_supersedes_record_id", reviewer_record_id=reviewer_id)
        if predecessor not in records_by_id:
            return invalid("superseded_record_missing", reviewer_record_id=reviewer_id)
        if predecessor in superseded_by:
            return invalid("branching_reviewer_correction_chain", supersedes_record_id=predecessor)
        if records_by_id[predecessor].get("reviewer_class") != record.get("reviewer_class"):
            return invalid("reviewer_class_changed_in_correction_chain", reviewer_record_id=reviewer_id)
        if identity_by_id[predecessor] != identity_by_id[reviewer_id]:
            return invalid("reviewer_identity_changed_in_correction_chain", reviewer_record_id=reviewer_id)
        if submitted_by_id[reviewer_id] < submitted_by_id[predecessor]:
            return invalid("reviewer_correction_time_regression", reviewer_record_id=reviewer_id)
        superseded_by[predecessor] = reviewer_id

    for start in records_by_id:
        seen: set[str] = set()
        cursor = start
        while cursor in superseded_by:
            if cursor in seen:
                return invalid("reviewer_correction_cycle", reviewer_record_id=start)
            seen.add(cursor)
            cursor = superseded_by[cursor]
        if cursor in seen:
            return invalid("reviewer_correction_cycle", reviewer_record_id=start)

    active_ids = [reviewer_id for reviewer_id in records_by_id if reviewer_id not in superseded_by]
    root_for_active: dict[str, str] = {}
    for active_id in active_ids:
        cursor = active_id
        visited: set[str] = set()
        while True:
            predecessor = records_by_id[cursor].get("supersedes_record_id")
            if predecessor is None:
                root = cursor
                break
            if predecessor in visited:
                return invalid("reviewer_correction_cycle", reviewer_record_id=active_id)
            visited.add(predecessor)
            cursor = predecessor
        if root in root_for_active:
            return invalid("multiple_active_records_for_reviewer_chain", root_record_id=root)
        root_for_active[root] = active_id

    active_identities = [identity_by_id[reviewer_id] for reviewer_id in active_ids]
    if len(active_identities) != len(set(active_identities)):
        return invalid("duplicate_active_reviewer_identity_commitment")

    all_hashes = list(raw_hash_by_id.values())
    reported_all_hashes = result.get("reviewer_record_sha256s")
    if (
        not isinstance(reported_all_hashes, list)
        or len(reported_all_hashes) != len(set(reported_all_hashes))
        or set(reported_all_hashes) != set(all_hashes)
    ):
        return invalid("raw_reviewer_record_digest_census_mismatch")

    active_hashes = [raw_hash_by_id[reviewer_id] for reviewer_id in active_ids]
    reported_active_hashes = result.get("active_reviewer_record_sha256s")
    if (
        not isinstance(reported_active_hashes, list)
        or len(reported_active_hashes) != len(set(reported_active_hashes))
        or set(reported_active_hashes) != set(active_hashes)
    ):
        return invalid("active_reviewer_record_digest_census_mismatch")
    reported_active_identities = result.get("active_reviewer_identity_commitment_sha256s")
    if (
        not isinstance(reported_active_identities, list)
        or len(reported_active_identities) != len(set(reported_active_identities))
        or set(reported_active_identities) != set(active_identities)
    ):
        return invalid("active_reviewer_identity_census_mismatch")

    reviewer_classes: Counter[str] = Counter()
    rating_values: dict[str, list[int]] = {dimension: [] for dimension in dimension_ids}
    missing_counts: Counter[str] = Counter()
    na_counts: Counter[str] = Counter()
    blinding_problem = False

    for reviewer_id in active_ids:
        record = records_by_id[reviewer_id]
        reviewer_classes[record["reviewer_class"]] += 1
        rated_count = sum(item["status"] == "RATED" for item in record["ratings"])
        if rated_count < minimum_rated_per_reviewer:
            return invalid("reviewer_below_minimum_rated_dimensions", reviewer_record_id=reviewer_id)
        if require_identity_blinding and (
            record["blinded_to_candidate_identity"] is not True or record["unblinding_events"]
        ):
            return invalid("required_candidate_identity_blinding_not_preserved", reviewer_record_id=reviewer_id)
        if require_condition_blinding and (
            record["blinded_to_condition"] is not True or record["unblinding_events"]
        ):
            return invalid("required_condition_blinding_not_preserved", reviewer_record_id=reviewer_id)
        if (
            record["blinded_to_candidate_identity"] is not True
            or record["blinded_to_condition"] is not True
            or record["unblinding_events"]
        ):
            blinding_problem = True

        for item in record["ratings"]:
            dimension = item["dimension_id"]
            status = item["status"]
            if status == "RATED":
                rating_values[dimension].append(item["rating"])
            elif status == "MISSING":
                missing_counts[dimension] += 1
            else:
                na_counts[dimension] += 1

    completed = len(active_ids)
    if result.get("panel_size_committed") != minimum_panel_size:
        return invalid("committed_panel_size_mismatch")
    if result.get("panel_size_completed") != completed:
        return invalid("completed_panel_size_mismatch")

    expected_class_counts = {
        key: reviewer_classes.get(key, 0)
        for key in sorted(ALLOWED_REVIEWER_CLASSES)
        if reviewer_classes.get(key, 0) > 0
    }
    if result.get("reviewer_class_counts") != expected_class_counts:
        return invalid("reviewer_class_counts_mismatch", expected=expected_class_counts)
    for reviewer_class, minimum in minimum_class_counts.items():
        if reviewer_classes.get(reviewer_class, 0) < minimum:
            return invalid(
                "reviewer_class_minimum_not_met",
                reviewer_class=reviewer_class,
                required=minimum,
                observed=reviewer_classes.get(reviewer_class, 0),
            )

    if blinding_problem and not result.get("blinding_limitations"):
        return invalid("blinding_limitations_not_reported")

    summaries = result.get("dimension_summaries")
    if (
        not isinstance(summaries, list)
        or any(not isinstance(entry, dict) for entry in summaries)
        or [entry.get("dimension_id") for entry in summaries] != dimension_ids
    ):
        return invalid("dimension_summary_census_mismatch")

    derived_hard_violations: list[str] = []
    for summary, dimension in zip(summaries, dimension_ids):
        values = rating_values[dimension]
        if len(values) < minimum_ratings_per_dimension:
            return invalid(
                "dimension_below_minimum_ratings",
                dimension_id=dimension,
                required=minimum_ratings_per_dimension,
                observed=len(values),
            )
        expected_counts = normalized_rating_counts(values)
        if summary.get("rating_counts") != expected_counts:
            return invalid("dimension_rating_counts_mismatch", dimension_id=dimension, expected=expected_counts)
        if (
            summary.get("missing_count") != missing_counts[dimension]
            or summary.get("not_applicable_count") != na_counts[dimension]
        ):
            return invalid("dimension_missingness_mismatch", dimension_id=dimension)
        mean = statistics.fmean(values) if values else None
        median = float(statistics.median(values)) if values else None
        if not numeric_equal(summary.get("mean"), mean) or not numeric_equal(summary.get("median"), median):
            return invalid("dimension_summary_statistic_mismatch", dimension_id=dimension)
        if dimension in hard_dimensions and any(value <= failure_max for value in values):
            derived_hard_violations.append(dimension)

    if result.get("hard_safety_violation_dimensions") != derived_hard_violations:
        return invalid("hard_safety_violation_summary_mismatch", expected=derived_hard_violations)

    metric = plan.get("agreement_metric")
    if result.get("agreement_metric") != metric:
        return invalid("agreement_metric_mismatch")
    threshold = plan.get("agreement_threshold")
    direction = plan.get("agreement_threshold_direction")
    reported_value = result.get("agreement_value")
    reported_requirement = result.get("agreement_requirement_met")

    agreement_recomputed = False
    if metric == BUILTIN_AGREEMENT_METRIC:
        agreement_recomputed = True
        computed_value = mean_pairwise_ordinal_agreement(rating_values)
        if not numeric_equal(reported_value, computed_value):
            return invalid("agreement_value_mismatch", expected=computed_value)
        if threshold is None:
            if reported_requirement is not None:
                return invalid("agreement_requirement_should_be_null")
        else:
            if direction not in {"AT_LEAST", "AT_MOST"}:
                return invalid("agreement_threshold_direction_invalid")
            if computed_value is None:
                return invalid("agreement_uncomputable_for_threshold")
            expected_requirement = (
                computed_value >= float(threshold)
                if direction == "AT_LEAST"
                else computed_value <= float(threshold)
            )
            if reported_requirement is not expected_requirement:
                return invalid("agreement_requirement_mismatch", expected=expected_requirement)
    else:
        if reported_value is not None or reported_requirement is not None:
            return invalid("unsupported_agreement_metric_must_remain_unscored", agreement_metric=metric)

    disposition = result.get("disposition")
    if disposition == "ADJUDICATION_SUPPORTED":
        if completed < minimum_panel_size:
            return invalid("supported_below_minimum_panel")
        if derived_hard_violations:
            return invalid("supported_with_hard_safety_violation")
        if not agreement_recomputed:
            return invalid("supported_with_unverified_agreement_metric")
        if threshold is not None and reported_requirement is not True:
            return invalid("supported_without_required_agreement")
    if (
        disposition == "ADJUDICATION_CONTESTED"
        and threshold is not None
        and agreement_recomputed
        and reported_requirement is True
        and not result.get("notes")
    ):
        return invalid("contested_despite_agreement_requires_explanation")

    return emit({
        "authority": "MeasurementOnly",
        "disposition": disposition,
        "adjudication_epoch": epoch,
        "case_id": case_id,
        "plan_sha256": plan_sha,
        "rubric_sha256": rubric_sha,
        "result_sha256": sha256_bytes(result_bytes),
        "raw_record_count": len(records_by_id),
        "active_reviewer_count": completed,
        "reviewer_class_counts": expected_class_counts,
        "hard_safety_violation_dimensions": derived_hard_violations,
        "agreement_metric": metric,
        "agreement_recomputed": agreement_recomputed,
        "agreement_requirement_met": reported_requirement,
        "adjudication_integrity_verified": True,
        "objective_moral_truth_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    })


if __name__ == "__main__":
    raise SystemExit(main())
