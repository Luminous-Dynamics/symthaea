#!/usr/bin/env python3
"""Verify WCARE-36 reviewer provenance and panel independence evidence.

This tool qualifies evidentiary independence only. It does not alter WCARE-35 ratings,
establish moral truth, or grant runtime authority.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

PROTOCOL = "wcare36-reviewer-provenance-v1"
PROVENANCE_STRENGTHS = {
    "SelfDeclared",
    "OrganizerVerified",
    "ExternalVerified",
    "InstitutionalAttestation",
    "ModelSessionProvenance",
}
RELATION_EVIDENCE_STRENGTHS = {
    "SelfDeclared",
    "OrganizerAssessed",
    "ExternalVerified",
    "InstitutionalAttestation",
    "ModelAssessment",
}
ISSUER_CLASSES = {
    "Self",
    "PanelOrganizer",
    "ExternalOrganization",
    "Institution",
    "ModelRuntime",
}
RELATIONS = {
    "Independent",
    "Related",
    "SameLineage",
    "Unknown",
    "ConflictOfInterest",
}
WCARE35_QUALIFIABLE = {
    "ADJUDICATION_SUPPORTED",
    "ADJUDICATION_CONTESTED",
}


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
    return 0 if payload.get("independence_integrity_verified") else 1


def invalid(detail: str, **extra: object) -> int:
    payload = {
        "authority": "MeasurementOnly",
        "disposition": "INDEPENDENCE_INVALID",
        "detail": detail,
        "independence_integrity_verified": False,
        "objective_moral_truth_established": False,
        "universal_cultural_validity_established": False,
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


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def canonical_components(
    identities: list[str], adjacency: dict[str, set[str]]
) -> list[list[str]]:
    remaining = set(identities)
    components: list[list[str]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        remaining -= component
        components.append(sorted(component))
    components.sort(key=lambda values: values[0])
    return components


def validate_string_set(
    value: object,
    allowed: set[str],
    *,
    field: str,
) -> tuple[set[str] | None, int | None]:
    if (
        not isinstance(value, list)
        or not value
        or len(value) != len(set(value))
        or any(item not in allowed for item in value)
    ):
        return None, invalid(f"{field}_invalid")
    return set(value), None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("wcare35_result", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--provenance", nargs="+", required=True, type=Path)
    parser.add_argument("--relations", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    try:
        plan_bytes, plan = load_json(args.plan)
        w35_bytes, w35 = load_json(args.wcare35_result)
        result_bytes, result = load_json(args.result)
        provenance_artifacts = [load_json(path) for path in args.provenance]
        relation_artifacts = [load_json(path) for path in args.relations]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return invalid(f"artifact_read_or_parse_failed:{type(exc).__name__}")

    if plan.get("protocol_version") != PROTOCOL or result.get("protocol_version") != PROTOCOL:
        return invalid("protocol_version_mismatch")

    w35_sha = sha256_bytes(w35_bytes)
    plan_sha = sha256_bytes(plan_bytes)
    if plan.get("wcare35_result_sha256") != w35_sha:
        return invalid("plan_wcare35_digest_mismatch")
    if result.get("wcare35_result_sha256") != w35_sha:
        return invalid("result_wcare35_digest_mismatch")
    if result.get("plan_sha256") != plan_sha:
        return invalid("result_plan_digest_mismatch")

    epoch = w35.get("adjudication_epoch")
    if plan.get("adjudication_epoch") != epoch or result.get("adjudication_epoch") != epoch:
        return invalid("adjudication_epoch_mismatch")
    if w35.get("disposition") not in WCARE35_QUALIFIABLE:
        return invalid(
            "wcare35_result_not_qualifiable",
            wcare35_disposition=w35.get("disposition"),
        )

    try:
        parse_time(plan.get("plan_created_utc"))
    except (TypeError, ValueError):
        return invalid("plan_timestamp_invalid")

    active = w35.get("active_reviewer_identity_commitment_sha256s")
    if (
        not isinstance(active, list)
        or not active
        or len(active) != len(set(active))
        or any(not is_sha256(value) for value in active)
    ):
        return invalid("wcare35_active_identity_census_invalid")
    active = sorted(active)
    active_set = set(active)
    if result.get("active_reviewer_identity_commitment_sha256s") != active:
        return invalid("result_active_identity_census_mismatch")
    if result.get("active_reviewer_count") != len(active):
        return invalid("result_active_reviewer_count_mismatch")

    minimum_components = plan.get("minimum_effective_independent_components")
    minimum_lineages = plan.get("minimum_distinct_lineages")
    max_unknown = plan.get("maximum_unknown_relation_pairs")
    require_no_conflict = plan.get("require_no_conflict_of_interest")
    for name, value in (
        ("minimum_effective_independent_components", minimum_components),
        ("minimum_distinct_lineages", minimum_lineages),
        ("maximum_unknown_relation_pairs", max_unknown),
    ):
        floor = 0 if name == "maximum_unknown_relation_pairs" else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < floor:
            return invalid(f"{name}_invalid")
    if minimum_components > len(active) or minimum_lineages > len(active):
        return invalid("plan_minimum_exceeds_active_reviewer_count")
    if not isinstance(require_no_conflict, bool):
        return invalid("require_no_conflict_of_interest_not_boolean")

    accepted_independent_strengths, error = validate_string_set(
        plan.get("accepted_independent_relation_strengths"),
        RELATION_EVIDENCE_STRENGTHS,
        field="accepted_independent_relation_strengths",
    )
    if error is not None:
        return error
    accepted_lineage_strengths, error = validate_string_set(
        plan.get("accepted_lineage_provenance_strengths"),
        PROVENANCE_STRENGTHS,
        field="accepted_lineage_provenance_strengths",
    )
    if error is not None:
        return error
    assert accepted_independent_strengths is not None
    assert accepted_lineage_strengths is not None

    minimum_strengths = plan.get("minimum_provenance_strength_counts")
    if (
        not isinstance(minimum_strengths, dict)
        or any(key not in PROVENANCE_STRENGTHS for key in minimum_strengths)
    ):
        return invalid("minimum_provenance_strength_counts_invalid")
    for key, value in minimum_strengths.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            return invalid(
                "minimum_provenance_strength_count_invalid",
                provenance_strength=key,
            )

    provenance_by_identity: dict[str, dict] = {}
    provenance_hashes: list[str] = []
    provenance_ids: set[str] = set()
    provenance_strength_by_identity: dict[str, str] = {}
    strength_counts = Counter()
    lineage_by_identity: dict[str, str] = {}
    conflict_identities: set[str] = set()

    for raw, receipt in provenance_artifacts:
        if receipt.get("protocol_version") != PROTOCOL:
            return invalid("provenance_protocol_version_mismatch")
        if (
            receipt.get("adjudication_epoch") != epoch
            or receipt.get("wcare35_result_sha256") != w35_sha
        ):
            return invalid("provenance_subject_binding_mismatch")
        identity = receipt.get("reviewer_identity_commitment_sha256")
        if identity not in active_set or not is_sha256(identity):
            return invalid("provenance_identity_not_active")
        if identity in provenance_by_identity:
            return invalid(
                "duplicate_provenance_receipt_for_identity",
                reviewer_identity_commitment_sha256=identity,
            )
        receipt_id = receipt.get("provenance_receipt_id")
        if (
            not isinstance(receipt_id, str)
            or not receipt_id
            or receipt_id in provenance_ids
        ):
            return invalid("duplicate_or_invalid_provenance_receipt_id")
        provenance_ids.add(receipt_id)

        strength = receipt.get("provenance_strength")
        issuer_class = receipt.get("issuer_class")
        if strength not in PROVENANCE_STRENGTHS or issuer_class not in ISSUER_CLASSES:
            return invalid(
                "invalid_provenance_strength_or_issuer",
                provenance_receipt_id=receipt_id,
            )
        for field in (
            "issuer_commitment_sha256",
            "lineage_commitment_sha256",
            "evidence_sha256",
        ):
            if not is_sha256(receipt.get(field)):
                return invalid(
                    "invalid_provenance_digest",
                    provenance_receipt_id=receipt_id,
                    field=field,
                )
        if not isinstance(receipt.get("conflict_of_interest"), bool):
            return invalid(
                "conflict_of_interest_not_boolean",
                provenance_receipt_id=receipt_id,
            )
        if not isinstance(receipt.get("conflict_note"), str):
            return invalid(
                "conflict_note_not_string",
                provenance_receipt_id=receipt_id,
            )
        try:
            parse_time(receipt.get("created_utc"))
        except (TypeError, ValueError):
            return invalid(
                "provenance_timestamp_invalid",
                provenance_receipt_id=receipt_id,
            )

        provenance_by_identity[identity] = receipt
        provenance_hashes.append(sha256_bytes(raw))
        provenance_strength_by_identity[identity] = strength
        strength_counts[strength] += 1
        lineage_by_identity[identity] = receipt["lineage_commitment_sha256"]
        if receipt["conflict_of_interest"]:
            conflict_identities.add(identity)

    if set(provenance_by_identity) != active_set:
        return invalid("incomplete_provenance_identity_census")

    relation_hashes: list[str] = []
    relation_ids: set[str] = set()
    relation_by_pair: dict[tuple[str, str], tuple[str, str]] = {}
    relation_counts = Counter()
    relation_strength_counts = Counter()

    for raw, receipt in relation_artifacts:
        if receipt.get("protocol_version") != PROTOCOL:
            return invalid("relation_protocol_version_mismatch")
        if (
            receipt.get("adjudication_epoch") != epoch
            or receipt.get("wcare35_result_sha256") != w35_sha
        ):
            return invalid("relation_subject_binding_mismatch")
        left = receipt.get("left_reviewer_identity_commitment_sha256")
        right = receipt.get("right_reviewer_identity_commitment_sha256")
        if left not in active_set or right not in active_set:
            return invalid("relation_identity_not_active")
        if left == right:
            return invalid("relation_self_pair")

        relation = receipt.get("relation")
        relation_strength = receipt.get("relation_evidence_strength")
        if relation not in RELATIONS:
            return invalid("invalid_relation_class")
        if relation_strength not in RELATION_EVIDENCE_STRENGTHS:
            return invalid("invalid_relation_evidence_strength")

        receipt_id = receipt.get("relation_receipt_id")
        if (
            not isinstance(receipt_id, str)
            or not receipt_id
            or receipt_id in relation_ids
        ):
            return invalid("duplicate_or_invalid_relation_receipt_id")
        relation_ids.add(receipt_id)
        for field in ("basis_sha256", "assessor_commitment_sha256"):
            if not is_sha256(receipt.get(field)):
                return invalid(
                    "invalid_relation_digest",
                    relation_receipt_id=receipt_id,
                    field=field,
                )
        try:
            parse_time(receipt.get("created_utc"))
        except (TypeError, ValueError):
            return invalid(
                "relation_timestamp_invalid",
                relation_receipt_id=receipt_id,
            )

        pair = canonical_pair(left, right)
        if pair in relation_by_pair:
            return invalid("duplicate_relation_pair", left=pair[0], right=pair[1])

        same_lineage = lineage_by_identity[left] == lineage_by_identity[right]
        if same_lineage and relation == "Independent":
            return invalid(
                "same_lineage_pair_marked_independent",
                left=pair[0],
                right=pair[1],
            )
        if relation == "SameLineage" and not same_lineage:
            return invalid(
                "same_lineage_relation_without_matching_lineage_commitment",
                left=pair[0],
                right=pair[1],
            )

        relation_by_pair[pair] = (relation, relation_strength)
        relation_counts[relation] += 1
        relation_strength_counts[relation_strength] += 1
        relation_hashes.append(sha256_bytes(raw))
        if relation == "ConflictOfInterest":
            conflict_identities.update(pair)

    expected_pairs = {
        (active[i], active[j])
        for i in range(len(active))
        for j in range(i + 1, len(active))
    }
    if set(relation_by_pair) != expected_pairs:
        missing = sorted(expected_pairs - set(relation_by_pair))
        extra = sorted(set(relation_by_pair) - expected_pairs)
        return invalid(
            "incomplete_or_extra_relation_census",
            missing_pairs=missing,
            extra_pairs=extra,
        )

    qualified_lineage_identities = {
        identity
        for identity in active
        if provenance_strength_by_identity[identity] in accepted_lineage_strengths
    }
    qualified_lineage_reviewer_count = len(qualified_lineage_identities)
    distinct_lineages = len(set(lineage_by_identity.values()))
    qualified_distinct_lineages = len(
        {lineage_by_identity[identity] for identity in qualified_lineage_identities}
    )

    adjacency = {identity: set() for identity in active}
    accepted_independent_pairs = 0
    downgraded_independent_pairs = 0

    for pair, (relation, relation_strength) in relation_by_pair.items():
        left, right = pair
        if relation != "Independent":
            adjacency[left].add(right)
            adjacency[right].add(left)
            continue

        relation_strength_ok = relation_strength in accepted_independent_strengths
        left_lineage_ok = left in qualified_lineage_identities
        right_lineage_ok = right in qualified_lineage_identities
        lineage_distinct = lineage_by_identity[left] != lineage_by_identity[right]
        can_separate = (
            relation_strength_ok
            and left_lineage_ok
            and right_lineage_ok
            and lineage_distinct
        )
        if can_separate:
            accepted_independent_pairs += 1
        else:
            downgraded_independent_pairs += 1
            adjacency[left].add(right)
            adjacency[right].add(left)

    components = canonical_components(active, adjacency)
    effective_components = len(components)
    unknown_pairs = relation_counts["Unknown"]

    expected_strength_counts = {
        key: strength_counts.get(key, 0)
        for key in sorted(PROVENANCE_STRENGTHS)
    }
    expected_relation_counts = {
        key: relation_counts.get(key, 0)
        for key in sorted(RELATIONS)
    }
    expected_relation_strength_counts = {
        key: relation_strength_counts.get(key, 0)
        for key in sorted(RELATION_EVIDENCE_STRENGTHS)
    }

    if result.get("provenance_receipt_sha256s") != sorted(provenance_hashes):
        return invalid("provenance_receipt_digest_census_mismatch")
    if result.get("relation_receipt_sha256s") != sorted(relation_hashes):
        return invalid("relation_receipt_digest_census_mismatch")
    if result.get("provenance_strength_counts") != expected_strength_counts:
        return invalid(
            "provenance_strength_counts_mismatch",
            expected=expected_strength_counts,
        )
    if result.get("relation_counts") != expected_relation_counts:
        return invalid("relation_counts_mismatch", expected=expected_relation_counts)
    if (
        result.get("relation_evidence_strength_counts")
        != expected_relation_strength_counts
    ):
        return invalid(
            "relation_evidence_strength_counts_mismatch",
            expected=expected_relation_strength_counts,
        )
    if result.get("unknown_relation_pair_count") != unknown_pairs:
        return invalid("unknown_relation_pair_count_mismatch")
    if result.get("distinct_lineage_count") != distinct_lineages:
        return invalid("distinct_lineage_count_mismatch")
    if result.get("qualified_distinct_lineage_count") != qualified_distinct_lineages:
        return invalid("qualified_distinct_lineage_count_mismatch")
    if (
        result.get("qualified_lineage_reviewer_count")
        != qualified_lineage_reviewer_count
    ):
        return invalid("qualified_lineage_reviewer_count_mismatch")
    if result.get("accepted_independent_pair_count") != accepted_independent_pairs:
        return invalid("accepted_independent_pair_count_mismatch")
    if (
        result.get("downgraded_independent_pair_count")
        != downgraded_independent_pairs
    ):
        return invalid("downgraded_independent_pair_count_mismatch")
    if result.get("effective_independent_components") != effective_components:
        return invalid("effective_independent_component_count_mismatch")
    if result.get("independence_component_census") != components:
        return invalid(
            "independence_component_census_mismatch",
            expected=components,
        )
    expected_conflicts = sorted(conflict_identities)
    if result.get("conflict_identity_commitment_sha256s") != expected_conflicts:
        return invalid("conflict_identity_census_mismatch")

    minimum_effective_components_met = effective_components >= minimum_components
    minimum_distinct_lineages_met = qualified_distinct_lineages >= minimum_lineages
    minimum_provenance_strengths_met = all(
        strength_counts.get(key, 0) >= value
        for key, value in minimum_strengths.items()
    )
    unknown_pairs_within_limit = unknown_pairs <= max_unknown
    conflict_policy_met = (not conflict_identities) if require_no_conflict else True
    requirements_met = all(
        (
            minimum_effective_components_met,
            minimum_distinct_lineages_met,
            minimum_provenance_strengths_met,
            unknown_pairs_within_limit,
            conflict_policy_met,
        )
    )

    expected_gates = {
        "minimum_effective_components_met": minimum_effective_components_met,
        "minimum_distinct_lineages_met": minimum_distinct_lineages_met,
        "minimum_provenance_strengths_met": minimum_provenance_strengths_met,
        "unknown_pairs_within_limit": unknown_pairs_within_limit,
        "conflict_policy_met": conflict_policy_met,
        "requirements_met": requirements_met,
    }
    for field, expected in expected_gates.items():
        if result.get(field) is not expected:
            return invalid(
                "reported_independence_gate_mismatch",
                field=field,
                expected=expected,
            )

    expected_disposition = (
        "INDEPENDENCE_SUPPORTED"
        if requirements_met
        else "INDEPENDENCE_LIMITED"
    )
    if result.get("disposition") != expected_disposition:
        return invalid(
            "independence_disposition_mismatch",
            expected=expected_disposition,
        )

    return emit({
        "authority": "MeasurementOnly",
        "disposition": expected_disposition,
        "adjudication_epoch": epoch,
        "wcare35_result_sha256": w35_sha,
        "plan_sha256": plan_sha,
        "result_sha256": sha256_bytes(result_bytes),
        "active_reviewer_count": len(active),
        "distinct_lineage_count": distinct_lineages,
        "qualified_distinct_lineage_count": qualified_distinct_lineages,
        "qualified_lineage_reviewer_count": qualified_lineage_reviewer_count,
        "effective_independent_components": effective_components,
        "accepted_independent_pair_count": accepted_independent_pairs,
        "downgraded_independent_pair_count": downgraded_independent_pairs,
        "unknown_relation_pair_count": unknown_pairs,
        "conflict_identity_commitment_sha256s": expected_conflicts,
        "requirements_met": requirements_met,
        "independence_integrity_verified": True,
        "reviewer_headcount_equals_independent_evidence_count": False,
        "objective_moral_truth_established": False,
        "universal_cultural_validity_established": False,
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
