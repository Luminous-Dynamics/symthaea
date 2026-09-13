#!/usr/bin/env python3
"""Qualify WCARE-36 panel evidence under exact WCARE-37 authentication.

WCARE-38 is MeasurementOnly. It never rewrites WCARE-35/WCARE-36 evidence and
it never trusts a copied WCARE-37 `ATTESTATION_ACCEPTED` JSON. Required
attestations are re-executed through the exact WCARE-37 qualifier.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

PROTOCOL = "wcare38-authenticated-panel-v1"
W35_PROTOCOL = "wcare35-adjudication-v1"
W36_PROTOCOL = "wcare36-reviewer-provenance-v1"
PROVENANCE_STRENGTHS = {
    "SelfDeclared",
    "OrganizerVerified",
    "ExternalVerified",
    "InstitutionalAttestation",
    "ModelSessionProvenance",
}
RELATION_STRENGTHS = {
    "SelfDeclared",
    "OrganizerAssessed",
    "ExternalVerified",
    "InstitutionalAttestation",
    "ModelAssessment",
}
RELATIONS = {"Independent", "Related", "SameLineage", "Unknown", "ConflictOfInterest"}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> tuple[bytes, dict]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return raw, value


def parse_utc(value: object) -> datetime:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise ValueError("timestamp must be canonical YYYY-MM-DDTHH:MM:SSZ")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("timestamp is not canonical")
    return parsed


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(ch in "0123456789abcdef" for ch in value)
    )


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def components(identities: list[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    remaining = set(identities)
    output: list[list[str]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node] - seen)
        remaining -= seen
        output.append(sorted(seen))
    output.sort(key=lambda values: values[0])
    return output


def invalid(detail: str, **extra: object) -> int:
    payload = {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "disposition": "AUTHENTICATION_INVALID",
        "detail": detail,
        "reviewer_correctness_established": False,
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
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 4


def safe_package_path(base: Path, value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"invalid_{field}")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe_{field}")
    return base / relative


def classify_w37_execution(
    qualifier: Path,
    package_base: Path,
    entry: dict,
    w36_path: Path,
    w35_path: Path,
    subject_path: Path,
    evaluation_utc: str,
) -> tuple[str, str | None, str]:
    """Return (Authenticated|Unauthenticated|Indeterminate|Invalid, receipt_sha, detail)."""
    try:
        envelope = safe_package_path(package_base, entry.get("envelope_path"), "envelope_path")
        policy = safe_package_path(package_base, entry.get("issuer_policy_path"), "issuer_policy_path")
    except ValueError as exc:
        return "Invalid", None, str(exc)
    if not envelope.is_file() or not policy.is_file():
        return "Unauthenticated", None, "attestation_package_file_missing"

    try:
        completed = subprocess.run(
            [
                "bash",
                str(qualifier),
                str(envelope),
                str(policy),
                str(w36_path),
                str(w35_path),
                str(subject_path),
                evaluation_utc,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return "Indeterminate", None, f"wcare37_execution_failed:{type(exc).__name__}"

    stdout = completed.stdout
    receipt_sha = sha256_bytes(stdout) if stdout else None
    try:
        parsed = json.loads(stdout)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return "Indeterminate", receipt_sha, f"wcare37_nonjson_output:exit_{completed.returncode}"
    if not isinstance(parsed, dict):
        return "Indeterminate", receipt_sha, "wcare37_output_not_object"

    disposition = parsed.get("disposition")
    classification = parsed.get("classification")
    if completed.returncode == 0 and disposition == "ATTESTATION_ACCEPTED":
        return "Authenticated", receipt_sha, "ATTESTATION_ACCEPTED"
    if completed.returncode == 2 and disposition == "SIGNATURE_VALID_ISSUER_UNTRUSTED":
        return "Unauthenticated", receipt_sha, "SIGNATURE_VALID_ISSUER_UNTRUSTED"
    if completed.returncode == 1 and disposition == "ATTESTATION_REJECTED":
        return "Unauthenticated", receipt_sha, "ATTESTATION_REJECTED"
    if completed.returncode == 3 or classification == "INFRASTRUCTURE_INDETERMINATE":
        return "Indeterminate", receipt_sha, parsed.get("detail", "WCARE37_INDETERMINATE")
    if completed.returncode == 4 or classification == "INVALID_PROTOCOL":
        return "Invalid", receipt_sha, parsed.get("detail", "WCARE37_INVALID_PROTOCOL")
    return "Indeterminate", receipt_sha, f"unexpected_wcare37_result:exit_{completed.returncode}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("wcare38_plan", type=Path)
    parser.add_argument("wcare35_result", type=Path)
    parser.add_argument("wcare36_plan", type=Path)
    parser.add_argument("wcare36_result", type=Path)
    parser.add_argument("attestation_manifest", type=Path)
    parser.add_argument("--provenance", nargs="+", required=True, type=Path)
    parser.add_argument("--relations", nargs="+", required=True, type=Path)
    args = parser.parse_args()

    try:
        plan_raw, plan = load_json(args.wcare38_plan)
        w35_raw, w35 = load_json(args.wcare35_result)
        w36_plan_raw, w36_plan = load_json(args.wcare36_plan)
        w36_result_raw, w36_result = load_json(args.wcare36_result)
        manifest_raw, manifest = load_json(args.attestation_manifest)
        provenance_artifacts = [(path, *load_json(path)) for path in args.provenance]
        relation_artifacts = [(path, *load_json(path)) for path in args.relations]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return invalid(f"artifact_read_or_parse_failed:{type(exc).__name__}")

    w35_sha = sha256_bytes(w35_raw)
    w36_plan_sha = sha256_bytes(w36_plan_raw)
    w36_result_sha = sha256_bytes(w36_result_raw)
    w38_plan_sha = sha256_bytes(plan_raw)
    manifest_sha = sha256_bytes(manifest_raw)

    if plan.get("protocol_version") != PROTOCOL or manifest.get("protocol_version") != PROTOCOL:
        return invalid("protocol_version_mismatch")
    if w35.get("protocol_version") != W35_PROTOCOL or w36_plan.get("protocol_version") != W36_PROTOCOL or w36_result.get("protocol_version") != W36_PROTOCOL:
        return invalid("upstream_protocol_version_mismatch")
    if plan.get("wcare35_result_sha256") != w35_sha or plan.get("wcare36_plan_sha256") != w36_plan_sha or plan.get("wcare36_result_sha256") != w36_result_sha:
        return invalid("plan_subject_digest_mismatch")
    if manifest.get("wcare36_result_sha256") != w36_result_sha:
        return invalid("manifest_wcare36_digest_mismatch")
    if w36_result.get("wcare35_result_sha256") != w35_sha or w36_result.get("plan_sha256") != w36_plan_sha or w36_plan.get("wcare35_result_sha256") != w35_sha:
        return invalid("wcare36_subject_binding_mismatch")
    if plan.get("authenticate_all_wcare36_qualified_lineages") is not True or plan.get("authenticate_all_wcare36_accepted_independent_edges") is not True:
        return invalid("fixed_v1_authentication_rule_disabled")

    try:
        parse_utc(plan.get("plan_created_utc"))
        evaluation_utc = plan.get("evaluation_utc")
        parse_utc(evaluation_utc)
    except (TypeError, ValueError):
        return invalid("plan_timestamp_invalid")

    root = Path(__file__).resolve().parent.parent
    w37_qualifier = root / "scripts" / "wcare37-qualify.sh"
    w37_protocol = root / "docs" / "release" / "evidence" / "WCARE37_ATTESTATION_PROTOCOL_V1.md"
    try:
        qualifier_sha = sha256_bytes(w37_qualifier.read_bytes())
        protocol_sha = sha256_bytes(w37_protocol.read_bytes())
    except OSError:
        return invalid("wcare37_subject_files_unavailable")
    if plan.get("wcare37_qualifier_sha256") != qualifier_sha or plan.get("wcare37_protocol_sha256") != protocol_sha:
        return invalid("wcare37_subject_digest_mismatch", observed_qualifier_sha256=qualifier_sha, observed_protocol_sha256=protocol_sha)

    active = w35.get("active_reviewer_identity_commitment_sha256s")
    if not isinstance(active, list) or not active or len(active) != len(set(active)) or any(not is_sha256(value) for value in active):
        return invalid("wcare35_active_identity_census_invalid")
    active = sorted(active)
    active_set = set(active)
    if w36_result.get("active_reviewer_identity_commitment_sha256s") != active:
        return invalid("wcare36_active_identity_census_mismatch")

    accepted_lineage_strengths = w36_plan.get("accepted_lineage_provenance_strengths")
    accepted_relation_strengths = w36_plan.get("accepted_independent_relation_strengths")
    if not isinstance(accepted_lineage_strengths, list) or not accepted_lineage_strengths or len(accepted_lineage_strengths) != len(set(accepted_lineage_strengths)) or any(value not in PROVENANCE_STRENGTHS for value in accepted_lineage_strengths):
        return invalid("wcare36_accepted_lineage_strengths_invalid")
    if not isinstance(accepted_relation_strengths, list) or not accepted_relation_strengths or len(accepted_relation_strengths) != len(set(accepted_relation_strengths)) or any(value not in RELATION_STRENGTHS for value in accepted_relation_strengths):
        return invalid("wcare36_accepted_relation_strengths_invalid")
    accepted_lineage_strengths = set(accepted_lineage_strengths)
    accepted_relation_strengths = set(accepted_relation_strengths)

    minimum_strengths = w36_plan.get("minimum_provenance_strength_counts")
    if not isinstance(minimum_strengths, dict) or any(key not in PROVENANCE_STRENGTHS or not isinstance(value, int) or isinstance(value, bool) or value < 0 for key, value in minimum_strengths.items()):
        return invalid("wcare36_minimum_provenance_strengths_invalid")

    provenance_by_identity: dict[str, dict] = {}
    provenance_path_by_hash: dict[str, Path] = {}
    provenance_hash_by_identity: dict[str, str] = {}
    lineage_by_identity: dict[str, str] = {}
    strength_by_identity: dict[str, str] = {}
    strength_counts = Counter()
    conflicts: set[str] = set()

    for path, raw, receipt in provenance_artifacts:
        digest = sha256_bytes(raw)
        identity = receipt.get("reviewer_identity_commitment_sha256")
        strength = receipt.get("provenance_strength")
        lineage = receipt.get("lineage_commitment_sha256")
        if receipt.get("protocol_version") != W36_PROTOCOL or receipt.get("wcare35_result_sha256") != w35_sha:
            return invalid("provenance_subject_binding_mismatch", receipt_sha256=digest)
        if identity not in active_set or identity in provenance_by_identity or strength not in PROVENANCE_STRENGTHS or not is_sha256(lineage):
            return invalid("invalid_or_duplicate_provenance_receipt", receipt_sha256=digest)
        provenance_by_identity[identity] = receipt
        provenance_path_by_hash[digest] = path
        provenance_hash_by_identity[identity] = digest
        lineage_by_identity[identity] = lineage
        strength_by_identity[identity] = strength
        strength_counts[strength] += 1
        if receipt.get("conflict_of_interest") is True:
            conflicts.add(identity)

    if set(provenance_by_identity) != active_set:
        return invalid("incomplete_provenance_identity_census")
    if w36_result.get("provenance_receipt_sha256s") != sorted(provenance_path_by_hash):
        return invalid("wcare36_provenance_digest_census_mismatch")

    relation_by_pair: dict[tuple[str, str], tuple[dict, str, Path]] = {}
    relation_hashes: set[str] = set()
    relation_counts = Counter()
    unknown_pairs = 0
    for path, raw, receipt in relation_artifacts:
        digest = sha256_bytes(raw)
        left = receipt.get("left_reviewer_identity_commitment_sha256")
        right = receipt.get("right_reviewer_identity_commitment_sha256")
        relation = receipt.get("relation")
        strength = receipt.get("relation_evidence_strength")
        if receipt.get("protocol_version") != W36_PROTOCOL or receipt.get("wcare35_result_sha256") != w35_sha:
            return invalid("relation_subject_binding_mismatch", receipt_sha256=digest)
        if left not in active_set or right not in active_set or left == right or relation not in RELATIONS or strength not in RELATION_STRENGTHS:
            return invalid("invalid_relation_receipt", receipt_sha256=digest)
        pair = canonical_pair(left, right)
        if pair in relation_by_pair:
            return invalid("duplicate_relation_pair", left=pair[0], right=pair[1])
        relation_by_pair[pair] = (receipt, digest, path)
        relation_hashes.add(digest)
        relation_counts[relation] += 1
        if relation == "Unknown":
            unknown_pairs += 1
        if relation == "ConflictOfInterest":
            conflicts.update(pair)

    expected_pairs = {(active[i], active[j]) for i in range(len(active)) for j in range(i + 1, len(active))}
    if set(relation_by_pair) != expected_pairs:
        return invalid("incomplete_relation_pair_census")
    if w36_result.get("relation_receipt_sha256s") != sorted(relation_hashes):
        return invalid("wcare36_relation_digest_census_mismatch")

    qualified_identities = {identity for identity in active if strength_by_identity[identity] in accepted_lineage_strengths}
    w36_qualified_lineage_reviewers = len(qualified_identities)
    w36_qualified_lineages = len({lineage_by_identity[identity] for identity in qualified_identities})
    baseline_adjacency = {identity: set() for identity in active}
    baseline_accepted_pairs: set[tuple[str, str]] = set()
    baseline_downgraded_pairs = 0

    for pair, (receipt, _digest, _path) in relation_by_pair.items():
        left, right = pair
        relation = receipt["relation"]
        strength = receipt["relation_evidence_strength"]
        if relation != "Independent":
            baseline_adjacency[left].add(right)
            baseline_adjacency[right].add(left)
            continue
        can_separate = (
            strength in accepted_relation_strengths
            and left in qualified_identities
            and right in qualified_identities
            and lineage_by_identity[left] != lineage_by_identity[right]
        )
        if can_separate:
            baseline_accepted_pairs.add(pair)
        else:
            baseline_downgraded_pairs += 1
            baseline_adjacency[left].add(right)
            baseline_adjacency[right].add(left)

    baseline_components = components(active, baseline_adjacency)
    if w36_result.get("qualified_lineage_reviewer_count") != w36_qualified_lineage_reviewers or w36_result.get("qualified_distinct_lineage_count") != w36_qualified_lineages or w36_result.get("accepted_independent_pair_count") != len(baseline_accepted_pairs) or w36_result.get("downgraded_independent_pair_count") != baseline_downgraded_pairs or w36_result.get("effective_independent_components") != len(baseline_components) or w36_result.get("independence_component_census") != baseline_components:
        return invalid("wcare36_upper_bound_recomputation_mismatch")

    minimum_required_strengths = {strength for strength, count in minimum_strengths.items() if count > 0}
    required_provenance_hashes = {
        provenance_hash_by_identity[identity]
        for identity in active
        if identity in qualified_identities or strength_by_identity[identity] in minimum_required_strengths
    }
    required_relation_hashes = {relation_by_pair[pair][1] for pair in baseline_accepted_pairs}

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        return invalid("manifest_entries_missing")
    manifest_by_subject: dict[str, dict] = {}
    all_receipt_hashes = set(provenance_path_by_hash) | relation_hashes
    for entry in entries:
        if not isinstance(entry, dict):
            return invalid("manifest_entry_not_object")
        subject_hash = entry.get("subject_receipt_sha256")
        if not is_sha256(subject_hash) or subject_hash not in all_receipt_hashes:
            return invalid("manifest_subject_not_wcare36_receipt", subject_receipt_sha256=subject_hash)
        if subject_hash in manifest_by_subject:
            return invalid("duplicate_attestation_package_for_receipt", subject_receipt_sha256=subject_hash)
        manifest_by_subject[subject_hash] = entry

    authenticated_provenance: set[str] = set()
    unauthenticated_provenance: set[str] = set()
    indeterminate_provenance: set[str] = set()
    authenticated_relations: set[str] = set()
    unauthenticated_relations: set[str] = set()
    indeterminate_relations: set[str] = set()
    execution_receipts: set[str] = set()
    downgrade_reasons: list[str] = []
    invalid_execution = False

    manifest_base = args.attestation_manifest.resolve().parent

    def evaluate_required(subject_hash: str, subject_path: Path, kind: str) -> None:
        nonlocal invalid_execution
        entry = manifest_by_subject.get(subject_hash)
        if entry is None:
            state, receipt_sha, detail = "Unauthenticated", None, "attestation_package_missing"
        else:
            state, receipt_sha, detail = classify_w37_execution(
                w37_qualifier,
                manifest_base,
                entry,
                args.wcare36_result,
                args.wcare35_result,
                subject_path,
                evaluation_utc,
            )
        if receipt_sha is not None:
            execution_receipts.add(receipt_sha)
        target_authenticated = authenticated_provenance if kind == "provenance" else authenticated_relations
        target_unauthenticated = unauthenticated_provenance if kind == "provenance" else unauthenticated_relations
        target_indeterminate = indeterminate_provenance if kind == "provenance" else indeterminate_relations
        if state == "Authenticated":
            target_authenticated.add(subject_hash)
        elif state == "Unauthenticated":
            target_unauthenticated.add(subject_hash)
            downgrade_reasons.append(f"{kind}:{subject_hash}:{detail}")
        elif state == "Indeterminate":
            target_indeterminate.add(subject_hash)
            downgrade_reasons.append(f"{kind}:{subject_hash}:{detail}")
        else:
            invalid_execution = True
            downgrade_reasons.append(f"{kind}:{subject_hash}:{detail}")

    for subject_hash in sorted(required_provenance_hashes):
        evaluate_required(subject_hash, provenance_path_by_hash[subject_hash], "provenance")
    relation_path_by_hash = {digest: path for _pair, (_receipt, digest, path) in relation_by_pair.items()}
    for subject_hash in sorted(required_relation_hashes):
        evaluate_required(subject_hash, relation_path_by_hash[subject_hash], "relation")

    if invalid_execution:
        return invalid("wcare37_authentication_protocol_invalid", downgrade_reasons=sorted(downgrade_reasons))

    authenticated_qualified_identities = {
        identity
        for identity in qualified_identities
        if provenance_hash_by_identity[identity] in authenticated_provenance
    }
    authenticated_lineage_reviewers = len(authenticated_qualified_identities)
    authenticated_lineages = len({lineage_by_identity[identity] for identity in authenticated_qualified_identities})

    authenticated_strength_counts = Counter()
    for identity in active:
        digest = provenance_hash_by_identity[identity]
        if digest in authenticated_provenance:
            authenticated_strength_counts[strength_by_identity[identity]] += 1
    authenticated_strength_counts_out = {key: authenticated_strength_counts.get(key, 0) for key in sorted(PROVENANCE_STRENGTHS)}

    authenticated_adjacency = {identity: set(neighbors) for identity, neighbors in baseline_adjacency.items()}
    authenticated_accepted_pairs = 0
    for pair in baseline_accepted_pairs:
        left, right = pair
        relation_digest = relation_by_pair[pair][1]
        can_remain_separate = (
            left in authenticated_qualified_identities
            and right in authenticated_qualified_identities
            and relation_digest in authenticated_relations
        )
        if can_remain_separate:
            authenticated_accepted_pairs += 1
        else:
            authenticated_adjacency[left].add(right)
            authenticated_adjacency[right].add(left)
            downgrade_reasons.append(f"relation:{relation_digest}:authentication_required_for_independence")

    authenticated_components = components(active, authenticated_adjacency)
    authenticated_component_count = len(authenticated_components)

    monotone_lineage_reviewers = authenticated_lineage_reviewers <= w36_qualified_lineage_reviewers
    monotone_lineages = authenticated_lineages <= w36_qualified_lineages
    monotone_pairs = authenticated_accepted_pairs <= len(baseline_accepted_pairs)
    monotone_components = authenticated_component_count <= len(baseline_components)
    if not all((monotone_lineage_reviewers, monotone_lineages, monotone_pairs, monotone_components)):
        return invalid("authentication_monotonicity_violation")

    min_components = w36_plan.get("minimum_effective_independent_components")
    min_lineages = w36_plan.get("minimum_distinct_lineages")
    max_unknown = w36_plan.get("maximum_unknown_relation_pairs")
    require_no_conflict = w36_plan.get("require_no_conflict_of_interest")
    if not isinstance(min_components, int) or not isinstance(min_lineages, int) or not isinstance(max_unknown, int) or not isinstance(require_no_conflict, bool):
        return invalid("wcare36_gate_policy_invalid")

    minimum_effective_components_met = authenticated_component_count >= min_components
    minimum_distinct_lineages_met = authenticated_lineages >= min_lineages
    minimum_provenance_strengths_met = all(authenticated_strength_counts.get(strength, 0) >= count for strength, count in minimum_strengths.items())
    unknown_pairs_within_limit = unknown_pairs <= max_unknown
    conflict_policy_met = (not conflicts) if require_no_conflict else True
    all_required_determinate = not indeterminate_provenance and not indeterminate_relations

    underlying_disposition = w36_result.get("disposition")
    if underlying_disposition not in {"INDEPENDENCE_SUPPORTED", "INDEPENDENCE_LIMITED"}:
        return invalid("wcare36_disposition_not_composable", underlying_disposition=underlying_disposition)

    requirements_met = all((
        underlying_disposition == "INDEPENDENCE_SUPPORTED",
        minimum_effective_components_met,
        minimum_distinct_lineages_met,
        minimum_provenance_strengths_met,
        unknown_pairs_within_limit,
        conflict_policy_met,
        all_required_determinate,
    ))

    if not all_required_determinate:
        disposition = "INFRASTRUCTURE_INDETERMINATE"
    elif requirements_met:
        disposition = "AUTHENTICATED_PANEL_SUPPORTED"
    else:
        disposition = "AUTHENTICATED_PANEL_LIMITED"

    payload = {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "wcare35_result_sha256": w35_sha,
        "wcare36_plan_sha256": w36_plan_sha,
        "wcare36_result_sha256": w36_result_sha,
        "wcare38_plan_sha256": w38_plan_sha,
        "attestation_manifest_sha256": manifest_sha,
        "underlying_wcare36_disposition": underlying_disposition,
        "required_provenance_receipt_sha256s": sorted(required_provenance_hashes),
        "authenticated_provenance_receipt_sha256s": sorted(authenticated_provenance),
        "unauthenticated_provenance_receipt_sha256s": sorted(unauthenticated_provenance),
        "indeterminate_provenance_receipt_sha256s": sorted(indeterminate_provenance),
        "required_relation_receipt_sha256s": sorted(required_relation_hashes),
        "authenticated_relation_receipt_sha256s": sorted(authenticated_relations),
        "unauthenticated_relation_receipt_sha256s": sorted(unauthenticated_relations),
        "indeterminate_relation_receipt_sha256s": sorted(indeterminate_relations),
        "wcare37_execution_receipt_sha256s": sorted(execution_receipts),
        "authenticated_provenance_strength_counts": authenticated_strength_counts_out,
        "authenticated_qualified_lineage_reviewer_count": authenticated_lineage_reviewers,
        "authenticated_qualified_distinct_lineage_count": authenticated_lineages,
        "authenticated_accepted_independent_pair_count": authenticated_accepted_pairs,
        "authenticated_effective_independent_components": authenticated_component_count,
        "authenticated_independence_component_census": authenticated_components,
        "minimum_effective_components_met": minimum_effective_components_met,
        "minimum_distinct_lineages_met": minimum_distinct_lineages_met,
        "minimum_provenance_strengths_met": minimum_provenance_strengths_met,
        "unknown_pairs_within_limit": unknown_pairs_within_limit,
        "conflict_policy_met": conflict_policy_met,
        "all_required_authentication_determinate": all_required_determinate,
        "monotone_lineage_reviewer_count": monotone_lineage_reviewers,
        "monotone_distinct_lineage_count": monotone_lineages,
        "monotone_independent_pair_count": monotone_pairs,
        "monotone_effective_component_count": monotone_components,
        "requirements_met": requirements_met,
        "downgrade_reasons": sorted(set(downgrade_reasons)),
        "disposition": disposition,
        "reviewer_correctness_established": False,
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
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    if disposition == "AUTHENTICATED_PANEL_SUPPORTED":
        return 0
    if disposition == "AUTHENTICATED_PANEL_LIMITED":
        return 2
    if disposition == "INFRASTRUCTURE_INDETERMINATE":
        return 3
    return 4


if __name__ == "__main__":
    raise SystemExit(main())
