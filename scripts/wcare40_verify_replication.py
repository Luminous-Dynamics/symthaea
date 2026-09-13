#!/usr/bin/env python3
"""Verify WCARE-40 execution replication and conservative builder independence."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

PROTOCOL = "wcare40-execution-replication-v1"
W39_PROTOCOL = "wcare39-execution-capsule-v1"
PROVENANCE_STRENGTHS = {"SelfDeclared", "OrganizerVerified", "ExternalVerified", "InstitutionalAttestation"}
RELATION_STRENGTHS = {"SelfDeclared", "OrganizerAssessed", "ExternalVerified", "InstitutionalAttestation"}
RELATIONS = {"Independent", "Related", "Unknown", "ConflictOfInterest"}
FAULT_DOMAINS = {"BuilderIdentity", "Organization", "Infrastructure", "ToolchainLineage", "OperatorProcess", "EvidenceSource"}
DOMAIN_FIELDS = {
    "BuilderIdentity": "builder_identity_commitment_sha256",
    "Organization": "organization_commitment_sha256",
    "Infrastructure": "infrastructure_commitment_sha256",
    "ToolchainLineage": "toolchain_lineage_commitment_sha256",
    "OperatorProcess": "operator_process_commitment_sha256",
    "EvidenceSource": "evidence_source_commitment_sha256",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


def is_git_sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(ch in "0123456789abcdef" for ch in value)


def parse_utc(value: object) -> None:
    if not isinstance(value, str) or len(value) != 20 or not value.endswith("Z"):
        raise ValueError("timestamp_not_canonical")
    parsed = datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    if parsed.strftime("%Y-%m-%dT%H:%M:%SZ") != value:
        raise ValueError("timestamp_not_canonical")


def read_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"json_root_not_object:{path}")
    return raw, value


def false_claims() -> dict[str, bool]:
    return {
        "builder_authentication_established": False,
        "independent_builder_identity_established_beyond_commitments": False,
        "reviewer_independence_established": False,
        "subject_correctness_established": False,
        "network_isolation_established": False,
        "sandbox_enforcement_established": False,
        "phenomenal_experience_established": False,
        "suffering_established": False,
        "moral_patienthood_established": False,
        "binding_consent_established": False,
        "veto_authority_granted": False,
        "self_preservation_authority_granted": False,
        "runtime_authority_granted": False,
    }


def emit_invalid(detail: str, **extra: Any) -> int:
    payload: dict[str, Any] = {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "disposition": "REPLICATION_INVALID",
        "detail": detail,
        **false_claims(),
    }
    payload.update(extra)
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return 4


def canonical_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def components(nodes: list[str], adjacency: dict[str, set[str]]) -> list[list[str]]:
    remaining = set(nodes)
    result: list[list[str]] = []
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
        result.append(sorted(seen))
    result.sort(key=lambda group: group[0])
    return result


def validate_plan(plan: dict[str, Any]) -> tuple[dict[str, str], set[str], set[str]]:
    required = {
        "protocol_version", "wcare39_protocol_sha256", "wcare39_runner_sha256", "wcare39_integrity_sha256",
        "plan_created_utc", "minimum_qualified_replicas", "minimum_effective_independent_components",
        "accepted_builder_provenance_strengths", "accepted_independent_relation_strengths",
        "required_receipt_stage_ids", "require_subject_outcome_agreement", "require_required_receipt_agreement",
        "require_no_conflict_of_interest", "replica_slots",
    }
    if set(plan) - (required | {"notes"}) or required - set(plan):
        raise ValueError("plan_field_set_invalid")
    if plan["protocol_version"] != PROTOCOL:
        raise ValueError("plan_protocol_mismatch")
    for field in ("wcare39_protocol_sha256", "wcare39_runner_sha256", "wcare39_integrity_sha256"):
        if not is_sha256(plan[field]):
            raise ValueError(f"plan_{field}_invalid")
    parse_utc(plan["plan_created_utc"])
    slots = plan["replica_slots"]
    if not isinstance(slots, list) or not slots:
        raise ValueError("plan_replica_slots_invalid")
    slot_builders: dict[str, str] = {}
    for slot in slots:
        if not isinstance(slot, dict) or set(slot) != {"replica_id", "builder_identity_commitment_sha256"}:
            raise ValueError("plan_replica_slot_fields_invalid")
        replica_id = slot["replica_id"]
        builder = slot["builder_identity_commitment_sha256"]
        if not isinstance(replica_id, str) or not replica_id or replica_id in slot_builders or not is_sha256(builder):
            raise ValueError("plan_replica_slot_invalid_or_duplicate")
        slot_builders[replica_id] = builder
    for field in ("minimum_qualified_replicas", "minimum_effective_independent_components"):
        value = plan[field]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1 or value > len(slot_builders):
            raise ValueError(f"plan_{field}_invalid")
    provenance_strengths = plan["accepted_builder_provenance_strengths"]
    relation_strengths = plan["accepted_independent_relation_strengths"]
    if not isinstance(provenance_strengths, list) or not provenance_strengths or len(provenance_strengths) != len(set(provenance_strengths)) or any(v not in PROVENANCE_STRENGTHS for v in provenance_strengths):
        raise ValueError("plan_provenance_strengths_invalid")
    if not isinstance(relation_strengths, list) or not relation_strengths or len(relation_strengths) != len(set(relation_strengths)) or any(v not in RELATION_STRENGTHS for v in relation_strengths):
        raise ValueError("plan_relation_strengths_invalid")
    stages = plan["required_receipt_stage_ids"]
    if not isinstance(stages, list) or len(stages) != len(set(stages)) or any(not isinstance(v, str) or not v for v in stages):
        raise ValueError("plan_required_receipt_stage_ids_invalid")
    if plan["require_subject_outcome_agreement"] is not True or plan["require_required_receipt_agreement"] is not True or plan["require_no_conflict_of_interest"] is not True:
        raise ValueError("plan_fixed_v1_policy_disabled")
    return slot_builders, set(provenance_strengths), set(relation_strengths)


def validate_provenance(receipt: dict[str, Any], plan_sha: str, expected_builder: str) -> None:
    required = {
        "protocol_version", "plan_sha256", "replica_id", "execution_observed",
        "wcare39_prepared_capsule_sha256", "wcare39_final_capsule_sha256",
        "builder_identity_commitment_sha256", "organization_commitment_sha256",
        "infrastructure_commitment_sha256", "toolchain_lineage_commitment_sha256",
        "operator_process_commitment_sha256", "evidence_source_commitment_sha256",
        "provenance_strength", "conflict_of_interest", "collected_utc",
    }
    if set(receipt) - (required | {"notes"}) or required - set(receipt):
        raise ValueError("provenance_field_set_invalid")
    if receipt["protocol_version"] != PROTOCOL or receipt["plan_sha256"] != plan_sha:
        raise ValueError("provenance_plan_binding_mismatch")
    if receipt["builder_identity_commitment_sha256"] != expected_builder:
        raise ValueError("provenance_builder_slot_mismatch")
    for field in DOMAIN_FIELDS.values():
        if not is_sha256(receipt[field]):
            raise ValueError(f"provenance_{field}_invalid")
    if receipt["provenance_strength"] not in PROVENANCE_STRENGTHS or not isinstance(receipt["conflict_of_interest"], bool):
        raise ValueError("provenance_policy_field_invalid")
    parse_utc(receipt["collected_utc"])
    observed = receipt["execution_observed"]
    prepared = receipt["wcare39_prepared_capsule_sha256"]
    final = receipt["wcare39_final_capsule_sha256"]
    if not isinstance(observed, bool):
        raise ValueError("provenance_execution_observed_invalid")
    if observed:
        if not is_sha256(prepared) or not is_sha256(final) or prepared == final:
            raise ValueError("provenance_observed_capsule_binding_invalid")
    elif prepared is not None or final is not None:
        raise ValueError("provenance_unobserved_capsules_must_be_null")


def shared_domains(left: dict[str, Any], right: dict[str, Any]) -> set[str]:
    return {label for label, field in DOMAIN_FIELDS.items() if left[field] == right[field]}


def validate_relation(
    receipt: dict[str, Any], plan_sha: str, slot_builders: dict[str, str], provenance: dict[str, dict[str, Any]]
) -> tuple[tuple[str, str], set[str]]:
    required = {
        "protocol_version", "plan_sha256", "left_replica_id", "right_replica_id",
        "left_builder_identity_commitment_sha256", "right_builder_identity_commitment_sha256",
        "relation", "declared_shared_fault_domains", "relation_evidence_strength", "assessed_utc",
    }
    if set(receipt) - (required | {"notes"}) or required - set(receipt):
        raise ValueError("relation_field_set_invalid")
    if receipt["protocol_version"] != PROTOCOL or receipt["plan_sha256"] != plan_sha:
        raise ValueError("relation_plan_binding_mismatch")
    left, right = receipt["left_replica_id"], receipt["right_replica_id"]
    if left not in slot_builders or right not in slot_builders or left == right:
        raise ValueError("relation_replica_id_invalid")
    if receipt["left_builder_identity_commitment_sha256"] != slot_builders[left] or receipt["right_builder_identity_commitment_sha256"] != slot_builders[right]:
        raise ValueError("relation_builder_identity_binding_mismatch")
    if receipt["relation"] not in RELATIONS or receipt["relation_evidence_strength"] not in RELATION_STRENGTHS:
        raise ValueError("relation_policy_value_invalid")
    declared = receipt["declared_shared_fault_domains"]
    if not isinstance(declared, list) or len(declared) != len(set(declared)) or any(v not in FAULT_DOMAINS for v in declared):
        raise ValueError("relation_declared_fault_domains_invalid")
    parse_utc(receipt["assessed_utc"])
    actual = shared_domains(provenance[left], provenance[right])
    if set(declared) != actual:
        raise ValueError(f"relation_shared_fault_domain_mismatch:{left}:{right}")
    return canonical_pair(left, right), actual


def environment_fingerprint(capsule: dict[str, Any]) -> str:
    payload = {
        "platform": capsule.get("platform"),
        "tools": capsule.get("tools"),
        "materials": capsule.get("materials"),
        "safe_environment": capsule.get("safe_environment"),
        "ambient_environment_sha256": capsule.get("ambient_environment_sha256"),
        "network_policy_declared": capsule.get("network_policy_declared"),
        "sandbox_policy_declared": capsule.get("sandbox_policy_declared"),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def same_subject_key(capsule: dict[str, Any]) -> bytes:
    if not is_git_sha(capsule.get("subject_git_head")) or not is_sha256(capsule.get("command_plan_sha256")):
        raise ValueError("wcare39_subject_identity_invalid")
    subject_digests = capsule.get("subject_digests")
    seeds = capsule.get("deterministic_seed_commitments")
    if not isinstance(subject_digests, dict) or any(not isinstance(k, str) or not is_sha256(v) for k, v in subject_digests.items()):
        raise ValueError("wcare39_subject_digests_invalid")
    if not isinstance(seeds, dict) or any(not isinstance(k, str) or not is_sha256(v) for k, v in seeds.items()):
        raise ValueError("wcare39_seed_commitments_invalid")
    return canonical_json_bytes({
        "subject_git_head": capsule["subject_git_head"],
        "command_plan_sha256": capsule["command_plan_sha256"],
        "subject_digests": subject_digests,
        "deterministic_seed_commitments": seeds,
    })


def stage_receipt(capsule: dict[str, Any], stage_id: str) -> str | None:
    commands = capsule.get("commands")
    if not isinstance(commands, list):
        raise ValueError("wcare39_commands_invalid")
    matches = [entry for entry in commands if isinstance(entry, dict) and entry.get("stage_id") == stage_id]
    if len(matches) != 1:
        raise ValueError(f"wcare39_required_stage_census_invalid:{stage_id}")
    digest = matches[0].get("output_receipt_sha256")
    if digest is not None and not is_sha256(digest):
        raise ValueError(f"wcare39_output_receipt_digest_invalid:{stage_id}")
    return digest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--capsules", nargs="*", default=[], type=Path)
    parser.add_argument("--provenance", nargs="*", default=[], type=Path)
    parser.add_argument("--relations", nargs="*", default=[], type=Path)
    args = parser.parse_args()

    try:
        plan_raw, plan = read_object(args.plan)
        slot_builders, accepted_provenance, accepted_relation = validate_plan(plan)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return emit_invalid(f"plan_read_or_validation_failed:{type(exc).__name__}:{exc}")

    plan_sha = sha256_bytes(plan_raw)
    root = Path(__file__).resolve().parent.parent
    w39_protocol_path = root / "docs/release/evidence/WCARE39_EXECUTION_CAPSULE_PROTOCOL_V1.md"
    w39_runner_path = root / "scripts/wcare39_execution_capsule.py"
    w39_integrity_path = root / "scripts/wcare39-integrity.sh"
    try:
        observed_subject_hashes = {
            "wcare39_protocol_sha256": sha256_file(w39_protocol_path),
            "wcare39_runner_sha256": sha256_file(w39_runner_path),
            "wcare39_integrity_sha256": sha256_file(w39_integrity_path),
        }
    except OSError as exc:
        return emit_invalid(f"wcare39_subject_unavailable:{type(exc).__name__}")
    for field, observed in observed_subject_hashes.items():
        if plan.get(field) != observed:
            return emit_invalid(f"wcare39_subject_digest_mismatch:{field}", observed_sha256=observed)

    expected_ids = sorted(slot_builders)
    expected_pairs = {canonical_pair(expected_ids[i], expected_ids[j]) for i in range(len(expected_ids)) for j in range(i + 1, len(expected_ids))}

    capsule_by_hash: dict[str, tuple[Path, dict[str, Any]]] = {}
    try:
        for path in args.capsules:
            raw, capsule = read_object(path)
            digest = sha256_bytes(raw)
            if digest in capsule_by_hash:
                return emit_invalid("duplicate_capsule_bytes", capsule_sha256=digest)
            capsule_by_hash[digest] = (path, capsule)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return emit_invalid(f"capsule_read_failed:{type(exc).__name__}:{exc}")

    provenance: dict[str, dict[str, Any]] = {}
    provenance_hashes: set[str] = set()
    try:
        for path in args.provenance:
            raw, receipt = read_object(path)
            digest = sha256_bytes(raw)
            replica_id = receipt.get("replica_id")
            if replica_id not in slot_builders or replica_id in provenance:
                raise ValueError("provenance_replica_invalid_or_duplicate")
            validate_provenance(receipt, plan_sha, slot_builders[replica_id])
            provenance[replica_id] = receipt
            provenance_hashes.add(digest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return emit_invalid(f"provenance_invalid:{type(exc).__name__}:{exc}")
    if set(provenance) != set(expected_ids):
        return emit_invalid("incomplete_builder_provenance_census", missing_replica_ids=sorted(set(expected_ids) - set(provenance)))

    referenced_capsules: set[str] = set()
    prepared_sha_by_replica: dict[str, str] = {}
    final_sha_by_replica: dict[str, str] = {}
    observed_ids: list[str] = []
    missing_ids: list[str] = []
    for replica_id in expected_ids:
        receipt = provenance[replica_id]
        if receipt["execution_observed"]:
            prepared_sha = receipt["wcare39_prepared_capsule_sha256"]
            final_sha = receipt["wcare39_final_capsule_sha256"]
            if prepared_sha not in capsule_by_hash or final_sha not in capsule_by_hash:
                return emit_invalid("provenance_references_unsupplied_capsule", replica_id=replica_id)
            if prepared_sha in referenced_capsules or final_sha in referenced_capsules:
                return emit_invalid("capsule_reused_across_replica_slots", replica_id=replica_id)
            referenced_capsules.update((prepared_sha, final_sha))
            prepared_sha_by_replica[replica_id] = prepared_sha
            final_sha_by_replica[replica_id] = final_sha
            observed_ids.append(replica_id)
        else:
            missing_ids.append(replica_id)
    if referenced_capsules != set(capsule_by_hash):
        return emit_invalid("unexpected_or_unreferenced_capsule_artifact", capsule_sha256s=sorted(set(capsule_by_hash) - referenced_capsules))

    relations: dict[tuple[str, str], tuple[dict[str, Any], set[str]]] = {}
    relation_hashes: set[str] = set()
    try:
        for path in args.relations:
            raw, receipt = read_object(path)
            digest = sha256_bytes(raw)
            pair, actual_shared = validate_relation(receipt, plan_sha, slot_builders, provenance)
            if pair in relations:
                raise ValueError("duplicate_relation_pair")
            relations[pair] = (receipt, actual_shared)
            relation_hashes.add(digest)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        return emit_invalid(f"relation_invalid:{type(exc).__name__}:{exc}")
    if set(relations) != expected_pairs:
        return emit_invalid("incomplete_pairwise_relation_census", missing_pairs=[list(pair) for pair in sorted(expected_pairs - set(relations))])

    qualified_environment: list[str] = []
    subject_eligible: list[str] = []
    pass_ids: list[str] = []
    fail_ids: list[str] = []
    noneligible: list[str] = []
    infra_ids: set[str] = set(missing_ids)
    final_by_replica: dict[str, dict[str, Any]] = {}
    compare_receipt_sha: dict[str, str] = {}
    env_fingerprints: dict[str, str] = {}

    for replica_id in observed_ids:
        prepared_sha = prepared_sha_by_replica[replica_id]
        final_sha = final_sha_by_replica[replica_id]
        prepared_path, prepared = capsule_by_hash[prepared_sha]
        final_path, final = capsule_by_hash[final_sha]
        if prepared.get("protocol_version") != W39_PROTOCOL or final.get("protocol_version") != W39_PROTOCOL:
            return emit_invalid("wcare39_capsule_protocol_mismatch", replica_id=replica_id)
        if prepared.get("capsule_phase") != "PREPARED" or final.get("capsule_phase") != "FINAL":
            return emit_invalid("wcare39_capsule_phase_mismatch", replica_id=replica_id)
        try:
            compared = subprocess.run(
                [sys.executable, str(w39_runner_path), "compare", str(prepared_path), str(final_path)],
                cwd=root, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=300,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            infra_ids.add(replica_id)
            continue
        compare_receipt_sha[replica_id] = sha256_bytes(compared.stdout)
        try:
            compare_result = json.loads(compared.stdout)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return emit_invalid("wcare39_compare_nonjson_output", replica_id=replica_id)
        if not isinstance(compare_result, dict):
            return emit_invalid("wcare39_compare_output_not_object", replica_id=replica_id)
        compare_class = compare_result.get("classification")
        final_class = final.get("classification")
        final_integrity = final.get("environment_integrity")
        final_drift = final.get("drift_fields")
        if compared.returncode == 4 or compare_class == "INVALID_CAPSULE":
            return emit_invalid("wcare39_compare_rejected_lineage", replica_id=replica_id)
        if compared.returncode == 3 or compare_class == "INFRASTRUCTURE_INDETERMINATE":
            infra_ids.add(replica_id)
            continue
        if compared.returncode == 2 or compare_class == "ENVIRONMENT_DRIFT":
            if final_class != "ENVIRONMENT_DRIFT" or final_integrity != "DRIFTED":
                return emit_invalid("wcare39_compare_final_classification_mismatch", replica_id=replica_id)
            noneligible.append(replica_id)
            continue
        if compared.returncode != 0 or compare_class != "QUALIFIED_EXECUTION":
            return emit_invalid("wcare39_compare_unexpected_result", replica_id=replica_id, exit_code=compared.returncode)
        if final_class == "INFRASTRUCTURE_INDETERMINATE" or final_integrity == "INDETERMINATE":
            infra_ids.add(replica_id)
            continue
        if final_class != "QUALIFIED_EXECUTION" or final_integrity != "QUALIFIED" or final_drift != []:
            return emit_invalid("wcare39_final_claim_inconsistent_with_compare", replica_id=replica_id)
        qualified_environment.append(replica_id)
        final_by_replica[replica_id] = final
        outcome = final.get("subject_outcome")
        if outcome == "PASS":
            subject_eligible.append(replica_id); pass_ids.append(replica_id)
        elif outcome == "FAIL":
            subject_eligible.append(replica_id); fail_ids.append(replica_id)
        else:
            noneligible.append(replica_id)
        env_fingerprints[replica_id] = environment_fingerprint(final)

    if subject_eligible:
        reference = same_subject_key(final_by_replica[subject_eligible[0]])
        for replica_id in subject_eligible[1:]:
            if same_subject_key(final_by_replica[replica_id]) != reference:
                return emit_invalid("replicas_do_not_share_exact_subject", replica_id=replica_id)

    outcome_agreement = bool(subject_eligible) and not (pass_ids and fail_ids)
    agreed_outcome: str | None = None
    if outcome_agreement:
        agreed_outcome = "PASS" if pass_ids else "FAIL"

    stage_agreement: dict[str, bool] = {}
    stage_distinct: dict[str, list[str]] = {}
    required_receipt_contradiction = False
    required_receipt_missing = False
    for stage_id in plan["required_receipt_stage_ids"]:
        digests: list[str] = []
        missing = False
        try:
            for replica_id in subject_eligible:
                digest = stage_receipt(final_by_replica[replica_id], stage_id)
                if digest is None:
                    missing = True
                else:
                    digests.append(digest)
        except ValueError as exc:
            return emit_invalid(f"required_receipt_stage_invalid:{exc}")
        distinct = sorted(set(digests))
        stage_distinct[stage_id] = distinct
        if len(distinct) > 1:
            required_receipt_contradiction = True
        if missing or len(digests) != len(subject_eligible):
            required_receipt_missing = True
        stage_agreement[stage_id] = bool(subject_eligible) and not missing and len(distinct) == 1 and len(digests) == len(subject_eligible)
    required_receipt_agreement = all(stage_agreement.values()) if plan["required_receipt_stage_ids"] else True

    env_groups_map: dict[str, list[str]] = defaultdict(list)
    for replica_id in subject_eligible:
        env_groups_map[env_fingerprints[replica_id]].append(replica_id)
    env_groups = [sorted(group) for _fingerprint, group in sorted(env_groups_map.items())]

    qualified_builder = sorted(
        replica_id for replica_id in subject_eligible
        if provenance[replica_id]["provenance_strength"] in accepted_provenance
    )
    qualified_set = set(qualified_builder)
    adjacency = {replica_id: set() for replica_id in qualified_builder}
    accepted_pairs = 0
    downgraded_independent = 0
    conflict_present = any(provenance[replica_id]["conflict_of_interest"] for replica_id in expected_ids)

    for pair in sorted(expected_pairs):
        relation, actual_shared = relations[pair]
        if relation["relation"] == "ConflictOfInterest":
            conflict_present = True
        left, right = pair
        if left not in qualified_set or right not in qualified_set:
            continue
        can_separate = (
            relation["relation"] == "Independent"
            and relation["relation_evidence_strength"] in accepted_relation
            and not actual_shared
            and not provenance[left]["conflict_of_interest"]
            and not provenance[right]["conflict_of_interest"]
        )
        if can_separate:
            accepted_pairs += 1
        else:
            adjacency[left].add(right); adjacency[right].add(left)
            if relation["relation"] == "Independent":
                downgraded_independent += 1

    component_census = components(qualified_builder, adjacency) if qualified_builder else []
    effective_components = len(component_census)
    min_qualified_met = len(qualified_builder) >= plan["minimum_qualified_replicas"]
    min_components_met = effective_components >= plan["minimum_effective_independent_components"]
    conflict_policy_met = not conflict_present
    all_observed = not missing_ids

    known_contradiction = bool(pass_ids and fail_ids) or required_receipt_contradiction
    requirements_met = all((
        all_observed,
        len(subject_eligible) == len(expected_ids),
        outcome_agreement,
        required_receipt_agreement,
        min_qualified_met,
        min_components_met,
        conflict_policy_met,
        not infra_ids,
    ))

    if known_contradiction:
        disposition = "REPLICATION_CONTRADICTED"
        exit_code = 1
    elif infra_ids:
        disposition = "INFRASTRUCTURE_INDETERMINATE"
        exit_code = 3
    elif requirements_met:
        disposition = "REPLICATION_SUPPORTED"
        exit_code = 0
    else:
        disposition = "REPLICATION_LIMITED"
        exit_code = 2

    payload: dict[str, Any] = {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "plan_sha256": plan_sha,
        **observed_subject_hashes,
        "expected_replica_ids": expected_ids,
        "observed_replica_ids": sorted(observed_ids),
        "missing_replica_ids": sorted(missing_ids),
        "prepared_capsule_sha256_by_replica": dict(sorted(prepared_sha_by_replica.items())),
        "final_capsule_sha256_by_replica": dict(sorted(final_sha_by_replica.items())),
        "wcare39_compare_receipt_sha256_by_replica": dict(sorted(compare_receipt_sha.items())),
        "qualified_environment_replica_ids": sorted(qualified_environment),
        "subject_eligible_replica_ids": sorted(subject_eligible),
        "pass_replica_ids": sorted(pass_ids),
        "fail_replica_ids": sorted(fail_ids),
        "subject_noneligible_replica_ids": sorted(set(noneligible)),
        "infrastructure_indeterminate_replica_ids": sorted(infra_ids),
        "subject_outcome_agreement_met": outcome_agreement,
        "agreed_subject_outcome": agreed_outcome,
        "required_receipt_stage_ids": list(plan["required_receipt_stage_ids"]),
        "required_receipt_stage_agreement": dict(sorted(stage_agreement.items())),
        "required_receipt_stage_distinct_sha256s": dict(sorted(stage_distinct.items())),
        "required_receipt_agreement_met": required_receipt_agreement and not required_receipt_missing,
        "environment_fingerprint_sha256_by_replica": {k: env_fingerprints[k] for k in sorted(subject_eligible)},
        "distinct_environment_fingerprint_count": len(env_groups_map),
        "exact_environment_replica_groups": env_groups,
        "builder_provenance_receipt_sha256s": sorted(provenance_hashes),
        "builder_relation_receipt_sha256s": sorted(relation_hashes),
        "raw_distinct_builder_identity_count": len(set(slot_builders.values())),
        "qualified_builder_replica_ids": qualified_builder,
        "qualified_distinct_builder_identity_count": len({slot_builders[r] for r in qualified_builder}),
        "accepted_independent_pair_count": accepted_pairs,
        "downgraded_independent_pair_count": downgraded_independent,
        "effective_independent_components": effective_components,
        "independence_component_census": component_census,
        "minimum_qualified_replicas_met": min_qualified_met,
        "minimum_effective_independent_components_met": min_components_met,
        "conflict_policy_met": conflict_policy_met,
        "all_expected_slots_observed": all_observed,
        "requirements_met": requirements_met,
        "disposition": disposition,
        **false_claims(),
    }
    sys.stdout.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
