#!/usr/bin/env python3
"""Pure WCARE-44 candidate aggregation kernel.

This kernel recomputes the WCARE-40 builder graph and applies conservative
candidate-authentication edge additions. It deliberately cannot establish child
verifier lineage or final WCARE-44 claims.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
from typing import Any

PROTOCOL = "wcare44-authenticated-replication-aggregation-v1"
WCARE40 = "wcare40-execution-replication-v1"
WCARE41 = "wcare41-authenticated-preregistration-v1"
HEX = set("0123456789abcdef")

FAULT_FIELDS = [
    ("builder_identity_commitment_sha256", "BuilderIdentity"),
    ("organization_commitment_sha256", "Organization"),
    ("infrastructure_commitment_sha256", "Infrastructure"),
    ("toolchain_lineage_commitment_sha256", "ToolchainLineage"),
    ("operator_process_commitment_sha256", "OperatorProcess"),
    ("evidence_source_commitment_sha256", "EvidenceSource"),
]


class InvalidEvidence(Exception):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise InvalidEvidence(f"read_failed:{path}:{exc}") from exc
    try:
        value = json.loads(raw)
    except Exception as exc:
        raise InvalidEvidence(f"json_parse_failed:{path}:{exc}") from exc
    if not isinstance(value, dict):
        raise InvalidEvidence(f"json_not_object:{path}")
    return value, sha256(raw)


def is_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in HEX for c in value)


def require_sha(value: Any, label: str) -> str:
    if not is_sha(value):
        raise InvalidEvidence(f"invalid_sha256:{label}")
    return value


def require_unique_strings(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise InvalidEvidence(f"invalid_string_array:{label}")
    if len(value) != len(set(value)):
        raise InvalidEvidence(f"duplicate_string_array:{label}")
    return list(value)


def pair_key(a: str, b: str) -> tuple[str, str]:
    if a == b:
        raise InvalidEvidence(f"self_pair:{a}")
    return tuple(sorted((a, b)))  # type: ignore[return-value]


def normalized_components(components: list[list[str]]) -> list[list[str]]:
    return sorted((sorted(component) for component in components), key=lambda x: tuple(x))


def connected_components(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        if left not in adjacency or right not in adjacency:
            raise InvalidEvidence(f"edge_outside_node_set:{left}:{right}")
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(nodes)
    components: list[list[str]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        seen: set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(sorted(adjacency[node] - seen, reverse=True))
        remaining -= seen
        components.append(sorted(seen))
    return normalized_components(components)


def baseline_component_containment(baseline: list[list[str]], overlay: list[list[str]]) -> bool:
    overlay_sets = [set(component) for component in overlay]
    return all(sum(set(component) <= target for target in overlay_sets) == 1 for component in baseline)


def base_result() -> dict[str, Any]:
    return {
        "authority": "MeasurementOnly",
        "protocol_version": PROTOCOL,
        "stage": "CandidateKernel",
        "disposition": "CANDIDATE_INVALID",
        "detail": "uninitialized",
        "wcare40_plan_sha256": None,
        "wcare40_result_sha256": None,
        "wcare41_authentication_plan_sha256": None,
        "wcare40_recomputed_matches_reported": False,
        "wcare40_qualified_builder_replica_ids": [],
        "wcare40_effective_independent_components": 0,
        "wcare40_component_census": [],
        "candidate_authenticated_effective_independent_components": 0,
        "candidate_component_census": [],
        "monotonicity_met": False,
        "baseline_partition_not_split": False,
        "builder_authentication_candidate_status": "INVALID",
        "candidate_builder_authentication_requirements_met": False,
        "candidate_complete_builder_coverage": False,
        "required_builder_subject_receipt_sha256s": [],
        "candidate_accepted_builder_subject_receipt_sha256s": [],
        "candidate_untrusted_builder_subject_receipt_sha256s": [],
        "candidate_rejected_builder_subject_receipt_sha256s": [],
        "candidate_indeterminate_builder_subject_receipt_sha256s": [],
        "candidate_missing_builder_subject_receipt_sha256s": [],
        "added_conservative_edges": [],
        "temporal_candidate_status": "INVALID",
        "candidate_temporal_requirements_met": False,
        "candidate_authenticated_preregistered_replication": False,
        "child_verifier_lineage_established": False,
        "wcare42_executable_qualification_established": False,
        "wcare43_external_execution_lineage_established": False,
        "builder_authentication_established": False,
        "preregistration_temporal_precedence_established": False,
        "authenticated_preregistered_replication_established": False,
        "subject_correctness_established": False,
        "reviewer_independence_established": False,
        "global_tsa_trust_established": False,
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


def load_receipt_census(paths: list[str], kind: str) -> list[tuple[dict[str, Any], str]]:
    if not paths:
        raise InvalidEvidence(f"missing_{kind}_receipts")
    result: list[tuple[dict[str, Any], str]] = []
    digests: set[str] = set()
    for raw_path in paths:
        value, digest = load(Path(raw_path))
        if digest in digests:
            raise InvalidEvidence(f"duplicate_{kind}_receipt_bytes:{digest}")
        digests.add(digest)
        result.append((value, digest))
    return result


def derive_shared_fault_domains(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    shared: list[str] = []
    for field, label in FAULT_FIELDS:
        l = require_sha(left.get(field), f"left.{field}")
        r = require_sha(right.get(field), f"right.{field}")
        if l == r:
            shared.append(label)
    return sorted(shared)


def derive_baseline(
    plan: dict[str, Any],
    result: dict[str, Any],
    plan_sha: str,
    provenance_receipts: list[tuple[dict[str, Any], str]],
    relation_receipts: list[tuple[dict[str, Any], str]],
) -> tuple[
    list[str],
    list[list[str]],
    set[tuple[str, str]],
    dict[str, str],
    dict[tuple[str, str], str],
]:
    if plan.get("protocol_version") != WCARE40 or result.get("protocol_version") != WCARE40:
        raise InvalidEvidence("wcare40_protocol_mismatch")
    if require_sha(result.get("plan_sha256"), "result.plan_sha256") != plan_sha:
        raise InvalidEvidence("wcare40_result_plan_sha256_mismatch")

    slots = plan.get("replica_slots")
    if not isinstance(slots, list) or len(slots) < 2:
        raise InvalidEvidence("invalid_replica_slots")
    planned_ids: list[str] = []
    planned_builder: dict[str, str] = {}
    for slot in slots:
        if not isinstance(slot, dict):
            raise InvalidEvidence("replica_slot_not_object")
        replica_id = slot.get("replica_id")
        if not isinstance(replica_id, str) or not replica_id or replica_id in planned_builder:
            raise InvalidEvidence(f"duplicate_or_invalid_replica_id:{replica_id}")
        planned_ids.append(replica_id)
        planned_builder[replica_id] = require_sha(slot.get("builder_identity_commitment_sha256"), f"slot_builder:{replica_id}")

    expected_ids = require_unique_strings(result.get("expected_replica_ids"), "result.expected_replica_ids")
    if set(expected_ids) != set(planned_ids):
        raise InvalidEvidence("expected_replica_ids_do_not_match_plan")

    result_prov_hashes = require_unique_strings(result.get("builder_provenance_receipt_sha256s"), "result.builder_provenance_receipt_sha256s")
    result_rel_hashes = require_unique_strings(result.get("builder_relation_receipt_sha256s"), "result.builder_relation_receipt_sha256s")
    supplied_prov_hashes = [digest for _, digest in provenance_receipts]
    supplied_rel_hashes = [digest for _, digest in relation_receipts]
    if set(result_prov_hashes) != set(supplied_prov_hashes):
        raise InvalidEvidence("builder_provenance_receipt_digest_census_mismatch")
    if set(result_rel_hashes) != set(supplied_rel_hashes):
        raise InvalidEvidence("builder_relation_receipt_digest_census_mismatch")

    provenance_by_replica: dict[str, tuple[dict[str, Any], str]] = {}
    for receipt, digest in provenance_receipts:
        if receipt.get("protocol_version") != WCARE40:
            raise InvalidEvidence("builder_provenance_protocol_mismatch")
        if require_sha(receipt.get("plan_sha256"), f"provenance_plan:{digest}") != plan_sha:
            raise InvalidEvidence("builder_provenance_plan_mismatch")
        replica_id = receipt.get("replica_id")
        if not isinstance(replica_id, str) or replica_id not in planned_builder or replica_id in provenance_by_replica:
            raise InvalidEvidence(f"invalid_or_duplicate_provenance_replica:{replica_id}")
        if require_sha(receipt.get("builder_identity_commitment_sha256"), f"provenance_builder:{replica_id}") != planned_builder[replica_id]:
            raise InvalidEvidence(f"provenance_builder_commitment_mismatch:{replica_id}")
        provenance_by_replica[replica_id] = (receipt, digest)
    if set(provenance_by_replica) != set(planned_ids):
        raise InvalidEvidence("provenance_replica_census_incomplete")

    relation_by_pair: dict[tuple[str, str], tuple[dict[str, Any], str]] = {}
    for receipt, digest in relation_receipts:
        if receipt.get("protocol_version") != WCARE40:
            raise InvalidEvidence("builder_relation_protocol_mismatch")
        if require_sha(receipt.get("plan_sha256"), f"relation_plan:{digest}") != plan_sha:
            raise InvalidEvidence("builder_relation_plan_mismatch")
        left = receipt.get("left_replica_id")
        right = receipt.get("right_replica_id")
        if not isinstance(left, str) or not isinstance(right, str) or left not in planned_builder or right not in planned_builder:
            raise InvalidEvidence("relation_unknown_replica")
        key = pair_key(left, right)
        if key in relation_by_pair:
            raise InvalidEvidence(f"duplicate_relation_pair:{key[0]}:{key[1]}")
        if require_sha(receipt.get("left_builder_identity_commitment_sha256"), f"relation_left_builder:{digest}") != planned_builder[left]:
            raise InvalidEvidence("relation_left_builder_commitment_mismatch")
        if require_sha(receipt.get("right_builder_identity_commitment_sha256"), f"relation_right_builder:{digest}") != planned_builder[right]:
            raise InvalidEvidence("relation_right_builder_commitment_mismatch")
        derived_shared = derive_shared_fault_domains(provenance_by_replica[left][0], provenance_by_replica[right][0])
        declared = receipt.get("declared_shared_fault_domains")
        if not isinstance(declared, list) or any(not isinstance(x, str) for x in declared) or len(declared) != len(set(declared)):
            raise InvalidEvidence("invalid_declared_shared_fault_domains")
        if sorted(declared) != derived_shared:
            raise InvalidEvidence(f"forged_shared_fault_domains:{key[0]}:{key[1]}")
        relation_by_pair[key] = (receipt, digest)

    expected_pairs = {pair_key(a, b) for a, b in itertools.combinations(planned_ids, 2)}
    if set(relation_by_pair) != expected_pairs:
        raise InvalidEvidence("relation_pair_census_incomplete")

    accepted_provenance = set(require_unique_strings(plan.get("accepted_builder_provenance_strengths"), "plan.accepted_builder_provenance_strengths"))
    accepted_relation = set(require_unique_strings(plan.get("accepted_independent_relation_strengths"), "plan.accepted_independent_relation_strengths"))
    eligible = set(require_unique_strings(result.get("subject_eligible_replica_ids"), "result.subject_eligible_replica_ids"))
    if not eligible <= set(planned_ids):
        raise InvalidEvidence("subject_eligible_replica_outside_plan")
    qualified = sorted(
        replica_id
        for replica_id in eligible
        if provenance_by_replica[replica_id][0].get("provenance_strength") in accepted_provenance
    )
    reported_qualified = sorted(require_unique_strings(result.get("qualified_builder_replica_ids"), "result.qualified_builder_replica_ids"))
    if qualified != reported_qualified:
        raise InvalidEvidence("qualified_builder_replica_set_mismatch")

    edges: set[tuple[str, str]] = set()
    accepted_pairs = 0
    downgraded_independent_pairs = 0
    for left, right in itertools.combinations(qualified, 2):
        key = pair_key(left, right)
        relation = relation_by_pair[key][0]
        left_prov = provenance_by_replica[left][0]
        right_prov = provenance_by_replica[right][0]
        shared = derive_shared_fault_domains(left_prov, right_prov)
        is_independent = (
            relation.get("relation") == "Independent"
            and relation.get("relation_evidence_strength") in accepted_relation
            and left_prov.get("provenance_strength") in accepted_provenance
            and right_prov.get("provenance_strength") in accepted_provenance
            and left_prov.get("conflict_of_interest") is False
            and right_prov.get("conflict_of_interest") is False
            and not shared
            and require_sha(left_prov.get("builder_identity_commitment_sha256"), f"left_builder:{left}")
            != require_sha(right_prov.get("builder_identity_commitment_sha256"), f"right_builder:{right}")
        )
        if is_independent:
            accepted_pairs += 1
        else:
            edges.add(key)
            if relation.get("relation") == "Independent":
                downgraded_independent_pairs += 1

    components = connected_components(qualified, edges)
    reported_components_raw = result.get("independence_component_census")
    if not isinstance(reported_components_raw, list) or any(not isinstance(x, list) for x in reported_components_raw):
        raise InvalidEvidence("invalid_reported_component_census")
    reported_components = normalized_components([[str(item) for item in component] for component in reported_components_raw])
    if components != reported_components:
        raise InvalidEvidence("wcare40_component_partition_mismatch")
    if result.get("effective_independent_components") != len(components):
        raise InvalidEvidence("wcare40_component_count_mismatch")
    if result.get("accepted_independent_pair_count") != accepted_pairs:
        raise InvalidEvidence("wcare40_accepted_independent_pair_count_mismatch")
    if result.get("downgraded_independent_pair_count") != downgraded_independent_pairs:
        raise InvalidEvidence("wcare40_downgraded_independent_pair_count_mismatch")

    raw_distinct = len({require_sha(receipt[0].get("builder_identity_commitment_sha256"), "raw_builder_identity") for receipt in provenance_by_replica.values()})
    if result.get("raw_distinct_builder_identity_count") != raw_distinct:
        raise InvalidEvidence("wcare40_raw_builder_identity_count_mismatch")
    qualified_distinct = len({require_sha(provenance_by_replica[x][0].get("builder_identity_commitment_sha256"), "qualified_builder_identity") for x in qualified})
    if result.get("qualified_distinct_builder_identity_count") != qualified_distinct:
        raise InvalidEvidence("wcare40_qualified_builder_identity_count_mismatch")

    no_conflict = all(receipt[0].get("conflict_of_interest") is False for receipt in provenance_by_replica.values()) and all(receipt[0].get("relation") != "ConflictOfInterest" for receipt in relation_by_pair.values())
    if result.get("conflict_policy_met") is not no_conflict:
        raise InvalidEvidence("wcare40_conflict_policy_mismatch")

    prov_digest_by_replica = {replica_id: digest for replica_id, (_, digest) in provenance_by_replica.items()}
    relation_digest_by_pair = {key: digest for key, (_, digest) in relation_by_pair.items()}
    return qualified, components, edges, prov_digest_by_replica, relation_digest_by_pair


def load_observations(paths: list[str], expected: dict[str, str]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    observations: dict[str, dict[str, Any]] = {}
    buckets = {"accepted": [], "untrusted": [], "rejected": [], "indeterminate": []}
    for raw_path in paths:
        observation, _ = load(Path(raw_path))
        if observation.get("protocol_version") != PROTOCOL:
            raise InvalidEvidence("builder_observation_protocol_mismatch")
        subject = require_sha(observation.get("subject_receipt_sha256"), "builder_observation.subject")
        if subject not in expected:
            raise InvalidEvidence(f"builder_observation_unknown_subject:{subject}")
        if subject in observations:
            raise InvalidEvidence(f"duplicate_builder_observation:{subject}")
        if observation.get("subject_kind") != expected[subject]:
            raise InvalidEvidence(f"builder_observation_subject_kind_mismatch:{subject}")
        status = observation.get("status")
        if status not in {"ACCEPTED", "UNTRUSTED", "REJECTED", "INDETERMINATE"}:
            raise InvalidEvidence(f"invalid_builder_observation_status:{subject}")
        usable = status == "ACCEPTED" and observation.get("verifier_execution_qualified") is True and observation.get("synthetic") is False
        observation = dict(observation)
        observation["_candidate_usable"] = usable
        observations[subject] = observation
        if usable:
            buckets["accepted"].append(subject)
        elif status == "INDETERMINATE" or (status == "ACCEPTED" and observation.get("verifier_execution_qualified") is not True):
            buckets["indeterminate"].append(subject)
        elif status == "REJECTED":
            buckets["rejected"].append(subject)
        else:
            buckets["untrusted"].append(subject)
    for values in buckets.values():
        values.sort()
    return observations, buckets


def temporal_candidate(path: Path, result_sha: str, auth_plan_sha: str) -> tuple[str, bool, bool]:
    observation, _ = load(path)
    if observation.get("protocol_version") != PROTOCOL:
        raise InvalidEvidence("temporal_observation_protocol_mismatch")
    if require_sha(observation.get("wcare40_result_sha256"), "temporal.wcare40_result") != result_sha:
        raise InvalidEvidence("temporal_wcare40_result_sha256_mismatch")
    if require_sha(observation.get("wcare41_authentication_plan_sha256"), "temporal.wcare41_authentication_plan") != auth_plan_sha:
        raise InvalidEvidence("temporal_wcare41_authentication_plan_sha256_mismatch")
    status = observation.get("status")
    if status == "INVALID":
        return "INVALID", False, True
    if status == "INDETERMINATE":
        return "INDETERMINATE", False, False
    if status == "NOT_ESTABLISHED":
        return "NOT_ESTABLISHED", False, False
    if status != "ESTABLISHED":
        raise InvalidEvidence("invalid_temporal_observation_status")
    if observation.get("synthetic") is True:
        return "NOT_ESTABLISHED", False, False
    if observation.get("verifier_execution_qualified") is not True:
        return "INDETERMINATE", False, False
    return "ESTABLISHED_CANDIDATE", True, False


def evaluate(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    out = base_result()
    try:
        plan, plan_sha = load(Path(args.wcare40_plan))
        result, result_sha = load(Path(args.wcare40_result))
        auth_plan, auth_plan_sha = load(Path(args.wcare41_authentication_plan))
        out["wcare40_plan_sha256"] = plan_sha
        out["wcare40_result_sha256"] = result_sha
        out["wcare41_authentication_plan_sha256"] = auth_plan_sha

        if auth_plan.get("protocol_version") != WCARE41:
            raise InvalidEvidence("wcare41_authentication_plan_protocol_mismatch")
        if require_sha(auth_plan.get("wcare40_plan_sha256"), "auth_plan.wcare40_plan") != plan_sha:
            raise InvalidEvidence("wcare41_auth_plan_wcare40_plan_sha256_mismatch")
        if require_sha(auth_plan.get("wcare40_result_sha256"), "auth_plan.wcare40_result") != result_sha:
            raise InvalidEvidence("wcare41_auth_plan_wcare40_result_sha256_mismatch")
        if auth_plan.get("require_complete_builder_attestation_coverage") is not True:
            raise InvalidEvidence("wcare41_complete_builder_coverage_not_required")
        if auth_plan.get("require_temporal_preregistration") is not True:
            raise InvalidEvidence("wcare41_temporal_preregistration_not_required")

        provenance_receipts = load_receipt_census(args.provenance, "provenance")
        relation_receipts = load_receipt_census(args.relation, "relation")
        qualified, baseline_components, baseline_edges, prov_digest_by_replica, relation_digest_by_pair = derive_baseline(
            plan, result, plan_sha, provenance_receipts, relation_receipts
        )
        out["wcare40_recomputed_matches_reported"] = True
        out["wcare40_qualified_builder_replica_ids"] = qualified
        out["wcare40_effective_independent_components"] = len(baseline_components)
        out["wcare40_component_census"] = baseline_components

        expected_subject_kind: dict[str, str] = {}
        for _, digest in provenance_receipts:
            expected_subject_kind[digest] = "BuilderProvenance"
        for _, digest in relation_receipts:
            expected_subject_kind[digest] = "BuilderRelation"
        required_subjects = sorted(expected_subject_kind)
        out["required_builder_subject_receipt_sha256s"] = required_subjects

        observations, buckets = load_observations(args.builder_observation, expected_subject_kind)
        observed_subjects = set(observations)
        missing = sorted(set(required_subjects) - observed_subjects)
        out["candidate_accepted_builder_subject_receipt_sha256s"] = buckets["accepted"]
        out["candidate_untrusted_builder_subject_receipt_sha256s"] = buckets["untrusted"]
        out["candidate_rejected_builder_subject_receipt_sha256s"] = buckets["rejected"]
        out["candidate_indeterminate_builder_subject_receipt_sha256s"] = buckets["indeterminate"]
        out["candidate_missing_builder_subject_receipt_sha256s"] = missing

        candidate_edges = set(baseline_edges)
        added: list[dict[str, Any]] = []
        for left, right in itertools.combinations(qualified, 2):
            key = pair_key(left, right)
            if key in baseline_edges:
                continue
            left_digest = prov_digest_by_replica[left]
            right_digest = prov_digest_by_replica[right]
            relation_digest = relation_digest_by_pair[key]
            reasons: list[str] = []
            if not observations.get(left_digest, {}).get("_candidate_usable", False):
                reasons.append("LeftProvenanceUnauthenticated")
            if not observations.get(right_digest, {}).get("_candidate_usable", False):
                reasons.append("RightProvenanceUnauthenticated")
            if not observations.get(relation_digest, {}).get("_candidate_usable", False):
                reasons.append("RelationUnauthenticated")
            if reasons:
                candidate_edges.add(key)
                added.append({
                    "left_replica_id": key[0],
                    "right_replica_id": key[1],
                    "reasons": reasons,
                })
        added.sort(key=lambda x: (x["left_replica_id"], x["right_replica_id"]))
        out["added_conservative_edges"] = added

        candidate_components = connected_components(qualified, candidate_edges)
        out["candidate_component_census"] = candidate_components
        out["candidate_authenticated_effective_independent_components"] = len(candidate_components)
        monotone = len(candidate_components) <= len(baseline_components)
        partition_not_split = baseline_component_containment(baseline_components, candidate_components)
        out["monotonicity_met"] = monotone
        out["baseline_partition_not_split"] = partition_not_split
        if not monotone or not partition_not_split:
            raise InvalidEvidence("candidate_overlay_violated_graph_monotonicity")

        complete_coverage = set(buckets["accepted"]) == set(required_subjects)
        out["candidate_complete_builder_coverage"] = complete_coverage
        minimum_components = plan.get("minimum_effective_independent_components")
        if not isinstance(minimum_components, int) or minimum_components < 2:
            raise InvalidEvidence("invalid_minimum_effective_independent_components")
        builder_requirements = complete_coverage and len(candidate_components) >= minimum_components
        out["candidate_builder_authentication_requirements_met"] = builder_requirements

        any_indeterminate = bool(buckets["indeterminate"])
        if any_indeterminate:
            builder_status = "INDETERMINATE"
        elif builder_requirements:
            builder_status = "AUTHENTICATED_CANDIDATE"
        elif not buckets["accepted"]:
            builder_status = "UNAUTHENTICATED"
        else:
            builder_status = "PARTIAL"
        out["builder_authentication_candidate_status"] = builder_status

        temporal_status, temporal_met, temporal_invalid = temporal_candidate(
            Path(args.temporal_observation), result_sha, auth_plan_sha
        )
        out["temporal_candidate_status"] = temporal_status
        out["candidate_temporal_requirements_met"] = temporal_met
        out["candidate_authenticated_preregistered_replication"] = builder_requirements and temporal_met

        if temporal_invalid:
            out["disposition"] = "CANDIDATE_INVALID"
            out["detail"] = "temporal_candidate_invalid"
            return out, 4
        if builder_status == "INDETERMINATE" or temporal_status == "INDETERMINATE":
            out["disposition"] = "CANDIDATE_INDETERMINATE"
            out["detail"] = "required_child_verifier_candidate_indeterminate"
            return out, 3
        out["disposition"] = "CANDIDATE_MEASURED"
        out["detail"] = "wcare40_graph_recomputed_and_candidate_authentication_overlay_measured"
        return out, 0
    except InvalidEvidence as exc:
        out["disposition"] = "CANDIDATE_INVALID"
        out["detail"] = str(exc)
        out["builder_authentication_candidate_status"] = "INVALID"
        out["temporal_candidate_status"] = "INVALID"
        return out, 4
    except Exception as exc:
        out["disposition"] = "CANDIDATE_INVALID"
        out["detail"] = f"unexpected_kernel_error:{type(exc).__name__}:{exc}"
        out["builder_authentication_candidate_status"] = "INVALID"
        out["temporal_candidate_status"] = "INVALID"
        return out, 4


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("wcare40_plan")
    p.add_argument("wcare40_result")
    p.add_argument("wcare41_authentication_plan")
    p.add_argument("temporal_observation")
    p.add_argument("--provenance", action="append", default=[], required=True)
    p.add_argument("--relation", action="append", default=[], required=True)
    p.add_argument("--builder-observation", action="append", default=[])
    return p


def main() -> int:
    result, code = evaluate(parser().parse_args())
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
