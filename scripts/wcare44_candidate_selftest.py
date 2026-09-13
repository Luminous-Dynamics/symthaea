#!/usr/bin/env python3
"""Dependency-free adversarial campaign for the WCARE-44 Stage-A kernel."""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
KERNEL = ROOT / "scripts/wcare44_candidate_kernel.py"
P = "wcare44-authenticated-replication-aggregation-v1"
W40 = "wcare40-execution-replication-v1"
W41 = "wcare41-authenticated-preregistration-v1"


def raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, value: object) -> str:
    data = raw(value)
    path.write_bytes(data)
    return digest(data)


def hx(ch: str) -> str:
    return ch * 64


def provenance(replica_id: str, n: int, plan_sha: str) -> dict:
    digits = [format((n + offset) % 16, "x") for offset in range(6)]
    return {
        "protocol_version": W40,
        "plan_sha256": plan_sha,
        "replica_id": replica_id,
        "execution_observed": True,
        "wcare39_prepared_capsule_sha256": hx(format((n + 6) % 16, "x")),
        "wcare39_final_capsule_sha256": hx(format((n + 7) % 16, "x")),
        "builder_identity_commitment_sha256": hx(digits[0]),
        "organization_commitment_sha256": hx(digits[1]),
        "infrastructure_commitment_sha256": hx(digits[2]),
        "toolchain_lineage_commitment_sha256": hx(digits[3]),
        "operator_process_commitment_sha256": hx(digits[4]),
        "evidence_source_commitment_sha256": hx(digits[5]),
        "provenance_strength": "ExternalVerified",
        "conflict_of_interest": False,
        "collected_utc": "2026-09-13T20:00:00Z",
    }


def relation(left: str, right: str, prov: dict[str, dict], plan_sha: str, kind: str = "Independent") -> dict:
    shared_labels = []
    field_labels = [
        ("builder_identity_commitment_sha256", "BuilderIdentity"),
        ("organization_commitment_sha256", "Organization"),
        ("infrastructure_commitment_sha256", "Infrastructure"),
        ("toolchain_lineage_commitment_sha256", "ToolchainLineage"),
        ("operator_process_commitment_sha256", "OperatorProcess"),
        ("evidence_source_commitment_sha256", "EvidenceSource"),
    ]
    for field, label in field_labels:
        if prov[left][field] == prov[right][field]:
            shared_labels.append(label)
    return {
        "protocol_version": W40,
        "plan_sha256": plan_sha,
        "left_replica_id": left,
        "right_replica_id": right,
        "left_builder_identity_commitment_sha256": prov[left]["builder_identity_commitment_sha256"],
        "right_builder_identity_commitment_sha256": prov[right]["builder_identity_commitment_sha256"],
        "relation": kind,
        "declared_shared_fault_domains": sorted(shared_labels),
        "relation_evidence_strength": "ExternalVerified",
        "assessed_utc": "2026-09-13T20:01:00Z",
    }


def component_census(nodes: list[str], related_pair: tuple[str, str] | None) -> list[list[str]]:
    if related_pair is None:
        return [[x] for x in sorted(nodes)]
    pair = sorted(related_pair)
    other = sorted(set(nodes) - set(pair))
    components = [pair]
    components.extend([[x] for x in other])
    return sorted(components, key=lambda x: tuple(x))


def fixture(tmp: Path, *, related_pair: tuple[str, str] | None = None) -> dict:
    ids = ["a", "b", "c"]
    placeholder_plan = {
        "protocol_version": W40,
        "wcare39_protocol_sha256": hx("a"),
        "wcare39_runner_sha256": hx("b"),
        "wcare39_integrity_sha256": hx("c"),
        "plan_created_utc": "2026-09-13T19:00:00Z",
        "minimum_qualified_replicas": 2,
        "minimum_effective_independent_components": 2,
        "accepted_builder_provenance_strengths": ["ExternalVerified"],
        "accepted_independent_relation_strengths": ["ExternalVerified"],
        "required_receipt_stage_ids": [],
        "require_subject_outcome_agreement": True,
        "require_required_receipt_agreement": True,
        "require_no_conflict_of_interest": True,
        "replica_slots": [],
    }
    seed_prov = {
        "a": provenance("a", 1, hx("0")),
        "b": provenance("b", 8, hx("0")),
        "c": provenance("c", 13, hx("0")),
    }
    placeholder_plan["replica_slots"] = [
        {"replica_id": x, "builder_identity_commitment_sha256": seed_prov[x]["builder_identity_commitment_sha256"]}
        for x in ids
    ]
    plan = tmp / "plan.json"
    plan_sha = write(plan, placeholder_plan)

    prov_values = {x: provenance(x, {"a": 1, "b": 8, "c": 13}[x], plan_sha) for x in ids}
    prov_paths: dict[str, Path] = {}
    prov_sha: dict[str, str] = {}
    for x in ids:
        path = tmp / f"prov-{x}.json"
        prov_paths[x] = path
        prov_sha[x] = write(path, prov_values[x])

    rel_paths: dict[tuple[str, str], Path] = {}
    rel_sha: dict[tuple[str, str], str] = {}
    for left, right in itertools.combinations(ids, 2):
        kind = "Related" if related_pair and set((left, right)) == set(related_pair) else "Independent"
        path = tmp / f"rel-{left}-{right}.json"
        key = (left, right)
        rel_paths[key] = path
        rel_sha[key] = write(path, relation(left, right, prov_values, plan_sha, kind))

    baseline_components = component_census(ids, related_pair)
    accepted_pairs = 3 - (1 if related_pair else 0)
    result_value = {
        "authority": "MeasurementOnly",
        "protocol_version": W40,
        "wcare40_frontdoor_sha256": hx("d"),
        "wcare40_core_verifier_sha256": hx("e"),
        "plan_sha256": plan_sha,
        "expected_replica_ids": ids,
        "subject_eligible_replica_ids": ids,
        "builder_provenance_receipt_sha256s": [prov_sha[x] for x in ids],
        "builder_relation_receipt_sha256s": [rel_sha[k] for k in sorted(rel_sha)],
        "raw_distinct_builder_identity_count": 3,
        "qualified_builder_replica_ids": ids,
        "qualified_distinct_builder_identity_count": 3,
        "accepted_independent_pair_count": accepted_pairs,
        "downgraded_independent_pair_count": 0,
        "effective_independent_components": len(baseline_components),
        "independence_component_census": baseline_components,
        "conflict_policy_met": True,
        "disposition": "REPLICATION_SUPPORTED",
        "preregistration_temporal_precedence_established": False,
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
    result = tmp / "result.json"
    result_sha = write(result, result_value)

    auth_value = {
        "protocol_version": W41,
        "wcare40_plan_sha256": plan_sha,
        "wcare40_result_sha256": result_sha,
        "wcare40_frontdoor_sha256": hx("d"),
        "wcare40_core_verifier_sha256": hx("e"),
        "builder_verifier": {"backend_id": "w42", "executable_sha256": hx("1"), "policy_sha256": hx("2")},
        "temporal_verifier": {"backend_id": "w43", "executable_sha256": hx("3"), "policy_sha256": hx("4")},
        "require_complete_builder_attestation_coverage": True,
        "require_temporal_preregistration": True,
        "evaluation_utc": "2026-09-14T00:00:00Z",
    }
    auth = tmp / "auth-plan.json"
    auth_sha = write(auth, auth_value)

    return {
        "ids": ids,
        "plan": plan,
        "plan_sha": plan_sha,
        "result": result,
        "result_sha": result_sha,
        "auth": auth,
        "auth_sha": auth_sha,
        "prov_paths": prov_paths,
        "prov_sha": prov_sha,
        "rel_paths": rel_paths,
        "rel_sha": rel_sha,
        "baseline_components": baseline_components,
    }


def builder_observation(path: Path, subject_sha: str, kind: str, *, status: str = "ACCEPTED", qualified: bool = True, synthetic: bool = False) -> Path:
    write(path, {
        "protocol_version": P,
        "subject_receipt_sha256": subject_sha,
        "subject_kind": kind,
        "status": status,
        "wcare42_result_sha256": hx("5"),
        "wcare42_verifier_sha256": hx("6"),
        "wcare42_qualification_receipt_sha256": hx("7"),
        "verifier_execution_qualified": qualified,
        "synthetic": synthetic,
    })
    return path


def temporal_observation(path: Path, f: dict, *, status: str = "ESTABLISHED", qualified: bool = True, synthetic: bool = False) -> Path:
    write(path, {
        "protocol_version": P,
        "wcare40_result_sha256": f["result_sha"],
        "wcare41_authentication_plan_sha256": f["auth_sha"],
        "status": status,
        "wcare43_result_sha256": hx("8"),
        "wcare43_verifier_sha256": hx("9"),
        "verifier_execution_qualified": qualified,
        "synthetic": synthetic,
    })
    return path


def all_builder_observations(tmp: Path, f: dict) -> list[Path]:
    result: list[Path] = []
    for replica_id in f["ids"]:
        result.append(builder_observation(tmp / f"obs-prov-{replica_id}.json", f["prov_sha"][replica_id], "BuilderProvenance"))
    for pair, subject_sha in f["rel_sha"].items():
        result.append(builder_observation(tmp / f"obs-rel-{pair[0]}-{pair[1]}.json", subject_sha, "BuilderRelation"))
    return result


def argv(f: dict, temporal: Path, observations: list[Path]) -> list[str]:
    cmd = [
        sys.executable,
        str(KERNEL),
        str(f["plan"]),
        str(f["result"]),
        str(f["auth"]),
        str(temporal),
    ]
    for replica_id in f["ids"]:
        cmd += ["--provenance", str(f["prov_paths"][replica_id])]
    for pair in sorted(f["rel_paths"]):
        cmd += ["--relation", str(f["rel_paths"][pair])]
    for path in observations:
        cmd += ["--builder-observation", str(path)]
    return cmd


def execute(cmd: list[str]) -> tuple[int, dict]:
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    try:
        value = json.loads(result.stdout)
    except Exception as exc:
        raise AssertionError(f"invalid kernel JSON: rc={result.returncode} out={result.stdout!r} err={result.stderr!r}") from exc
    return result.returncode, value


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="wcare44-selftest-") as raw_tmp:
        root = Path(raw_tmp)

        full = root / "full"
        full.mkdir()
        f = fixture(full)
        observations = all_builder_observations(full, f)
        temporal = temporal_observation(full / "temporal.json", f)
        rc, value = execute(argv(f, temporal, observations))
        assert rc == 0 and value["disposition"] == "CANDIDATE_MEASURED", value
        assert value["wcare40_effective_independent_components"] == 3, value
        assert value["candidate_authenticated_effective_independent_components"] == 3, value
        assert value["candidate_builder_authentication_requirements_met"] is True, value
        assert value["candidate_temporal_requirements_met"] is True, value
        assert value["candidate_authenticated_preregistered_replication"] is True, value
        assert value["builder_authentication_established"] is False, value
        assert value["preregistration_temporal_precedence_established"] is False, value
        assert value["authenticated_preregistered_replication_established"] is False, value

        missing_rel = root / "missing-rel"
        missing_rel.mkdir()
        f2 = fixture(missing_rel)
        obs2 = all_builder_observations(missing_rel, f2)
        missing_digest = f2["rel_sha"][("a", "b")]
        obs2 = [p for p in obs2 if json.loads(p.read_text())["subject_receipt_sha256"] != missing_digest]
        temporal2 = temporal_observation(missing_rel / "temporal.json", f2)
        rc, value = execute(argv(f2, temporal2, obs2))
        assert rc == 0, value
        assert value["candidate_authenticated_effective_independent_components"] == 2, value
        assert value["candidate_complete_builder_coverage"] is False, value
        assert any(e["left_replica_id"] == "a" and e["right_replica_id"] == "b" and e["reasons"] == ["RelationUnauthenticated"] for e in value["added_conservative_edges"]), value

        missing_endpoint = root / "missing-endpoint"
        missing_endpoint.mkdir()
        f3 = fixture(missing_endpoint)
        obs3 = all_builder_observations(missing_endpoint, f3)
        missing_prov = f3["prov_sha"]["a"]
        obs3 = [p for p in obs3 if json.loads(p.read_text())["subject_receipt_sha256"] != missing_prov]
        temporal3 = temporal_observation(missing_endpoint / "temporal.json", f3)
        rc, value = execute(argv(f3, temporal3, obs3))
        assert rc == 0, value
        assert value["candidate_authenticated_effective_independent_components"] == 1, value
        assert len(value["added_conservative_edges"]) == 2, value

        baseline_edge = root / "baseline-edge"
        baseline_edge.mkdir()
        f4 = fixture(baseline_edge, related_pair=("a", "b"))
        obs4 = all_builder_observations(baseline_edge, f4)
        temporal4 = temporal_observation(baseline_edge / "temporal.json", f4)
        rc, value = execute(argv(f4, temporal4, obs4))
        assert rc == 0, value
        assert value["wcare40_component_census"] == [["a", "b"], ["c"]], value
        assert value["candidate_component_census"] == [["a", "b"], ["c"]], value
        assert value["baseline_partition_not_split"] is True, value

        temporal_only = root / "temporal-only"
        temporal_only.mkdir()
        f5 = fixture(temporal_only)
        temporal5 = temporal_observation(temporal_only / "temporal.json", f5)
        rc, value = execute(argv(f5, temporal5, []))
        assert rc == 0, value
        assert value["candidate_temporal_requirements_met"] is True, value
        assert value["candidate_authenticated_effective_independent_components"] == 1, value
        assert value["candidate_builder_authentication_requirements_met"] is False, value
        assert value["candidate_authenticated_preregistered_replication"] is False, value

        builder_only = root / "builder-only"
        builder_only.mkdir()
        f6 = fixture(builder_only)
        obs6 = all_builder_observations(builder_only, f6)
        temporal6 = temporal_observation(builder_only / "temporal.json", f6, status="NOT_ESTABLISHED")
        rc, value = execute(argv(f6, temporal6, obs6))
        assert rc == 0, value
        assert value["candidate_builder_authentication_requirements_met"] is True, value
        assert value["candidate_temporal_requirements_met"] is False, value
        assert value["candidate_authenticated_preregistered_replication"] is False, value

        synthetic_temporal = root / "synthetic-temporal"
        synthetic_temporal.mkdir()
        f7 = fixture(synthetic_temporal)
        obs7 = all_builder_observations(synthetic_temporal, f7)
        temporal7 = temporal_observation(synthetic_temporal / "temporal.json", f7, synthetic=True)
        rc, value = execute(argv(f7, temporal7, obs7))
        assert rc == 0 and value["temporal_candidate_status"] == "NOT_ESTABLISHED", value
        assert value["candidate_authenticated_preregistered_replication"] is False, value

        indeterminate = root / "indeterminate"
        indeterminate.mkdir()
        f8 = fixture(indeterminate)
        obs8 = all_builder_observations(indeterminate, f8)
        first = obs8[0]
        first_data = json.loads(first.read_text())
        first_data["status"] = "INDETERMINATE"
        write(first, first_data)
        temporal8 = temporal_observation(indeterminate / "temporal.json", f8)
        rc, value = execute(argv(f8, temporal8, obs8))
        assert rc == 3 and value["disposition"] == "CANDIDATE_INDETERMINATE", value
        assert value["builder_authentication_candidate_status"] == "INDETERMINATE", value

        duplicate = root / "duplicate"
        duplicate.mkdir()
        f9 = fixture(duplicate)
        obs9 = all_builder_observations(duplicate, f9)
        duplicate_path = duplicate / "duplicate-observation.json"
        duplicate_path.write_bytes(obs9[0].read_bytes())
        temporal9 = temporal_observation(duplicate / "temporal.json", f9)
        rc, value = execute(argv(f9, temporal9, obs9 + [duplicate_path]))
        assert rc == 4 and value["disposition"] == "CANDIDATE_INVALID", value
        assert "duplicate_builder_observation" in value["detail"], value

        forged = root / "forged-count"
        forged.mkdir()
        f10 = fixture(forged)
        forged_result = json.loads(f10["result"].read_text())
        forged_result["effective_independent_components"] = 2
        f10["result_sha"] = write(f10["result"], forged_result)
        auth_data = json.loads(f10["auth"].read_text())
        auth_data["wcare40_result_sha256"] = f10["result_sha"]
        f10["auth_sha"] = write(f10["auth"], auth_data)
        obs10 = all_builder_observations(forged, f10)
        temporal10 = temporal_observation(forged / "temporal.json", f10)
        rc, value = execute(argv(f10, temporal10, obs10))
        assert rc == 4 and value["disposition"] == "CANDIDATE_INVALID", value
        assert value["detail"] == "wcare40_component_count_mismatch", value

        temporal_invalid = root / "temporal-invalid"
        temporal_invalid.mkdir()
        f11 = fixture(temporal_invalid)
        obs11 = all_builder_observations(temporal_invalid, f11)
        temporal11 = temporal_observation(temporal_invalid / "temporal.json", f11, status="INVALID")
        rc, value = execute(argv(f11, temporal11, obs11))
        assert rc == 4 and value["disposition"] == "CANDIDATE_INVALID", value
        assert value["detail"] == "temporal_candidate_invalid", value

    print(json.dumps({
        "authority": "MeasurementOnly",
        "classification": "PASS_WCARE44_CANDIDATE_SELFTEST",
        "full_candidate_conjunction_without_final_promotion_verified": True,
        "missing_relation_collapses_one_separation": True,
        "missing_endpoint_collapses_multiple_separations": True,
        "baseline_edge_cannot_be_removed": True,
        "temporal_only_cannot_authenticate_builders": True,
        "builder_only_cannot_establish_temporal_precedence": True,
        "synthetic_temporal_cannot_establish_candidate_precedence": True,
        "indeterminate_builder_state_preserved": True,
        "duplicate_observation_rejected": True,
        "forged_wcare40_component_count_rejected": True,
        "invalid_temporal_state_rejected": True,
        "runtime_authority_granted": False,
    }, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
