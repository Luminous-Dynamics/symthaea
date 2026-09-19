#!/usr/bin/env python3
"""Static self-test for REL-005A QualificationDurabilityContractOnly."""
from __future__ import annotations
import argparse, json, pathlib
from typing import Any

SCHEMA = "symthaea.rel.qualification-durability-contract.v1"
AUTHORITY = "QualificationDurabilityContractOnly"
ALLOWED_RESULTS = ["ALL_PREDICATES_PASS", "PREDICATE_FAILURES"]
ORDER = [
    "QualificationInputFirewallOnly",
    "QualificationOnly",
    "PostQualificationFirewallOnly",
    "QualificationEvidencePersistenceOnly",
    "QualificationCapsulePackagingOnly",
    "AttestationOnly",
]
REQUIRED_FILES = {
    "predicate-contract-receipt.json",
    "execution-v3-receipt.json",
    "observation-seal-v3.json",
    "comparison-only-qualification-receipt.json",
    "qualification-input-manifest.json",
    "qualification-assembly-receipt.json",
    "qualification-only.json",
    "post-qualification-firewall-receipt.json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), "contract must be a JSON object")
    return value


def validate(contract: dict[str, Any]) -> dict[str, Any]:
    require(contract.get("schema") == SCHEMA, "schema mismatch")
    require(contract.get("authority") == AUTHORITY, "authority mismatch")
    require(contract.get("relation") == "REL-005A", "relation mismatch")
    require(contract.get("allowed_comparison_results") == ALLOWED_RESULTS, "allowed outcomes changed")
    require(contract.get("selected_comparison_result") is None, "durability contract selected an outcome")
    require(contract.get("outcome_independent") is True, "durability policy is not outcome-independent")
    require("qualification_mapping" not in contract, "durability layer must not contain scientific mapping")
    require(contract.get("authority_order") == ORDER, "authority order mismatch")

    jobs = contract["job_boundaries"]
    require(jobs == {
        "qualification_job_ends_after_durable_evidence": True,
        "capsule_packaging_runs_in_fresh_downstream_job": True,
        "capsule_job_may_receive_only_durable_qualification_evidence": True,
        "attestation_runs_in_fresh_downstream_job": True,
        "attestation_job_may_receive_only_final_capsule": True,
    }, "job-boundary contract mismatch")

    evidence = contract["durable_qualification_evidence"]
    require(evidence["exact_file_census"] is True, "durable artifact census is not exact")
    require(set(evidence["required_files"]) == REQUIRED_FILES, "durable evidence census mismatch")
    require(len(evidence["required_files"]) == len(REQUIRED_FILES), "duplicate durable evidence file")
    for key in (
        "raw_observation_allowed",
        "execution_logs_allowed",
        "detailed_predicate_values_allowed",
        "scientific_thresholds_allowed",
        "transport_identity_is_scientific_authority",
    ):
        require(evidence[key] is False, f"forbidden evidence property enabled: {key}")
    require(evidence["inner_content_commitments_are_authoritative"] is True, "content authority missing")
    require(evidence["upload_must_complete_before_capsule_packaging"] is True, "persistence is not ordered before packaging")

    failure = contract["failure_semantics"]
    require(failure["persistence_failure_before_durable_artifact"] == "QUALIFICATION_EVIDENCE_NOT_DURABLE", "persistence failure semantics mismatch")
    require(failure["capsule_failure_after_durable_artifact_mutates_qualification_result"] is False, "capsule failure may mutate qualification")
    require(failure["attestation_failure_after_durable_artifact_mutates_qualification_result"] is False, "attestation failure may mutate qualification")
    require(failure["workflow_conclusion_is_scientific_authority"] is False, "workflow conclusion promoted to scientific authority")
    require(failure["downstream_packaging_may_not_rewrite_qualification_only_json"] is True, "packaging may rewrite qualification output")

    require(contract["forbidden_result_dependent_durability_policy"] is True, "result-dependent durability not forbidden")
    require(not any(contract["claims"].values()), "contract makes runtime/scientific claims")

    # Both legal scientific outcomes must encounter the exact same persistence topology.
    topology_by_result = {result: tuple(ORDER) for result in ALLOWED_RESULTS}
    census_by_result = {result: frozenset(REQUIRED_FILES) for result in ALLOWED_RESULTS}
    require(len(set(topology_by_result.values())) == 1, "result-dependent authority topology")
    require(len(set(census_by_result.values())) == 1, "result-dependent evidence census")

    return {
        "schema": "symthaea.rel.qualification-durability-contract-static-receipt.v1",
        "authority": AUTHORITY,
        "self_tests_passed": True,
        "outcome_blind": True,
        "allowed_result_count": 2,
        "required_evidence_file_count": len(REQUIRED_FILES),
        "qualification_mapping_present": False,
        "persistence_precedes_packaging": True,
        "packaging_failure_mutates_qualification": False,
        "attestation_failure_mutates_qualification": False,
        "claims": {
            "durability_implemented": False,
            "qualification_completed": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=pathlib.Path, required=True)
    args = parser.parse_args()
    print(json.dumps(validate(load(args.contract)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
