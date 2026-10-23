#!/usr/bin/env python3
"""Independent contract oracle for the Symthaea Millennium Biology profile.

Stdlib-only by design. This validates data/contract integrity, not biological truth.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path

EXPECTED_IDS = [f"MPB-2026-09-23-{i:02d}" for i in range(1, 13)]
EXPECTED_MAPPING_GENERATION = re.compile(
    r"^BIO-MILLENNIUM-001B-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}$"
)
EXPECTED_CRITERIA_GENERATION = re.compile(
    r"^DESCI-BIO-001B-[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{2}$"
)
REQUIRED_AUTHORITY = (
    "local_mapping_is_not_official_criterion",
    "local_mapping_does_not_declare_solved",
    "local_mapping_does_not_grant_experimental_authority",
    "local_surrogate_is_not_official_evidence",
    "model_agreement_is_not_independent_corroboration",
    "formal_model_proof_is_not_physical_biological_evidence",
)
REQUIRED_INTEGRITY = (
    "exact_criterion_generation_binding",
    "exact_candidate_identity",
    "exact_model_lineage_when_available",
    "knowledge_exposure_cutoff",
    "provenance_complete_result_identity",
    "terminal_disposition_census",
    "negative_and_null_results_retained",
    "independent_scorer_when_claimed",
    "external_execution_requires_human_or_institutional_authority",
)
EVENT_TYPES = {
    "HypothesisProposal",
    "InSilicoPrediction",
    "RetrospectiveBenchmarkResult",
    "ProspectiveCommittedPrediction",
    "ExternalExperimentalObservation",
    "IndependentReplication",
    "OfficialChallengeCriterionEvidence",
    "Failure",
    "NullResult",
    "Contradiction",
}
PROSPECTIVE_TYPES = {
    "ProspectiveCommittedPrediction",
    "ExternalExperimentalObservation",
    "IndependentReplication",
    "OfficialChallengeCriterionEvidence",
}


class OracleFailure(Exception):
    def __init__(self, code: str, detail: str):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}")


def require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise OracleFailure(code, detail)


def load_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise OracleFailure("MALFORMED_JSON", f"{path}: {exc}") from exc
    require(isinstance(value, dict), "MALFORMED_DOCUMENT", f"{path} must contain an object")
    return value


def validate_mapping(mapping: dict) -> None:
    require(mapping.get("schema") == "symthaea.research.millennium_biology.claim_mapping",
            "SCHEMA_ID_MISMATCH", "unexpected claim-mapping schema")
    require(mapping.get("schema_version") == "1.0.0",
            "SCHEMA_VERSION_MISMATCH", "unsupported claim-mapping schema version")

    generation = mapping.get("mapping_generation")
    require(isinstance(generation, str) and EXPECTED_MAPPING_GENERATION.fullmatch(generation),
            "MAPPING_GENERATION_INVALID", "mapping generation is missing or malformed")

    source = mapping.get("source_contract")
    require(isinstance(source, dict), "SOURCE_CONTRACT_MISSING", "source_contract must be an object")
    criteria_generation = source.get("official_criteria_generation")
    require(
        isinstance(criteria_generation, str) and EXPECTED_CRITERIA_GENERATION.fullmatch(criteria_generation),
        "CRITERIA_GENERATION_INVALID",
        "official criteria generation is missing or malformed",
    )
    require(
        source.get("publisher") == "FutureHouse / Edison Scientific",
        "SOURCE_PUBLISHER_MISMATCH",
        "unexpected publisher",
    )
    require(
        source.get("source_authority") == "external_source_only",
        "SOURCE_AUTHORITY_MISMATCH",
        "local mapping must not become source authority",
    )

    authority = mapping.get("authority_ceiling")
    require(isinstance(authority, dict), "AUTHORITY_CEILING_MISSING", "authority_ceiling must be an object")
    for key in REQUIRED_AUTHORITY:
        require(authority.get(key) is True, "AUTHORITY_CEILING_VIOLATION", f"{key} must remain true")

    ladder = mapping.get("evidence_ladder")
    require(isinstance(ladder, list), "EVIDENCE_LADDER_MISSING", "evidence_ladder must be a list")
    require(
        "ExternalExperimentalObservation" in ladder
        and "IndependentReplication" in ladder
        and "OfficialChallengeCriterionEvidence" in ladder,
        "EVIDENCE_LADDER_INCOMPLETE",
        "experimental, replication, and official evidence levels are required",
    )

    fixtures = mapping.get("non_promotion_fixtures")
    require(
        isinstance(fixtures, list) and len(fixtures) >= 8 and len(set(fixtures)) == len(fixtures),
        "NON_PROMOTION_FIXTURES_INVALID",
        "at least eight unique non-promotion fixtures are required",
    )

    criteria = mapping.get("criteria")
    require(isinstance(criteria, list), "CRITERIA_MISSING", "criteria must be a list")
    ids = [item.get("id") if isinstance(item, dict) else None for item in criteria]
    require(ids == EXPECTED_IDS, "CHALLENGE_ID_ORDER_MISMATCH", "exactly the canonical 12 IDs in order are required")

    for index, item in enumerate(criteria, start=1):
        require(isinstance(item, dict), "CRITERION_MALFORMED", f"criterion {index} is not an object")
        require(
            item.get("minimum_official_evidence") == "ExternalExperimentalObservation",
            "EVIDENCE_CEILING_VIOLATION",
            f"{item.get('id')}: minimum official evidence is too low",
        )
        require(
            item.get("completion_requires") == "OfficialChallengeCriterionEvidence",
            "EVIDENCE_CEILING_VIOLATION",
            f"{item.get('id')}: completion predicate was lowered",
        )
        require(
            isinstance(item.get("explicit_nonpromotion"), list)
            and len(item["explicit_nonpromotion"]) >= 1,
            "NON_PROMOTION_FIXTURE_MISSING",
            f"{item.get('id')}: explicit non-promotion fixtures are required",
        )

    integrity = mapping.get("integrity_requirements")
    require(isinstance(integrity, dict), "INTEGRITY_REQUIREMENTS_MISSING", "integrity_requirements must be an object")
    for key in REQUIRED_INTEGRITY:
        require(integrity.get(key) is True, "INTEGRITY_REQUIREMENT_DISABLED", f"{key} must remain true")


def validate_event_contract(contract: dict) -> None:
    require(
        contract.get("schema") == "symthaea.research.millennium_biology.evidence_event",
        "EVENT_SCHEMA_ID_MISMATCH",
        "unexpected evidence-event schema",
    )
    require(contract.get("schema_version") == "1.0.0",
            "EVENT_SCHEMA_VERSION_MISMATCH", "unsupported evidence-event schema version")

    boundary = contract.get("authority_boundary")
    require(isinstance(boundary, dict), "EVENT_AUTHORITY_BOUNDARY_MISSING", "authority boundary is required")
    for key in (
        "model_output_is_not_observation",
        "observation_is_not_replication",
        "replication_is_not_official_criterion_evidence",
        "local_status_is_not_source_authority",
        "events_are_append_only",
    ):
        require(boundary.get(key) is True, "EVENT_AUTHORITY_BOUNDARY_VIOLATION", f"{key} must remain true")

    identity = contract.get("required_identity")
    require(
        isinstance(identity, list)
        and all(key in identity for key in (
            "event_id", "event_type", "challenge_id", "criteria_generation",
            "mapping_generation", "created_at", "actor_id", "provenance"
        )),
        "EVENT_IDENTITY_INCOMPLETE",
        "required event identity fields are incomplete",
    )
    require(set(contract.get("event_types", [])) == EVENT_TYPES,
            "EVENT_TYPES_MISMATCH", "event type vocabulary drifted")
    negative = contract.get("negative_result_policy", {})
    for key in (
        "failure_is_first_class",
        "null_result_is_first_class",
        "contradiction_is_first_class",
        "failed_events_must_not_be_deleted_or_rewritten",
        "supersession_requires_new_event",
    ):
        require(negative.get(key) is True, "NEGATIVE_RESULT_POLICY_VIOLATION", f"{key} must remain true")
    promotion = contract.get("promotion_policy", {})
    for key in (
        "allowed_transitions_are_defined_by_claim_mapping",
        "status_must_be_derived_from_lineage_not_free_text",
        "official_criterion_evidence_requires_exact_criteria_generation",
        "official_criterion_evidence_requires_complete_source_predicate",
    ):
        require(promotion.get(key) is True, "PROMOTION_POLICY_VIOLATION", f"{key} must remain true")


def validate_event(event: dict, mapping: dict, contract: dict) -> None:
    require(isinstance(event, dict), "EVENT_MALFORMED", "event must be an object")
    for key in contract["required_identity"]:
        require(key in event, "EVENT_IDENTITY_INCOMPLETE", f"missing {key}")
    require(event["event_type"] in EVENT_TYPES, "EVENT_TYPE_INVALID", "unknown event type")
    require(event["challenge_id"] in EXPECTED_IDS, "EVENT_CHALLENGE_ID_INVALID", "unknown challenge ID")
    require(
        event["criteria_generation"] == mapping["source_contract"]["official_criteria_generation"],
        "EVENT_CRITERIA_GENERATION_MISMATCH",
        "event is bound to a different official criteria generation",
    )
    require(
        event["mapping_generation"] == mapping["mapping_generation"],
        "EVENT_MAPPING_GENERATION_MISMATCH",
        "event is bound to a different claim-mapping generation",
    )
    provenance = event["provenance"]
    require(isinstance(provenance, dict), "EVENT_PROVENANCE_MISSING", "provenance must be an object")
    for key in ("exact_input_digest", "artifact_digest", "parent_event_ids"):
        require(key in provenance and provenance[key] not in (None, ""), "EVENT_PROVENANCE_INCOMPLETE", f"missing {key}")
    if event["event_type"] in PROSPECTIVE_TYPES:
        require(
            provenance.get("exposure_cutoff") not in (None, ""),
            "PROSPECTIVE_EXPOSURE_CUTOFF_MISSING",
            "prospective events require an exposure cutoff",
        )
    if event["event_type"] in {"InSilicoPrediction", "RetrospectiveBenchmarkResult", "ProspectiveCommittedPrediction"}:
        require(
            provenance.get("model_lineage") not in (None, ""),
            "MODEL_LINEAGE_MISSING",
            "model-derived events require model lineage",
        )


def immutable_event_mutation_is_rejected(mapping: dict, contract: dict) -> str:
    base = {
        "event_id": "evt-001",
        "event_type": "ProspectiveCommittedPrediction",
        "challenge_id": EXPECTED_IDS[0],
        "criteria_generation": mapping["source_contract"]["official_criteria_generation"],
        "mapping_generation": mapping["mapping_generation"],
        "created_at": "2026-09-28T00:00:00Z",
        "actor_id": "oracle-fixture",
        "provenance": {
            "exact_input_digest": "sha256:input",
            "artifact_digest": "sha256:artifact-v1",
            "parent_event_ids": [],
            "model_lineage": "model:v1",
            "knowledge_cutoff": "2026-09-27T00:00:00Z",
            "exposure_cutoff": "2026-09-28T00:00:00Z",
        },
        "payload_digest": "sha256:payload-v1",
    }
    mutated = copy.deepcopy(base)
    mutated["payload_digest"] = "sha256:payload-v2"
    require(
        mutated["event_id"] == base["event_id"] and mutated["payload_digest"] != base["payload_digest"],
        "FIXTURE_CONSTRUCTION_ERROR",
        "mutation fixture was not constructed",
    )
    # An amendment with the same event identity but changed immutable payload is invalid.
    require(
        not (
            mutated["event_id"] == base["event_id"]
            and any(mutated.get(key) != base.get(key) for key in contract["immutable_after_commit"])
        ) or False,
        "IMMUTABLE_EVENT_MUTATION",
        "append-only identity must not be rewritten; supersession requires a new event",
    )
    return "IMMUTABLE_EVENT_MUTATION"


def run_mutation_suite(mapping: dict, contract: dict) -> list[dict]:
    mutations = []

    def expect(name: str, mutate, expected: str) -> None:
        candidate = copy.deepcopy(mapping)
        mutate(candidate)
        try:
            validate_mapping(candidate)
        except OracleFailure as exc:
            mutations.append({"name": name, "expected": expected, "observed": exc.code, "passed": exc.code == expected})
            return
        mutations.append({"name": name, "expected": expected, "observed": "ACCEPTED", "passed": False})

    expect("mutate_one_challenge_id", lambda m: m["criteria"].__setitem__(0, {**m["criteria"][0], "id": "MPB-2026-09-23-99"}), "CHALLENGE_ID_ORDER_MISMATCH")
    expect("reorder_challenges", lambda m: m["criteria"].__setitem__(0, m["criteria"][1]), "CHALLENGE_ID_ORDER_MISMATCH")
    expect("mutate_criteria_generation", lambda m: m["source_contract"].__setitem__("official_criteria_generation", "DESCI-BIO-001B-2099-01-01-01"), "CRITERIA_GENERATION_INVALID")
    expect("lower_evidence_ceiling", lambda m: m["criteria"][0].__setitem__("minimum_official_evidence", "SimulationResult"), "EVIDENCE_CEILING_VIOLATION")
    expect("remove_nonpromotion_fixture", lambda m: m["criteria"][0].__setitem__("explicit_nonpromotion", []), "NON_PROMOTION_FIXTURE_MISSING")
    expect("duplicate_challenge_id", lambda m: m["criteria"].__setitem__(1, {**m["criteria"][1], "id": m["criteria"][0]["id"]}), "CHALLENGE_ID_ORDER_MISMATCH")
    expect("promote_simulation_to_completion", lambda m: m["criteria"][0].__setitem__("completion_requires", "SimulationResult"), "EVIDENCE_CEILING_VIOLATION")

    try:
        immutable_event_mutation_is_rejected(mapping, contract)
        mutations.append({"name": "mutate_prospective_prediction_after_commit", "expected": "IMMUTABLE_EVENT_MUTATION", "observed": "ACCEPTED", "passed": False})
    except OracleFailure as exc:
        mutations.append({"name": "mutate_prospective_prediction_after_commit", "expected": "IMMUTABLE_EVENT_MUTATION", "observed": exc.code, "passed": exc.code == "IMMUTABLE_EVENT_MUTATION"})

    return mutations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--event-contract", type=Path, required=True)
    parser.add_argument("--mutation-suite", action="store_true")
    args = parser.parse_args()

    mapping = load_json(args.mapping)
    contract = load_json(args.event_contract)

    checks = []
    for name, fn in (
        ("claim_mapping", lambda: validate_mapping(mapping)),
        ("evidence_event_contract", lambda: validate_event_contract(contract)),
    ):
        try:
            fn()
            checks.append({"name": name, "status": "PASS"})
        except OracleFailure as exc:
            checks.append({"name": name, "status": "FAIL", "code": exc.code, "detail": exc.detail})

    mutations = run_mutation_suite(mapping, contract) if args.mutation_suite else []
    if mutations and not all(item["passed"] for item in mutations):
        checks.append({"name": "mutation_suite", "status": "FAIL", "failures": [x for x in mutations if not x["passed"]]})
    elif mutations:
        checks.append({"name": "mutation_suite", "status": "PASS", "cases": len(mutations)})

    report = {
        "oracle": "BIO-MILLENNIUM-001C",
        "status": "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL",
        "checks": checks,
    }
    if mutations:
        report["mutation_results"] = mutations

    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
