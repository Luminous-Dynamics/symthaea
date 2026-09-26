#!/usr/bin/env python3
"""Independent reference validator for CIV-VALUE-PILOT-001A.

No Symthaea production code is imported. The validator binds both the exact
CIV-VALUE-000A domain matrix and the exact seven-chain synthetic benchmark.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MATRIX_PATH = Path("docs/release/evidence/civ-value-000-critical-domain-matrix-v1.json")
CHAIN_PATH = Path("docs/release/evidence/civ-value-pilot-001-reference-chains-v1.json")

EXPECTED_MATRIX_SHA256 = "67abb09708153f10879d06121213dc1dae0f5de558adae857e670fee51cc5ae6"
EXPECTED_CHAIN_SHA256 = "3bf4501f80c73274f50fa64c03f1a6dcc3bb5a81f022b0bd4cc5d791d477f815"
EXPECTED_MATRIX_SCHEMA = "civ-value-000-critical-domain-matrix-v1"
EXPECTED_CHAIN_SCHEMA = "civ-value-pilot-001-reference-chains-v1"
EXPECTED_AUTHORITY = "analysis_only_no_allocation_or_execution_authority"

EXPECTED_MATRIX_DOMAINS = {
    "water_sanitation",
    "food_agriculture_cold_chain",
    "energy_systems",
    "construction_shelter",
    "mining_beneficiation",
    "metals_alloys",
    "industrial_chemicals",
    "polymers_elastomers",
    "glass_ceramics_cement",
    "wood_pulp_paper_packaging",
    "textiles_hygiene",
    "machine_tools_tooling",
    "precision_mechanical",
    "fluid_machinery",
    "electromechanical_power",
    "electronics_interconnect",
    "semiconductors_compute",
    "electrochemical_storage",
    "sensors_metrology_optics",
    "robotics_automation",
    "building_services",
    "health_products",
    "transport_logistics",
    "communications_data",
    "waste_recovery",
    "repair_field_service",
    "emergency_resilience",
}

EXPECTED_STAGE_ROLES = [
    "primary_or_input_feedstock",
    "refining_or_material_grade",
    "intermediate_or_component",
    "manufactured_asset",
    "installed_system",
    "operational_service",
    "maintenance_or_replacement",
    "circular_return",
]

EXPECTED_CHAINS = {
    "water_service_infrastructure": (
        {"water_sanitation", "fluid_machinery", "sensors_metrology_optics", "industrial_chemicals"},
        "service_reachability_not_potable_water_safety_or_capacity",
    ),
    "electric_power_distribution_service": (
        {"energy_systems", "electromechanical_power", "electronics_interconnect", "electrochemical_storage"},
        "conductor_or_generation_not_transformer_switchgear_or_service_continuity",
    ),
    "productive_machine_repair_service": (
        {"machine_tools_tooling", "precision_mechanical", "electromechanical_power", "repair_field_service"},
        "machine_frame_or_single_repair_not_productive_or_reproductive_closure",
    ),
    "compute_network_service": (
        {"semiconductors_compute", "electronics_interconnect", "communications_data", "repair_field_service"},
        "board_assembly_or_working_server_not_semiconductor_or_resilient_service_closure",
    ),
    "food_cold_chain_service": (
        {"food_agriculture_cold_chain", "water_sanitation", "transport_logistics", "wood_pulp_paper_packaging"},
        "food_output_not_safety_nutrition_cold_chain_integrity_or_continuity",
    ),
    "health_diagnostic_manufacturing_service": (
        {"health_products", "industrial_chemicals", "electronics_interconnect", "sensors_metrology_optics"},
        "manufacturing_or_device_availability_not_clinical_diagnostic_sterility_regulatory_or_service_evidence",
    ),
    "circular_critical_material_route": (
        {"waste_recovery", "mining_beneficiation", "metals_alloys", "electronics_interconnect"},
        "recyclable_or_recovered_constituent_not_specification_grade_secondary_feedstock_or_component",
    ),
}

EXPECTED_ADVERSARIAL = {
    "feedstock_without_refining": (
        "primary_feedstock_present_refining_route_absent",
        "downstream_component_unavailable",
    ),
    "refined_material_missing_precision_component": (
        "refined_material_present_precision_component_absent",
        "asset_route_blocked",
    ),
    "asset_missing_service_dependency": (
        "asset_present_required_utility_or_consumable_absent",
        "service_unavailable",
    ),
    "one_service_episode_without_maintenance": (
        "service_observed_once_maintenance_spares_absent",
        "continuity_unresolved",
    ),
    "g1_imported_spares_only": (
        "g1_supported_by_stocked_import_g2_renewal_absent",
        "multigeneration_closure_absent",
    ),
    "recovered_without_grade": (
        "recovered_constituent_present_grade_evidence_absent",
        "qualified_reentry_blocked",
    ),
    "bounded_local_alternative": (
        "local_alternative_satisfies_profile_a_not_profile_b",
        "profile_relative_sufficiency_only",
    ),
    "mycelix_fact_not_process_qualification": (
        "inventory_or_work_order_fact_present_engineering_qualification_absent",
        "engineering_qualification_unestablished",
    ),
    "model_result_not_operational_fact": (
        "closure_model_result_present_mycelix_fact_absent",
        "operational_fact_unestablished",
    ),
    "no_authority_minting": (
        "benchmark_or_model_result_present",
        "no_procurement_allocation_or_execution_authority",
    ),
}

FORBIDDEN_KEY_FRAGMENTS = (
    "priority",
    "score",
    "rank",
    "self_sufficiency",
    "allocation_weight",
    "procurement_authority",
    "execution_authority",
    "allocation_authority",
    "operating_setpoint",
    "process_recipe",
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def reject_forbidden_keys_and_numbers(value: Any, path: str = "$") -> None:
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        fail(f"numeric process/performance value forbidden at {path}")
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            for fragment in FORBIDDEN_KEY_FRAGMENTS:
                if fragment in normalized:
                    fail(f"forbidden field at {path}.{key}: {fragment}")
            reject_forbidden_keys_and_numbers(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys_and_numbers(child, f"{path}[{index}]")


def unique_strings(value: Any, label: str) -> list[str]:
    require(isinstance(value, list), f"{label} must be a list")
    require(all(isinstance(item, str) and item for item in value), f"{label} entries must be non-empty strings")
    require(len(value) == len(set(value)), f"{label} contains duplicates")
    return value


def load_exact(path: Path, expected_digest: str, expected_schema: str) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    require(digest == expected_digest, f"{path}: digest drift: {digest}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"{path}: invalid JSON: {exc}")
    require(isinstance(value, dict), f"{path}: root must be object")
    require(value.get("schema") == expected_schema, f"{path}: schema drift")
    require(value.get("authority") == EXPECTED_AUTHORITY, f"{path}: authority drift")
    return value, digest


def validate_matrix(matrix: dict[str, Any]) -> set[str]:
    domains = matrix.get("domains")
    require(isinstance(domains, list), "matrix domains must be list")
    ids = []
    for entry in domains:
        require(isinstance(entry, dict), "matrix domain must be object")
        domain_id = entry.get("id")
        require(isinstance(domain_id, str) and domain_id, "matrix domain id invalid")
        ids.append(domain_id)
    require(len(ids) == len(set(ids)), "matrix contains duplicate domain IDs")
    require(set(ids) == EXPECTED_MATRIX_DOMAINS, "matrix domain identity set drift")
    return set(ids)


def validate_chains(chains_doc: dict[str, Any], matrix_domains: set[str]) -> None:
    require(chains_doc.get("stage_roles") == EXPECTED_STAGE_ROLES, "stage role vocabulary/order drift")
    reject_forbidden_keys_and_numbers(chains_doc)

    chains = chains_doc.get("chains")
    require(isinstance(chains, list), "chains must be list")
    require(len(chains) == 7, f"expected 7 chains, found {len(chains)}")

    by_id: dict[str, dict[str, Any]] = {}
    for chain in chains:
        require(isinstance(chain, dict), "chain entry must be object")
        require(set(chain) == {"claim_ceiling", "domain_ids", "id", "service_dependencies", "stages"}, "chain field set drift")
        chain_id = chain["id"]
        require(isinstance(chain_id, str) and chain_id, "chain id invalid")
        require(chain_id not in by_id, f"duplicate chain id: {chain_id}")
        by_id[chain_id] = chain

    require(set(by_id) == set(EXPECTED_CHAINS), "chain identity set drift")

    all_subjects: set[str] = set()
    for chain_id, chain in by_id.items():
        expected_domains, expected_ceiling = EXPECTED_CHAINS[chain_id]
        domain_ids = unique_strings(chain["domain_ids"], f"{chain_id}.domain_ids")
        require(set(domain_ids) == expected_domains, f"{chain_id}: domain coverage drift")
        require(set(domain_ids) <= matrix_domains, f"{chain_id}: unresolved domain ID")

        service_dependencies = unique_strings(chain["service_dependencies"], f"{chain_id}.service_dependencies")
        require(set(service_dependencies) <= matrix_domains, f"{chain_id}: unresolved service dependency")

        require(chain["claim_ceiling"] == expected_ceiling, f"{chain_id}: claim ceiling drift")

        stages = chain["stages"]
        require(isinstance(stages, list) and len(stages) == len(EXPECTED_STAGE_ROLES), f"{chain_id}: stage count drift")
        roles = [stage.get("role") if isinstance(stage, dict) else None for stage in stages]
        require(roles == EXPECTED_STAGE_ROLES, f"{chain_id}: stage role ordering drift")

        chain_subjects: set[str] = set()
        for index, stage in enumerate(stages):
            require(isinstance(stage, dict), f"{chain_id}.stages[{index}] must be object")
            require(set(stage) == {"owners", "role", "subject_ref"}, f"{chain_id}.stages[{index}]: field set drift")
            subject = stage["subject_ref"]
            require(isinstance(subject, str) and subject.startswith("synthetic:"), f"{chain_id}: non-synthetic subject ref")
            require(subject not in chain_subjects, f"{chain_id}: duplicate stage subject ref")
            require(subject not in all_subjects, f"cross-chain subject ref alias: {subject}")
            chain_subjects.add(subject)
            all_subjects.add(subject)

            owners = unique_strings(stage["owners"], f"{chain_id}.stages[{index}].owners")
            for owner in owners:
                require(" " not in owner, f"{chain_id}: malformed owner ref with whitespace")
                require("#" in owner or owner == "Mycelix-manufacturing", f"{chain_id}: owner ref lacks explicit subject form: {owner}")

    adversarial = chains_doc.get("adversarial_cases")
    require(isinstance(adversarial, list), "adversarial_cases must be list")
    require(len(adversarial) == 10, f"expected 10 adversarial cases, found {len(adversarial)}")
    observed: dict[str, tuple[str, str]] = {}
    for case in adversarial:
        require(isinstance(case, dict), "adversarial case must be object")
        require(set(case) == {"expected", "id", "premise"}, "adversarial case field set drift")
        case_id = case["id"]
        require(isinstance(case_id, str) and case_id, "adversarial case id invalid")
        require(case_id not in observed, f"duplicate adversarial case id: {case_id}")
        observed[case_id] = (case["premise"], case["expected"])
    require(observed == EXPECTED_ADVERSARIAL, "adversarial premise/disposition semantics drift")

    nonclaims = unique_strings(chains_doc.get("nonclaims"), "nonclaims")
    require(any("self-sufficiency" in item for item in nonclaims), "missing no-self-sufficiency nonclaim")
    require(any("procurement or physical execution authority" in item for item in nonclaims), "missing no-procurement/no-execution nonclaim")

    require(
        observed["mycelix_fact_not_process_qualification"][1] == "engineering_qualification_unestablished",
        "Mycelix operational facts must not mint engineering qualification",
    )
    require(
        observed["model_result_not_operational_fact"][1] == "operational_fact_unestablished",
        "model results must not mint Mycelix operational facts",
    )
    require(
        observed["no_authority_minting"][1] == "no_procurement_allocation_or_execution_authority",
        "benchmark must not mint authority",
    )


def main() -> None:
    matrix, _ = load_exact(MATRIX_PATH, EXPECTED_MATRIX_SHA256, EXPECTED_MATRIX_SCHEMA)
    chains, chain_digest = load_exact(CHAIN_PATH, EXPECTED_CHAIN_SHA256, EXPECTED_CHAIN_SCHEMA)
    matrix_domains = validate_matrix(matrix)
    validate_chains(chains, matrix_domains)
    print(f"ok chains=7 adversarial=10 digest={chain_digest}")


if __name__ == "__main__":
    main()
