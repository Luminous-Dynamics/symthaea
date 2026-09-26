#!/usr/bin/env python3
"""Independent validator for CIV-VALUE-PILOT-002 frozen weak-link corpus.

This script intentionally imports only the Python standard library and does not
call Symthaea production code or the owner-matrix validator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CHILD_PATH = ROOT / "docs/release/evidence/civ-value-pilot-002-weak-link-chains-v1.json"
PARENT_PATH = ROOT / "docs/release/evidence/civ-value-pilot-001-reference-chains-v1.json"

CHILD_SHA256 = "b263c517c244ab11d38312c7ac28f11eea7151564d6527e551e60d3fa9745a47"
PARENT_SHA256 = "3bf4501f80c73274f50fa64c03f1a6dcc3bb5a81f022b0bd4cc5d791d477f815"
OWNER_MATRIX_SHA256 = "45056ab59a16137b48325338a552b109b40eaef263936af96fb52eec1fab892a"
OWNER_MATRIX_HEAD = "7fc655b12eb6decf1fb77a4055fa7b20741bba79"
OWNER_MATRIX_SCHEMA = "civ-domain-gap-001a-owner-coverage-matrix-v1"
PILOT_001_HEAD = "d85bab23ff192d83337d4576cf4fb8f718c303a6"

EXPECTED_CHAIN_IDS = [
    "water_service_infrastructure",
    "electric_power_distribution_service",
    "productive_machine_repair_service",
    "compute_network_service",
    "food_cold_chain_service",
    "health_diagnostic_manufacturing_service",
    "circular_critical_material_route",
]

EXPECTED_STAGE_ROLES = [
    "resource_occurrence",
    "recoverability_or_accessibility",
    "acquisition_or_harvest",
    "beneficiation_or_preparation",
    "refining_or_specification_grade",
    "intermediate_or_component",
    "productive_or_manufacturing_asset",
    "installed_or_commissioned_system",
    "operational_service_bundle",
    "maintenance_spares_calibration_tooling_renewal",
    "circular_recovery",
    "secondary_feedstock_requalification",
]

EXPECTED_DOMAIN_CODES = {
    "D00": "building_services",
    "D01": "communications_data",
    "D02": "electrochemical_storage",
    "D03": "electromechanical_power",
    "D04": "electronics_interconnect",
    "D05": "energy_systems",
    "D06": "fluid_machinery",
    "D07": "food_agriculture_cold_chain",
    "D08": "health_products",
    "D09": "industrial_chemicals",
    "D10": "machine_tools_tooling",
    "D11": "metals_alloys",
    "D12": "mining_beneficiation",
    "D13": "polymers_elastomers",
    "D14": "precision_mechanical",
    "D15": "repair_field_service",
    "D16": "semiconductors_compute",
    "D17": "sensors_metrology_optics",
    "D18": "transport_logistics",
    "D19": "waste_recovery",
    "D20": "water_sanitation",
    "D21": "wood_pulp_paper_packaging",
}

EXPECTED_UTILITY_NAMES = {
    "communications_data",
    "electric_power",
    "metrology_calibration",
    "process_gas_or_chemical",
    "refrigeration",
    "thermal_service",
    "vacuum_or_pressure_service",
    "waste_treatment",
    "water_service",
}

EXPECTED_DISPOSITIONS = [
    "OwnerDefinedCapabilityUnresolved",
    "DomainOwnerGap",
    "FeedstockUnavailable",
    "UpstreamEvidenceUnresolved",
    "GradeOrPurityUnresolved",
    "MidstreamCapabilityUnavailable",
    "ComponentDependencyUnavailable",
    "UtilityEnvelopeUnresolved",
    "UtilityCapacityOrTimingInsufficient",
    "InstalledAssetUnqualified",
    "ServiceReachableButContinuityUnresolved",
    "MaintenanceOrRenewalUnresolved",
    "CircularReentryUnqualified",
    "EvidenceInsufficientOrStale",
    "ProfileBoundedAlternativeOnly",
    "FullyRepresentedSyntheticRouteUnderProfile",
    "NoProcurementAllocationOrExecutionAuthority",
]

EXPECTED_GAPS = {
    "water_service_infrastructure": [],
    "electric_power_distribution_service": ["D02"],
    "productive_machine_repair_service": [],
    "compute_network_service": [],
    "food_cold_chain_service": ["D21", "D18"],
    "health_diagnostic_manufacturing_service": ["D18"],
    "circular_critical_material_route": [],
}

EXPECTED_ADVERSARIAL = {
    "resource_occurrence_without_assay": "UpstreamEvidenceUnresolved",
    "feedstock_without_refining": "MidstreamCapabilityUnavailable",
    "commodity_without_spec_grade": "GradeOrPurityUnresolved",
    "materials_complete_utility_unqualified": "UtilityEnvelopeUnresolved",
    "average_energy_without_instantaneous_capacity": "UtilityCapacityOrTimingInsufficient",
    "redundant_utility_shared_failure_root": "EvidenceInsufficientOrStale",
    "asset_without_commissioning": "InstalledAssetUnqualified",
    "single_service_without_renewal": "ServiceReachableButContinuityUnresolved",
    "g1_imported_consumable_only": "MaintenanceOrRenewalUnresolved",
    "recovered_without_secondary_grade": "CircularReentryUnqualified",
    "bounded_local_substitute": "ProfileBoundedAlternativeOnly",
    "source_lot_change": "EvidenceInsufficientOrStale",
    "mycelix_fact_not_engineering": "OwnerDefinedCapabilityUnresolved",
    "model_not_operational_fact": "OwnerDefinedCapabilityUnresolved",
    "shared_utility_cross_chain_common_mode": "UtilityEnvelopeUnresolved",
    "missing_domain_theorem_not_physical_shortage": "DomainOwnerGap",
    "no_authority_minting": "NoProcurementAllocationOrExecutionAuthority",
}

FORBIDDEN_KEY_FRAGMENTS = (
    "priority_score",
    "ranking_score",
    "readiness_score",
    "self_sufficiency_score",
    "resilience_score",
    "civilization_score",
    "bootstrap_score",
    "investment_score",
    "allocation_score",
    "recommendation_score",
)


def fail(message: str) -> None:
    raise SystemExit(f"FAIL_CIV_VALUE_PILOT_002: {message}")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def require_unique(values: list[str], label: str) -> None:
    require(len(values) == len(set(values)), f"duplicate {label}")


def reject_forbidden_keys(value: Any, path: str = "$" ) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(fragment in lowered for fragment in FORBIDDEN_KEY_FRAGMENTS):
                fail(f"forbidden score/ranking key at {path}.{key}")
            reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, f"{path}[{index}]")


def load_exact(path: Path, expected_sha: str, label: str) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    require(digest(raw) == expected_sha, f"{label} digest drift")
    obj = json.loads(raw.decode("utf-8"))
    require(raw == canonical_bytes(obj), f"{label} is not canonical compact sorted-key JSON + newline")
    return raw, obj


def main() -> None:
    _, parent = load_exact(PARENT_PATH, PARENT_SHA256, "PILOT-001 parent")
    _, child = load_exact(CHILD_PATH, CHILD_SHA256, "PILOT-002 child")

    require(parent.get("schema") == "civ-value-pilot-001-reference-chains-v1", "parent schema drift")
    require(child.get("schema") == "civ-value-pilot-002-weak-link-chains-v1", "child schema drift")
    require(
        child.get("authority") == "analysis_only_no_priority_allocation_procurement_or_execution_authority",
        "authority ceiling drift",
    )

    parent_chain_ids = [entry.get("id") for entry in parent.get("chains", [])]
    require(parent_chain_ids == EXPECTED_CHAIN_IDS, "PILOT-001 chain identity/order drift")

    parents = child.get("parents", {})
    p1 = parents.get("pilot_001", {})
    require(p1.get("head") == PILOT_001_HEAD, "PILOT-001 head reference drift")
    require(p1.get("sha256") == PARENT_SHA256, "PILOT-001 digest reference drift")
    require(p1.get("schema") == "civ-value-pilot-001-reference-chains-v1", "PILOT-001 schema reference drift")
    require(p1.get("pr") == 5955, "PILOT-001 PR reference drift")

    owner = parents.get("owner_matrix", {})
    require(owner.get("head") == OWNER_MATRIX_HEAD, "owner matrix head drift")
    require(owner.get("sha256") == OWNER_MATRIX_SHA256, "owner matrix digest drift")
    require(owner.get("schema") == OWNER_MATRIX_SCHEMA, "owner matrix schema drift")
    require(owner.get("pr") == 5975, "owner matrix PR reference drift")

    require(child.get("stage_roles") == EXPECTED_STAGE_ROLES, "stage-role vocabulary/order drift")
    require(child.get("weak_link_dispositions") == EXPECTED_DISPOSITIONS, "weak-link disposition drift")
    require(child.get("codes", {}).get("domains") == EXPECTED_DOMAIN_CODES, "domain codebook drift")

    utilities = child.get("utility_profiles", {})
    require(set(utilities) == EXPECTED_UTILITY_NAMES, "utility registry drift")
    require(len(utilities) == 9, "utility profile count != 9")
    for utility_name, profile in utilities.items():
        require(profile.get("owner") == "CIV-UTILITY#5959", f"{utility_name}: utility owner drift")
        domains = profile.get("domains")
        require(isinstance(domains, list) and domains, f"{utility_name}: missing domain refs")
        require_unique(domains, f"{utility_name} utility domain refs")
        require(all(code in EXPECTED_DOMAIN_CODES for code in domains), f"{utility_name}: unknown domain code")

    chains = child.get("chains", [])
    require(len(chains) == 7, "chain count != 7")
    child_chain_ids = [entry.get("id") for entry in chains]
    require(child_chain_ids == EXPECTED_CHAIN_IDS, "child chain identity/order drift")
    require_unique(child_chain_ids, "chain IDs")

    for chain in chains:
        chain_id = chain["id"]
        require(chain.get("parent") == chain_id, f"{chain_id}: parent-chain binding drift")
        domains = chain.get("domains")
        require(isinstance(domains, list) and domains, f"{chain_id}: missing domain refs")
        require_unique(domains, f"{chain_id} domain refs")
        require(all(code in EXPECTED_DOMAIN_CODES for code in domains), f"{chain_id}: unknown domain code")

        gaps = chain.get("gaps")
        require(gaps == EXPECTED_GAPS[chain_id], f"{chain_id}: owner-gap mapping drift")
        require(all(code in domains for code in gaps), f"{chain_id}: gap ref outside declared chain domains")

        chain_utilities = chain.get("utilities")
        require(isinstance(chain_utilities, list) and chain_utilities, f"{chain_id}: missing utilities")
        require_unique(chain_utilities, f"{chain_id} utility refs")
        require(all(name in utilities for name in chain_utilities), f"{chain_id}: unknown utility ref")

        stages = chain.get("stages")
        require(isinstance(stages, list) and len(stages) == 12, f"{chain_id}: stage count != 12")
        stage_names: list[str] = []
        for stage in stages:
            require(isinstance(stage, list) and len(stage) == 2, f"{chain_id}: malformed stage encoding")
            stage_name, stage_domains = stage
            require(isinstance(stage_name, str) and stage_name, f"{chain_id}: empty stage subject")
            require(isinstance(stage_domains, list) and stage_domains, f"{chain_id}/{stage_name}: no domain refs")
            require_unique(stage_domains, f"{chain_id}/{stage_name} domain refs")
            require(all(code in EXPECTED_DOMAIN_CODES for code in stage_domains), f"{chain_id}/{stage_name}: unknown domain code")
            stage_names.append(stage_name)
        require_unique(stage_names, f"{chain_id} stage subjects")
        require(isinstance(chain.get("ceiling"), str) and chain["ceiling"], f"{chain_id}: missing claim ceiling")

    adversarial = child.get("adversarial_cases", [])
    require(len(adversarial) == 17, "adversarial case count != 17")
    adversarial_ids = [case.get("id") for case in adversarial]
    require_unique(adversarial_ids, "adversarial IDs")
    require(set(adversarial_ids) == set(EXPECTED_ADVERSARIAL), "adversarial case identity drift")
    for case in adversarial:
        require(case.get("expected") == EXPECTED_ADVERSARIAL[case["id"]], f"{case['id']}: expected disposition drift")
        require(case.get("expected") in EXPECTED_DISPOSITIONS, f"{case['id']}: disposition outside frozen vocabulary")
        require(isinstance(case.get("premise"), str) and case["premise"], f"{case['id']}: missing premise")

    # Prove the architecture-owner gap and physical/capability unresolved states remain distinct.
    expected_values = {case["expected"] for case in adversarial}
    require("DomainOwnerGap" in expected_values, "DomainOwnerGap distinction lost")
    require("OwnerDefinedCapabilityUnresolved" in expected_values, "OwnerDefinedCapabilityUnresolved distinction lost")

    rules = set(child.get("rules", []))
    required_rules = {
        "owner coverage disposition is architectural metadata, not physical availability evidence",
        "physical or import dependency failure does not create a new ontology or domain owner",
        "Mycelix operational facts cannot create engineering process grade calibration or qualification evidence",
        "Symthaea model closure or planning outputs cannot create Mycelix inventory custody work order production or service facts",
        "no weak-link disposition is a priority readiness resilience self-sufficiency or investment score",
    }
    require(required_rules.issubset(rules), "required non-laundering/anti-score rules missing")

    evidence_state = child.get("codes", {}).get("evidence_state")
    require(evidence_state == "all stage subjects are synthetic_unresolved", "synthetic evidence-state ceiling drift")

    nonclaims = child.get("nonclaims", [])
    require(isinstance(nonclaims, list) and len(nonclaims) >= 4, "nonclaims missing")
    require(any("procurement" in item and "physical-execution" in item for item in nonclaims), "execution-authority nonclaim missing")

    reject_forbidden_keys(child)

    print(
        "PASS_CIV_VALUE_PILOT_002_SOURCE_VALIDATION "
        f"parent_sha256={PARENT_SHA256} child_sha256={CHILD_SHA256} "
        f"owner_matrix_sha256={OWNER_MATRIX_SHA256} chains=7 stages=12 utilities=9 adversarial=17"
    )


if __name__ == "__main__":
    main()
