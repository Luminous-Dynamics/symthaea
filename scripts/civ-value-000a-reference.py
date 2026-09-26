#!/usr/bin/env python3
"""Independent reference validator for CIV-VALUE-000A.

This script intentionally imports no Symthaea production code. It validates the
exact frozen terrestrial critical-domain matrix and a small set of semantic
invariants that prevent coverage classifications from becoming hidden ranking,
allocation, capability, or authority claims.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

MATRIX_PATH = Path("docs/release/evidence/civ-value-000-critical-domain-matrix-v1.json")
EXPECTED_SHA256 = "67abb09708153f10879d06121213dc1dae0f5de558adae857e670fee51cc5ae6"
EXPECTED_SCHEMA = "civ-value-000-critical-domain-matrix-v1"
EXPECTED_AUTHORITY = "analysis_only_no_allocation_or_execution_authority"

EXPECTED_DOMAINS = {
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

EXPECTED_TAGS = {
    "H": "human_essential",
    "M": "shared_industrial_multiplier",
    "S": "symthaea_continuity",
}

EXPECTED_ROLES = {
    "A": "acquisition_or_harvest",
    "B": "beneficiation_or_preparation",
    "C": "intermediate_or_component",
    "F": "manufactured_asset",
    "I": "installed_system",
    "P": "primary_resource",
    "R": "refining_or_material_grade",
    "V": "operational_service",
    "X": "maintenance_repair",
    "Y": "circular_recovery",
}

FORBIDDEN_KEY_FRAGMENTS = (
    "priority",
    "score",
    "rank",
    "allocation_weight",
    "self_sufficiency",
    "resilience_score",
    "strategic_priority",
    "strategic_rank",
    "procurement_authority",
    "execution_authority",
    "allocation_authority",
)


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def reject_forbidden_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized = str(key).lower()
            for fragment in FORBIDDEN_KEY_FRAGMENTS:
                if fragment in normalized:
                    fail(f"forbidden ranking/allocation key at {path}.{key}: {fragment}")
            reject_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            reject_forbidden_keys(child, f"{path}[{index}]")


def domain_map(domains: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for entry in domains:
        require(isinstance(entry, dict), "each domain entry must be an object")
        domain_id = entry.get("id")
        require(isinstance(domain_id, str) and domain_id, "domain id must be non-empty string")
        require(domain_id not in result, f"duplicate domain id: {domain_id}")
        result[domain_id] = entry
    return result


def require_unique_strings(values: Any, label: str) -> list[str]:
    require(isinstance(values, list), f"{label} must be a list")
    require(all(isinstance(item, str) and item for item in values), f"{label} entries must be non-empty strings")
    require(len(values) == len(set(values)), f"{label} contains duplicate references")
    return values


def validate_domain(
    entry: dict[str, Any],
    dependency_codes: set[str],
    owner_codes: set[str],
    boundary_profiles: set[str],
) -> None:
    domain_id = entry["id"]
    require(set(entry) == {"d", "id", "o", "q", "r", "t"}, f"{domain_id}: unexpected/missing fields")

    tags = entry["t"]
    require(isinstance(tags, str) and tags, f"{domain_id}: t must be non-empty string")
    require(len(tags) == len(set(tags)), f"{domain_id}: duplicate classification tag")
    require(set(tags) <= set(EXPECTED_TAGS), f"{domain_id}: unknown classification tag")

    roles = entry["r"]
    require(isinstance(roles, str) and roles, f"{domain_id}: r must be non-empty string")
    require(len(roles) == len(set(roles)), f"{domain_id}: duplicate value-chain role")
    require(set(roles) <= set(EXPECTED_ROLES), f"{domain_id}: unknown value-chain role")

    dependencies = require_unique_strings(entry["d"], f"{domain_id}.d")
    owners = require_unique_strings(entry["o"], f"{domain_id}.o")
    require(set(dependencies) <= dependency_codes, f"{domain_id}: unresolved dependency code")
    require(set(owners) <= owner_codes, f"{domain_id}: unresolved owner code")

    boundary = entry["q"]
    require(isinstance(boundary, str) and boundary, f"{domain_id}: q must be non-empty string")
    require(boundary in boundary_profiles, f"{domain_id}: unresolved boundary profile {boundary}")


def assert_semantic_anchors(domains: dict[str, dict[str, Any]]) -> None:
    water = domains["water_sanitation"]
    require(set(water["t"]) == {"H", "S", "M"}, "water must remain human/Symthaea/shared multi-tagged")
    require("V" in water["r"], "water must represent an operational-service role")
    require(water["q"] == "service_not_safety_or_capacity", "water service claim ceiling drift")

    machine_tools = domains["machine_tools_tooling"]
    require(set(machine_tools["t"]) == {"S", "M"}, "machine tools must remain Symthaea/shared multiplier")
    require(machine_tools["q"] == "machine_exists_not_productive_or_renewal_closure", "machine-tool closure boundary drift")

    health = domains["health_products"]
    require("H" in health["t"], "health products must retain human-essential classification")
    require(health["q"] == "manufacturing_not_clinical_or_regulatory_evidence", "health manufacturing claim ceiling drift")

    semiconductors = domains["semiconductors_compute"]
    require(set(semiconductors["t"]) == {"S", "M"}, "semiconductor/compute must remain Symthaea/shared")
    require(semiconductors["q"] == "design_fab_package_board_and_service_are_distinct", "semiconductor stage-separation boundary drift")

    metrology = domains["sensors_metrology_optics"]
    require(set(metrology["t"]) == {"H", "S", "M"}, "metrology must remain human/Symthaea/shared")
    require(metrology["q"] == "instrument_exists_not_calibrated_current_measurement", "metrology calibration boundary drift")

    repair = domains["repair_field_service"]
    require(set(repair["t"]) == {"H", "S", "M"}, "repair must remain human/Symthaea/shared")
    require(repair["q"] == "repairable_repaired_and_reproducible_are_distinct", "repair/reproduction boundary drift")

    mining = domains["mining_beneficiation"]
    require("P" in mining["r"] and "A" in mining["r"] and "B" in mining["r"], "mining must preserve primary/acquisition/beneficiation stages")
    require(mining["q"] == "occurrence_not_recoverable_qualified_feedstock", "resource occurrence boundary drift")

    waste = domains["waste_recovery"]
    require("Y" in waste["r"], "waste/recovery must retain circular-recovery role")
    require(waste["q"] == "waste_recyclable_recovered_and_feedstock_are_distinct", "circular feedstock boundary drift")


def main() -> None:
    raw = MATRIX_PATH.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    require(digest == EXPECTED_SHA256, f"matrix digest drift: {digest}")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON: {exc}")

    require(isinstance(data, dict), "matrix root must be object")
    require(data.get("schema") == EXPECTED_SCHEMA, "schema drift")
    require(data.get("authority") == EXPECTED_AUTHORITY, "authority drift")

    reject_forbidden_keys(data)

    codes = data.get("codes")
    require(isinstance(codes, dict), "codes must be object")
    require(set(codes) == {"dependencies", "owners", "roles", "tags"}, "codebook families drift")
    require(codes["tags"] == EXPECTED_TAGS, "classification tag codebook drift")
    require(codes["roles"] == EXPECTED_ROLES, "value-chain role codebook drift")
    require(isinstance(codes["dependencies"], dict) and codes["dependencies"], "dependency codebook missing")
    require(isinstance(codes["owners"], dict) and codes["owners"], "owner codebook missing")

    boundaries = data.get("boundary_profiles")
    require(isinstance(boundaries, dict) and boundaries, "boundary_profiles missing")
    require(all(isinstance(k, str) and k for k in boundaries), "boundary profile IDs invalid")
    require(all(isinstance(v, str) and v for v in boundaries.values()), "boundary profile descriptions invalid")

    domains_raw = data.get("domains")
    require(isinstance(domains_raw, list), "domains must be list")
    require(len(domains_raw) == 27, f"expected 27 domains, found {len(domains_raw)}")
    domains = domain_map(domains_raw)
    require(set(domains) == EXPECTED_DOMAINS, "domain identity set drift")

    names = data.get("names")
    require(isinstance(names, dict), "names must be object")
    require(set(names) == EXPECTED_DOMAINS, "domain name coverage drift")
    require(all(isinstance(name, str) and name for name in names.values()), "domain names must be non-empty strings")

    dependency_codes = set(codes["dependencies"])
    owner_codes = set(codes["owners"])
    boundary_profiles = set(boundaries)
    for entry in domains_raw:
        validate_domain(entry, dependency_codes, owner_codes, boundary_profiles)

    nonclaims = data.get("nonclaims")
    require_unique_strings(nonclaims, "nonclaims")
    require(any("priority/score" in item for item in nonclaims), "missing explicit no-priority/no-score nonclaim")
    require(any("resource-allocation/execution authority" in item for item in nonclaims), "missing no-allocation/no-execution nonclaim")

    assert_semantic_anchors(domains)

    # Classification must actually be multi-label in the frozen corpus; otherwise
    # the three tags could silently devolve into mutually exclusive tiers.
    require(any(len(entry["t"]) >= 3 for entry in domains_raw), "matrix lost multi-label classification evidence")

    print(f"ok domains=27 digest={digest}")


if __name__ == "__main__":
    main()
