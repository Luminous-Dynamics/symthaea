#!/usr/bin/env python3
"""Independent validator for PROD-EQP-001A frozen source corpus."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/release/evidence/prod-eqp-001a-productive-machine-reference-v1.json"
SHA256 = "2cfcd09a899599e0bec8c7da7424e07c89324608d94365330010a6ac2e996ea7"
SCHEMA = "prod-eqp-001a-productive-machine-reference-v1"
AUTHORITY = "analysis_only_no_machine_execution_authority"

EXPECTED_PLANES = [
    "design_configuration","as_built_generation","installed_subsystems","geometry_alignment",
    "bounded_machine_capability","tooling_workholding","metrology_calibration",
    "process_family_compatibility","commissioning","produced_part_observation",
    "maintenance_spares","tooling_reference_renewal","import_dependencies",
    "generation_profile","prior_generation_envelope_relation",
]

EXPECTED_DISPOSITIONS = [
    "DesignOnly","AsBuiltUncommissioned","PartialProductiveClosure",
    "MetrologyRenewalUnresolved","ToolingRenewalUnresolved","CapabilityUnresolved",
    "ProcessCompatibilityUnresolved","GenerationEnvelopeDegraded",
    "DegradedButSufficientUnderProfile","CapabilityDimensionRestoredPartialClosure",
    "NewConfigurationSubjectRequired","SemanticIdentityUnchanged","InvalidDuplicateReference",
    "HistoryPreserved","NoAsBuiltOrCommissionedState","NoMachineExecutionAuthority",
]

EXPECTED_OWNERS = {
    "generational_envelope":"CIV-BOOT#5782",
    "generic_process":"MFG-PROC#5686",
    "lifecycle":"MFG-LIFE#5705",
    "observation":"FIELD/SE-OBS",
    "productive_closure":"CIV-BOOT#5774",
    "realization":"ROB-REALIZE#4859",
    "tolerance_metrology":"ENG-TOL#4883",
}

EXPECTED_CASES = {
    "design_only":"DesignOnly",
    "as_built_uncommissioned":"AsBuiltUncommissioned",
    "local_frame_imported_critical_subsystems":"PartialProductiveClosure",
    "motion_without_calibration_reference_renewal":"MetrologyRenewalUnresolved",
    "operational_machine_nonrenewable_tooling":"ToolingRenewalUnresolved",
    "substitute_bearing_fit_without_function_evidence":"CapabilityUnresolved",
    "single_part_success_not_process_capability":"ProcessCompatibilityUnresolved",
    "g1_brackets_not_spindle_route":"GenerationEnvelopeDegraded",
    "degraded_successor_sufficient_for_repair_profile":"DegradedButSufficientUnderProfile",
    "metrology_upgrade_restores_one_dimension":"CapabilityDimensionRestoredPartialClosure",
    "installed_subsystem_change":"NewConfigurationSubjectRequired",
    "display_label_change_only":"SemanticIdentityUnchanged",
    "duplicate_required_subsystem_ref":"InvalidDuplicateReference",
    "later_success_preserves_prior_failure":"HistoryPreserved",
    "optimizer_proposal_not_as_built":"NoAsBuiltOrCommissionedState",
    "no_execution_authority_minting":"NoMachineExecutionAuthority",
}

REQUIRED_RULES = {
    "friendly labels are non-semantic",
    "installed subsystem or as-built generation changes are semantic",
    "duplicate required references reject instead of silently deduplicating",
    "one successful produced part does not establish general process capability",
    "later success does not erase earlier failure degradation or missing evidence",
    "same nominal machine class does not establish equivalent productive capability",
    "generation one success does not establish indefinite productive closure",
    "degraded capability may remain sufficient only for an exact declared profile",
    "metrology or tooling renewal may restore one capability dimension without establishing whole-machine closure",
    "no analysis validation or qualification artifact can mint physical machine execution authority",
}

FORBIDDEN = (
    "priority_score","ranking_score","readiness_score","self_sufficiency_score",
    "productive_closure_score","machine_quality_score","investment_score","allocation_score",
)

def fail(msg):
    raise SystemExit(f"FAIL_PROD_EQP_001A: {msg}")

def require(cond, msg):
    if not cond:
        fail(msg)

def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()

def scan_keys(value, path="$"):
    if isinstance(value, dict):
        for key, child in value.items():
            low = str(key).lower()
            if any(fragment in low for fragment in FORBIDDEN):
                fail(f"forbidden scalar/ranking key {path}.{key}")
            scan_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for i, child in enumerate(value):
            scan_keys(child, f"{path}[{i}]")

raw = PATH.read_bytes()
require(hashlib.sha256(raw).hexdigest() == SHA256, "corpus digest drift")
obj = json.loads(raw.decode())
require(raw == canonical(obj), "corpus not canonical compact sorted-key JSON + newline")
require(obj.get("schema") == SCHEMA, "schema drift")
require(obj.get("authority") == AUTHORITY, "authority drift")
require(obj.get("evidence_planes") == EXPECTED_PLANES, "evidence-plane drift")
require(obj.get("dispositions") == EXPECTED_DISPOSITIONS, "disposition drift")
require(obj.get("owners") == EXPECTED_OWNERS, "owner-ref drift")

cases = obj.get("cases", [])
require(len(cases) == 16, "case count != 16")
ids = [c.get("id") for c in cases]
require(len(ids) == len(set(ids)), "duplicate case id")
require(set(ids) == set(EXPECTED_CASES), "case identity drift")
for case in cases:
    require(case.get("expected") == EXPECTED_CASES[case["id"]], f"{case['id']}: expected disposition drift")
    require(case.get("expected") in EXPECTED_DISPOSITIONS, f"{case['id']}: unknown disposition")
    require(isinstance(case.get("premise"), str) and case["premise"], f"{case['id']}: missing premise")

rules = set(obj.get("rules", []))
require(REQUIRED_RULES.issubset(rules), "required precision/history/authority rules missing")
require("NoMachineExecutionAuthority" in {c["expected"] for c in cases}, "zero-authority fixture missing")
require("GenerationEnvelopeDegraded" in {c["expected"] for c in cases}, "precision-ratchet fixture missing")
require("ToolingRenewalUnresolved" in {c["expected"] for c in cases}, "tooling-renewal fixture missing")
require("MetrologyRenewalUnresolved" in {c["expected"] for c in cases}, "metrology-renewal fixture missing")
require(isinstance(obj.get("nonclaims"), list) and len(obj["nonclaims"]) >= 4, "nonclaims missing")
scan_keys(obj)

print(f"PASS_PROD_EQP_001A_SOURCE_VALIDATION sha256={SHA256} cases=16 planes=15")
