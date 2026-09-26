#!/usr/bin/env python3
"""Independent validator for IND-COMP-001A frozen source corpus."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "docs/release/evidence/ind-comp-001a-functional-component-reference-v1.json"
SHA256 = "11cd7e2eb13c22493341edfd29ea4e37905506637612880ffe21e7c8fb9ae799"
SCHEMA = "ind-comp-001a-functional-component-reference-v1"
AUTHORITY = "analysis_only_no_component_or_system_execution_authority"

EXPECTED_FAMILIES = [
    "bearing","gear_or_transmission","seal_or_gasket","shaft_or_coupling",
    "spring_or_fastener","low_energy_pump","valve","compressor_or_blower",
]
EXPECTED_PLANES = [
    "component_article_configuration_identity","material_surface_process_state",
    "dimensional_fit_evidence","installation_assembly_context","functional_observation",
    "duty_applicability_profile","quantity_measurement_refs","wear_lifetime_maintenance",
    "repair_remanufacture_generation","metrology_calibration_currentness",
    "import_dependencies","productive_closure",
]
EXPECTED_DISPOSITIONS = [
    "DimensionalConformanceFunctionUnresolved","SurfaceOrProcessStateUnresolved",
    "LeakageUnresolved","DutyEnvelopeUnresolved","ProfileBoundedSubstitute",
    "RepairObservedDurabilityUnresolved","PartialProductiveClosure","MetrologyBlocked",
    "NewConfigurationSubjectRequired","SemanticIdentityUnchanged","InvalidDuplicateReference",
    "HistoryPreserved","PredictionOnlyNoPhysicalObservation","NoComponentOrSystemExecutionAuthority",
]
EXPECTED_OWNERS = {
    "component_subsystems":"ENG-DEVICE#5670",
    "generational_envelope":"CIV-BOOT#5782",
    "lifecycle":"MFG-LIFE#5705",
    "manufacturing":"MFG-PROC#5686",
    "observation":"FIELD/SE-OBS",
    "productive_closure":"CIV-BOOT#5774",
    "realization":"ROB-REALIZE#4859",
    "tolerance_metrology":"ENG-TOL#4883",
}
EXPECTED_CASES = {
    "bearing_fit_without_runout_life": ("bearing", "DimensionalConformanceFunctionUnresolved"),
    "gear_geometry_without_surface_state": ("gear_or_transmission", "SurfaceOrProcessStateUnresolved"),
    "seal_nominal_material_without_leakage": ("seal_or_gasket", "LeakageUnresolved"),
    "pump_runs_without_duty_envelope": ("low_energy_pump", "DutyEnvelopeUnresolved"),
    "valve_actuates_without_tightness_flow": ("valve", "DutyEnvelopeUnresolved"),
    "compressor_runs_without_delivered_envelope": ("compressor_or_blower", "DutyEnvelopeUnresolved"),
    "bounded_local_substitute": ("shaft_or_coupling", "ProfileBoundedSubstitute"),
    "repair_once_without_durability": ("bearing", "RepairObservedDurabilityUnresolved"),
    "imported_critical_component_material": ("seal_or_gasket", "PartialProductiveClosure"),
    "shared_metrology_reference_loss": ("gear_or_transmission", "MetrologyBlocked"),
    "changed_installed_article": ("bearing", "NewConfigurationSubjectRequired"),
    "display_label_change_only": ("bearing", "SemanticIdentityUnchanged"),
    "duplicate_required_evidence_ref": ("valve", "InvalidDuplicateReference"),
    "later_success_preserves_failure": ("low_energy_pump", "HistoryPreserved"),
    "model_prediction_not_field_observation": ("compressor_or_blower", "PredictionOnlyNoPhysicalObservation"),
    "no_execution_authority_minting": ("spring_or_fastener", "NoComponentOrSystemExecutionAuthority"),
}
REQUIRED_RULES = {
    "component design or fabrication does not establish functional performance",
    "dimensional fit does not establish runout friction leakage flow load or lifetime",
    "one functional observation does not establish a requested duty envelope or durability",
    "substitution is exact-profile relative and never a universal equivalence flag",
    "predicted or simulated performance is not FIELD physical observation",
    "later success preserves earlier negative or unresolved evidence",
    "friendly labels are non-semantic while installed article or configuration changes are semantic",
    "duplicate required evidence references reject",
    "partial local manufacture remains import-dependent when critical component or material routes are external",
    "no analysis validation or qualification artifact can mint physical component or system operation authority",
}
FORBIDDEN = (
    "rating_score","priority_score","ranking_score","readiness_score",
    "self_sufficiency_score","productive_closure_score","component_quality_score",
    "investment_score","allocation_score",
)

def fail(msg):
    raise SystemExit(f"FAIL_IND_COMP_001A: {msg}")

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
require(obj.get("component_families") == EXPECTED_FAMILIES, "component-family drift")
require(obj.get("evidence_planes") == EXPECTED_PLANES, "evidence-plane drift")
require(obj.get("dispositions") == EXPECTED_DISPOSITIONS, "disposition drift")
require(obj.get("owners") == EXPECTED_OWNERS, "owner-ref drift")

cases = obj.get("cases", [])
require(len(cases) == 16, "case count != 16")
ids = [c.get("id") for c in cases]
require(len(ids) == len(set(ids)), "duplicate case id")
require(set(ids) == set(EXPECTED_CASES), "case identity drift")
for case in cases:
    family, expected = EXPECTED_CASES[case["id"]]
    require(case.get("family") == family, f"{case['id']}: family drift")
    require(case.get("expected") == expected, f"{case['id']}: expected disposition drift")
    require(case.get("expected") in EXPECTED_DISPOSITIONS, f"{case['id']}: unknown disposition")
    require(isinstance(case.get("premise"), str) and case["premise"], f"{case['id']}: missing premise")

rules = set(obj.get("rules", []))
require(REQUIRED_RULES.issubset(rules), "required function/evidence/history/authority rules missing")
outcomes = {c["expected"] for c in cases}
require("PredictionOnlyNoPhysicalObservation" in outcomes, "prediction-vs-FIELD fixture missing")
require("ProfileBoundedSubstitute" in outcomes, "profile-bounded substitution fixture missing")
require("MetrologyBlocked" in outcomes, "metrology common-mode fixture missing")
require("NoComponentOrSystemExecutionAuthority" in outcomes, "zero-authority fixture missing")
require(isinstance(obj.get("nonclaims"), list) and len(obj["nonclaims"]) >= 4, "nonclaims missing")
scan_keys(obj)

print(f"PASS_IND_COMP_001A_SOURCE_VALIDATION sha256={SHA256} cases=16 families=8 planes=12")
