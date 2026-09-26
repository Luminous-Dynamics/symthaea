#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

PATH = Path("docs/release/evidence/water-mfg-001a-nonpotable-module-reference-v1.json")
EXPECTED_SHA256 = "75c74323afb879f0af4eb5917d74e620a01905beba7af50bbbb6301c757438c1"
EXPECTED_SCHEMA = "water-mfg-001a-nonpotable-module-reference-v1"
EXPECTED_AUTHORITY = "analysis_only_no_water_quality_service_or_physical_operation_authority"
EXPECTED_PLANES = ["source_or_feedstream_identity","feedstream_characterization","component_article_and_configuration_identity","equipment_as_built_lineage","hydraulic_component_function","media_or_filter_identity_and_currentness","calibrated_flow_pressure_level_observation","treatment_effect_observation","water_quality_or_safety_external_evidence","commissioning_state","capacity_availability_and_continuity","maintenance_media_and_consumable_renewal","common_mode_dependency","residual_waste_or_recovery_route","productive_or_reproductive_closure"]
EXPECTED_OWNERS = {"root":"WATER-MFG#5966","components":"IND-COMP#5965","process":"MFG-PROC#5686","observation":"FIELD/SE-OBS","sensing":"SENSE#5820","service":"CIV-SERVICE#5949","closure":"CIV-BOOT#5774","lifecycle":"MFG-LIFE#5705","operations":"Mycelix-operations"}
EXPECTED_CASES = {"source_present_characterization_absent":"FeedstreamCharacterizationUnresolved","pump_present_duty_unmeasured":"CapacityOrDutyUnresolved","filter_media_identity_unknown":"MediaCurrentnessUnresolved","clear_output_no_qualified_measurement":"MeasurementQualificationUnresolved","treatment_effect_not_water_safety":"WaterQualityAuthorityExternal","asset_without_commissioning":"CommissioningUnresolved","shared_power_redundant_pumps":"CommonModeDependencyUnresolved","one_success_no_media_renewal":"ContinuityOrRenewalUnresolved","reconditioned_component_no_requalification":"ReplacementQualificationUnresolved","feedstream_changed_outside_profile":"ApplicabilityTransferBlocked","imported_pump_local_housing":"ProductiveClosurePartial","low_head_profile_only":"ProfileBoundedFunctionOnly","mycelix_event_not_engineering":"ComponentFunctionUnresolved","model_not_physical_observation":"TreatmentEffectUnresolved","synthetic_complete_route":"SyntheticModuleRepresentedUnderProfile","no_authority_minting":"NoPhysicalOperationAuthority"}
REQUIRED_RULES = ["equipment availability does not establish water quality or safety","Mycelix operational facts cannot create engineering treatment or water quality evidence","model predictions cannot become FIELD observations","no result may authorize dosing disinfection public-water operation procurement allocation or physical actuation"]
FORBIDDEN = ("score","priority","readiness","self_sufficiency","self-sufficiency","execute","actuate")

def walk_keys(x, prefix=""):
    if isinstance(x, dict):
        for k,v in x.items():
            yield prefix+k
            yield from walk_keys(v, prefix+k+".")
    elif isinstance(x, list):
        for i,v in enumerate(x):
            yield from walk_keys(v, prefix+str(i)+".")

raw = PATH.read_bytes()
assert hashlib.sha256(raw).hexdigest() == EXPECTED_SHA256, "corpus digest drift"
data = json.loads(raw)
assert raw == (json.dumps(data, sort_keys=True, separators=(",",":")) + "\n").encode(), "non-canonical JSON"
assert data["schema"] == EXPECTED_SCHEMA
assert data["authority"] == EXPECTED_AUTHORITY
assert data["module_profile"] == "synthetic_nonpotable_low_head_circulation_filter_observation_module"
assert data["evidence_planes"] == EXPECTED_PLANES
assert data["owners"] == EXPECTED_OWNERS
assert len(data["cases"]) == 16
case_map = {}
for case in data["cases"]:
    assert set(case) == {"id","premise","expected"}
    assert case["id"] not in case_map
    assert case["expected"] in data["dispositions"]
    case_map[case["id"]] = case["expected"]
assert case_map == EXPECTED_CASES, "case identity/outcome drift"
assert len(data["dispositions"]) == len(set(data["dispositions"]))
for rule in REQUIRED_RULES:
    assert rule in data["rules"], f"missing rule: {rule}"
for key in walk_keys(data):
    leaf = key.rsplit(".",1)[-1].lower()
    if leaf == "authority":
        continue
    assert not any(f in leaf for f in FORBIDDEN), f"forbidden authority/score key: {key}"
assert "no procurement allocation or physical operation authority" in data["nonclaims"]
print("PASS_WATER_MFG_001A_REFERENCE_QUALIFICATION", EXPECTED_SHA256, "cases=16", "planes=15")
