#!/usr/bin/env python3
import hashlib, json
from pathlib import Path

PATH = Path("docs/release/evidence/agri-mfg-001a-benign-infrastructure-reference-v1.json")
EXPECTED_SHA256 = "e6608d872353ce80190488ae4f0ba7e9e3cad5a2b5a8cb475ebe70dc854200bf"
EXPECTED_SCHEMA = "agri-mfg-001a-benign-infrastructure-reference-v1"
EXPECTED_AUTHORITY = "analysis_only_no_agronomic_food_safety_resource_allocation_or_physical_actuation_authority"
EXPECTED_PLANES = ["water_and_input_profile","equipment_article_and_configuration_identity","irrigation_component_function","environmental_observation","greenhouse_mechanism_function","production_or_harvest_operational_facts","agronomic_inference","storage_or_cold_chain_state","packaging_and_logistics_dependency","maintenance_and_spares","seasonal_or_time_availability","common_mode_dependency","productive_or_reproductive_closure","food_safety_and_nutrition_external_evidence","authority_ceiling"]
EXPECTED_OWNERS = {"root":"AGRI-MFG#5967","water":"WATER-MFG#5966","components":"IND-COMP#5965","process":"MFG-PROC#5686","observation":"FIELD/SE-OBS","sensing":"SENSE#5820","service":"CIV-SERVICE#5949","closure":"CIV-BOOT#5774","lifecycle":"MFG-LIFE#5705","operations":"Mycelix-operations"}
EXPECTED_CASES = {"irrigation_operates_water_profile_unresolved":"WaterOrInputProfileUnresolved","sensor_recommendation_no_actuation":"NoPhysicalActuationOrAllocationAuthority","shade_mechanism_effect_unobserved":"EnvironmentalEffectUnresolved","high_output_not_food_truth":"FoodSafetyOrNutritionAuthorityExternal","cold_room_no_current_temperature_history":"ColdChainEvidenceUnresolved","imported_packaging_refrigeration_consumables":"ProductiveClosurePartial","seasonal_input_availability":"SeasonalAvailabilityUnresolved","implement_nonrenewable_bearing_seal_tooling":"MaintenanceOrRenewalUnresolved","field_or_site_profile_changed":"ApplicabilityTransferBlocked","mycelix_harvest_not_engineering":"OperationalFactNotEngineeringEvidence","model_not_field_observation":"ModelNotPhysicalObservation","shared_power_water_failure_root":"CommonModeDependencyUnresolved","bounded_greenhouse_profile_only":"ProfileBoundedInfrastructureOnly","equipment_present_function_unmeasured":"InfrastructureFunctionUnresolved","synthetic_complete_route":"SyntheticInfrastructureRepresentedUnderProfile","no_authority_minting":"NoPhysicalActuationOrAllocationAuthority"}
REQUIRED_RULES = ["infrastructure function does not establish agronomic outcome","crop or harvest quantity does not establish food safety or nutrition","sensor or model recommendation cannot authorize irrigation nutrient application harvesting or food handling","Mycelix operational facts cannot create engineering agronomic food safety or nutrition evidence","model predictions cannot become FIELD observations","seasonal availability remains distinct from structural route existence","no result may authorize procurement resource allocation physical actuation or public food service"]
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
assert data["module_profile"] == "synthetic_irrigation_environment_monitoring_greenhouse_shade_ventilation_module"
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
assert "no procurement resource allocation irrigation chemical application or physical execution authority" in data["nonclaims"]
print("PASS_AGRI_MFG_001A_REFERENCE_QUALIFICATION", EXPECTED_SHA256, "cases=16", "planes=15")
