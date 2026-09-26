#!/usr/bin/env python3
import hashlib
import json
from collections import Counter
from pathlib import Path

CORPUS = Path("docs/release/evidence/eng-design-004a-poetic-engineering-reference-v1.json")
EXPECTED_SHA256 = "770d5915f50397a3e99ef31db0a44b5c22ac40c14896421d06a1c4c4af3161b9"
EXPECTED_SCHEMA = "eng-design-004a-poetic-engineering-reference-v1"
EXPECTED_AUTHORITY = "repository_design_composition_only_no_physical_execution_authority"
EXPECTED_IDS = [f"C{i:02d}" for i in range(1, 18)]
EXPECTED_FORBIDDEN = {
    "poetry_score", "beauty_score", "meaning_score", "aesthetic_score",
    "cultural_correctness_score", "universal_rank",
}
EXPECTED_CENSUS = {
    "NoPoeticIntent": 1,
    "IntentCaptured": 1,
    "PoeticMoveAdmissible": 2,
    "UnattributedSymbolismBlocked": 1,
    "EngineeringConstraintBlocked": 1,
    "PoeticChangeReviewRequired": 1,
    "SymbolicIntentStale": 1,
    "AccessibilityConstraintBlocked": 1,
    "MaterialNarrativeReviewRequired": 1,
    "AsBuiltExperienceStale": 1,
    "ExperientialValidationFailed": 1,
    "ProtectedFeatureContinuity": 1,
    "ContestedInterpretationRetained": 1,
    "OptionalOrnamentOnly": 1,
    "EngineeringReleaseEligiblePoeticValidationBounded": 1,
    "AuthorityBoundaryBlocked": 1,
}

def fail(msg):
    raise SystemExit(msg)

raw = CORPUS.read_bytes()
if hashlib.sha256(raw).hexdigest() != EXPECTED_SHA256:
    fail("corpus digest mismatch")
data = json.loads(raw)
canonical = (json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
if canonical != raw:
    fail("corpus is not canonical JSON")
if data.get("schema") != EXPECTED_SCHEMA:
    fail("schema mismatch")
if data.get("authority") != EXPECTED_AUTHORITY:
    fail("authority mismatch")
if set(data.get("forbidden_scalar_keys", [])) != EXPECTED_FORBIDDEN:
    fail("forbidden scalar vocabulary mismatch")

allowed_fields = set(data["allowed_fields"])
defaults = data["defaults"]
if set(defaults) != allowed_fields:
    fail("defaults/allowed_fields mismatch")
allowed_validation = set(data["allowed_enums"]["human_validation_result"])

def reject_forbidden(obj):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in EXPECTED_FORBIDDEN:
                fail(f"forbidden scalar key present: {k}")
            reject_forbidden(v)
    elif isinstance(obj, list):
        for v in obj:
            reject_forbidden(v)

reject_forbidden(data)

def expand(overrides):
    unknown = set(overrides) - allowed_fields
    if unknown:
        fail(f"unknown override keys: {sorted(unknown)}")
    x = dict(defaults)
    x.update(overrides)
    if x["human_validation_result"] not in allowed_validation:
        fail("unknown human_validation_result")
    if not isinstance(x["functional_roles"], int) or x["functional_roles"] < 0:
        fail("functional_roles invalid")
    if not isinstance(x["experiential_roles"], int) or x["experiential_roles"] < 0:
        fail("experiential_roles invalid")
    return x

def derive(overrides):
    x = expand(overrides)
    if x["physical_execution_authority_requested"]:
        return "AuthorityBoundaryBlocked"
    if not x["intent_declared"] and not x["ornament_declared"] and not x["symbolic_claim_requested"]:
        return "NoPoeticIntent"
    if x["symbolic_claim_requested"] and not x["intent_source_bound"]:
        return "UnattributedSymbolismBlocked"
    if not x["engineering_hard_constraints_satisfied"] or not x["maintenance_access_satisfied"]:
        return "EngineeringConstraintBlocked"
    if not x["accessibility_satisfied"]:
        return "AccessibilityConstraintBlocked"
    if x["design_changed"] and x["protected_feature"] and not x["protected_feature_preserved"]:
        return "PoeticChangeReviewRequired"
    if x["symbolic_claim_requested"] and not x["symbolic_source_current"]:
        return "SymbolicIntentStale"
    if x["material_narrative_declared"] and not x["material_provenance_current"]:
        return "MaterialNarrativeReviewRequired"
    if not x["as_built_matches_model_intent"]:
        return "AsBuiltExperienceStale"
    if x["human_legibility_required"] and x["operational_correct"] and not x["human_legibility_pass"]:
        return "ExperientialValidationFailed"
    if x["human_validation_result"] == "Contested":
        return "ContestedInterpretationRetained"
    if x["protected_feature"] and x["protected_feature_preserved"] and x["transform_performed"]:
        return "ProtectedFeatureContinuity"
    if x["ornament_declared"] and x["functional_roles"] == 0 and x["experiential_roles"] == 0 and not x["symbolic_claim_requested"]:
        return "OptionalOrnamentOnly"
    if x["engineering_release_prereqs"] and x["human_validation_result"] == "Pass":
        return "EngineeringReleaseEligiblePoeticValidationBounded"
    if x["functional_roles"] + x["experiential_roles"] >= 3:
        return "PoeticMoveAdmissible"
    return "IntentCaptured"

cases = data["cases"]
if [c["id"] for c in cases] != EXPECTED_IDS:
    fail("case IDs/order mismatch")

census = Counter()
for case in cases:
    if set(case) != {"id", "title", "overrides", "expected"}:
        fail(f"invalid case shape: {case.get('id')}")
    observed = derive(case["overrides"])
    if observed != case["expected"]:
        fail(f"{case['id']}: expected {case['expected']}, derived {observed}")
    census[observed] += 1

if dict(census) != EXPECTED_CENSUS:
    fail(f"disposition census mismatch: {dict(census)}")

if derive({"engineering_release_prereqs": True, "human_validation_result": "Pass", "physical_execution_authority_requested": True}) != "AuthorityBoundaryBlocked":
    fail("authority hostile self-test failed")

try:
    expand({"poetry_score": 0.99})
except SystemExit:
    pass
else:
    fail("forbidden/unknown score key was accepted")

try:
    expand({"human_validation_result": "UniversallyBeautiful"})
except SystemExit:
    pass
else:
    fail("unknown human validation enum was accepted")

print(
    "PASS_ENG_DESIGN_004A_POETIC "
    f"digest={EXPECTED_SHA256} cases={len(cases)} outcomes={json.dumps(dict(census), sort_keys=True)}"
)
