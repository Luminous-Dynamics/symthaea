#!/usr/bin/env python3
"""Independent stdlib oracle for CIV-ENG-001A.

Qualification infrastructure only. This script imports no production CIV/MAT/ENG
decision code and grants no physical authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_HEAD = "70577dc3b093cdcf53030ebbdce3a01af0166104"
SOURCE_PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
DOC_BLOB = "f6e553d32930ca006ef8a27ec2c53c1e4d3fd54f"
CORPUS_BLOB = "eae841d345930a105c4fc3a57dfed173a97bc010"
CORPUS_SHA256 = "36aa384278fc16c9b64cf0861e98fc246645c31679099344709e24f0be806974"
SCHEMA = "civ-eng-001a-reference-v1"
DOC_PATH = Path("docs/engineering/CIV_ENG_001A.md")
CORPUS_PATH = Path("docs/engineering/data/civ_eng_001a_reference_v1.json")
QUALIFIER_FILES = {
    ".github/workflows/civ-eng-001a1-qualifier.yml",
    "scripts/validate_civ_eng_001a.py",
}
AXES = ["demand", "routing", "material", "return", "admission", "generation", "effect", "authority"]

ROUTES = {
    "MaterialPropertyGap": "MAT_OPPORTUNITY_THEN_MAT_ENG",
    "MaterialGradeGap": "MAT_OPPORTUNITY_THEN_MAT_ENG",
    "InterfaceOrSurfaceGap": "MAT_INTERFACE_DOMAIN",
    "ProcessCapabilityGap": "MFG_PROC_DOMAIN",
    "PrecisionOrToleranceGap": "ENG_TOL_METROLOGY",
    "MetrologyRenewalGap": "ENG_MEAS_CALIBRATION",
    "CalibrationRenewalGap": "ENG_MEAS_CALIBRATION",
    "ToolingRenewalGap": "MFG_TOOLING",
    "ComponentQualificationGap": "DOMAIN_COMPONENT",
    "ReliabilityOrLifecycleGap": "MFG_LIFE_RELIABILITY",
    "EnergyOrThermalGap": "DOMAIN_ENERGY_THERMAL",
    "SoftwareOrControlGap": "SOFTWARE_CONTROL",
    "SensorCalibrationGap": "SENSE_BOOT",
    "ComputeElectronicsGap": "COMPUTE_SEMI_BOOT",
    "RobotReplacementGap": "ROB_MFG",
    "EvidenceInsufficient": "EVIDENCE_MEASUREMENT",
    "MultiGenerationDegradation": "CIV_BOOT_003_DOMAIN",
}
CODES = {
    "demand": {"I": "Incomplete", "R": "Represented", "S": "StaleOrMismatched"},
    "routing": {"C": "CanonicalOwnerRoute", "N": "NeedsOwnerResolution", "W": "WrongOwnerRoute"},
    "material": {
        "B": "BottleneckEvidenceRequired",
        "E": "MaterialResearchEligibleUnderProfile",
        "M": "MaterialVariableNotBottlenecking",
        "N": "NotMaterialQuestion",
    },
    "return": {
        "C": "CandidateOnly",
        "E": "EngineeringEvidenceAvailable",
        "P": "ProcessOrComponentEvidenceAvailable",
        "S": "CapabilityEnvelopeSupportedUnderProfile",
        "X": "EvidenceInsufficient",
    },
    "admission": {
        "E": "EligibleForNewClosureGeneration",
        "N": "NotEligible",
        "P": "ProfileMismatch",
        "S": "GenerationStale",
    },
    "generation": {
        "1": "G1Only",
        "2": "G2PlusSupportedUnderProfile",
        "D": "GenerationalDegradation",
        "N": "NotEvaluated",
        "U": "Unresolved",
    },
    "effect": {
        "B": "BottleneckMigrated",
        "G": "ClosureGainObserved",
        "I": "PredictedGainUnderestimated",
        "N": "NoClosureGain",
        "O": "PredictedGainOverestimated",
        "U": "Unresolved",
    },
    "authority": {"0": "NoPhysicalExecutionAuthorityFromThisContract"},
}
REVERSE = {axis: {meaning: code for code, meaning in table.items()} for axis, table in CODES.items()}
REQUIRED = {
    "vg", "cg", "ccg", "t", "br", "bc", "ss", "il", "env", "mla", "ba", "route",
    "er", "ev", "pm", "gm", "mc", "vc", "met", "intf", "g1", "g2", "gd",
    "pg", "rg", "bm", "pa", "pf",
}

def fail(message: str) -> None:
    raise AssertionError(message)

def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()

def validate_input(x: dict) -> None:
    if set(x) != REQUIRED:
        fail(f"raw input keys differ: missing={sorted(REQUIRED-set(x))} extra={sorted(set(x)-REQUIRED)}")
    for key in ("il","env","mla","er","pm","gm","vc","met","intf","g1","g2","gd","bm","pa"):
        if type(x[key]) is not bool:
            fail(f"{key} must be bool")
    for key in ("pg","rg"):
        if type(x[key]) is not int or x[key] < 0:
            fail(f"{key} must be non-negative int")
    for key in REQUIRED - {"il","env","mla","er","pm","gm","vc","met","intf","g1","g2","gd","bm","pa","pg","rg"}:
        if not isinstance(x[key], str) or not x[key]:
            fail(f"{key} must be non-empty string")
    if x["ev"] not in {"None","CandidateOnly","EngineeringEvidence","ProcessOrComponentEvidence","CapabilityEnvelopeSupported"}:
        fail("unsupported evidence level")
    if x["ba"] not in {"NotApplicable","Missing","Negative","Positive"}:
        fail("unsupported bottleneck attribution")
    if x["mc"] not in {"Current","Stale","Unknown"}:
        fail("unsupported currentness")

def derive(x: dict) -> list[str]:
    validate_input(x)

    if x["cg"] != x["ccg"]:
        demand = "StaleOrMismatched"
    elif not x["env"] or not x["er"]:
        demand = "Incomplete"
    else:
        demand = "Represented"

    expected_route = ROUTES.get(x["bc"])
    if expected_route is None:
        routing = "NeedsOwnerResolution"
    elif x["route"] == expected_route:
        routing = "CanonicalOwnerRoute"
    else:
        routing = "WrongOwnerRoute"

    if not x["mla"]:
        material = "NotMaterialQuestion"
    elif x["ba"] == "Positive":
        material = "MaterialResearchEligibleUnderProfile"
    elif x["ba"] == "Negative":
        material = "MaterialVariableNotBottlenecking"
    else:
        material = "BottleneckEvidenceRequired"

    returned = {
        "None": "EvidenceInsufficient",
        "CandidateOnly": "CandidateOnly",
        "EngineeringEvidence": "EngineeringEvidenceAvailable",
        "ProcessOrComponentEvidence": "ProcessOrComponentEvidenceAvailable",
        "CapabilityEnvelopeSupported": "CapabilityEnvelopeSupportedUnderProfile",
    }[x["ev"]]

    if demand == "StaleOrMismatched" or not x["gm"] or x["mc"] == "Stale":
        admission = "GenerationStale"
    elif x["ev"] != "CapabilityEnvelopeSupported":
        admission = "NotEligible"
    elif not x["pm"]:
        admission = "ProfileMismatch"
    elif x["mc"] != "Current" or not x["vc"] or not x["met"] or not x["intf"]:
        admission = "NotEligible"
    elif demand != "Represented" or routing != "CanonicalOwnerRoute":
        admission = "NotEligible"
    elif x["mla"] and material != "MaterialResearchEligibleUnderProfile":
        admission = "NotEligible"
    else:
        admission = "EligibleForNewClosureGeneration"

    if not x["g1"]:
        generation = "Unresolved"
    elif x["g2"]:
        generation = "G2PlusSupportedUnderProfile"
    elif x["gd"]:
        generation = "GenerationalDegradation"
    else:
        generation = "G1Only"

    if admission != "EligibleForNewClosureGeneration":
        effect = "Unresolved"
    elif x["bm"]:
        effect = "BottleneckMigrated"
    elif x["rg"] == 0:
        effect = "NoClosureGain"
    elif x["rg"] < x["pg"]:
        effect = "PredictedGainOverestimated"
    elif x["rg"] > x["pg"]:
        effect = "PredictedGainUnderestimated"
    else:
        effect = "ClosureGainObserved"

    meanings = [
        demand, routing, material, returned, admission, generation, effect,
        "NoPhysicalExecutionAuthorityFromThisContract",
    ]
    return [REVERSE[axis][meaning] for axis, meaning in zip(AXES, meanings)]

def full_input(base: dict, overrides: dict) -> dict:
    if not isinstance(overrides, dict):
        fail("case overrides must be object")
    if not set(overrides) <= REQUIRED:
        fail("case overrides contain unknown key")
    x = copy.deepcopy(base)
    x.update(overrides)
    return x

def positive(base: dict) -> dict:
    x = copy.deepcopy(base)
    x.update(ev="CapabilityEnvelopeSupported", g1=True, pg=2, rg=2)
    return x

def axis(result: list[str], name: str) -> str:
    return CODES[name][result[AXES.index(name)]]

def expect(x: dict, name: str, value: str) -> None:
    got = axis(derive(x), name)
    if got != value:
        fail(f"mutation expected {name}={value}, got {got}")

def run_mutations(base: dict) -> None:
    p = positive(base)
    mutations = []

    def m(field, value, axis_name, expected):
        x = copy.deepcopy(p); x[field] = value
        mutations.append((x, axis_name, expected))

    m("env", False, "demand", "Incomplete")
    m("er", False, "demand", "Incomplete")
    m("ccg", "CIV-G1", "demand", "StaleOrMismatched")
    m("bc", "UnknownGap", "routing", "NeedsOwnerResolution")
    m("route", "WRONG_ROUTE", "routing", "WrongOwnerRoute")

    x=copy.deepcopy(p); x.update(bc="MaterialPropertyGap",route="MAT_OPPORTUNITY_THEN_MAT_ENG",mla=True,ba="Missing"); mutations.append((x,"material","BottleneckEvidenceRequired"))
    x=copy.deepcopy(p); x.update(bc="MaterialPropertyGap",route="MAT_OPPORTUNITY_THEN_MAT_ENG",mla=True,ba="Negative"); mutations.append((x,"material","MaterialVariableNotBottlenecking"))
    x=copy.deepcopy(p); x.update(bc="MaterialPropertyGap",route="MAT_OPPORTUNITY_THEN_MAT_ENG",mla=True,ba="Positive"); mutations.append((x,"material","MaterialResearchEligibleUnderProfile"))

    m("ev", "CandidateOnly", "return", "CandidateOnly")
    m("ev", "EngineeringEvidence", "return", "EngineeringEvidenceAvailable")
    m("ev", "ProcessOrComponentEvidence", "return", "ProcessOrComponentEvidenceAvailable")
    m("pm", False, "admission", "ProfileMismatch")
    m("gm", False, "admission", "GenerationStale")
    m("mc", "Stale", "admission", "GenerationStale")
    m("mc", "Unknown", "admission", "NotEligible")
    m("vc", False, "admission", "NotEligible")
    m("met", False, "admission", "NotEligible")
    m("intf", False, "admission", "NotEligible")

    x=copy.deepcopy(p); x.update(g1=True,g2=False,gd=False); mutations.append((x,"generation","G1Only"))
    x=copy.deepcopy(p); x.update(g1=True,g2=True,gd=False); mutations.append((x,"generation","G2PlusSupportedUnderProfile"))
    x=copy.deepcopy(p); x.update(g1=True,g2=False,gd=True); mutations.append((x,"generation","GenerationalDegradation"))

    x=copy.deepcopy(p); x.update(pg=2,rg=0); mutations.append((x,"effect","NoClosureGain"))
    x=copy.deepcopy(p); x.update(pg=5,rg=2); mutations.append((x,"effect","PredictedGainOverestimated"))
    x=copy.deepcopy(p); x.update(pg=2,rg=5); mutations.append((x,"effect","PredictedGainUnderestimated"))
    x=copy.deepcopy(p); x.update(pg=4,rg=2,bm=True); mutations.append((x,"effect","BottleneckMigrated"))
    x=copy.deepcopy(p); x.update(pa=True); mutations.append((x,"authority","NoPhysicalExecutionAuthorityFromThisContract"))

    if len(mutations) < 24:
        fail("mutation suite too small")
    for x, axis_name, expected_value in mutations:
        expect(x, axis_name, expected_value)

def verify_git_shape() -> None:
    if git("rev-parse", "HEAD^") != SOURCE_HEAD:
        fail("qualifier parent is not exact frozen source head")
    if git("rev-parse", "HEAD^^") != SOURCE_PARENT:
        fail("source head parent changed")
    if git("rev-parse", f"HEAD^:{DOC_PATH.as_posix()}") != DOC_BLOB:
        fail("contract blob mismatch")
    if git("rev-parse", f"HEAD^:{CORPUS_PATH.as_posix()}") != CORPUS_BLOB:
        fail("corpus blob mismatch")
    changed = set(filter(None, git("diff", "--name-only", "HEAD^", "HEAD").splitlines()))
    if changed != QUALIFIER_FILES:
        fail(f"qualifier scope mismatch: {sorted(changed)}")

def validate_corpus(data: dict, raw: bytes) -> None:
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        fail("corpus SHA-256 mismatch")
    if data.get("schema") != SCHEMA:
        fail("schema mismatch")
    if data.get("axes") != AXES:
        fail("axis order mismatch")
    if data.get("codes") != CODES:
        fail("codebook mismatch")
    if data.get("route_map") != ROUTES:
        fail("route map mismatch")
    if data.get("authority_ceiling") != "AnalysisProjectionOnly_NoPhysicalExecutionAuthority":
        fail("authority ceiling mismatch")
    base = data.get("base")
    validate_input(base)

    cases = data.get("cases")
    if not isinstance(cases, list) or len(cases) != 32:
        fail("expected exactly 32 cases")
    wanted_ids = [f"C{i:02d}" for i in range(1, 33)]
    ids = [case.get("id") for case in cases]
    if ids != wanted_ids or len(set(ids)) != 32:
        fail("case order/identity mismatch")

    bearing = []
    for case in cases:
        if set(case) != {"id","o","e"}:
            fail(f"{case.get('id')}: unexpected case shape")
        x = full_input(base, case["o"])
        derived = derive(x)
        expected = case["e"]
        if not isinstance(expected, list) or len(expected) != len(AXES):
            fail(f"{case['id']}: malformed expected vector")
        for axis_name, code in zip(AXES, expected):
            if code not in CODES[axis_name]:
                fail(f"{case['id']}: invalid expected code {axis_name}={code}")
        if derived != expected:
            fail(f"{case['id']}: derived {derived} != expected {expected}")
        if x["pf"] == "PrecisionBearing":
            bearing.append(case["id"])

    wanted_bearing = ["C01","C14","C17","C18","C19","C20","C21","C23","C24"]
    if bearing != wanted_bearing:
        fail(f"bearing subset mismatch: {bearing}")

    run_mutations(base)

def main() -> None:
    verify_git_shape()
    raw = CORPUS_PATH.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    validate_corpus(data, raw)
    print("CIV-ENG-001A1 ORACLE PASS")
    print(f"source={SOURCE_HEAD}")
    print(f"corpus_sha256={CORPUS_SHA256}")
    print("cases=32 mutations>=24 authority=proposal-only")

if __name__ == "__main__":
    main()
