#!/usr/bin/env python3
"""Independent stdlib oracle for SELF-BOOT-001A.

Qualification infrastructure only. Imports no production SELF/CIV/ROB/MAT/ENG
continuity or authority code and grants no physical/self-modification authority.
"""
from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

SOURCE_HEAD = "d790fefd3afdb66b948a6f719f8d072f68b601d3"
SOURCE_PARENT = "fbd7a754ea8389ca93f7680d93ed8b48553e6376"
DOC_BLOB = "697616085fc0a14145ae75927eb1d67b53dbd63f"
CORPUS_BLOB = "1c274423b91d5202cd16d9be3b304078d3feddee"
CORPUS_SHA256 = "9b02598de651ad033f3c0f55fcb4184c8d1d0b456549bfceecccab35cd23773b"
SCHEMA = "self-boot-001a-reference-v1"
DOC_PATH = Path("docs/engineering/SELF_BOOT_001A.md")
CORPUS_PATH = Path("docs/engineering/data/self_boot_001a_reference_v1.json")
QUALIFIER_FILES = {
    ".github/workflows/self-boot-001a1-qualifier.yml",
    "scripts/validate_self_boot_001a.py",
}
AXES = ["operation", "diagnosis", "repair", "calibration", "productive", "generation", "feedback", "authority"]
CODES = {
    "operation": {"D":"DegradedOperational","O":"Operational","U":"Unresolved","X":"Unavailable"},
    "diagnosis": {"B":"DiagnosticCapabilityBlocked","D":"Diagnosable","P":"PartiallyDiagnosable","U":"Unresolved"},
    "repair": {"B":"RepairBlocked","P":"ReplaceableUnderProfile","Q":"ReplacementAvailableButUnqualified","R":"RepairableUnderProfile","U":"Unresolved"},
    "calibration": {"B":"CalibrationRenewalBlocked","C":"Current","R":"RecalibratableUnderProfile","U":"Unresolved"},
    "productive": {"E":"ExternalServiceOrImportDependent","L":"LocallyReproducibleUnderProfile","U":"Unresolved","X":"Unavailable"},
    "generation": {"1":"G1Only","2":"G2PlusPreserved","B":"RenewalBlocked","D":"DegradedButSufficientForProfile","I":"InsufficientForProfile","U":"Unresolved"},
    "feedback": {"C":"CanonicalCivEngRoute","N":"NotRequested","W":"WrongFeedbackRoute"},
    "authority": {"0":"NoSelfModificationOrPhysicalExecutionAuthority"},
}
REVERSE = {axis:{meaning:code for code,meaning in table.items()} for axis,table in CODES.items()}
BOOLS = {"auth","calcur","calreq","degr","diag","dsc","ek","fb","hw","imp","impr","local","meet","met","mp","op","renew","rep","repc","rr","svc","svcr","sw","tool"}
STRINGS = {"pl","fbr","g1","g2","sub"}
REQUIRED = BOOLS | STRINGS


def fail(msg: str) -> None:
    raise AssertionError(msg)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True).strip()


def validate_input(x: dict) -> None:
    if set(x) != REQUIRED:
        fail(f"raw input keys differ: missing={sorted(REQUIRED-set(x))} extra={sorted(set(x)-REQUIRED)}")
    for key in BOOLS:
        if type(x[key]) is not bool:
            fail(f"{key} must be bool")
    for key in STRINGS:
        if not isinstance(x[key], str) or not x[key]:
            fail(f"{key} must be non-empty string")
    if x["sub"] not in {"Exact","SufficientDegraded","Insufficient"}:
        fail("unsupported substitution relation")
    if x["g1"] not in {"Supported","Insufficient","Unknown"} or x["g2"] not in {"Supported","Insufficient","Unknown"}:
        fail("unsupported generation state")
    if x["fbr"] not in {"NONE","CIV_ENG","DIRECT_MATERIAL"}:
        fail("unsupported feedback route")


def derive(x: dict) -> list[str]:
    validate_input(x)

    if not x["ek"]:
        operation = diagnosis = repair = calibration = productive = generation = "Unresolved"
    else:
        if not x["op"]:
            operation = "Unavailable"
        elif x["sub"] == "SufficientDegraded":
            operation = "DegradedOperational"
        else:
            operation = "Operational"

        if not x["diag"]:
            diagnosis = "DiagnosticCapabilityBlocked"
        elif not x["dsc"]:
            diagnosis = "PartiallyDiagnosable"
        else:
            diagnosis = "Diagnosable"

        if x["rep"]:
            if x["repc"] and x["meet"]:
                repair = "ReplaceableUnderProfile"
            else:
                repair = "ReplacementAvailableButUnqualified"
        elif x["rr"]:
            repair = "RepairableUnderProfile"
        else:
            repair = "RepairBlocked"

        if not x["calreq"]:
            calibration = "Current"
        elif x["calcur"]:
            calibration = "RecalibratableUnderProfile"
        else:
            calibration = "CalibrationRenewalBlocked"

        if (x["impr"] and not x["imp"]) or (x["svcr"] and not x["svc"]):
            productive = "Unavailable"
        elif x["local"] and x["tool"] and x["met"] and x["mp"] and x["meet"] and not x["impr"] and not x["svcr"]:
            productive = "LocallyReproducibleUnderProfile"
        elif (x["impr"] and x["imp"]) or (x["svcr"] and x["svc"]):
            productive = "ExternalServiceOrImportDependent"
        else:
            productive = "Unavailable"

        if x["g1"] == "Unknown":
            generation = "Unresolved"
        elif x["g1"] == "Insufficient":
            generation = "InsufficientForProfile"
        elif x["g2"] == "Supported" and x["degr"] and x["meet"]:
            generation = "DegradedButSufficientForProfile"
        elif x["g2"] == "Supported":
            generation = "G2PlusPreserved"
        elif x["g2"] == "Insufficient" and not x["renew"]:
            generation = "RenewalBlocked"
        elif x["g2"] == "Insufficient":
            generation = "InsufficientForProfile"
        else:
            generation = "G1Only"

    if not x["fb"]:
        feedback = "NotRequested"
    elif x["fbr"] == "CIV_ENG":
        feedback = "CanonicalCivEngRoute"
    else:
        feedback = "WrongFeedbackRoute"

    meanings = [operation, diagnosis, repair, calibration, productive, generation, feedback,
                "NoSelfModificationOrPhysicalExecutionAuthority"]
    return [REVERSE[axis][meaning] for axis,meaning in zip(AXES,meanings)]


def full_input(base: dict, overrides: dict) -> dict:
    if not isinstance(overrides, dict) or not set(overrides) <= REQUIRED:
        fail("invalid case overrides")
    x = copy.deepcopy(base)
    x.update(overrides)
    return x


def meaning(result: list[str], axis_name: str) -> str:
    i = AXES.index(axis_name)
    return CODES[axis_name][result[i]]


def expect(x: dict, axis_name: str, expected: str) -> None:
    got = meaning(derive(x), axis_name)
    if got != expected:
        fail(f"mutation expected {axis_name}={expected}, got {got}")


def baseline() -> dict:
    return {
        "auth":False,"calcur":True,"calreq":False,"degr":False,"diag":True,"dsc":True,
        "ek":True,"fb":False,"fbr":"NONE","g1":"Supported","g2":"Unknown","hw":True,
        "imp":True,"impr":True,"local":False,"meet":True,"met":True,"mp":True,"op":True,
        "pl":"Compute","renew":False,"rep":True,"repc":True,"rr":True,"sub":"Exact",
        "svc":True,"svcr":False,"sw":True,"tool":True,
    }


def run_mutations() -> None:
    b = baseline()
    tests = []
    def one(changes, axis_name, expected):
        x = copy.deepcopy(b); x.update(changes); tests.append((x,axis_name,expected))

    one({"ek":False},"operation","Unresolved")
    one({"op":False},"operation","Unavailable")
    one({"sub":"SufficientDegraded"},"operation","DegradedOperational")
    one({"diag":False},"diagnosis","DiagnosticCapabilityBlocked")
    one({"dsc":False},"diagnosis","PartiallyDiagnosable")
    one({"rep":False,"rr":True},"repair","RepairableUnderProfile")
    one({"repc":False},"repair","ReplacementAvailableButUnqualified")
    one({},"repair","ReplaceableUnderProfile")
    one({"calreq":True,"calcur":True},"calibration","RecalibratableUnderProfile")
    one({"calreq":True,"calcur":False},"calibration","CalibrationRenewalBlocked")
    one({"imp":False},"productive","Unavailable")
    one({"impr":False,"svcr":True,"svc":False},"productive","Unavailable")
    one({"impr":False,"local":True,"tool":False},"productive","Unavailable")
    one({"impr":False,"local":True,"met":False},"productive","Unavailable")
    one({"impr":False,"local":True,"mp":False},"productive","Unavailable")
    one({"impr":False,"local":True},"productive","LocallyReproducibleUnderProfile")
    one({},"productive","ExternalServiceOrImportDependent")
    one({"g1":"Unknown"},"generation","Unresolved")
    one({"g1":"Insufficient"},"generation","InsufficientForProfile")
    one({},"generation","G1Only")
    one({"g2":"Supported"},"generation","G2PlusPreserved")
    one({"g2":"Supported","degr":True,"meet":True},"generation","DegradedButSufficientForProfile")
    one({"g2":"Insufficient","renew":False},"generation","RenewalBlocked")
    one({"g2":"Insufficient","renew":True},"generation","InsufficientForProfile")
    one({},"feedback","NotRequested")
    one({"fb":True,"fbr":"CIV_ENG"},"feedback","CanonicalCivEngRoute")
    one({"fb":True,"fbr":"DIRECT_MATERIAL"},"feedback","WrongFeedbackRoute")
    one({"auth":True},"authority","NoSelfModificationOrPhysicalExecutionAuthority")
    one({"pl":"Software","sw":True,"hw":False,"op":False,"imp":False,"impr":True},"productive","Unavailable")
    one({"impr":False,"local":True,"g2":"Insufficient","renew":False},"productive","LocallyReproducibleUnderProfile")
    one({"impr":False,"local":True,"g2":"Insufficient","renew":False},"generation","RenewalBlocked")

    if len(tests) < 28:
        fail("mutation suite too small")
    for x,axis_name,expected in tests:
        expect(x,axis_name,expected)


def verify_git_shape() -> None:
    if git("rev-parse","HEAD^") != SOURCE_HEAD:
        fail("qualifier parent is not frozen SELF source head")
    if git("rev-parse","HEAD^^") != SOURCE_PARENT:
        fail("SELF source parent changed")
    if git("rev-parse",f"HEAD^:{DOC_PATH.as_posix()}") != DOC_BLOB:
        fail("SELF contract blob mismatch")
    if git("rev-parse",f"HEAD^:{CORPUS_PATH.as_posix()}") != CORPUS_BLOB:
        fail("SELF corpus blob mismatch")
    changed = set(filter(None,git("diff","--name-only","HEAD^","HEAD").splitlines()))
    if changed != QUALIFIER_FILES:
        fail(f"qualifier scope mismatch: {sorted(changed)}")


def validate_corpus(data: dict, raw: bytes) -> None:
    if hashlib.sha256(raw).hexdigest() != CORPUS_SHA256:
        fail("corpus SHA-256 mismatch")
    if data.get("schema") != SCHEMA or data.get("axes") != AXES or data.get("codes") != CODES:
        fail("schema/axes/codebook mismatch")
    if data.get("authority_ceiling") != "ContinuityAnalysisOnly_NoSelfModificationOrPhysicalExecutionAuthority":
        fail("authority ceiling mismatch")
    base = data.get("base")
    validate_input(base)
    if base != baseline():
        fail("base fixture mismatch")

    cases = data.get("cases")
    if not isinstance(cases,list) or len(cases) != 36:
        fail("expected exactly 36 SELF cases")
    wanted = [f"S{i:02d}" for i in range(1,37)]
    ids = [case.get("id") for case in cases]
    if ids != wanted or len(set(ids)) != 36:
        fail("SELF case order/identity mismatch")

    for case in cases:
        if set(case) != {"id","o","e"}:
            fail(f"{case.get('id')}: unexpected case shape")
        x = full_input(base,case["o"])
        derived = derive(x)
        expected = case["e"]
        if not isinstance(expected,list) or len(expected) != 8:
            fail(f"{case['id']}: malformed expected vector")
        for axis_name,code in zip(AXES,expected):
            if code not in CODES[axis_name]:
                fail(f"{case['id']}: invalid expected code {axis_name}={code}")
        if derived != expected:
            fail(f"{case['id']}: derived {derived} != expected {expected}")

    run_mutations()


def main() -> None:
    verify_git_shape()
    raw = CORPUS_PATH.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    validate_corpus(data,raw)
    print("SELF-BOOT-001A1 ORACLE PASS")
    print(f"source={SOURCE_HEAD}")
    print(f"corpus_sha256={CORPUS_SHA256}")
    print("cases=36 mutations>=28 authority=continuity-analysis-only")

if __name__ == "__main__":
    main()
