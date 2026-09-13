#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK-3C native analytical-evidence reference theorem.

Authority ladder:
  native calculation != admitted analytical evidence
  != historical discharge receipt != current analytical discharge fact
  != requirement satisfaction / qualification / actuation authority

V2 currentness theorem:
  same currentness assertion != currently applicable forever

This is a stdlib-only reference implementation. It freezes authority semantics,
not Euler-Bernoulli physical applicability, premise authenticity, or model qualification.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import struct

REQ_D = b"symthaea.etk-accepted-requirement.v1\0"
OBL_D = b"symthaea.etk-proof-obligation-snapshot.v1\0"
CURRENTNESS_D = b"symthaea.etk-currentness-assertion.v2\0"
METHOD_D = b"symthaea.etk-native-analytical-method.v1\0"
INPUT_D = b"symthaea.etk-native-analytical-input.v1\0"
POLICY_D = b"symthaea.etk-native-analytical-policy.v1\0"
PLAN_D = b"symthaea.etk-native-analytical-plan.v1\0"
ADMIT_D = b"symthaea.etk-native-analytical-admitted-evidence.v1\0"
RECEIPT_D = b"symthaea.etk-native-analytical-discharge-receipt.v1\0"
FACT_D = b"symthaea.etk-current-native-analytical-discharge-fact.v2\0"

SUBJECT = "sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e"
TWIN = "sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9"
VALIDITY = "sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890"
OBL_ID = "00000000-0000-4000-8000-000000000042"
CLAIM = "stress remains below allowable under service load"
REQUIREMENT_MAX_STRESS_PA = 250e6
REL_TOL = 1e-12

OBSERVED_AT_UNIX_MS = 1_789_123_456_000
VALID_UNTIL_UNIX_MS = 1_789_209_856_000
EVALUATED_AT_UNIX_MS = 1_789_123_457_000

EXPECTED = {
    "requirement_revision_id": "sha256:10891514f6551c85b10674a3df04ab2c6d19f74f014adeacc671fa31deecd419",
    "obligation_revision_id": "sha256:ea6e2f0f5e3ec37524325eda24b95ead75fa502f5d2a0b9410a407aed3a12a8c",
    "method_revision_id": "sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3",
    "input_revision_id": "sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86",
    "policy_revision_id": "sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5",
    "currentness_assertion_id": "sha256:deab1cf24d0cb23c7e72697f9bc41277b82997696a13ea1c25fb3c7a7d0994c6",
    "analytical_plan_id": "sha256:bc308f5399ca9032131dcbed3f53991ab104465b15471521b0388be7f611b5e1",
    "admitted_evidence_id": "sha256:228f0ceb1566fa133d8647ccc5859ffd45957ab5138d67fa4b943a79fcdf0ec7",
    "historical_receipt_id": "sha256:ce0b9d96d83e829d3878e7303e35ac607dbed8512a2202e7c224ca4c32ee0b81",
    "current_discharge_fact_id": "sha256:2d6e90e20460efc8f2f3d24c72e2c7a07be77e0143119d38f23385a868dad6eb",
}


class Denied(ValueError):
    pass


def cj(value):
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def dh(domain, value):
    return "sha256:" + hashlib.sha256(domain + cj(value).encode()).hexdigest()


def d(ch):
    return "sha256:" + ch * 64


def f(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Denied("non_finite")
    if not math.isfinite(float(value)):
        raise Denied("non_finite")
    normalized = 0.0 if float(value) == 0.0 else float(value)
    return "f64:" + struct.pack(">d", normalized).hex()


def dg(value, field):
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise Denied("invalid_digest:" + field)
    hex_part = value[7:]
    if hex_part.lower() != hex_part or any(c not in "0123456789abcdef" for c in hex_part):
        raise Denied("invalid_digest:" + field)
    return value


def timestamp_ms(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise Denied("invalid_timestamp:" + field)
    return value


def close(actual, expected):
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1.0) <= REL_TOL


def currentness(
    attestation="4",
    observed_at_unix_ms=OBSERVED_AT_UNIX_MS,
    valid_until_unix_ms=VALID_UNTIL_UNIX_MS,
):
    observed = timestamp_ms(observed_at_unix_ms, "observed_at_unix_ms")
    valid_until = timestamp_ms(valid_until_unix_ms, "valid_until_unix_ms")
    if valid_until <= observed:
        raise Denied("invalid_currentness_window")
    assertion_id = dh(
        CURRENTNESS_D,
        {
            "attestation_digest": d(attestation),
            "observed_at_unix_ms": observed,
            "schema": "symthaea.etk-currentness-assertion.v2",
            "twin_revision_id": TWIN,
            "valid_until_unix_ms": valid_until,
            "validity_domain_revision_id": VALIDITY,
        },
    )
    return {
        "id": assertion_id,
        "observed_at_unix_ms": observed,
        "valid_until_unix_ms": valid_until,
    }


def context(current=None):
    current = currentness() if current is None else current
    if not isinstance(current, dict):
        raise Denied("invalid_currentness_record")
    current_id = dg(current.get("id"), "currentness_assertion_id")
    observed = timestamp_ms(current.get("observed_at_unix_ms"), "observed_at_unix_ms")
    valid_until = timestamp_ms(current.get("valid_until_unix_ms"), "valid_until_unix_ms")
    if valid_until <= observed:
        raise Denied("invalid_currentness_window")

    requirement = dh(
        REQ_D,
        {
            "acceptance_record_digest": d("a"),
            "criticality": "Blocking",
            "domain": "Civil",
            "expected_evidence_kind": "Analysis",
            "logical_requirement_id": "REQ-STRESS",
            "schema": "symthaea.etk-accepted-requirement.v1",
            "statement": "stress remains below allowable",
            "structural_invariants": ["stress <= 250 MPa"],
        },
    )
    obligation = dh(
        OBL_D,
        {
            "claim": CLAIM,
            "expected_evidence_kind": "Analysis",
            "obligation_id": OBL_ID,
            "schema": "symthaea.etk-proof-obligation-snapshot.v1",
        },
    )
    return {
        "requirement": requirement,
        "requirement_max_stress_pa": REQUIREMENT_MAX_STRESS_PA,
        "obligation": obligation,
        "subject": SUBJECT,
        "twin": TWIN,
        "validity": VALIDITY,
        "currentness": current_id,
        "currentness_observed_at_unix_ms": observed,
        "currentness_valid_until_unix_ms": valid_until,
    }


def method():
    return dh(
        METHOD_D,
        {
            "algorithm_revision_digest": d("b"),
            "assumptions": [
                "euler_bernoulli_kinematics",
                "linear_elastic_material",
                "prismatic_beam",
                "single_span",
                "small_deflection",
                "statically_determinate",
            ],
            "implementation_artifact_digest": d("a"),
            "method_key": "symthaea-structural/euler-bernoulli-beam",
            "outputs": [
                {"name": "factor_of_safety", "unit": "1"},
                {"name": "max_bending_stress", "unit": "Pa"},
                {"name": "max_deflection", "unit": "m"},
                {"name": "max_moment", "unit": "N*m"},
            ],
            "schema": "symthaea.etk-native-analytical-method.v1",
            "supported_load_cases": [
                "cantilever_end_point",
                "cantilever_udl",
                "simply_supported_center_point",
                "simply_supported_udl",
            ],
            "unit_system": "SI",
        },
    )


def inputs(load=1000.0):
    return {
        "beam": {
            "length_m": 2.0,
            "material": {
                "youngs_modulus_pa": 200e9,
                "yield_strength_pa": 250e6,
            },
            "section": {
                "height_m": 0.1,
                "kind": "rectangular",
                "width_m": 0.05,
            },
        },
        "load": {
            "kind": "cantilever_end_point",
            "unit": "N",
            "value": load,
        },
    }


def input_id(x):
    beam = x["beam"]
    section = beam["section"]
    material = beam["material"]
    load = x["load"]
    values = [
        beam["length_m"],
        section["width_m"],
        section["height_m"],
        material["youngs_modulus_pa"],
        material["yield_strength_pa"],
        load["value"],
    ]
    if (
        section["kind"] != "rectangular"
        or load["kind"] != "cantilever_end_point"
        or load["unit"] != "N"
        or any(float(value) <= 0 for value in values)
    ):
        raise Denied("unsupported_or_invalid_input")
    return dh(
        INPUT_D,
        {
            "beam": {
                "length_m": f(beam["length_m"]),
                "material": {
                    "youngs_modulus_pa": f(material["youngs_modulus_pa"]),
                    "yield_strength_pa": f(material["yield_strength_pa"]),
                },
                "section": {
                    "height_m": f(section["height_m"]),
                    "kind": "rectangular",
                    "width_m": f(section["width_m"]),
                },
            },
            "load": {
                "kind": load["kind"],
                "unit": "N",
                "value": f(load["value"]),
            },
            "method_revision_id": method(),
            "schema": "symthaea.etk-native-analytical-input.v1",
        },
    )


def policy(threshold=2.0, max_error=0.05):
    if not math.isfinite(float(threshold)) or float(threshold) <= 0:
        raise Denied("invalid_policy_threshold")
    if not math.isfinite(float(max_error)) or not 0 <= float(max_error) <= 1:
        raise Denied("invalid_policy_error")
    return {
        "id": dh(
            POLICY_D,
            {
                "max_model_relative_error_bound": f(max_error),
                "metric": "factor_of_safety",
                "model_qualification_record_digest": d("9"),
                "operator": ">=",
                "schema": "symthaea.etk-native-analytical-policy.v1",
                "threshold": f(threshold),
            },
        ),
        "threshold": float(threshold),
        "max_error": float(max_error),
    }


def plan(x, current=None, pol=None):
    ctx = context(current)
    pol = policy() if pol is None else pol
    material = x["beam"]["material"]
    worst = float(material["yield_strength_pa"]) / pol["threshold"] * (1.0 + pol["max_error"])
    if worst > ctx["requirement_max_stress_pa"]:
        raise Denied("policy_requirement_mismatch")
    method_id = method()
    input_revision = input_id(x)
    preimage = {
        "acceptance_policy_revision_id": pol["id"],
        "currentness_assertion_id": ctx["currentness"],
        "input_revision_id": input_revision,
        "method_revision_id": method_id,
        "obligation_id": OBL_ID,
        "obligation_revision_id": ctx["obligation"],
        "requirement_revision_id": ctx["requirement"],
        "schema": "symthaea.etk-native-analytical-plan.v1",
        "subject_revision_id": ctx["subject"],
        "twin_revision_id": ctx["twin"],
        "validity_domain_revision_id": ctx["validity"],
    }
    return {
        **ctx,
        "method": method_id,
        "input": input_revision,
        "policy": pol["id"],
        "policy_spec": pol,
        "plan": dh(PLAN_D, preimage),
    }


def expected_outputs(x):
    beam = x["beam"]
    section = beam["section"]
    material = beam["material"]
    load = x["load"]
    moment = float(load["value"]) * float(beam["length_m"])
    section_modulus = float(section["width_m"]) * float(section["height_m"]) ** 2 / 6.0
    stress = moment / section_modulus
    inertia = float(section["width_m"]) * float(section["height_m"]) ** 3 / 12.0
    deflection = (
        float(load["value"])
        * float(beam["length_m"]) ** 3
        / (3.0 * float(material["youngs_modulus_pa"]) * inertia)
    )
    return moment, stress, deflection, float(material["yield_strength_pa"]) / stress


def result(x=None, fos=None, err=0.02, artifact=None, deflection=None, moment=None):
    x = inputs() if x is None else x
    expected_moment, stress, expected_deflection, expected_fos = expected_outputs(x)
    return {
        "method_revision_id": method(),
        "input_revision_id": input_id(x),
        "execution_artifact_digest": d("8") if artifact is None else artifact,
        "factor_of_safety": expected_fos if fos is None else fos,
        "max_bending_stress_pa": stress,
        "max_deflection_m": expected_deflection if deflection is None else deflection,
        "max_moment_nm": expected_moment if moment is None else moment,
        "model_relative_error_bound": err,
    }


def admit(p, x, r):
    pol = p["policy_spec"]
    if (
        method() != p["method"]
        or input_id(x) != p["input"]
        or r.get("method_revision_id") != p["method"]
        or r.get("input_revision_id") != p["input"]
    ):
        raise Denied("plan_binding_mismatch")
    dg(r["execution_artifact_digest"], "execution_artifact_digest")
    keys = [
        "factor_of_safety",
        "max_bending_stress_pa",
        "max_deflection_m",
        "max_moment_nm",
        "model_relative_error_bound",
    ]
    if not all(math.isfinite(float(r[key])) for key in keys):
        raise Denied("invalid_result")
    expected_moment, expected_stress, expected_deflection, expected_fos = expected_outputs(x)
    if not close(r["max_moment_nm"], expected_moment):
        raise Denied("equation_mismatch:max_moment")
    if not close(r["max_bending_stress_pa"], expected_stress):
        raise Denied("equation_mismatch:max_stress")
    if not close(r["max_deflection_m"], expected_deflection):
        raise Denied("equation_mismatch:max_deflection")
    if not close(r["factor_of_safety"], expected_fos):
        raise Denied("inconsistent_factor_of_safety")
    error_bound = float(r["model_relative_error_bound"])
    if not 0 <= error_bound <= pol["max_error"]:
        raise Denied("model_error_budget")
    if (
        float(r["max_bending_stress_pa"]) * (1.0 + error_bound)
        > p["requirement_max_stress_pa"]
    ):
        raise Denied("requirement_stress_predicate")
    conservative = float(r["factor_of_safety"]) / (1.0 + error_bound)
    if conservative < pol["threshold"]:
        raise Denied("acceptance_predicate")
    normalized = {
        "execution_artifact_digest": r["execution_artifact_digest"],
        "factor_of_safety": f(r["factor_of_safety"]),
        "max_bending_stress_pa": f(r["max_bending_stress_pa"]),
        "max_deflection_m": f(r["max_deflection_m"]),
        "max_moment_nm": f(r["max_moment_nm"]),
        "model_relative_error_bound": f(error_bound),
    }
    return dh(
        ADMIT_D,
        {
            "analytical_plan_id": p["plan"],
            "conservative_factor_of_safety": f(conservative),
            "normalized_result": normalized,
            "schema": "symthaea.etk-native-analytical-admitted-evidence.v1",
        },
    )


def receipt(p, admitted):
    return dh(
        RECEIPT_D,
        {
            "admitted_analytical_evidence_id": admitted,
            "analytical_plan_id": p["plan"],
            "obligation_id": OBL_ID,
            "obligation_revision_id": p["obligation"],
            "schema": "symthaea.etk-native-analytical-discharge-receipt.v1",
        },
    )


def current_fact(current, historical, rcpt, evaluated_at_unix_ms=EVALUATED_AT_UNIX_MS):
    if current["plan"] != historical["plan"]:
        raise Denied("historical_plan")
    evaluated_at = timestamp_ms(evaluated_at_unix_ms, "evaluated_at_unix_ms")
    if evaluated_at < current["currentness_observed_at_unix_ms"]:
        raise Denied("freshness_not_yet_valid")
    if evaluated_at > current["currentness_valid_until_unix_ms"]:
        raise Denied("freshness_expired")
    return dh(
        FACT_D,
        {
            "acceptance_policy_revision_id": current["policy"],
            "analytical_plan_id": current["plan"],
            "currentness_assertion_id": current["currentness"],
            "currentness_observed_at_unix_ms": current["currentness_observed_at_unix_ms"],
            "currentness_valid_until_unix_ms": current["currentness_valid_until_unix_ms"],
            "evaluated_at_unix_ms": evaluated_at,
            "input_revision_id": current["input"],
            "method_revision_id": current["method"],
            "native_analytical_discharge_receipt_id": rcpt,
            "obligation_id": OBL_ID,
            "obligation_revision_id": current["obligation"],
            "requirement_revision_id": current["requirement"],
            "schema": "symthaea.etk-current-native-analytical-discharge-fact.v2",
            "subject_revision_id": current["subject"],
            "twin_revision_id": current["twin"],
            "validity_domain_revision_id": current["validity"],
        },
    )


def expect_denied(fn, reason):
    try:
        fn()
    except Denied as error:
        assert reason in str(error), (reason, str(error))
        return
    raise AssertionError(reason)


def self_test():
    assert f(-0.0) == "f64:0000000000000000"
    assert f(0.1) == "f64:3fb999999999999a"

    baseline_currentness = currentness("4")
    assert baseline_currentness["id"] == EXPECTED["currentness_assertion_id"]

    x = inputs()
    p = plan(x, baseline_currentness)
    candidate = result(x, fos=10.416666666666666)
    candidate.update(
        max_bending_stress_pa=24e6,
        max_deflection_m=0.0032,
        max_moment_nm=2000.0,
    )
    admitted = admit(p, x, candidate)
    rcpt = receipt(p, admitted)
    current = current_fact(p, p, rcpt)

    vectors = {
        "requirement_revision_id": p["requirement"],
        "obligation_revision_id": p["obligation"],
        "method_revision_id": p["method"],
        "input_revision_id": p["input"],
        "policy_revision_id": p["policy"],
        "currentness_assertion_id": p["currentness"],
        "analytical_plan_id": p["plan"],
        "admitted_evidence_id": admitted,
        "historical_receipt_id": rcpt,
        "current_discharge_fact_id": current,
    }
    assert vectors == EXPECTED, (EXPECTED, vectors)

    reordered = copy.deepcopy(x)
    reordered["beam"] = {
        "section": reordered["beam"]["section"],
        "material": reordered["beam"]["material"],
        "length_m": reordered["beam"]["length_m"],
    }
    assert input_id(reordered) == p["input"]

    changed = inputs(1000.0000000000001)
    expect_denied(lambda: admit(p, changed, result(changed)), "plan_binding_mismatch")
    expect_denied(lambda: plan(x, baseline_currentness, policy(0.5, 0.05)), "policy_requirement_mismatch")

    strict = plan(x, baseline_currentness, policy(10.3, 0.05))
    expect_denied(lambda: admit(strict, x, result(x, err=0.02)), "acceptance_predicate")
    expect_denied(lambda: admit(p, x, result(x, err=0.06)), "model_error_budget")
    expect_denied(lambda: admit(p, x, result(x, fos=10.0)), "inconsistent_factor_of_safety")
    expect_denied(lambda: admit(p, x, result(x, deflection=0.004)), "equation_mismatch:max_deflection")
    expect_denied(lambda: admit(p, x, result(x, moment=1999.0)), "equation_mismatch:max_moment")
    expect_denied(
        lambda: admit(p, x, result(x, artifact="sha256:not-a-digest")),
        "invalid_digest",
    )
    wrong_binding = result(x)
    wrong_binding["input_revision_id"] = d("0")
    expect_denied(lambda: admit(p, x, wrong_binding), "plan_binding_mismatch")

    expect_denied(
        lambda: currentness("4", OBSERVED_AT_UNIX_MS, OBSERVED_AT_UNIX_MS),
        "invalid_currentness_window",
    )
    expect_denied(
        lambda: current_fact(p, p, rcpt, OBSERVED_AT_UNIX_MS - 1),
        "freshness_not_yet_valid",
    )
    expect_denied(
        lambda: current_fact(p, p, rcpt, VALID_UNTIL_UNIX_MS + 1),
        "freshness_expired",
    )
    # Both exact boundaries remain valid.
    current_fact(p, p, rcpt, OBSERVED_AT_UNIX_MS)
    current_fact(p, p, rcpt, VALID_UNTIL_UNIX_MS)

    refreshed = plan(x, currentness("5"))
    assert refreshed["plan"] != p["plan"]
    expect_denied(lambda: current_fact(refreshed, p, rcpt), "historical_plan")
    return vectors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--vectors", action="store_true")
    args = parser.parse_args()
    try:
        vectors = self_test()
        if args.vectors:
            print(cj(vectors))
        elif args.self_test:
            for key in sorted(vectors):
                print(f"ok {key}={vectors[key]}")
        else:
            print(cj({"decision": "SelfTest", "vectors": vectors}))
        return 0
    except (Denied, AssertionError, KeyError, TypeError, ValueError) as error:
        print(cj({"decision": "Deny", "reason": str(error)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
