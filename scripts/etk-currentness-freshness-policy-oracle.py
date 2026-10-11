#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK-3C freshness-policy binding reference theorem.

Core law:
  finite currentness window
  != policy-bounded currentness
  != authorized freshness policy
  != trusted evaluation clock
  != current analytical authority.

This stdlib-only oracle proves only deterministic binding semantics.
"""
from __future__ import annotations
import argparse, hashlib, json, math, struct

REQ_D=b"symthaea.etk-accepted-requirement.v1\0"
OBL_D=b"symthaea.etk-proof-obligation-snapshot.v1\0"
METHOD_D=b"symthaea.etk-native-analytical-method.v1\0"
INPUT_D=b"symthaea.etk-native-analytical-input.v1\0"
POLICY_D=b"symthaea.etk-native-analytical-policy.v1\0"
FRESH_POLICY_D=b"symthaea.etk-currentness-freshness-policy.v1\0"
CURRENTNESS_D=b"symthaea.etk-currentness-assertion.v3\0"
PLAN_D=b"symthaea.etk-native-analytical-plan.v1\0"
ADMIT_D=b"symthaea.etk-native-analytical-admitted-evidence.v1\0"
RECEIPT_D=b"symthaea.etk-native-analytical-discharge-receipt.v1\0"
FACT_D=b"symthaea.etk-current-native-analytical-discharge-fact.v3\0"

SUBJECT="sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e"
TWIN="sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9"
VALIDITY="sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890"
OBL_ID="00000000-0000-4000-8000-000000000042"
CLAIM="stress remains below allowable under service load"
OBSERVED=1789123456000
MAX_AGE_MS=86_400_000
VALID_UNTIL=OBSERVED+MAX_AGE_MS
EVALUATED=OBSERVED+1000
REL_TOL=1e-12

UPSTREAM={
 "requirement_revision_id":"sha256:10891514f6551c85b10674a3df04ab2c6d19f74f014adeacc671fa31deecd419",
 "obligation_revision_id":"sha256:ea6e2f0f5e3ec37524325eda24b95ead75fa502f5d2a0b9410a407aed3a12a8c",
 "method_revision_id":"sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3",
 "input_revision_id":"sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86",
 "policy_revision_id":"sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5",
}

class Denied(ValueError):
    pass

def cj(v):
    return json.dumps(v, sort_keys=True, separators=(",",":"), ensure_ascii=False, allow_nan=False)

def dh(domain,v):
    return "sha256:"+hashlib.sha256(domain+cj(v).encode()).hexdigest()

def d(ch):
    return "sha256:"+ch*64

def f(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):
        raise Denied("non_finite")
    value=0.0 if float(v)==0.0 else float(v)
    return "f64:"+struct.pack(">d",value).hex()

def requirement():
    return dh(REQ_D,{
      "acceptance_record_digest":d("a"),"criticality":"Blocking","domain":"Civil",
      "expected_evidence_kind":"Analysis","logical_requirement_id":"REQ-STRESS",
      "schema":"symthaea.etk-accepted-requirement.v1","statement":"stress remains below allowable",
      "structural_invariants":["stress <= 250 MPa"]})

def obligation():
    return dh(OBL_D,{"claim":CLAIM,"expected_evidence_kind":"Analysis","obligation_id":OBL_ID,
                     "schema":"symthaea.etk-proof-obligation-snapshot.v1"})

def method():
    return dh(METHOD_D,{
      "algorithm_revision_digest":d("b"),
      "assumptions":["euler_bernoulli_kinematics","linear_elastic_material","prismatic_beam",
                     "single_span","small_deflection","statically_determinate"],
      "implementation_artifact_digest":d("a"),
      "method_key":"symthaea-structural/euler-bernoulli-beam",
      "outputs":[{"name":"factor_of_safety","unit":"1"},{"name":"max_bending_stress","unit":"Pa"},
                 {"name":"max_deflection","unit":"m"},{"name":"max_moment","unit":"N*m"}],
      "schema":"symthaea.etk-native-analytical-method.v1",
      "supported_load_cases":["cantilever_end_point","cantilever_udl",
                              "simply_supported_center_point","simply_supported_udl"],
      "unit_system":"SI"})

def inputs():
    return {"beam":{"length_m":2.0,
                    "material":{"youngs_modulus_pa":200e9,"yield_strength_pa":250e6},
                    "section":{"height_m":0.1,"kind":"rectangular","width_m":0.05}},
            "load":{"kind":"cantilever_end_point","unit":"N","value":1000.0}}

def input_id(x):
    b=x["beam"]; sec=b["section"]; mat=b["material"]; load=x["load"]
    return dh(INPUT_D,{"beam":{"length_m":f(b["length_m"]),
      "material":{"youngs_modulus_pa":f(mat["youngs_modulus_pa"]),
                  "yield_strength_pa":f(mat["yield_strength_pa"])},
      "section":{"height_m":f(sec["height_m"]),"kind":"rectangular","width_m":f(sec["width_m"])}},
      "load":{"kind":"cantilever_end_point","unit":"N","value":f(load["value"])},
      "method_revision_id":method(),"schema":"symthaea.etk-native-analytical-input.v1"})

def analytical_policy():
    threshold=2.0; max_error=.05
    return dh(POLICY_D,{"max_model_relative_error_bound":f(max_error),
      "metric":"factor_of_safety","model_qualification_record_digest":d("9"),"operator":">=",
      "schema":"symthaea.etk-native-analytical-policy.v1","threshold":f(threshold)})

def freshness_policy(max_age_ms=MAX_AGE_MS, record="7"):
    if not isinstance(max_age_ms,int) or isinstance(max_age_ms,bool) or max_age_ms<=0:
        raise Denied("invalid_freshness_policy")
    return {"id":dh(FRESH_POLICY_D,{
        "clock_basis":"unix_epoch_ms",
        "max_validity_ms":max_age_ms,
        "policy_key":"native-analytical-currentness",
        "policy_record_digest":d(record),
        "schema":"symthaea.etk-currentness-freshness-policy.v1"}),
        "max_age_ms":max_age_ms}

def currentness(policy, attestation="4", observed=OBSERVED, valid_until=VALID_UNTIL):
    if not isinstance(observed,int) or not isinstance(valid_until,int):
        raise Denied("invalid_currentness_time")
    if valid_until<=observed:
        raise Denied("invalid_currentness_window")
    if valid_until-observed>policy["max_age_ms"]:
        raise Denied("freshness_policy_exceeded")
    return {"id":dh(CURRENTNESS_D,{
        "attestation_digest":d(attestation),
        "freshness_policy_revision_id":policy["id"],
        "observed_at_unix_ms":observed,
        "schema":"symthaea.etk-currentness-assertion.v3",
        "twin_revision_id":TWIN,
        "valid_until_unix_ms":valid_until,
        "validity_domain_revision_id":VALIDITY}),
        "observed":observed,"valid_until":valid_until,"freshness_policy":policy["id"]}

def plan(cur):
    pre={"acceptance_policy_revision_id":analytical_policy(),
         "currentness_assertion_id":cur["id"],
         "input_revision_id":input_id(inputs()),"method_revision_id":method(),
         "obligation_id":OBL_ID,"obligation_revision_id":obligation(),
         "requirement_revision_id":requirement(),
         "schema":"symthaea.etk-native-analytical-plan.v1",
         "subject_revision_id":SUBJECT,"twin_revision_id":TWIN,
         "validity_domain_revision_id":VALIDITY}
    return dh(PLAN_D,pre)

def expected_outputs(x):
    b=x["beam"]; sec=b["section"]; mat=b["material"]; load=x["load"]
    moment=load["value"]*b["length_m"]
    z=sec["width_m"]*sec["height_m"]**2/6.0
    stress=moment/z
    inertia=sec["width_m"]*sec["height_m"]**3/12.0
    deflection=load["value"]*b["length_m"]**3/(3.0*mat["youngs_modulus_pa"]*inertia)
    return moment,stress,deflection,mat["yield_strength_pa"]/stress

def admitted(plan_id):
    x=inputs()
    moment,stress,deflection,fos=expected_outputs(x)
    err=.02
    conservative=fos/(1.0+err)
    normalized={"execution_artifact_digest":d("8"),"factor_of_safety":f(fos),
                "max_bending_stress_pa":f(stress),"max_deflection_m":f(deflection),
                "max_moment_nm":f(moment),"model_relative_error_bound":f(err)}
    return dh(ADMIT_D,{"analytical_plan_id":plan_id,
      "conservative_factor_of_safety":f(conservative),"normalized_result":normalized,
      "schema":"symthaea.etk-native-analytical-admitted-evidence.v1"})

def receipt(plan_id, admitted_id):
    return dh(RECEIPT_D,{"admitted_analytical_evidence_id":admitted_id,
      "analytical_plan_id":plan_id,"obligation_id":OBL_ID,
      "obligation_revision_id":obligation(),
      "schema":"symthaea.etk-native-analytical-discharge-receipt.v1"})

def current_fact(cur, plan_id, receipt_id, evaluated=EVALUATED):
    if evaluated<cur["observed"]:
        raise Denied("freshness_not_yet_valid")
    if evaluated>cur["valid_until"]:
        raise Denied("freshness_expired")
    return dh(FACT_D,{
      "acceptance_policy_revision_id":analytical_policy(),
      "analytical_plan_id":plan_id,
      "currentness_assertion_id":cur["id"],
      "evaluated_at_unix_ms":evaluated,
      "freshness_policy_revision_id":cur["freshness_policy"],
      "input_revision_id":input_id(inputs()),
      "method_revision_id":method(),
      "native_analytical_discharge_receipt_id":receipt_id,
      "obligation_id":OBL_ID,"obligation_revision_id":obligation(),
      "observed_at_unix_ms":cur["observed"],
      "requirement_revision_id":requirement(),
      "schema":"symthaea.etk-current-native-analytical-discharge-fact.v3",
      "subject_revision_id":SUBJECT,"twin_revision_id":TWIN,
      "valid_until_unix_ms":cur["valid_until"],
      "validity_domain_revision_id":VALIDITY})

def expect_denied(fn, reason):
    try:
        fn()
    except Denied as e:
        assert reason in str(e),(reason,str(e))
        return
    raise AssertionError(reason)

def self_test():
    assert requirement()==UPSTREAM["requirement_revision_id"]
    assert obligation()==UPSTREAM["obligation_revision_id"]
    assert method()==UPSTREAM["method_revision_id"]
    assert input_id(inputs())==UPSTREAM["input_revision_id"]
    assert analytical_policy()==UPSTREAM["policy_revision_id"]

    fp=freshness_policy()
    cur=currentness(fp)
    p=plan(cur); a=admitted(p); r=receipt(p,a); fact=current_fact(cur,p,r)
    vectors={**UPSTREAM,
      "freshness_policy_revision_id":fp["id"],
      "currentness_revision_id":cur["id"],
      "analytical_plan_id":p,
      "admitted_evidence_id":a,
      "historical_receipt_id":r,
      "current_fact_id":fact}
    assert f(-0.0)=="f64:0000000000000000"
    expect_denied(lambda: freshness_policy(0),"invalid_freshness_policy")
    expect_denied(lambda: currentness(fp,valid_until=OBSERVED),"invalid_currentness_window")
    expect_denied(lambda: currentness(fp,valid_until=VALID_UNTIL+1),"freshness_policy_exceeded")
    assert currentness(fp,valid_until=VALID_UNTIL)["id"]==cur["id"]
    assert currentness(fp,valid_until=OBSERVED+1)["id"]!=cur["id"]
    expect_denied(lambda: current_fact(cur,p,r,OBSERVED-1),"freshness_not_yet_valid")
    expect_denied(lambda: current_fact(cur,p,r,VALID_UNTIL+1),"freshness_expired")
    assert current_fact(cur,p,r,OBSERVED) != current_fact(cur,p,r,OBSERVED+1)
    fp2=freshness_policy(record="8")
    cur2=currentness(fp2)
    assert fp2["id"]!=fp["id"] and cur2["id"]!=cur["id"]
    refreshed=currentness(fp,attestation="5")
    assert refreshed["id"]!=cur["id"] and plan(refreshed)!=p
    return vectors

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--self-test",action="store_true")
    ap.add_argument("--vectors",action="store_true")
    args=ap.parse_args()
    try:
        v=self_test()
        if args.vectors:
            print(cj(v))
        elif args.self_test:
            for k in sorted(v):
                print(f"ok {k}={v[k]}")
        else:
            print(cj({"decision":"SelfTest","vectors":v}))
        return 0
    except (Denied,AssertionError,KeyError,TypeError,ValueError) as e:
        print(cj({"decision":"Deny","reason":str(e)}))
        return 2

if __name__=="__main__":
    raise SystemExit(main())
