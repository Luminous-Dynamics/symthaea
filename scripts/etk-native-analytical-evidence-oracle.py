#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK-3C native analytical-evidence reference theorem.

Authority ladder:
  native calculation != admitted analytical evidence
  != historical discharge receipt != current analytical discharge fact
  != requirement satisfaction / qualification / actuation authority

This is a stdlib-only reference implementation. It freezes authority semantics,
not Euler-Bernoulli physical applicability or model qualification.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, struct

REQ_D=b"symthaea.etk-accepted-requirement.v1\0"
OBL_D=b"symthaea.etk-proof-obligation-snapshot.v1\0"
CURRENTNESS_D=b"symthaea.etk-currentness-assertion.v1\0"
METHOD_D=b"symthaea.etk-native-analytical-method.v1\0"
INPUT_D=b"symthaea.etk-native-analytical-input.v1\0"
POLICY_D=b"symthaea.etk-native-analytical-policy.v1\0"
PLAN_D=b"symthaea.etk-native-analytical-plan.v1\0"
ADMIT_D=b"symthaea.etk-native-analytical-admitted-evidence.v1\0"
RECEIPT_D=b"symthaea.etk-native-analytical-discharge-receipt.v1\0"
FACT_D=b"symthaea.etk-current-native-analytical-discharge-fact.v1\0"

SUBJECT="sha256:38a9505d423fa3464020107b4e8abc8acc6ac6af33bcc98a2418480e1d33390e"
TWIN="sha256:19d558d6e7579f0f44c71398c7ddac659677aca0fa0e65ed46227670e4876cd9"
VALIDITY="sha256:90bff4adf8917f2ae071f405d5c46afb310b12a98dd2e4eec845de1261a24890"
EXPECTED_CURRENT="sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9"
OBL_ID="00000000-0000-4000-8000-000000000042"
CLAIM="stress remains below allowable under service load"
REQUIREMENT_MAX_STRESS_PA=250e6
REL_TOL=1e-12

EXPECTED={
 "requirement_revision_id":"sha256:10891514f6551c85b10674a3df04ab2c6d19f74f014adeacc671fa31deecd419",
 "obligation_revision_id":"sha256:ea6e2f0f5e3ec37524325eda24b95ead75fa502f5d2a0b9410a407aed3a12a8c",
 "method_revision_id":"sha256:73bb060e12b166cecfea9a73c1f274f4443f1a540a777e7ac8260c65808bedc3",
 "input_revision_id":"sha256:3b9a62ea2ce031b73c188e8c0138fb1ebcda20e77d7f2401502ec493323d3c86",
 "policy_revision_id":"sha256:7edc8fcbc03184413cc9c275537bdefb2d1c18f281406dc1adacd9a53a0a3bb5",
 "analytical_plan_id":"sha256:b86b76c504bd7b7601981d40988ceafd6c0a6580c941e4c91a65054c57749432",
 "admitted_evidence_id":"sha256:7ea504bb399216df14d5792b44f574e18c62163049c4406e23e4dc3783e86bde",
 "historical_receipt_id":"sha256:5d740e2cfe67fa0bc0145cc7d79a9410e9c36b276ac78f12243ad5e013f7a1ec",
 "current_discharge_fact_id":"sha256:47e1d641da233d0d70e6084a97bbe55fbf9d0487a3b6f7978ca03790fbeb494c",
}

class Denied(ValueError): pass

def cj(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False)
def dh(domain,v): return "sha256:"+hashlib.sha256(domain+cj(v).encode()).hexdigest()
def d(ch): return "sha256:"+ch*64

def f(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)):
        raise Denied("non_finite")
    value=0.0 if float(v)==0.0 else float(v)
    return "f64:"+struct.pack(">d",value).hex()

def dg(v,field):
    if not isinstance(v,str) or len(v)!=71 or not v.startswith("sha256:"):
        raise Denied("invalid_digest:"+field)
    h=v[7:]
    if h.lower()!=h or any(c not in "0123456789abcdef" for c in h):
        raise Denied("invalid_digest:"+field)
    return v

def close(actual,expected):
    return abs(float(actual)-float(expected))/max(abs(float(expected)),1.0)<=REL_TOL

def currentness(attestation="4"):
    return dh(CURRENTNESS_D,{"attestation_digest":d(attestation),"observed_at_unix_ms":1789123456000,
       "schema":"symthaea.etk-currentness-assertion.v1","twin_revision_id":TWIN,
       "validity_domain_revision_id":VALIDITY})

def context(current=None):
    if current is None: current=currentness("4")
    req=dh(REQ_D,{"acceptance_record_digest":d("a"),"criticality":"Blocking","domain":"Civil",
       "expected_evidence_kind":"Analysis","logical_requirement_id":"REQ-STRESS",
       "schema":"symthaea.etk-accepted-requirement.v1","statement":"stress remains below allowable",
       "structural_invariants":["stress <= 250 MPa"]})
    obl=dh(OBL_D,{"claim":CLAIM,"expected_evidence_kind":"Analysis","obligation_id":OBL_ID,
                  "schema":"symthaea.etk-proof-obligation-snapshot.v1"})
    return {"requirement":req,"requirement_max_stress_pa":REQUIREMENT_MAX_STRESS_PA,
            "obligation":obl,"subject":SUBJECT,"twin":TWIN,"validity":VALIDITY,"currentness":current}

def method():
    return dh(METHOD_D,{"algorithm_revision_digest":d("b"),
      "assumptions":["euler_bernoulli_kinematics","linear_elastic_material","prismatic_beam",
                     "single_span","small_deflection","statically_determinate"],
      "implementation_artifact_digest":d("a"),"method_key":"symthaea-structural/euler-bernoulli-beam",
      "outputs":[{"name":"factor_of_safety","unit":"1"},{"name":"max_bending_stress","unit":"Pa"},
                 {"name":"max_deflection","unit":"m"},{"name":"max_moment","unit":"N*m"}],
      "schema":"symthaea.etk-native-analytical-method.v1",
      "supported_load_cases":["cantilever_end_point","cantilever_udl",
                              "simply_supported_center_point","simply_supported_udl"],"unit_system":"SI"})

def inputs(load=1000.0):
    return {"beam":{"length_m":2.0,"material":{"youngs_modulus_pa":200e9,"yield_strength_pa":250e6},
                    "section":{"height_m":0.1,"kind":"rectangular","width_m":0.05}},
            "load":{"kind":"cantilever_end_point","unit":"N","value":load}}

def input_id(x):
    b=x["beam"]; sec=b["section"]; mat=b["material"]; load=x["load"]
    vals=[b["length_m"],sec["width_m"],sec["height_m"],mat["youngs_modulus_pa"],mat["yield_strength_pa"],load["value"]]
    if sec["kind"]!="rectangular" or load["kind"]!="cantilever_end_point" or load["unit"]!="N" or any(float(v)<=0 for v in vals):
        raise Denied("unsupported_or_invalid_input")
    return dh(INPUT_D,{"beam":{"length_m":f(b["length_m"]),
       "material":{"youngs_modulus_pa":f(mat["youngs_modulus_pa"]),"yield_strength_pa":f(mat["yield_strength_pa"])},
       "section":{"height_m":f(sec["height_m"]),"kind":"rectangular","width_m":f(sec["width_m"])}},
       "load":{"kind":load["kind"],"unit":"N","value":f(load["value"])},
       "method_revision_id":method(),"schema":"symthaea.etk-native-analytical-input.v1"})

def policy(threshold=2.0,max_error=.05):
    if not math.isfinite(float(threshold)) or float(threshold)<=0: raise Denied("invalid_policy_threshold")
    if not math.isfinite(float(max_error)) or not 0<=float(max_error)<=1: raise Denied("invalid_policy_error")
    return {"id":dh(POLICY_D,{"max_model_relative_error_bound":f(max_error),"metric":"factor_of_safety",
       "model_qualification_record_digest":d("9"),"operator":">=",
       "schema":"symthaea.etk-native-analytical-policy.v1","threshold":f(threshold)}),
       "threshold":float(threshold),"max_error":float(max_error)}

def plan(x,current=None,pol=None):
    c=context(current); pol=policy() if pol is None else pol; mat=x["beam"]["material"]
    worst=float(mat["yield_strength_pa"])/pol["threshold"]*(1.0+pol["max_error"])
    if worst>c["requirement_max_stress_pa"]: raise Denied("policy_requirement_mismatch")
    mid=method(); iid=input_id(x)
    pre={"acceptance_policy_revision_id":pol["id"],"currentness_assertion_id":c["currentness"],
         "input_revision_id":iid,"method_revision_id":mid,"obligation_id":OBL_ID,
         "obligation_revision_id":c["obligation"],"requirement_revision_id":c["requirement"],
         "schema":"symthaea.etk-native-analytical-plan.v1","subject_revision_id":c["subject"],
         "twin_revision_id":c["twin"],"validity_domain_revision_id":c["validity"]}
    return {**c,"method":mid,"input":iid,"policy":pol["id"],"policy_spec":pol,"plan":dh(PLAN_D,pre)}

def expected_outputs(x):
    b=x["beam"]; sec=b["section"]; mat=b["material"]; load=x["load"]
    moment=float(load["value"])*float(b["length_m"])
    section_modulus=float(sec["width_m"])*float(sec["height_m"])**2/6.0
    stress=moment/section_modulus
    inertia=float(sec["width_m"])*float(sec["height_m"])**3/12.0
    deflection=float(load["value"])*float(b["length_m"])**3/(3.0*float(mat["youngs_modulus_pa"])*inertia)
    return moment,stress,deflection,float(mat["yield_strength_pa"])/stress

def result(x=None,fos=None,err=.02,artifact=None,deflection=None,moment=None):
    x=inputs() if x is None else x
    expected_moment,stress,expected_deflection,expected_fos=expected_outputs(x)
    return {"method_revision_id":method(),"input_revision_id":input_id(x),
            "execution_artifact_digest":d("8") if artifact is None else artifact,
            "factor_of_safety":expected_fos if fos is None else fos,"max_bending_stress_pa":stress,
            "max_deflection_m":expected_deflection if deflection is None else deflection,
            "max_moment_nm":expected_moment if moment is None else moment,"model_relative_error_bound":err}

def admit(p,x,r):
    pol=p["policy_spec"]
    if method()!=p["method"] or input_id(x)!=p["input"] or r.get("method_revision_id")!=p["method"] or r.get("input_revision_id")!=p["input"]:
        raise Denied("plan_binding_mismatch")
    dg(r["execution_artifact_digest"],"execution_artifact_digest")
    keys=["factor_of_safety","max_bending_stress_pa","max_deflection_m","max_moment_nm","model_relative_error_bound"]
    if not all(math.isfinite(float(r[k])) for k in keys): raise Denied("invalid_result")
    expected_moment,expected_stress,expected_deflection,expected_fos=expected_outputs(x)
    if not close(r["max_moment_nm"],expected_moment): raise Denied("equation_mismatch:max_moment")
    if not close(r["max_bending_stress_pa"],expected_stress): raise Denied("equation_mismatch:max_stress")
    if not close(r["max_deflection_m"],expected_deflection): raise Denied("equation_mismatch:max_deflection")
    if not close(r["factor_of_safety"],expected_fos): raise Denied("inconsistent_factor_of_safety")
    err=float(r["model_relative_error_bound"])
    if not 0<=err<=pol["max_error"]: raise Denied("model_error_budget")
    if float(r["max_bending_stress_pa"])*(1.0+err)>p["requirement_max_stress_pa"]: raise Denied("requirement_stress_predicate")
    conservative=float(r["factor_of_safety"])/(1.0+err)
    if conservative<pol["threshold"]: raise Denied("acceptance_predicate")
    norm={"execution_artifact_digest":r["execution_artifact_digest"],"factor_of_safety":f(r["factor_of_safety"]),
          "max_bending_stress_pa":f(r["max_bending_stress_pa"]),"max_deflection_m":f(r["max_deflection_m"]),
          "max_moment_nm":f(r["max_moment_nm"]),"model_relative_error_bound":f(err)}
    return dh(ADMIT_D,{"analytical_plan_id":p["plan"],"conservative_factor_of_safety":f(conservative),
                       "normalized_result":norm,"schema":"symthaea.etk-native-analytical-admitted-evidence.v1"})

def receipt(p,a):
    return dh(RECEIPT_D,{"admitted_analytical_evidence_id":a,"analytical_plan_id":p["plan"],
       "obligation_id":OBL_ID,"obligation_revision_id":p["obligation"],
       "schema":"symthaea.etk-native-analytical-discharge-receipt.v1"})

def current_fact(current,historical,rcpt):
    if current["plan"]!=historical["plan"]: raise Denied("historical_plan")
    return dh(FACT_D,{"acceptance_policy_revision_id":current["policy"],"analytical_plan_id":current["plan"],
       "currentness_assertion_id":current["currentness"],"input_revision_id":current["input"],
       "method_revision_id":current["method"],"native_analytical_discharge_receipt_id":rcpt,
       "obligation_id":OBL_ID,"obligation_revision_id":current["obligation"],
       "requirement_revision_id":current["requirement"],
       "schema":"symthaea.etk-current-native-analytical-discharge-fact.v1",
       "subject_revision_id":current["subject"],"twin_revision_id":current["twin"],
       "validity_domain_revision_id":current["validity"]})

def expect_denied(fn,reason):
    try: fn()
    except Denied as e:
        assert reason in str(e),(reason,str(e)); return
    raise AssertionError(reason)

def self_test():
    assert f(-0.0)=="f64:0000000000000000" and f(.1)=="f64:3fb999999999999a"
    assert currentness("4")==EXPECTED_CURRENT
    x=inputs(); p=plan(x); candidate=result(x,fos=10.416666666666666); candidate.update(max_bending_stress_pa=24e6,max_deflection_m=.0032,max_moment_nm=2000.0); a=admit(p,x,candidate); r=receipt(p,a); cf=current_fact(p,p,r)
    v={"requirement_revision_id":p["requirement"],"obligation_revision_id":p["obligation"],
       "method_revision_id":p["method"],"input_revision_id":p["input"],"policy_revision_id":p["policy"],
       "analytical_plan_id":p["plan"],"admitted_evidence_id":a,"historical_receipt_id":r,
       "current_discharge_fact_id":cf}
    assert v==EXPECTED,(EXPECTED,v)
    y=copy.deepcopy(x); y["beam"]={"section":y["beam"]["section"],"material":y["beam"]["material"],"length_m":y["beam"]["length_m"]}
    assert input_id(y)==p["input"]

    changed=inputs(1000.0000000000001)
    expect_denied(lambda: admit(p,changed,result(changed)),"plan_binding_mismatch")
    expect_denied(lambda: plan(x,pol=policy(.5,.05)),"policy_requirement_mismatch")

    strict=plan(x,pol=policy(10.3,.05))
    expect_denied(lambda: admit(strict,x,result(x,err=.02)),"acceptance_predicate")
    expect_denied(lambda: admit(p,x,result(x,err=.06)),"model_error_budget")
    expect_denied(lambda: admit(p,x,result(x,fos=10.0)),"inconsistent_factor_of_safety")
    expect_denied(lambda: admit(p,x,result(x,deflection=.004)),"equation_mismatch:max_deflection")
    expect_denied(lambda: admit(p,x,result(x,moment=1999.0)),"equation_mismatch:max_moment")
    expect_denied(lambda: admit(p,x,result(x,artifact="sha256:not-a-digest")),"invalid_digest")
    wrong_binding=result(x); wrong_binding["input_revision_id"]=d("0")
    expect_denied(lambda: admit(p,x,wrong_binding),"plan_binding_mismatch")

    refreshed=plan(x,currentness("5"))
    assert refreshed["plan"]!=p["plan"]
    expect_denied(lambda: current_fact(refreshed,p,r),"historical_plan")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true"); args=ap.parse_args()
    try:
        v=self_test()
        if args.vectors: print(cj(v))
        elif args.self_test:
            for k in sorted(v): print(f"ok {k}={v[k]}")
        else: print(cj({"decision":"SelfTest","vectors":v}))
        return 0
    except (Denied,AssertionError,KeyError,TypeError,ValueError) as e:
        print(cj({"decision":"Deny","reason":str(e)})); return 2

if __name__=="__main__": raise SystemExit(main())
