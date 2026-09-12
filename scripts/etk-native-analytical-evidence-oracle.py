#!/usr/bin/env python3
# Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Independent ETK-3C native analytical-evidence reference theorem.

Authority ladder:
  native calculation != admitted analytical evidence
  != historical discharge receipt != current analytical discharge fact

The canary uses Euler-Bernoulli beam outputs only to freeze admission semantics.
It does not establish physical correctness, calibration, or model qualification.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, struct
from typing import Any

REQ_D=b"symthaea.etk-accepted-requirement.v1\0"
OBL_D=b"symthaea.etk-proof-obligation-snapshot.v1\0"
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
CURRENT="sha256:f85bd7d5129b08a3256cfdd5b3506f3cae1dbc625b097a8bf95bf1f6fe709dd9"
REFRESHED="sha256:8afc4da94154375097adee0603968f356f3807e97be96864c5f79acf90e081d6"
OBL_ID="00000000-0000-4000-8000-000000000042"
CLAIM="stress remains below allowable under service load"

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
def dh(d,v): return "sha256:"+hashlib.sha256(d+cj(v).encode()).hexdigest()
def d(c): return "sha256:"+c*64
def f(v):
    if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(float(v)): raise Denied("non_finite")
    v=0.0 if float(v)==0.0 else float(v)
    return "f64:"+struct.pack(">d",v).hex()
def dg(v,field):
    if not isinstance(v,str) or len(v)!=71 or not v.startswith("sha256:"): raise Denied("invalid_digest:"+field)
    h=v[7:]
    if h.lower()!=h or any(c not in "0123456789abcdef" for c in h): raise Denied("invalid_digest:"+field)
    return v

def context(current=CURRENT):
    req=dh(REQ_D,{
      "acceptance_record_digest":d("a"),"criticality":"Blocking","domain":"Civil",
      "expected_evidence_kind":"Analysis","logical_requirement_id":"REQ-STRESS",
      "schema":"symthaea.etk-accepted-requirement.v1","statement":"stress remains below allowable",
      "structural_invariants":["stress <= 250 MPa"]})
    obl=dh(OBL_D,{"claim":CLAIM,"expected_evidence_kind":"Analysis",
                  "obligation_id":OBL_ID,"schema":"symthaea.etk-proof-obligation-snapshot.v1"})
    return {"requirement":req,"obligation":obl,"subject":SUBJECT,"twin":TWIN,
            "validity":VALIDITY,"currentness":current}

def method():
    return dh(METHOD_D,{
      "algorithm_revision_digest":d("b"),
      "assumptions":sorted(["euler_bernoulli_kinematics","linear_elastic_material","prismatic_beam",
                            "single_span","small_deflection","statically_determinate"]),
      "implementation_artifact_digest":d("a"),
      "method_key":"symthaea-structural/euler-bernoulli-beam",
      "outputs":[{"name":"factor_of_safety","unit":"1"},{"name":"max_bending_stress","unit":"Pa"},
                 {"name":"max_deflection","unit":"m"},{"name":"max_moment","unit":"N*m"}],
      "schema":"symthaea.etk-native-analytical-method.v1",
      "supported_load_cases":sorted(["cantilever_end_point","cantilever_udl",
                                     "simply_supported_center_point","simply_supported_udl"]),
      "unit_system":"SI"})

def inputs(load=1000.0):
    return {"beam":{"length_m":2.0,"material":{"youngs_modulus_pa":200e9,"yield_strength_pa":250e6},
                    "section":{"height_m":0.1,"kind":"rectangular","width_m":0.05}},
            "load":{"kind":"cantilever_end_point","unit":"N","value":load}}

def input_id(x):
    b=x["beam"]; s=b["section"]; m=b["material"]; l=x["load"]
    vals=[b["length_m"],s["width_m"],s["height_m"],m["youngs_modulus_pa"],m["yield_strength_pa"],l["value"]]
    if s["kind"]!="rectangular" or l["kind"]!="cantilever_end_point" or l["unit"]!="N" or any(float(v)<=0 for v in vals):
        raise Denied("unsupported_or_invalid_input")
    return dh(INPUT_D,{"beam":{"length_m":f(b["length_m"]),
       "material":{"youngs_modulus_pa":f(m["youngs_modulus_pa"]),"yield_strength_pa":f(m["yield_strength_pa"])},
       "section":{"height_m":f(s["height_m"]),"kind":"rectangular","width_m":f(s["width_m"])}},
       "load":{"kind":l["kind"],"unit":"N","value":f(l["value"])},
       "method_revision_id":method(),"schema":"symthaea.etk-native-analytical-input.v1"})

def policy():
    return dh(POLICY_D,{"max_model_relative_error_bound":f(.05),"metric":"factor_of_safety",
       "model_qualification_record_digest":d("9"),"operator":">=",
       "schema":"symthaea.etk-native-analytical-policy.v1","threshold":f(2.0)})

def plan(x,current=CURRENT):
    c=context(current); mid=method(); iid=input_id(x); pid=policy()
    pre={"acceptance_policy_revision_id":pid,"currentness_assertion_id":c["currentness"],
         "input_revision_id":iid,"method_revision_id":mid,"obligation_id":OBL_ID,
         "obligation_revision_id":c["obligation"],"requirement_revision_id":c["requirement"],
         "schema":"symthaea.etk-native-analytical-plan.v1","subject_revision_id":c["subject"],
         "twin_revision_id":c["twin"],"validity_domain_revision_id":c["validity"]}
    return {**c,"method":mid,"input":iid,"policy":pid,"plan":dh(PLAN_D,pre)}

def result(fos=10.416666666666666,err=.02,artifact=None):
    return {"execution_artifact_digest":d("8") if artifact is None else artifact,
            "factor_of_safety":fos,"max_bending_stress_pa":24e6,
            "max_deflection_m":.0032,"max_moment_nm":2000.0,"model_relative_error_bound":err}

def admit(p,x,r):
    if method()!=p["method"] or input_id(x)!=p["input"] or policy()!=p["policy"]: raise Denied("plan_binding_mismatch")
    dg(r["execution_artifact_digest"],"execution_artifact_digest")
    fos=float(r["factor_of_safety"]); stress=float(r["max_bending_stress_pa"]); err=float(r["model_relative_error_bound"])
    if not all(math.isfinite(float(r[k])) for k in ["factor_of_safety","max_bending_stress_pa","max_deflection_m","max_moment_nm","model_relative_error_bound"]):
        raise Denied("invalid_result")
    derived=float(x["beam"]["material"]["yield_strength_pa"])/stress
    if abs(derived-fos)/max(abs(derived),1.0)>1e-12: raise Denied("inconsistent_factor_of_safety")
    if not 0<=err<=.05: raise Denied("model_error_budget")
    conservative=fos/(1.0+err)
    if conservative<2.0: raise Denied("acceptance_predicate")
    norm={"execution_artifact_digest":r["execution_artifact_digest"],"factor_of_safety":f(fos),
          "max_bending_stress_pa":f(stress),"max_deflection_m":f(r["max_deflection_m"]),
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

def self_test():
    assert f(-0.0)=="f64:0000000000000000" and f(.1)=="f64:3fb999999999999a"
    x=inputs(); p=plan(x); a=admit(p,x,result()); r=receipt(p,a); cf=current_fact(p,p,r)
    v={"requirement_revision_id":p["requirement"],"obligation_revision_id":p["obligation"],
       "method_revision_id":p["method"],"input_revision_id":p["input"],"policy_revision_id":p["policy"],
       "analytical_plan_id":p["plan"],"admitted_evidence_id":a,"historical_receipt_id":r,
       "current_discharge_fact_id":cf}
    assert v==EXPECTED,(EXPECTED,v)
    y=copy.deepcopy(x); y["beam"]={"section":y["beam"]["section"],"material":y["beam"]["material"],"length_m":y["beam"]["length_m"]}
    assert input_id(y)==p["input"]
    try: admit(p,inputs(1000.0000000000001),result())
    except Denied as e: assert "plan_binding_mismatch" in str(e)
    else: raise AssertionError("input drift")
    low=result(2.01,.02); low["max_bending_stress_pa"]=250e6/2.01
    try: admit(p,x,low)
    except Denied as e: assert "acceptance_predicate" in str(e)
    else: raise AssertionError("conservative margin")
    for bad,reason in [(result(err=.06),"model_error_budget"),
                       ({**result(),"max_bending_stress_pa":25e6},"inconsistent_factor_of_safety"),
                       (result(artifact="sha256:not-a-digest"),"invalid_digest")]:
        try: admit(p,x,bad)
        except Denied as e: assert reason in str(e)
        else: raise AssertionError(reason)
    refreshed=plan(x,REFRESHED)
    assert refreshed["plan"]!=p["plan"]
    try: current_fact(refreshed,p,r)
    except Denied as e: assert "historical_plan" in str(e)
    else: raise AssertionError("stale receipt")
    return v

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--vectors",action="store_true"); a=ap.parse_args()
    try:
        v=self_test()
        if a.vectors: print(cj(v))
        elif a.self_test:
            for k in sorted(v): print(f"ok {k}={v[k]}")
        else: print(cj({"decision":"SelfTest","vectors":v}))
        return 0
    except (Denied,AssertionError,KeyError,TypeError,ValueError) as e:
        print(cj({"decision":"Deny","reason":str(e)})); return 2
if __name__=="__main__": raise SystemExit(main())