#!/usr/bin/env python3
"""Compose one validated V20 human-reference bundle into one V19 case candidate."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import scip_text_claim_annotation_bundle as v20

V19_POLICY_SHA="9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0"
V20_POLICY_SHA="12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9"
ADMISSION_POLICY_SHA="452f4201c230ca3278d1ddc744fb70bca5944fa0d77b89e494096cfbf9c3dfb9"
AUTHORITY="case-admission-only"
CASE_FIELDS={"assignment_key_sha256","case_id","kind","dimension","polarity","discourse_family","surface_sha256","source_inventory_sha256","expected_inventory_sha256","annotation_receipt_sha256","template_sha256","named_entity_tuple_sha256","numeric_tuple_sha256","exact_sentence_sha256","split"}
CASE_ID_FIELDS=("kind","dimension","polarity","discourse_family","surface_sha256","source_inventory_sha256","expected_inventory_sha256","annotation_receipt_sha256","template_sha256","named_entity_tuple_sha256","numeric_tuple_sha256","exact_sentence_sha256")
class E(ValueError): pass

def pairs(xs):
    d={}
    for k,v in xs:
        if k in d: raise E(f"duplicate JSON key: {k}")
        d[k]=v
    return d

def load(path):
    try: v=json.loads(Path(path).read_text("utf-8"),object_pairs_hook=pairs)
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as e: raise E(str(e)) from e
    if not isinstance(v,dict): raise E("top level must be object")
    return v

def canon(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def semantic(v): return hashlib.sha256(canon(v)).hexdigest()
def exact(v,ks,w):
    if not isinstance(v,dict) or set(v)!=set(ks): raise E(f"{w} schema mismatch")
    return v

def yes(v,w):
    if v is not True: raise E(f"{w} must be true")
def hx(v,w,none=False):
    if none and v is None: return None
    if not isinstance(v,str) or len(v)!=64 or v=="0"*64 or any(c not in "0123456789abcdef" for c in v): raise E(f"{w} must be non-zero lowercase sha256")
    return v

def admission_policy_ok(p):
    exact(p,{"schema","authority","v19_manifest_policy_sha256","v20_annotation_policy_sha256","binding","receipt","claim_boundary"},"admission policy")
    if p["schema"]!="symthaea.scip-text-claim-case-admission-policy/v1" or p["authority"]!=AUTHORITY: raise E("admission policy identity drift")
    if p["v19_manifest_policy_sha256"]!=V19_POLICY_SHA or p["v20_annotation_policy_sha256"]!=V20_POLICY_SHA: raise E("admission upstream binding drift")
    b=exact(p["binding"],{"surface_sha256_matches_v20","source_inventory_sha256_matches_v20","expected_inventory_sha256_is_v20_frozen_surface_inventory","annotation_receipt_sha256_matches_v20","v19_case_id_recomputed","assignment_key_bound_in_admission_receipt","split_bound_in_admission_receipt","hidden_alignment_inventory_not_substituted_for_expected_inventory"},"admission binding")
    for k,v in b.items(): yes(v,"admission binding."+k)
    r=exact(p["receipt"],{"schema","algorithm","domain","self_hash_field_absent"},"admission receipt")
    if r["schema"]!="symthaea.scip-text-claim-case-admission/v1" or r["algorithm"]!="sha256-domain-separated-canonical-json/v1" or r["domain"]!="symthaea-scip-text-claim-case-admission-v1\0": raise E("admission receipt policy drift")
    yes(r["self_hash_field_absent"],"admission receipt self hash")
    c=exact(p["claim_boundary"],{"single_case_admission_does_not_verify_split_assignment","single_case_admission_does_not_verify_cross_split_leakage","single_case_admission_does_not_qualify_v19_manifest","single_case_admission_does_not_prove_human_correctness","single_case_admission_does_not_establish_surface_fidelity","single_case_admission_does_not_authorize_confirmatory_execution"},"admission boundary")
    for k,v in c.items(): yes(v,"admission boundary."+k)
    if semantic(p)!=ADMISSION_POLICY_SHA: raise E("admission policy semantic sha drift")

def v19_policy_ok(p):
    if semantic(p)!=V19_POLICY_SHA: raise E("V19 policy semantic sha drift")
    ci=p.get("case_identity")
    if not isinstance(ci,dict) or ci.get("algorithm")!="sha256-domain-separated-canonical-json/v1" or ci.get("domain")!="symthaea-scip-text-claim-case-v1\0" or ci.get("split_excluded") is not True or ci.get("assignment_key_excluded") is not True: raise E("V19 case identity policy drift")
    if p.get("dimensions") is None or p.get("discourse_families") is None or p.get("dimension_stratum",{}).get("polarities")!=["positive","negative"]: raise E("V19 case taxonomy drift")

def case_ok(c,p):
    exact(c,CASE_FIELDS,"case")
    for k in ("assignment_key_sha256","case_id","surface_sha256","source_inventory_sha256","expected_inventory_sha256","annotation_receipt_sha256","template_sha256","exact_sentence_sha256"): hx(c[k],"case."+k)
    hx(c["named_entity_tuple_sha256"],"case.named_entity_tuple_sha256",True); hx(c["numeric_tuple_sha256"],"case.numeric_tuple_sha256",True)
    if c["split"] not in ("calibration","confirmatory"): raise E("case split invalid")
    if c["kind"]=="dimension":
        if c["dimension"] not in p["dimensions"] or c["polarity"] not in p["dimension_stratum"]["polarities"] or c["discourse_family"] is not None: raise E("case dimension stratum invalid")
    elif c["kind"]=="discourse":
        if c["dimension"] is not None or c["polarity"] is not None or c["discourse_family"] not in p["discourse_families"]: raise E("case discourse stratum invalid")
    else: raise E("case kind invalid")
    payload={k:c[k] for k in CASE_ID_FIELDS}; expected=hashlib.sha256(p["case_identity"]["domain"].encode()+canon(payload)).hexdigest()
    if c["case_id"]!=expected: raise E("case_id does not match V19 canonical content")

def validate(case,bundle,v19p,v20p,ap):
    admission_policy_ok(ap); v19_policy_ok(v19p)
    try: vr=v20.validate(bundle,v20p)
    except Exception as e: raise E(f"V20 bundle invalid: {e}") from e
    case_ok(case,v19p)
    checks=(("surface_sha256","surface_sha256"),("source_inventory_sha256","source_inventory_sha256"),("expected_inventory_sha256","frozen_surface_inventory_sha256"),("annotation_receipt_sha256","annotation_receipt_sha256"))
    for ck,vk in checks:
        if case[ck]!=vr[vk]: raise E(f"case {ck} does not bind V20 {vk}")
    if case["expected_inventory_sha256"]==vr["final_expected_inventory_sha256"] and vr["final_expected_inventory_sha256"]!=vr["frozen_surface_inventory_sha256"]: raise E("hidden alignment inventory cannot substitute for V19 expected extraction inventory")
    pre={"schema":ap["receipt"]["schema"],"policy_semantic_sha256":ADMISSION_POLICY_SHA,"v19_manifest_policy_sha256":V19_POLICY_SHA,"v20_annotation_policy_sha256":V20_POLICY_SHA,"case_id":case["case_id"],"assignment_key_sha256":case["assignment_key_sha256"],"split":case["split"],"annotation_receipt_sha256":vr["annotation_receipt_sha256"],"hidden_alignment_inventory_sha256":vr["final_expected_inventory_sha256"]}
    receipt=hashlib.sha256(ap["receipt"]["domain"].encode()+canon(pre)).hexdigest()
    return {"schema":"symthaea.scip-text-claim-case-admission-validation/v1","authority":AUTHORITY,"admission_policy_sha256":ADMISSION_POLICY_SHA,"case_id":case["case_id"],"annotation_receipt_sha256":vr["annotation_receipt_sha256"],"admission_receipt_sha256":receipt,"case_reference_binding_admitted":True,"eligible_for_v19_manifest_membership_subject_to_global_validation":True,"split_assignment_verified":False,"cross_split_leakage_verified":False,"v19_manifest_qualified":False,"human_correctness_established":False,"surface_fidelity_established":False,"confirmatory_execution_authorized":False}

def main():
    a=argparse.ArgumentParser(); a.add_argument("case"); a.add_argument("bundle"); a.add_argument("--v19-policy",required=True); a.add_argument("--v20-policy",required=True); a.add_argument("--admission-policy",required=True); x=a.parse_args()
    try: print(json.dumps(validate(load(x.case),load(x.bundle),load(x.v19_policy),load(x.v20_policy),load(x.admission_policy)),sort_keys=True,separators=(",",":")))
    except E as e: print("ERROR:",e,file=sys.stderr); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
