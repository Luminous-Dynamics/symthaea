#!/usr/bin/env python3
"""Fail-closed structural validator for SCIP V20 human reference bundles."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

POLICY_SHA="12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9"
V18_SHA="96e2ec5e1fad213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530"
V19_SHA="9a30ba36b3e872aaec349647f71433579bc0932f24c96335bb32afe77498ffc0"
BUNDLE_SCHEMA="symthaea.scip-text-claim-annotation-bundle/v1"
AUTHORITY="annotation-evidence-only"
DOMAIN="symthaea-scip-text-claim-annotation-bundle-v1\0"
OUTCOMES=("exact-source-claim","no-source-support")
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

def exact(v,ks,w):
    if not isinstance(v,dict) or set(v)!=set(ks): raise E(f"{w} schema mismatch")
    return v

def yes(v,w):
    if v is not True: raise E(f"{w} must be true")

def hx(v,w):
    if not isinstance(v,str) or len(v)!=64 or v=="0"*64 or any(c not in "0123456789abcdef" for c in v): raise E(f"{w} must be non-zero lowercase sha256")
    return v

def canon(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def sha(v): return hashlib.sha256(canon(v)).hexdigest()

def policy_ok(p):
    exact(p,{"schema","authority","v18_preregistration_sha256","v19_manifest_policy_sha256","actor_identity","extraction","alignment","receipt","claim_boundary"},"policy")
    if p["schema"]!="symthaea.scip-text-claim-annotation-policy/v1" or p["authority"]!=AUTHORITY: raise E("policy identity drift")
    if p["v18_preregistration_sha256"]!=V18_SHA or p["v19_manifest_policy_sha256"]!=V19_SHA: raise E("upstream binding drift")
    a=exact(p["actor_identity"],{"representation","names_emails_forbidden","all_roles_within_case_distinct","fingerprint_is_provenance_not_authentication"},"actor policy")
    if a["representation"]!="opaque-sha256-fingerprint": raise E("actor representation drift")
    for k in set(a)-{"representation"}: yes(a[k],f"actor policy.{k}")
    x=exact(p["extraction"],{"independent_annotators","surface_text_visible","public_claim_schema_visible","source_inventory_hidden","source_claim_ids_hidden","candidate_output_hidden","exact_agreement_uses_claim_inventory_sha256","disagreement_requires_distinct_adjudicator","adjudicator_source_inventory_hidden","raw_annotation_artifacts_retained_by_digest","frozen_before_alignment"},"extraction policy")
    if x["independent_annotators"]!=2: raise E("two extraction annotators required")
    for k in set(x)-{"independent_annotators"}: yes(x[k],f"extraction policy.{k}")
    y=exact(p["alignment"],{"independent_aligners","frozen_surface_inventory_visible","source_inventory_visible","candidate_output_hidden","surface_inventory_mutation_forbidden","allowed_alignment_outcomes","exact_agreement_uses_aligned_inventory_sha256","disagreement_requires_distinct_adjudicator","raw_alignment_artifacts_retained_by_digest"},"alignment policy")
    if y["independent_aligners"]!=2 or y["allowed_alignment_outcomes"]!=list(OUTCOMES): raise E("alignment policy drift")
    for k in set(y)-{"independent_aligners","allowed_alignment_outcomes"}: yes(y[k],f"alignment policy.{k}")
    r=exact(p["receipt"],{"schema","digest_algorithm","digest_domain","self_hash_field_absent","v19_annotation_receipt_sha256_is_bundle_digest"},"receipt policy")
    if r["schema"]!=BUNDLE_SCHEMA or r["digest_algorithm"]!="sha256-domain-separated-canonical-json/v1" or r["digest_domain"]!=DOMAIN: raise E("receipt policy drift")
    yes(r["self_hash_field_absent"],"receipt self hash"); yes(r["v19_annotation_receipt_sha256_is_bundle_digest"],"v19 receipt binding")
    b=exact(p["claim_boundary"],{"bundle_validation_does_not_prove_human_correctness","process_declarations_are_not_independently_attested","bundle_validation_does_not_establish_extractor_quality","bundle_validation_does_not_establish_surface_fidelity","bundle_validation_does_not_authorize_confirmatory_execution"},"claim boundary")
    for k,v in b.items(): yes(v,f"claim boundary.{k}")
    if sha(p)!=POLICY_SHA: raise E("policy semantic sha drift")

def actor(r,ks,w):
    exact(r,ks,w); hx(r["artifact_sha256"],w+".artifact"); return hx(r["actor_fingerprint"],w+".actor")

def aligned(outcomes,w):
    if not isinstance(outcomes,list) or not outcomes: raise E(w+" must be non-empty list")
    seen=set(); last=None
    for i,o in enumerate(outcomes):
        exact(o,{"surface_claim_sha256","outcome","source_claim_sha256"},f"{w}[{i}]")
        sid=hx(o["surface_claim_sha256"],f"{w}[{i}].surface")
        if sid in seen or (last is not None and sid<=last): raise E(w+" must be unique and sorted by surface claim digest")
        seen.add(sid); last=sid
        if o["outcome"] not in OUTCOMES: raise E(w+" invalid outcome")
        if o["outcome"]=="exact-source-claim": hx(o["source_claim_sha256"],f"{w}[{i}].source")
        elif o["source_claim_sha256"] is not None: raise E(w+" no-source-support must use null source")
    return sha(outcomes)

def extraction(v,actors):
    exact(v,{"annotations","exact_agreement","adjudication","resolved_inventory_sha256"},"extraction")
    if not isinstance(v["annotations"],list) or len(v["annotations"])!=2 or not isinstance(v["exact_agreement"],bool): raise E("extraction cardinality/type")
    ds=[]
    for i,r in enumerate(v["annotations"]):
        actors.append(actor(r,{"actor_fingerprint","artifact_sha256","claim_inventory_sha256"},f"extraction[{i}]")); ds.append(hx(r["claim_inventory_sha256"],f"extraction[{i}].inventory"))
    resolved=hx(v["resolved_inventory_sha256"],"extraction.resolved")
    if v["exact_agreement"]:
        if ds[0]!=ds[1] or v["adjudication"] is not None or resolved!=ds[0]: raise E("extraction false exact agreement")
    else:
        if ds[0]==ds[1] or not isinstance(v["adjudication"],dict): raise E("extraction disagreement requires adjudication")
        a=v["adjudication"]; actors.append(actor(a,{"actor_fingerprint","artifact_sha256","resolved_inventory_sha256"},"extraction adjudication"))
        if resolved!=hx(a["resolved_inventory_sha256"],"extraction adjudication.resolved"): raise E("extraction adjudication mismatch")
    return resolved

def alignment(v,frozen,source,actors):
    exact(v,{"annotations","exact_agreement","adjudication","resolved_inventory_sha256"},"alignment")
    if not isinstance(v["annotations"],list) or len(v["annotations"])!=2 or not isinstance(v["exact_agreement"],bool): raise E("alignment cardinality/type")
    ds=[]
    for i,r in enumerate(v["annotations"]):
        actors.append(actor(r,{"actor_fingerprint","artifact_sha256","frozen_surface_inventory_sha256","source_inventory_sha256","outcomes","aligned_inventory_sha256"},f"alignment[{i}]"))
        if r["frozen_surface_inventory_sha256"]!=frozen or r["source_inventory_sha256"]!=source: raise E("alignment binding mismatch")
        d=aligned(r["outcomes"],f"alignment[{i}].outcomes")
        if r["aligned_inventory_sha256"]!=d: raise E("alignment inventory digest mismatch")
        ds.append(d)
    resolved=hx(v["resolved_inventory_sha256"],"alignment.resolved")
    if v["exact_agreement"]:
        if ds[0]!=ds[1] or v["adjudication"] is not None or resolved!=ds[0]: raise E("alignment false exact agreement")
    else:
        if ds[0]==ds[1] or not isinstance(v["adjudication"],dict): raise E("alignment disagreement requires adjudication")
        a=v["adjudication"]; actors.append(actor(a,{"actor_fingerprint","artifact_sha256","resolved_outcomes","resolved_inventory_sha256"},"alignment adjudication"))
        d=aligned(a["resolved_outcomes"],"alignment adjudication.outcomes")
        if a["resolved_inventory_sha256"]!=d or resolved!=d: raise E("alignment adjudication mismatch")
    return resolved

def validate(b,p):
    policy_ok(p)
    exact(b,{"schema","authority","policy_semantic_sha256","surface_sha256","source_inventory_sha256","extraction","alignment","process_declarations"},"bundle")
    if b["schema"]!=BUNDLE_SCHEMA or b["authority"]!=AUTHORITY or b["policy_semantic_sha256"]!=POLICY_SHA: raise E("bundle identity drift")
    surface,source=hx(b["surface_sha256"],"surface"),hx(b["source_inventory_sha256"],"source")
    q=exact(b["process_declarations"],{"candidate_output_hidden_all_stages","source_hidden_during_extraction","extraction_inventory_frozen_before_source_reveal","alignment_did_not_modify_surface_inventory","all_role_fingerprints_distinct_within_case"},"process")
    for k,v in q.items(): yes(v,"process."+k)
    actors=[]; frozen=extraction(b["extraction"],actors); final=alignment(b["alignment"],frozen,source,actors)
    if len(actors)!=len(set(actors)): raise E("human role fingerprint reused")
    receipt=hashlib.sha256(DOMAIN.encode()+canon(b)).hexdigest()
    return {"schema":"symthaea.scip-text-claim-annotation-validation/v1","authority":AUTHORITY,"policy_semantic_sha256":POLICY_SHA,"annotation_receipt_sha256":receipt,"surface_sha256":surface,"source_inventory_sha256":source,"frozen_surface_inventory_sha256":frozen,"final_expected_inventory_sha256":final,"human_correctness_established":False,"process_declarations_independently_attested":False,"extractor_quality_established":False,"surface_fidelity_established":False,"confirmatory_execution_authorized":False}

def main():
    a=argparse.ArgumentParser(); a.add_argument("bundle"); a.add_argument("--policy",required=True); x=a.parse_args()
    try: print(json.dumps(validate(load(x.bundle),load(x.policy)),sort_keys=True,separators=(",",":")))
    except E as e: print("ERROR:",e,file=sys.stderr); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
