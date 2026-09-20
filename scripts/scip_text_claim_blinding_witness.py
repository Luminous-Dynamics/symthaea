#!/usr/bin/env python3
"""Structural V22 operational-blinding witness validator."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

POLICY_SHA = "edb837213b56f4423bc096d3bb11c90dd07de03cf5b672398b1a17de2a6b6cc0"
V20_POLICY_SHA = "12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9"
V21_POLICY_SHA = "452f4201c230ca3278d1ddc744fb70bca5944fa0d77b89e494096cfbf9c3dfb9"
AUTHORITY = "process-attestation-only"
WITNESS_SCHEMA = "symthaea.scip-text-claim-blinding-witness/v1"
WITNESS_FIELDS = {
    "schema","authority","role_slot","participant_actor_fingerprint","witness_issuer_actor_fingerprint",
    "annotation_receipt_sha256","case_admission_receipt_sha256","session_id_sha256","challenge_nonce_sha256",
    "public_claim_schema_sha256","allowed_artifacts","session_manifest_sha256","access_control_snapshot_sha256",
    "audit_capture_sha256","access_control_snapshot_bound_before_session","audit_capture_covers_full_session",
    "audit_capture_complete","forbidden_access_event_count","evidence_retained_outside_annotation_bundle",
}
ARTIFACT_FIELDS = {"class","sha256"}
BUNDLE_FIELDS = {"schema","authority","policy_semantic_sha256","surface_sha256","source_inventory_sha256","extraction","alignment","process_declarations"}
PROCESS_FIELDS = {
    "candidate_output_hidden_all_stages","source_hidden_during_extraction",
    "extraction_inventory_frozen_before_source_reveal","alignment_did_not_modify_surface_inventory",
    "all_role_fingerprints_distinct_within_case",
}
EXTRACTION_FIELDS = {"annotations","exact_agreement","adjudication","resolved_inventory_sha256"}
ALIGNMENT_FIELDS = {"annotations","exact_agreement","adjudication","resolved_inventory_sha256"}
EX_ANN_FIELDS = {"actor_fingerprint","artifact_sha256","claim_inventory_sha256"}
AL_ANN_FIELDS = {"actor_fingerprint","artifact_sha256","frozen_surface_inventory_sha256","source_inventory_sha256","outcomes","aligned_inventory_sha256"}
ADJ_EX_FIELDS = {"actor_fingerprint","artifact_sha256","resolved_inventory_sha256"}
ADJ_AL_FIELDS = {"actor_fingerprint","artifact_sha256","resolved_outcomes","resolved_inventory_sha256"}

class WitnessError(ValueError): pass

def strict_object(pairs):
    d={}
    for k,v in pairs:
        if k in d: raise WitnessError(f"duplicate JSON key: {k}")
        d[k]=v
    return d

def load(path):
    try:
        v=json.loads(Path(path).read_text("utf-8"),object_pairs_hook=strict_object)
    except (OSError,UnicodeDecodeError,json.JSONDecodeError) as e:
        raise WitnessError(str(e)) from e
    if not isinstance(v,dict): raise WitnessError("top level must be object")
    return v

def exact(v, expected, where):
    if not isinstance(v,dict) or set(v)!=set(expected):
        raise WitnessError(f"{where} schema mismatch")
    return v

def hx(v, where):
    if not isinstance(v,str) or len(v)!=64 or v=="0"*64 or any(c not in "0123456789abcdef" for c in v):
        raise WitnessError(f"{where} must be non-zero lowercase sha256")
    return v

def yes(v, where):
    if v is not True: raise WitnessError(f"{where} must be true")

def canon(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()

def sem(v): return hashlib.sha256(canon(v)).hexdigest()

def annotation_receipt(bundle):
    return hashlib.sha256(b"symthaea-scip-text-claim-annotation-bundle-v1\0"+canon(bundle)).hexdigest()

def validate_policy(p):
    if sem(p)!=POLICY_SHA: raise WitnessError("V22 policy semantic identity drift")
    if p.get("schema")!="symthaea.scip-text-claim-blinding-witness-policy/v1" or p.get("authority")!=AUTHORITY:
        raise WitnessError("V22 policy identity drift")
    if p.get("v20_annotation_policy_sha256")!=V20_POLICY_SHA or p.get("v21_case_admission_policy_sha256")!=V21_POLICY_SHA:
        raise WitnessError("upstream policy binding drift")
    if p.get("receipt",{}).get("domain")!="symthaea-scip-text-claim-blinding-witness-v1\0":
        raise WitnessError("receipt domain drift")
    if not all(p.get("evidence",{}).values()) or not all(p.get("claim_boundary",{}).values()):
        raise WitnessError("policy evidence/boundary weakened")

def validate_bundle_shape(b):
    exact(b,BUNDLE_FIELDS,"bundle")
    if b["schema"]!="symthaea.scip-text-claim-annotation-bundle/v1" or b["authority"]!="annotation-evidence-only" or b["policy_semantic_sha256"]!=V20_POLICY_SHA:
        raise WitnessError("bundle identity drift")
    hx(b["surface_sha256"],"bundle.surface_sha256"); hx(b["source_inventory_sha256"],"bundle.source_inventory_sha256")
    ex=exact(b["extraction"],EXTRACTION_FIELDS,"bundle.extraction")
    al=exact(b["alignment"],ALIGNMENT_FIELDS,"bundle.alignment")
    if not isinstance(ex["annotations"],list) or len(ex["annotations"])!=2: raise WitnessError("extraction annotations must be two")
    if not isinstance(al["annotations"],list) or len(al["annotations"])!=2: raise WitnessError("alignment annotations must be two")
    for i,r in enumerate(ex["annotations"]):
        exact(r,EX_ANN_FIELDS,f"extraction.annotations[{i}]"); hx(r["actor_fingerprint"],"ex actor"); hx(r["artifact_sha256"],"ex artifact"); hx(r["claim_inventory_sha256"],"ex inventory")
    for i,r in enumerate(al["annotations"]):
        exact(r,AL_ANN_FIELDS,f"alignment.annotations[{i}]"); hx(r["actor_fingerprint"],"al actor"); hx(r["artifact_sha256"],"al artifact")
        if r["frozen_surface_inventory_sha256"]!=ex["resolved_inventory_sha256"] or r["source_inventory_sha256"]!=b["source_inventory_sha256"]:
            raise WitnessError("alignment binding mismatch")
        hx(r["aligned_inventory_sha256"],"aligned inventory")
    hx(ex["resolved_inventory_sha256"],"extraction.resolved_inventory_sha256"); hx(al["resolved_inventory_sha256"],"alignment.resolved_inventory_sha256")
    if ex["adjudication"] is not None:
        a=exact(ex["adjudication"],ADJ_EX_FIELDS,"extraction.adjudication"); hx(a["actor_fingerprint"],"ex adjudicator"); hx(a["artifact_sha256"],"ex adj artifact")
        if a["resolved_inventory_sha256"]!=ex["resolved_inventory_sha256"]: raise WitnessError("ex adjudication resolved mismatch")
    if al["adjudication"] is not None:
        a=exact(al["adjudication"],ADJ_AL_FIELDS,"alignment.adjudication"); hx(a["actor_fingerprint"],"al adjudicator"); hx(a["artifact_sha256"],"al adj artifact")
        if a["resolved_inventory_sha256"]!=al["resolved_inventory_sha256"]: raise WitnessError("al adjudication resolved mismatch")
    p=exact(b["process_declarations"],PROCESS_FIELDS,"bundle.process_declarations")
    for k,v in p.items(): yes(v,"process."+k)
    return ex,al

def expected_profile(role_slot,b,p):
    ex,al=validate_bundle_shape(b)
    pub=None
    if role_slot=="extraction-annotator-0":
        actor=ex["annotations"][0]["actor_fingerprint"]; role="extraction-annotator"
    elif role_slot=="extraction-annotator-1":
        actor=ex["annotations"][1]["actor_fingerprint"]; role="extraction-annotator"
    elif role_slot=="extraction-adjudicator":
        if ex["adjudication"] is None: raise WitnessError("no extraction adjudicator exists in bundle")
        actor=ex["adjudication"]["actor_fingerprint"]; role="extraction-adjudicator"
    elif role_slot=="alignment-annotator-0":
        actor=al["annotations"][0]["actor_fingerprint"]; role="alignment-annotator"
    elif role_slot=="alignment-annotator-1":
        actor=al["annotations"][1]["actor_fingerprint"]; role="alignment-annotator"
    elif role_slot=="alignment-adjudicator":
        if al["adjudication"] is None: raise WitnessError("no alignment adjudicator exists in bundle")
        actor=al["adjudication"]["actor_fingerprint"]; role="alignment-adjudicator"
    else:
        raise WitnessError("invalid role_slot")
    allowed=p["roles"][role]["allowed_artifact_classes"]
    values={"surface-text":b["surface_sha256"],"frozen-surface-inventory":ex["resolved_inventory_sha256"],"source-inventory":b["source_inventory_sha256"],
            "extraction-annotation-a":ex["annotations"][0]["artifact_sha256"],"extraction-annotation-b":ex["annotations"][1]["artifact_sha256"],
            "alignment-annotation-a":al["annotations"][0]["artifact_sha256"],"alignment-annotation-b":al["annotations"][1]["artifact_sha256"]}
    return actor,role,allowed,values

def validate(w,b,p):
    validate_policy(p)
    exact(w,WITNESS_FIELDS,"witness")
    if w["schema"]!=WITNESS_SCHEMA or w["authority"]!=AUTHORITY: raise WitnessError("witness identity drift")
    ann=annotation_receipt(b)
    if w["annotation_receipt_sha256"]!=ann: raise WitnessError("annotation receipt mismatch")
    hx(w["case_admission_receipt_sha256"],"case_admission_receipt_sha256")
    actor,role,allowed,values=expected_profile(w["role_slot"],b,p)
    if w["participant_actor_fingerprint"]!=actor: raise WitnessError("participant fingerprint does not match V20 role")
    hx(w["witness_issuer_actor_fingerprint"],"witness issuer")
    if w["witness_issuer_actor_fingerprint"]==actor: raise WitnessError("witness issuer must differ from participant")
    pub=hx(w["public_claim_schema_sha256"],"public_claim_schema_sha256")
    values["public-claim-schema"]=pub
    arts=w["allowed_artifacts"]
    if not isinstance(arts,list) or len(arts)!=len(allowed): raise WitnessError("allowed artifact cardinality mismatch")
    seen=set(); digests=[]
    for i,(item,klass) in enumerate(zip(arts,allowed)):
        exact(item,ARTIFACT_FIELDS,f"allowed_artifacts[{i}]")
        if item["class"]!=klass: raise WitnessError("allowed artifact class/order drift")
        if klass in seen: raise WitnessError("duplicate artifact class")
        seen.add(klass)
        got=hx(item["sha256"],f"allowed_artifacts[{i}].sha256")
        if got!=values[klass]: raise WitnessError(f"artifact digest mismatch for {klass}")
        digests.append(got)
    if len(set(digests))!=len(digests): raise WitnessError("allowed artifact digests must be distinct")
    roots=[hx(w[k],k) for k in ("session_manifest_sha256","access_control_snapshot_sha256","audit_capture_sha256")]
    if len(set(roots))!=3: raise WitnessError("evidence roots must be distinct")
    sid=hx(w["session_id_sha256"],"session_id_sha256"); nonce=hx(w["challenge_nonce_sha256"],"challenge_nonce_sha256")
    if sid==nonce: raise WitnessError("session and challenge identities must differ")
    yes(w["access_control_snapshot_bound_before_session"],"access_control_snapshot_bound_before_session")
    yes(w["audit_capture_covers_full_session"],"audit_capture_covers_full_session")
    yes(w["audit_capture_complete"],"audit_capture_complete")
    yes(w["evidence_retained_outside_annotation_bundle"],"evidence_retained_outside_annotation_bundle")
    if type(w["forbidden_access_event_count"]) is not int or w["forbidden_access_event_count"]!=0:
        raise WitnessError("forbidden_access_event_count must be integer zero")
    profile={"role":role,"allowed_artifact_classes":allowed,"forbidden_artifact_classes":p["roles"][role]["forbidden_artifact_classes"]}
    pre={"policy_semantic_sha256":POLICY_SHA,"annotation_receipt_sha256":ann,"case_admission_receipt_sha256":w["case_admission_receipt_sha256"],
          "role_slot":w["role_slot"],"participant_actor_fingerprint":actor,"witness":w,"access_profile_sha256":sem(profile)}
    receipt=hashlib.sha256(p["receipt"]["domain"].encode()+canon(pre)).hexdigest()
    return {
      "schema":"symthaea.scip-text-claim-blinding-witness-validation/v1","authority":AUTHORITY,
      "policy_semantic_sha256":POLICY_SHA,"annotation_receipt_sha256":ann,
      "case_admission_receipt_sha256":w["case_admission_receipt_sha256"],"role_slot":w["role_slot"],
      "participant_actor_fingerprint":actor,"public_claim_schema_sha256":pub,"access_profile_sha256":sem(profile),
      "blinding_witness_receipt_sha256":receipt,"process_boundary_structurally_attested_under_witness":True,
      "role_artifact_profile_matches_v20":True,"evidence_roots_bound":True,"case_admission_binding_present":True,
      "case_admission_receipt_recomputed":False,"actor_authentication_established":False,"audit_truthfulness_established":False,
      "audit_completeness_in_reality_established":False,"human_independence_established":False,"human_expertise_established":False,
      "human_correctness_established":False,"surface_fidelity_established":False,"confirmatory_execution_authorized":False
    }

def main():
    a=argparse.ArgumentParser(); a.add_argument("witness"); a.add_argument("bundle"); a.add_argument("--policy",required=True); x=a.parse_args()
    try: print(json.dumps(validate(load(x.witness),load(x.bundle),load(x.policy)),sort_keys=True,separators=(",",":")))
    except WitnessError as e: print("ERROR:",e,file=sys.stderr); return 2
    return 0
if __name__=="__main__": raise SystemExit(main())
