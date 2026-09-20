#!/usr/bin/env python3
"""Compose the complete structurally valid V22 witness set for one V20 bundle."""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path
import scip_text_claim_blinding_witness as v22

POLICY_SHA="f50117a1dbe55574f65608194943595b298e66fde6e7dea727e57a51267f1a8f"
V22_POLICY_SHA=v22.POLICY_SHA
V21_POLICY_SHA="452f4201c230ca3278d1ddc744fb70bca5944fa0d77b89e494096cfbf9c3dfb9"
AUTHORITY="process-completeness-only"
SET_SCHEMA="symthaea.scip-text-claim-blinding-witness-set/v1"
SET_FIELDS={"schema","authority","policy_semantic_sha256","witnesses"}
POLICY_FIELDS={"schema","authority","v22_blinding_witness_policy_sha256","v21_case_admission_policy_sha256","composition","receipt","claim_boundary"}
COMPOSITION_FIELDS={
 "required_role_slots_derived_from_v20_bundle","one_witness_per_required_role_slot","missing_role_slots_rejected","extra_role_slots_rejected",
 "common_annotation_receipt_required","common_case_admission_receipt_required","common_public_claim_schema_sha256_required",
 "distinct_session_ids_required","distinct_challenge_nonces_required","distinct_witness_receipts_required","distinct_session_manifest_roots_required",
 "distinct_access_control_snapshot_roots_required","distinct_audit_capture_roots_required","canonical_role_slot_order_required"}
RECEIPT_FIELDS={"schema","algorithm","domain","self_hash_field_absent"}
BOUNDARY_FIELDS={
 "witness_set_completeness_does_not_recompute_v21_case_admission","witness_set_completeness_does_not_authenticate_evidence",
 "witness_set_completeness_does_not_authenticate_actors","witness_set_completeness_does_not_prove_audit_truthfulness",
 "witness_set_completeness_does_not_prove_audit_completeness_in_reality","witness_set_completeness_does_not_prove_human_independence",
 "witness_set_completeness_does_not_prove_human_expertise","witness_set_completeness_does_not_prove_human_correctness",
 "witness_set_completeness_does_not_establish_surface_fidelity","witness_set_completeness_does_not_authorize_confirmatory_execution"}

class SetError(ValueError): pass

def strict_object(pairs):
 d={}
 for k,v in pairs:
  if k in d: raise SetError(f"duplicate JSON key: {k}")
  d[k]=v
 return d

def load(path):
 try: v=json.loads(Path(path).read_text("utf-8"),object_pairs_hook=strict_object)
 except (OSError,UnicodeDecodeError,json.JSONDecodeError) as e: raise SetError(str(e)) from e
 if not isinstance(v,dict): raise SetError("top level must be object")
 return v

def exact(v, expected, where):
 if not isinstance(v,dict) or set(v)!=set(expected): raise SetError(f"{where} schema mismatch")
 return v

def yes(v,where):
 if v is not True: raise SetError(f"{where} must be true")
def canon(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def sem(v): return hashlib.sha256(canon(v)).hexdigest()

def validate_policy(p):
 exact(p,POLICY_FIELDS,"policy")
 if p["schema"]!="symthaea.scip-text-claim-blinding-witness-set-policy/v1" or p["authority"]!=AUTHORITY: raise SetError("policy identity drift")
 if p["v22_blinding_witness_policy_sha256"]!=V22_POLICY_SHA or p["v21_case_admission_policy_sha256"]!=V21_POLICY_SHA: raise SetError("upstream policy binding drift")
 c=exact(p["composition"],COMPOSITION_FIELDS,"policy.composition")
 for k,v in c.items(): yes(v,"policy.composition."+k)
 r=exact(p["receipt"],RECEIPT_FIELDS,"policy.receipt")
 if r["schema"]!=SET_SCHEMA or r["algorithm"]!="sha256-domain-separated-canonical-json/v1" or r["domain"]!="symthaea-scip-text-claim-blinding-witness-set-v1\0": raise SetError("receipt policy drift")
 yes(r["self_hash_field_absent"],"policy.receipt.self_hash_field_absent")
 b=exact(p["claim_boundary"],BOUNDARY_FIELDS,"policy.claim_boundary")
 for k,v in b.items(): yes(v,"policy.claim_boundary."+k)
 if sem(p)!=POLICY_SHA: raise SetError("policy semantic identity drift")

def required_slots(bundle):
 ex,al=v22.validate_bundle_shape(bundle)
 slots=["extraction-annotator-0","extraction-annotator-1"]
 if ex["adjudication"] is not None: slots.append("extraction-adjudicator")
 slots += ["alignment-annotator-0","alignment-annotator-1"]
 if al["adjudication"] is not None: slots.append("alignment-adjudicator")
 return slots

def distinct(values,where):
 if len(values)!=len(set(values)): raise SetError(f"{where} must be distinct across role sessions")

def validate(wset,bundle,v22_policy,policy):
 validate_policy(policy)
 try: v22.validate_policy(v22_policy); v22.validate_bundle_shape(bundle)
 except Exception as e: raise SetError(f"inherited V22 input invalid: {e}") from e
 exact(wset,SET_FIELDS,"witness set")
 if wset["schema"]!=SET_SCHEMA or wset["authority"]!=AUTHORITY or wset["policy_semantic_sha256"]!=POLICY_SHA: raise SetError("witness set identity drift")
 witnesses=wset["witnesses"]
 if not isinstance(witnesses,list): raise SetError("witnesses must be list")
 slots=required_slots(bundle)
 got_slots=[]; results=[]
 for i,w in enumerate(witnesses):
  if not isinstance(w,dict): raise SetError(f"witnesses[{i}] must be object")
  got_slots.append(w.get("role_slot"))
  try: results.append(v22.validate(w,bundle,v22_policy))
  except Exception as e: raise SetError(f"witnesses[{i}] invalid: {e}") from e
 if got_slots!=slots: raise SetError(f"role-slot census/order mismatch: expected={slots} got={got_slots}")
 anns=[r["annotation_receipt_sha256"] for r in results]
 cases=[r["case_admission_receipt_sha256"] for r in results]
 pubs=[r["public_claim_schema_sha256"] for r in results]
 if len(set(anns))!=1: raise SetError("annotation receipt mismatch across witnesses")
 if len(set(cases))!=1: raise SetError("case admission receipt mismatch across witnesses")
 if len(set(pubs))!=1: raise SetError("public claim schema mismatch across witnesses")
 distinct([w["session_id_sha256"] for w in witnesses],"session ids")
 distinct([w["challenge_nonce_sha256"] for w in witnesses],"challenge nonces")
 distinct([r["blinding_witness_receipt_sha256"] for r in results],"witness receipts")
 distinct([w["session_manifest_sha256"] for w in witnesses],"session manifest roots")
 distinct([w["access_control_snapshot_sha256"] for w in witnesses],"access-control roots")
 distinct([w["audit_capture_sha256"] for w in witnesses],"audit roots")
 role_records=[{
  "role_slot":r["role_slot"],"participant_actor_fingerprint":r["participant_actor_fingerprint"],
  "blinding_witness_receipt_sha256":r["blinding_witness_receipt_sha256"],
  "access_profile_sha256":r["access_profile_sha256"]} for r in results]
 pre={"policy_semantic_sha256":POLICY_SHA,"annotation_receipt_sha256":anns[0],"case_admission_receipt_sha256":cases[0],
      "public_claim_schema_sha256":pubs[0],"role_records":role_records}
 receipt=hashlib.sha256(policy["receipt"]["domain"].encode()+canon(pre)).hexdigest()
 return {
  "schema":"symthaea.scip-text-claim-blinding-witness-set-validation/v1","authority":AUTHORITY,"policy_semantic_sha256":POLICY_SHA,
  "annotation_receipt_sha256":anns[0],"case_admission_receipt_sha256":cases[0],"public_claim_schema_sha256":pubs[0],
  "required_role_slots":slots,"role_count":len(slots),"witness_set_receipt_sha256":receipt,
  "process_evidence_set_structurally_complete":True,"all_required_role_slots_present_exactly_once":True,"common_case_binding_established":True,
  "case_admission_receipt_recomputed":False,"evidence_authenticity_established":False,"actor_authentication_established":False,
  "audit_truthfulness_established":False,"audit_completeness_in_reality_established":False,"human_independence_established":False,
  "human_expertise_established":False,"human_correctness_established":False,"surface_fidelity_established":False,
  "confirmatory_execution_authorized":False}

def main():
 a=argparse.ArgumentParser(); a.add_argument("witness_set"); a.add_argument("bundle"); a.add_argument("--v22-policy",required=True); a.add_argument("--policy",required=True); x=a.parse_args()
 try: print(json.dumps(validate(load(x.witness_set),load(x.bundle),load(x.v22_policy),load(x.policy)),sort_keys=True,separators=(",",":")))
 except SetError as e: print("ERROR:",e,file=sys.stderr); return 2
 return 0
if __name__=="__main__": raise SystemExit(main())
