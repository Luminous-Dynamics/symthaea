#!/usr/bin/env python3
from __future__ import annotations
import copy, hashlib, json, subprocess, sys, tempfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/"scripts/scip_text_claim_blinding_witness.py"
P=ROOT/"scripts/qualification/scip_text_claim_blinding_witness_policy_v1.json"

def h(s): return hashlib.sha256(s.encode()).hexdigest()
def canon(v): return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(",",":")).encode()
def inv(xs): return hashlib.sha256(canon(xs)).hexdigest()
def receipt(b): return hashlib.sha256(b"symthaea-scip-text-claim-annotation-bundle-v1\0"+canon(b)).hexdigest()

def bundle():
    ex_res=h("frozen-surface")
    src=h("source")
    o1=[{"surface_claim_sha256":h("claim-a"),"outcome":"exact-source-claim","source_claim_sha256":h("src-a")}]
    o2=[{"surface_claim_sha256":h("claim-a"),"outcome":"no-source-support","source_claim_sha256":None}]
    resolved=[{"surface_claim_sha256":h("claim-a"),"outcome":"exact-source-claim","source_claim_sha256":h("src-a")}]
    return {
      "schema":"symthaea.scip-text-claim-annotation-bundle/v1","authority":"annotation-evidence-only",
      "policy_semantic_sha256":"12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9",
      "surface_sha256":h("surface"),"source_inventory_sha256":src,
      "extraction":{
        "annotations":[
          {"actor_fingerprint":h("ex-a"),"artifact_sha256":h("ex-art-a"),"claim_inventory_sha256":h("ex-inv-a")},
          {"actor_fingerprint":h("ex-b"),"artifact_sha256":h("ex-art-b"),"claim_inventory_sha256":h("ex-inv-b")}
        ],
        "exact_agreement":False,
        "adjudication":{"actor_fingerprint":h("ex-adj"),"artifact_sha256":h("ex-adj-art"),"resolved_inventory_sha256":ex_res},
        "resolved_inventory_sha256":ex_res
      },
      "alignment":{
        "annotations":[
          {"actor_fingerprint":h("al-a"),"artifact_sha256":h("al-art-a"),"frozen_surface_inventory_sha256":ex_res,"source_inventory_sha256":src,"outcomes":o1,"aligned_inventory_sha256":inv(o1)},
          {"actor_fingerprint":h("al-b"),"artifact_sha256":h("al-art-b"),"frozen_surface_inventory_sha256":ex_res,"source_inventory_sha256":src,"outcomes":o2,"aligned_inventory_sha256":inv(o2)}
        ],
        "exact_agreement":False,
        "adjudication":{"actor_fingerprint":h("al-adj"),"artifact_sha256":h("al-adj-art"),"resolved_outcomes":resolved,"resolved_inventory_sha256":inv(resolved)},
        "resolved_inventory_sha256":inv(resolved)
      },
      "process_declarations":{
        "candidate_output_hidden_all_stages":True,"source_hidden_during_extraction":True,
        "extraction_inventory_frozen_before_source_reveal":True,"alignment_did_not_modify_surface_inventory":True,
        "all_role_fingerprints_distinct_within_case":True
      }
    }

def role_actor(b,slot):
    if slot=="extraction-annotator-0": return b["extraction"]["annotations"][0]["actor_fingerprint"]
    if slot=="extraction-annotator-1": return b["extraction"]["annotations"][1]["actor_fingerprint"]
    if slot=="extraction-adjudicator": return b["extraction"]["adjudication"]["actor_fingerprint"]
    if slot=="alignment-annotator-0": return b["alignment"]["annotations"][0]["actor_fingerprint"]
    if slot=="alignment-annotator-1": return b["alignment"]["annotations"][1]["actor_fingerprint"]
    if slot=="alignment-adjudicator": return b["alignment"]["adjudication"]["actor_fingerprint"]
    raise AssertionError(slot)

def allowed_for(b,slot,pub):
    ex=b["extraction"]; al=b["alignment"]
    m={
      "surface-text":b["surface_sha256"],"public-claim-schema":pub,
      "frozen-surface-inventory":ex["resolved_inventory_sha256"],"source-inventory":b["source_inventory_sha256"],
      "extraction-annotation-a":ex["annotations"][0]["artifact_sha256"],"extraction-annotation-b":ex["annotations"][1]["artifact_sha256"],
      "alignment-annotation-a":al["annotations"][0]["artifact_sha256"],"alignment-annotation-b":al["annotations"][1]["artifact_sha256"]
    }
    classes={
      "extraction-annotator-0":["surface-text","public-claim-schema"],
      "extraction-annotator-1":["surface-text","public-claim-schema"],
      "extraction-adjudicator":["surface-text","public-claim-schema","extraction-annotation-a","extraction-annotation-b"],
      "alignment-annotator-0":["frozen-surface-inventory","source-inventory","public-claim-schema"],
      "alignment-annotator-1":["frozen-surface-inventory","source-inventory","public-claim-schema"],
      "alignment-adjudicator":["frozen-surface-inventory","source-inventory","public-claim-schema","alignment-annotation-a","alignment-annotation-b"]
    }[slot]
    return [{"class":c,"sha256":m[c]} for c in classes]

def witness(b,slot):
    pub=h("public-schema")
    return {
      "schema":"symthaea.scip-text-claim-blinding-witness/v1","authority":"process-attestation-only",
      "role_slot":slot,"participant_actor_fingerprint":role_actor(b,slot),
      "witness_issuer_actor_fingerprint":h("issuer-"+slot),
      "annotation_receipt_sha256":receipt(b),"case_admission_receipt_sha256":h("case-admission"),
      "session_id_sha256":h("session-"+slot),"challenge_nonce_sha256":h("challenge-"+slot),
      "public_claim_schema_sha256":pub,"allowed_artifacts":allowed_for(b,slot,pub),
      "session_manifest_sha256":h("session-manifest-"+slot),"access_control_snapshot_sha256":h("access-"+slot),
      "audit_capture_sha256":h("audit-"+slot),"access_control_snapshot_bound_before_session":True,
      "audit_capture_covers_full_session":True,"audit_capture_complete":True,"forbidden_access_event_count":0,
      "evidence_retained_outside_annotation_bundle":True
    }

def run(w,b):
    with tempfile.TemporaryDirectory() as t:
        t=Path(t); wp=t/"w.json"; bp=t/"b.json"
        wp.write_text(json.dumps(w)); bp.write_text(json.dumps(b))
        return subprocess.run([sys.executable,"-B",str(V),str(wp),str(bp),"--policy",str(P)],capture_output=True,text=True)

def ok(w,b):
    r=run(w,b); assert r.returncode==0,r.stderr
    return json.loads(r.stdout)

def reject(w,b):
    r=run(w,b); assert r.returncode!=0 and "ERROR:" in r.stderr,(r.returncode,r.stdout,r.stderr)

def main():
    b=bundle()
    receipts=[]
    for slot in ("extraction-annotator-0","extraction-adjudicator","alignment-annotator-1","alignment-adjudicator"):
        out=ok(witness(b,slot),b)
        assert out["process_boundary_structurally_attested_under_witness"] is True
        assert out["role_artifact_profile_matches_v20"] is True
        assert out["case_admission_receipt_recomputed"] is False
        assert out["actor_authentication_established"] is False
        assert out["audit_truthfulness_established"] is False
        assert out["human_correctness_established"] is False
        assert out["surface_fidelity_established"] is False
        assert out["confirmatory_execution_authorized"] is False
        receipts.append(out["blinding_witness_receipt_sha256"])
    assert len(set(receipts))==len(receipts)

    w=witness(b,"extraction-annotator-0")
    z=copy.deepcopy(w); z["participant_actor_fingerprint"]=h("wrong"); reject(z,b)
    z=copy.deepcopy(w); z["witness_issuer_actor_fingerprint"]=z["participant_actor_fingerprint"]; reject(z,b)
    z=copy.deepcopy(w); z["annotation_receipt_sha256"]=h("wrong-ann"); reject(z,b)
    z=copy.deepcopy(w); z["case_admission_receipt_sha256"]="0"*64; reject(z,b)
    z=copy.deepcopy(w); z["allowed_artifacts"].append({"class":"source-inventory","sha256":b["source_inventory_sha256"]}); reject(z,b)
    z=copy.deepcopy(w); z["allowed_artifacts"][0]["sha256"]=h("wrong-surface"); reject(z,b)
    z=copy.deepcopy(w); z["allowed_artifacts"].reverse(); reject(z,b)
    z=copy.deepcopy(w); z["audit_capture_complete"]=False; reject(z,b)
    z=copy.deepcopy(w); z["audit_capture_covers_full_session"]=False; reject(z,b)
    z=copy.deepcopy(w); z["access_control_snapshot_bound_before_session"]=False; reject(z,b)
    z=copy.deepcopy(w); z["forbidden_access_event_count"]=1; reject(z,b)
    z=copy.deepcopy(w); z["audit_capture_sha256"]=z["session_manifest_sha256"]; reject(z,b)
    z=copy.deepcopy(w); z["challenge_nonce_sha256"]=z["session_id_sha256"]; reject(z,b)
    z=copy.deepcopy(w); z["participant_name"]="Alice"; reject(z,b)

    noadj=copy.deepcopy(b); noadj["extraction"]["adjudication"]=None
    z=witness(b,"extraction-adjudicator"); reject(z,noadj)
    noadj2=copy.deepcopy(b); noadj2["alignment"]["adjudication"]=None
    z=witness(b,"alignment-adjudicator"); reject(z,noadj2)

    # Duplicate-key rejection
    raw=json.dumps(w)
    raw=raw.replace('"authority": "process-attestation-only"', '"authority": "process-attestation-only", "authority": "qualified"',1)
    with tempfile.TemporaryDirectory() as t:
        t=Path(t); wp=t/"w.json"; bp=t/"b.json"; wp.write_text(raw); bp.write_text(json.dumps(b))
        r=subprocess.run([sys.executable,"-B",str(V),str(wp),str(bp),"--policy",str(P)],capture_output=True,text=True)
        assert r.returncode!=0 and "ERROR:" in r.stderr
    print("PASS_BLINDING_WITNESS_ADVERSARIAL")

if __name__=="__main__": raise SystemExit(main())
