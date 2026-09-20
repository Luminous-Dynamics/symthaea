#!/usr/bin/env python3
from __future__ import annotations
import copy, hashlib, json, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; V=ROOT/"scripts/scip_text_claim_annotation_bundle.py"; P=ROOT/"scripts/qualification/scip_text_claim_annotation_policy_v1.json"
def h(s): return hashlib.sha256(s.encode()).hexdigest()
def inv(xs): return hashlib.sha256(json.dumps(xs,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def outcomes(tag="x"):
    return [
        {"surface_claim_sha256":h("claim-a"),"outcome":"exact-source-claim","source_claim_sha256":h("source-a"+tag)},
        {"surface_claim_sha256":h("claim-b"),"outcome":"no-source-support","source_claim_sha256":None},
    ]
def bundle():
    f=h("frozen"); s=h("source"); o=outcomes(""); a=inv(o)
    return {"schema":"symthaea.scip-text-claim-annotation-bundle/v1","authority":"annotation-evidence-only","policy_semantic_sha256":"12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9","surface_sha256":h("surface"),"source_inventory_sha256":s,"extraction":{"annotations":[{"actor_fingerprint":h("a"),"artifact_sha256":h("aa"),"claim_inventory_sha256":h("i1")},{"actor_fingerprint":h("b"),"artifact_sha256":h("bb"),"claim_inventory_sha256":h("i2")}],"exact_agreement":False,"adjudication":{"actor_fingerprint":h("c"),"artifact_sha256":h("cc"),"resolved_inventory_sha256":f},"resolved_inventory_sha256":f},"alignment":{"annotations":[{"actor_fingerprint":h("d"),"artifact_sha256":h("dd"),"frozen_surface_inventory_sha256":f,"source_inventory_sha256":s,"outcomes":o,"aligned_inventory_sha256":a},{"actor_fingerprint":h("e"),"artifact_sha256":h("ee"),"frozen_surface_inventory_sha256":f,"source_inventory_sha256":s,"outcomes":copy.deepcopy(o),"aligned_inventory_sha256":a}],"exact_agreement":True,"adjudication":None,"resolved_inventory_sha256":a},"process_declarations":{"candidate_output_hidden_all_stages":True,"source_hidden_during_extraction":True,"extraction_inventory_frozen_before_source_reveal":True,"alignment_did_not_modify_surface_inventory":True,"all_role_fingerprints_distinct_within_case":True}}
def run(b):
    with tempfile.TemporaryDirectory() as t:
        q=Path(t)/"b.json"; q.write_text(json.dumps(b),"utf-8"); return subprocess.run([sys.executable,"-B",str(V),str(q),"--policy",str(P)],capture_output=True,text=True)
def ok(b):
    r=run(b); assert r.returncode==0,r.stderr; return json.loads(r.stdout)
def reject(b):
    r=run(b); assert r.returncode!=0 and "ERROR:" in r.stderr

def main():
    b=bundle(); x=ok(b); assert not any(x[k] for k in ("human_correctness_established","process_declarations_independently_attested","extractor_quality_established","surface_fidelity_established","confirmatory_execution_authorized"))
    # Exact extraction agreement is valid without an adjudicator.
    z=copy.deepcopy(b); same=h("exact-extraction"); z["extraction"]["annotations"][0]["claim_inventory_sha256"]=same; z["extraction"]["annotations"][1]["claim_inventory_sha256"]=same; z["extraction"]["exact_agreement"]=True; z["extraction"]["adjudication"]=None; z["extraction"]["resolved_inventory_sha256"]=same
    for r in z["alignment"]["annotations"]: r["frozen_surface_inventory_sha256"]=same
    ok(z)
    # Alignment disagreement is valid only with a distinct adjudicator and resolved outcomes.
    d=copy.deepcopy(b); alt=outcomes("-alt"); d["alignment"]["annotations"][1]["outcomes"]=alt; d["alignment"]["annotations"][1]["aligned_inventory_sha256"]=inv(alt); resolved=outcomes("-resolved"); rd=inv(resolved); d["alignment"]["exact_agreement"]=False; d["alignment"]["adjudication"]={"actor_fingerprint":h("f"),"artifact_sha256":h("ff"),"resolved_outcomes":resolved,"resolved_inventory_sha256":rd}; d["alignment"]["resolved_inventory_sha256"]=rd; ok(d)
    missing=copy.deepcopy(d); missing["alignment"]["adjudication"]=None; reject(missing)
    reused=copy.deepcopy(d); reused["alignment"]["adjudication"]["actor_fingerprint"]=d["extraction"]["annotations"][0]["actor_fingerprint"]; reject(reused)
    muts=(lambda z:z["extraction"].__setitem__("adjudication",None),lambda z:z["extraction"]["annotations"][1].__setitem__("actor_fingerprint",z["extraction"]["annotations"][0]["actor_fingerprint"]),lambda z:z["alignment"]["annotations"][0].__setitem__("actor_fingerprint",z["extraction"]["annotations"][0]["actor_fingerprint"]),lambda z:z["alignment"]["annotations"][0].__setitem__("frozen_surface_inventory_sha256",h("other")),lambda z:z["alignment"]["annotations"][0].__setitem__("source_inventory_sha256",h("other")),lambda z:z["process_declarations"].__setitem__("source_hidden_during_extraction",False),lambda z:z["process_declarations"].__setitem__("candidate_output_hidden_all_stages",False),lambda z:z.__setitem__("surface_fidelity_established",True),lambda z:z["extraction"]["annotations"][0].__setitem__("annotator_name","Alice"))
    for mut in muts: z=copy.deepcopy(b); mut(z); reject(z)
    z=copy.deepcopy(b); z["alignment"]["annotations"][0]["outcomes"][0]["outcome"]="similar-source-claim"; reject(z)
    z=copy.deepcopy(b); z["alignment"]["annotations"][0]["outcomes"][1]["source_claim_sha256"]=h("forged"); reject(z)
    z=copy.deepcopy(b); z["alignment"]["annotations"][0]["outcomes"].reverse(); reject(z)
    z=copy.deepcopy(b); z["alignment"]["annotations"][0]["aligned_inventory_sha256"]=h("caller-chosen"); reject(z)
    old=ok(b)["annotation_receipt_sha256"]; z=copy.deepcopy(b); n=h("new"); z["extraction"]["adjudication"]["resolved_inventory_sha256"]=n; z["extraction"]["resolved_inventory_sha256"]=n; [r.__setitem__("frozen_surface_inventory_sha256",n) for r in z["alignment"]["annotations"]]; assert ok(z)["annotation_receipt_sha256"]!=old
    raw=json.dumps(b,separators=(",",":")); raw=raw.replace('"authority":"annotation-evidence-only"','"authority":"annotation-evidence-only","authority":"qualified"',1)
    with tempfile.TemporaryDirectory() as t:
        q=Path(t)/"b.json"; q.write_text(raw); r=subprocess.run([sys.executable,"-B",str(V),str(q),"--policy",str(P)],capture_output=True,text=True); assert r.returncode!=0
    print("PASS_ANNOTATION_BUNDLE_ADVERSARIAL")
if __name__=="__main__": raise SystemExit(main())
