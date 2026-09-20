#!/usr/bin/env python3
from __future__ import annotations
import copy,hashlib,json,subprocess,sys,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; V=ROOT/"scripts/scip_text_claim_case_admission.py"; V19=ROOT/"scripts/qualification/scip_text_claim_corpus_manifest_policy_v1.json"; V20=ROOT/"scripts/qualification/scip_text_claim_annotation_policy_v1.json"; AP=ROOT/"scripts/qualification/scip_text_claim_case_admission_policy_v1.json"
def h(s): return hashlib.sha256(s.encode()).hexdigest()
def canon(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
def inv(xs): return hashlib.sha256(canon(xs)).hexdigest()
def outs(): return [{"surface_claim_sha256":h("claim-a"),"outcome":"exact-source-claim","source_claim_sha256":h("source-a")},{"surface_claim_sha256":h("claim-b"),"outcome":"no-source-support","source_claim_sha256":None}]
def bundle():
    f=h("frozen"); s=h("source"); o=outs(); a=inv(o)
    return {"schema":"symthaea.scip-text-claim-annotation-bundle/v1","authority":"annotation-evidence-only","policy_semantic_sha256":"12a8431fd8a93dd8ba158758bb4089ef951960240fcab11a5178159e67f19bb9","surface_sha256":h("surface"),"source_inventory_sha256":s,"extraction":{"annotations":[{"actor_fingerprint":h("a"),"artifact_sha256":h("aa"),"claim_inventory_sha256":h("i1")},{"actor_fingerprint":h("b"),"artifact_sha256":h("bb"),"claim_inventory_sha256":h("i2")}],"exact_agreement":False,"adjudication":{"actor_fingerprint":h("c"),"artifact_sha256":h("cc"),"resolved_inventory_sha256":f},"resolved_inventory_sha256":f},"alignment":{"annotations":[{"actor_fingerprint":h("d"),"artifact_sha256":h("dd"),"frozen_surface_inventory_sha256":f,"source_inventory_sha256":s,"outcomes":o,"aligned_inventory_sha256":a},{"actor_fingerprint":h("e"),"artifact_sha256":h("ee"),"frozen_surface_inventory_sha256":f,"source_inventory_sha256":s,"outcomes":copy.deepcopy(o),"aligned_inventory_sha256":a}],"exact_agreement":True,"adjudication":None,"resolved_inventory_sha256":a},"process_declarations":{"candidate_output_hidden_all_stages":True,"source_hidden_during_extraction":True,"extraction_inventory_frozen_before_source_reveal":True,"alignment_did_not_modify_surface_inventory":True,"all_role_fingerprints_distinct_within_case":True}}
def receipt(b): return hashlib.sha256(b"symthaea-scip-text-claim-annotation-bundle-v1\0"+canon(b)).hexdigest()
CID_FIELDS=("kind","dimension","polarity","discourse_family","surface_sha256","source_inventory_sha256","expected_inventory_sha256","annotation_receipt_sha256","template_sha256","named_entity_tuple_sha256","numeric_tuple_sha256","exact_sentence_sha256")
def recase(c): c["case_id"]=hashlib.sha256(b"symthaea-scip-text-claim-case-v1\0"+canon({k:c[k] for k in CID_FIELDS})).hexdigest(); return c
def case(b):
    return recase({"assignment_key_sha256":h("assignment"),"case_id":"","kind":"dimension","dimension":"entity-reference","polarity":"positive","discourse_family":None,"surface_sha256":b["surface_sha256"],"source_inventory_sha256":b["source_inventory_sha256"],"expected_inventory_sha256":b["extraction"]["resolved_inventory_sha256"],"annotation_receipt_sha256":receipt(b),"template_sha256":h("template"),"named_entity_tuple_sha256":h("entities"),"numeric_tuple_sha256":h("numbers"),"exact_sentence_sha256":h("sentence"),"split":"calibration"})
def run(c,b,v19=V19,v20=V20,ap=AP):
    with tempfile.TemporaryDirectory() as t:
        t=Path(t); cp=t/"c.json"; bp=t/"b.json"; cp.write_text(json.dumps(c)); bp.write_text(json.dumps(b)); return subprocess.run([sys.executable,"-B",str(V),str(cp),str(bp),"--v19-policy",str(v19),"--v20-policy",str(v20),"--admission-policy",str(ap)],capture_output=True,text=True)
def ok(c,b): r=run(c,b); assert r.returncode==0,r.stderr; return json.loads(r.stdout)
def reject(c,b,**kw): r=run(c,b,**kw); assert r.returncode!=0 and "ERROR:" in r.stderr

def main():
    b=bundle(); c=case(b); x=ok(c,b); assert x["case_reference_binding_admitted"] and x["eligible_for_v19_manifest_membership_subject_to_global_validation"]; assert not x["split_assignment_verified"] and not x["cross_split_leakage_verified"] and not x["v19_manifest_qualified"] and not x["surface_fidelity_established"] and not x["confirmatory_execution_authorized"]
    base_receipt=x["admission_receipt_sha256"]
    for field,new in (("surface_sha256",h("other-surface")),("source_inventory_sha256",h("other-source")),("annotation_receipt_sha256",h("other-receipt"))):
        z=copy.deepcopy(c); z[field]=new; recase(z); reject(z,b)
    z=copy.deepcopy(c); z["expected_inventory_sha256"]=b["alignment"]["resolved_inventory_sha256"]; recase(z); reject(z,b)
    z=copy.deepcopy(c); z["case_id"]=h("forged-case"); reject(z,b)
    z=copy.deepcopy(c); z["assignment_key_sha256"]=h("new-assignment"); y=ok(z,b); assert z["case_id"]==c["case_id"] and y["admission_receipt_sha256"]!=base_receipt
    z=copy.deepcopy(c); z["split"]="confirmatory"; y=ok(z,b); assert z["case_id"]==c["case_id"] and y["admission_receipt_sha256"]!=base_receipt and not y["confirmatory_execution_authorized"]
    z=copy.deepcopy(c); z["template_sha256"]=h("new-template"); recase(z); y=ok(z,b); assert z["case_id"]!=c["case_id"] and y["admission_receipt_sha256"]!=base_receipt
    bad=copy.deepcopy(b); bad["process_declarations"]["source_hidden_during_extraction"]=False; reject(c,bad)
    with tempfile.TemporaryDirectory() as t:
        t=Path(t); vp=json.loads(V19.read_text()); vp["case_identity"]["domain"]="wrong\0"; q=t/"v19.json"; q.write_text(json.dumps(vp)); reject(c,b,v19=q)
        ap=json.loads(AP.read_text()); ap["claim_boundary"]["single_case_admission_does_not_qualify_v19_manifest"]=False; q2=t/"ap.json"; q2.write_text(json.dumps(ap)); reject(c,b,ap=q2)
    print("PASS_CASE_ADMISSION_ADVERSARIAL")
if __name__=="__main__": raise SystemExit(main())
