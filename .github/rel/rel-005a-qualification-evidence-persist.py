#!/usr/bin/env python3
"""Build a domain-separated content-addressed REL-005A durable qualification evidence receipt.

Authority: QualificationEvidencePersistenceOnly.
Hashes already-validated qualification evidence bytes without parsing scientific payload JSON.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, tempfile
from typing import Any

SCHEMA="symthaea.rel.qualification-durability-contract.v4"
AUTHORITY="QualificationEvidencePersistenceOnly"
RECEIPT="qualification-evidence-persistence-receipt.json"
DOMAIN="symthaea.rel.qualification-evidence-payload-set.v1"

def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:pathlib.Path)->dict[str,Any]:
    v=json.loads(p.read_text()); require(isinstance(v,dict),f"{p}: expected JSON object"); return v
def sha256(p:pathlib.Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def commitment_object(contract:dict[str,Any],entries:list[dict[str,Any]])->dict[str,Any]:
    return {"domain":DOMAIN,"relation":contract["relation"],"source_pipeline_head":contract["source_pipeline_head"],"source_pipeline_tree":contract["source_pipeline_tree"],"payload_files":entries}

def build(contract_path:pathlib.Path,input_dir:pathlib.Path,output:pathlib.Path)->dict[str,Any]:
    c=load(contract_path)
    require(c.get("schema")==SCHEMA,"contract schema mismatch")
    require(c.get("authority")=="QualificationDurabilityContractOnly","contract authority mismatch")
    require(c.get("selected_comparison_result") is None,"durability contract selected an outcome")
    require(c.get("outcome_independent") is True,"durability contract is outcome-dependent")
    require(not any(c.get("claims",{}).values()),"durability contract makes runtime/scientific claims")
    for key in c.get("forbidden_result_dependent_fields",[]):
        require(key not in c,f"forbidden result-dependent field present: {key}")
    spec=c["durable_qualification_evidence"]; commit_spec=spec["payload_commitment"]
    require(commit_spec["domain_separator"]==DOMAIN and commit_spec["schema"]==DOMAIN,"commitment domain mismatch")
    require(commit_spec["algorithm"]=="sha256","commitment algorithm mismatch")
    require(all(commit_spec[k] is True for k in ("bind_relation","bind_source_pipeline_head","bind_source_pipeline_tree","bind_ordered_payload_entries")),"commitment binding weakened")
    expected=set(spec["payload_files"])
    require(spec["persistence_receipt"]==RECEIPT,"persistence receipt name mismatch")
    require(spec["artifact_file_count"]==len(expected)+1,"artifact file count mismatch")
    files=[p for p in input_dir.rglob("*") if p.is_file()]
    rel={p.relative_to(input_dir).as_posix() for p in files}
    require(rel==expected,f"payload census mismatch: {sorted(rel)}")
    require(all(len(p.relative_to(input_dir).parts)==1 for p in files),"nested payload file forbidden")
    entries=[]
    for name in sorted(expected,key=lambda s:s.encode()):
        p=input_dir/name
        entries.append({"basename":name,"byte_length":p.stat().st_size,"sha256":sha256(p)})
    commitment=hashlib.sha256(canonical(commitment_object(c,entries))).hexdigest()
    receipt={
        "schema":"symthaea.rel.qualification-evidence-persistence-receipt.v2",
        "authority":AUTHORITY,"relation":c["relation"],
        "source_pipeline_head":c["source_pipeline_head"],"source_pipeline_tree":c["source_pipeline_tree"],
        "payload_file_count":len(entries),"payload_files":entries,
        "payload_commitment_domain":DOMAIN,"payload_set_commitment_sha256":commitment,
        "scientific_payload_parsed":False,"scientific_result_interpreted":False,
        "transport_identity_scientific_authority":False,
        "claims":{"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}
    }
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt

def self_test()->dict[str,Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp); payload=root/"payload"; payload.mkdir(); cp=root/"contract.json"
        names=["predicate-contract-receipt.json","execution-v3-receipt.json","observation-seal-v3.json","comparison-only-qualification-receipt.json","qualification-input-manifest.json","qualification-assembly-receipt.json","qualification-only.json","post-qualification-firewall-receipt.json"]
        c={"schema":SCHEMA,"authority":"QualificationDurabilityContractOnly","relation":"REL-005A","source_pipeline_head":"46"*20,"source_pipeline_tree":"d8"*20,
           "allowed_comparison_results":["ALL_PREDICATES_PASS","PREDICATE_FAILURES"],"selected_comparison_result":None,"outcome_independent":True,
           "forbidden_result_dependent_fields":["qualification_mapping","per_result_policy"],
           "durable_qualification_evidence":{"payload_files":names,"persistence_receipt":RECEIPT,"artifact_file_count":9,
             "payload_commitment":{"schema":DOMAIN,"algorithm":"sha256","domain_separator":DOMAIN,"bind_relation":True,"bind_source_pipeline_head":True,"bind_source_pipeline_tree":True,"bind_ordered_payload_entries":True}},
           "claims":{"durability_implemented":False,"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}}
        cp.write_text(json.dumps(c)+"\n")
        for i,n in enumerate(names):(payload/n).write_bytes(f"{n}\nsynthetic-{i}\n".encode())
        a=build(cp,payload,root/"a.json"); b=build(cp,payload,root/"b.json")
        require(a["payload_set_commitment_sha256"]==b["payload_set_commitment_sha256"],"commitment nondeterministic")
        c2=dict(c); c2["source_pipeline_head"]="47"*20; cp.write_text(json.dumps(c2)+"\n")
        changed=build(cp,payload,root/"c.json")
        require(changed["payload_set_commitment_sha256"]!=a["payload_set_commitment_sha256"],"source identity not bound")
        cp.write_text(json.dumps(c)+"\n"); (payload/names[-1]).unlink()
        try: build(cp,payload,root/"missing.json")
        except ValueError as e: require("payload census mismatch" in str(e),"missing-file rejection wrong")
        else: raise ValueError("missing payload accepted")
    return {"schema":"symthaea.rel.qualification-evidence-persistence-self-test.v2","authority":AUTHORITY,"deterministic_commitment":True,"source_identity_bound":True,"exact_payload_census":True,"missing_payload_rejected":True,"scientific_payload_parsed":False,"scientific_result_interpreted":False}

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=pathlib.Path); p.add_argument("--input-dir",type=pathlib.Path); p.add_argument("--output",type=pathlib.Path); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: print(json.dumps(self_test(),indent=2,sort_keys=True)); return
    require(a.contract is not None and a.input_dir is not None and a.output is not None,"--contract, --input-dir, and --output are required")
    print(json.dumps(build(a.contract,a.input_dir,a.output),indent=2,sort_keys=True))
if __name__=="__main__":main()
