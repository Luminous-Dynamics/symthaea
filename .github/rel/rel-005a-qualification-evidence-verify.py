#!/usr/bin/env python3
"""Independently verify self-describing durable REL-005A qualification evidence before packaging.

Authority: QualificationEvidencePersistenceVerificationOnly.
Recomputes file census, sizes, hashes, and a domain-separated payload-set commitment
without parsing scientific payload JSON.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, tempfile
from typing import Any

CONTRACT_SCHEMA="symthaea.rel.qualification-durability-contract.v5"
RECEIPT_NAME="qualification-evidence-persistence-receipt.json"
DOMAIN="symthaea.rel.qualification-evidence-payload-set.v1"
FORBIDDEN_RESULT_FIELDS=("qualification_mapping","per_result_policy","result_specific_artifact_policy","selected_result_policy")
CANON={"sort_keys":True,"separators":[",",":"],"ensure_ascii":True}

def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:pathlib.Path)->dict[str,Any]:
    v=json.loads(p.read_text()); require(isinstance(v,dict),f"{p}: expected JSON object"); return v
def sha256(p:pathlib.Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()
def commitment_object(c:dict[str,Any],entries:list[dict[str,Any]])->dict[str,Any]:
    return {"domain":DOMAIN,"relation":c["relation"],"source_pipeline_head":c["source_pipeline_head"],"source_pipeline_tree":c["source_pipeline_tree"],"payload_files":entries}

def verify(contract_path:pathlib.Path,durable_dir:pathlib.Path)->dict[str,Any]:
    c=load(contract_path)
    require(c.get("schema")==CONTRACT_SCHEMA,"contract schema mismatch")
    require(c.get("authority")=="QualificationDurabilityContractOnly","contract authority mismatch")
    require(c.get("selected_comparison_result") is None and c.get("outcome_independent") is True,"outcome-blindness mismatch")
    require(tuple(c.get("forbidden_result_dependent_fields",[]))==FORBIDDEN_RESULT_FIELDS,"forbidden result-dependent policy declaration mismatch")
    for key in FORBIDDEN_RESULT_FIELDS:
        require(key not in c,f"forbidden result-dependent field present: {key}")
    spec=c["durable_qualification_evidence"]; commit_spec=spec["payload_commitment"]
    require(spec["persistence_receipt_verification_required_before_packaging"] is True,"verification requirement missing")
    require(spec["packaging_may_not_substitute_transport_digest_for_inner_verification"] is True,"transport substitution allowed")
    require(spec.get("runtime_tools_must_hardcode_forbidden_result_fields") is True,"runtime forbidden-field policy missing")
    require(spec.get("persistence_receipt_must_self_describe_commitment") is True,"receipt self-description requirement missing")
    require(commit_spec["domain_separator"]==DOMAIN and commit_spec["schema"]==DOMAIN,"commitment domain mismatch")
    require(commit_spec.get("algorithm")=="sha256","commitment algorithm mismatch")
    require(commit_spec.get("canonical_json")==CANON,"commitment canonicalization mismatch")
    payload=set(spec["payload_files"]); expected=payload|{RECEIPT_NAME}
    files=[p for p in durable_dir.rglob("*") if p.is_file()]
    rel={p.relative_to(durable_dir).as_posix() for p in files}
    require(rel==expected,f"durable artifact census mismatch: {sorted(rel)}")
    require(all(len(p.relative_to(durable_dir).parts)==1 for p in files),"nested durable artifact file forbidden")

    receipt=load(durable_dir/RECEIPT_NAME)
    require(receipt.get("schema")=="symthaea.rel.qualification-evidence-persistence-receipt.v3","persistence receipt schema mismatch")
    require(receipt.get("authority")=="QualificationEvidencePersistenceOnly","persistence receipt authority mismatch")
    require(receipt.get("relation")==c["relation"],"relation mismatch")
    require(receipt.get("source_pipeline_head")==c["source_pipeline_head"],"source pipeline head mismatch")
    require(receipt.get("source_pipeline_tree")==c["source_pipeline_tree"],"source pipeline tree mismatch")
    require(receipt.get("payload_commitment_schema")==DOMAIN,"payload commitment schema mismatch")
    require(receipt.get("payload_commitment_algorithm")=="sha256","payload commitment algorithm mismatch")
    require(receipt.get("payload_commitment_domain")==DOMAIN,"payload commitment domain mismatch")
    require(receipt.get("payload_commitment_canonicalization")==CANON,"payload commitment canonicalization mismatch")
    require(receipt.get("payload_file_count")==len(payload),"payload count mismatch")
    require(receipt.get("scientific_payload_parsed") is False and receipt.get("scientific_result_interpreted") is False,"persistence receipt exceeded authority")
    require(receipt.get("transport_identity_scientific_authority") is False,"transport promoted to scientific authority")
    require(not any(receipt.get("claims",{}).values()),"persistence receipt makes scientific/runtime claims")

    entries=receipt.get("payload_files"); require(isinstance(entries,list),"payload entries missing")
    names=[e.get("basename") for e in entries]
    require(names==sorted(payload,key=lambda s:s.encode()),"payload order/census mismatch")
    require(len(names)==len(set(names)),"duplicate payload basename")
    rebuilt=[]
    for entry in entries:
        name=entry["basename"]; path=durable_dir/name
        require(type(entry.get("byte_length")) is int and entry["byte_length"]==path.stat().st_size,f"{name}: byte length mismatch")
        digest=entry.get("sha256"); require(isinstance(digest,str) and digest==sha256(path),f"{name}: sha256 mismatch")
        rebuilt.append({"basename":name,"byte_length":path.stat().st_size,"sha256":digest})
    commitment=hashlib.sha256(canonical(commitment_object(c,rebuilt))).hexdigest()
    require(receipt.get("payload_set_commitment_sha256")==commitment,"payload-set commitment mismatch")
    return {"schema":"symthaea.rel.qualification-evidence-persistence-verification-receipt.v3","authority":"QualificationEvidencePersistenceVerificationOnly","source_pipeline_head":c["source_pipeline_head"],"source_pipeline_tree":c["source_pipeline_tree"],"payload_commitment_schema":DOMAIN,"payload_commitment_algorithm":"sha256","payload_commitment_domain":DOMAIN,"payload_commitment_canonicalization":CANON,"payload_file_count":len(payload),"payload_set_commitment_sha256":commitment,"exact_file_census_verified":True,"byte_lengths_verified":True,"sha256_commitments_verified":True,"source_identity_bound":True,"scientific_payload_parsed":False,"scientific_result_interpreted":False,"claims":{"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}}

def self_test()->dict[str,Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp); durable=root/"durable"; durable.mkdir(); cp=root/"contract.json"
        names=["predicate-contract-receipt.json","execution-v3-receipt.json","observation-seal-v3.json","comparison-only-qualification-receipt.json","qualification-input-manifest.json","qualification-assembly-receipt.json","qualification-only.json","post-qualification-firewall-receipt.json"]
        c={"schema":CONTRACT_SCHEMA,"authority":"QualificationDurabilityContractOnly","relation":"REL-005A","source_pipeline_head":"46"*20,"source_pipeline_tree":"d8"*20,"allowed_comparison_results":["ALL_PREDICATES_PASS","PREDICATE_FAILURES"],"selected_comparison_result":None,"outcome_independent":True,"forbidden_result_dependent_fields":list(FORBIDDEN_RESULT_FIELDS),"durable_qualification_evidence":{"payload_files":names,"persistence_receipt":RECEIPT_NAME,"artifact_file_count":9,"persistence_receipt_verification_required_before_packaging":True,"packaging_may_not_substitute_transport_digest_for_inner_verification":True,"runtime_tools_must_hardcode_forbidden_result_fields":True,"persistence_receipt_must_self_describe_commitment":True,"payload_commitment":{"schema":DOMAIN,"algorithm":"sha256","domain_separator":DOMAIN,"canonical_json":CANON}}}
        cp.write_text(json.dumps(c)+"\n")
        entries=[]
        for i,n in enumerate(sorted(names,key=lambda s:s.encode())):
            p=durable/n; p.write_bytes(f"{n}\nsynthetic-{i}\n".encode()); entries.append({"basename":n,"byte_length":p.stat().st_size,"sha256":sha256(p)})
        commitment=hashlib.sha256(canonical(commitment_object(c,entries))).hexdigest()
        receipt={"schema":"symthaea.rel.qualification-evidence-persistence-receipt.v3","authority":"QualificationEvidencePersistenceOnly","relation":"REL-005A","source_pipeline_head":c["source_pipeline_head"],"source_pipeline_tree":c["source_pipeline_tree"],"payload_commitment_schema":DOMAIN,"payload_commitment_algorithm":"sha256","payload_commitment_domain":DOMAIN,"payload_commitment_canonicalization":CANON,"payload_file_count":8,"payload_files":entries,"payload_set_commitment_sha256":commitment,"scientific_payload_parsed":False,"scientific_result_interpreted":False,"transport_identity_scientific_authority":False,"claims":{"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}}
        rp=durable/RECEIPT_NAME; rp.write_text(json.dumps(receipt,sort_keys=True)+"\n")
        r=verify(cp,durable); require(r["payload_set_commitment_sha256"]==commitment,"valid artifact rejected")
        victim=durable/names[0]; old=victim.read_bytes(); victim.write_bytes(old+b"tamper")
        try: verify(cp,durable)
        except ValueError as e: require("sha256 mismatch" in str(e) or "byte length mismatch" in str(e),"payload tamper rejection wrong")
        else: raise ValueError("tampered payload accepted")
        victim.write_bytes(old)
        receipt["payload_set_commitment_sha256"]="00"*32; rp.write_text(json.dumps(receipt)+"\n")
        try: verify(cp,durable)
        except ValueError as e: require("payload-set commitment mismatch" in str(e),"receipt tamper rejection wrong")
        else: raise ValueError("tampered receipt accepted")
    return {"schema":"symthaea.rel.qualification-evidence-persistence-verification-self-test.v3","authority":"QualificationEvidencePersistenceVerificationOnly","valid_artifact_accepted":True,"tampered_payload_rejected":True,"tampered_receipt_rejected":True,"source_identity_bound":True,"commitment_self_description_verified":True,"scientific_payload_parsed":False,"scientific_result_interpreted":False}

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=pathlib.Path); p.add_argument("--durable-dir",type=pathlib.Path); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: print(json.dumps(self_test(),indent=2,sort_keys=True)); return
    require(a.contract is not None and a.durable_dir is not None,"--contract and --durable-dir are required")
    print(json.dumps(verify(a.contract,a.durable_dir),indent=2,sort_keys=True))
if __name__=="__main__":main()
