#!/usr/bin/env python3
"""Verify a durable REL-005A qualification evidence artifact before packaging.

Authority: QualificationEvidencePersistenceVerificationOnly.
This verifier checks file census, byte lengths, SHA-256 values, and the payload
set commitment recorded by QualificationEvidencePersistenceOnly. It does not
parse or reinterpret scientific payload JSON.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, tempfile
from typing import Any

CONTRACT_SCHEMA="symthaea.rel.qualification-durability-contract.v3"
RECEIPT_NAME="qualification-evidence-persistence-receipt.json"


def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:pathlib.Path)->dict[str,Any]:
    v=json.loads(p.read_text()); require(isinstance(v,dict),f"{p}: expected JSON object"); return v
def sha256(p:pathlib.Path)->str:return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v:Any)->bytes:return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()


def verify(contract_path:pathlib.Path,durable_dir:pathlib.Path)->dict[str,Any]:
    contract=load(contract_path)
    require(contract.get("schema")==CONTRACT_SCHEMA,"contract schema mismatch")
    require(contract.get("authority")=="QualificationDurabilityContractOnly","contract authority mismatch")
    spec=contract["durable_qualification_evidence"]
    require(spec["persistence_receipt_verification_required_before_packaging"] is True,"verification requirement missing")
    require(spec["packaging_may_not_substitute_transport_digest_for_inner_verification"] is True,"transport substitution allowed")
    payload=set(spec["payload_files"]); expected=payload|{RECEIPT_NAME}
    files=[p for p in durable_dir.rglob("*") if p.is_file()]
    rel={p.relative_to(durable_dir).as_posix() for p in files}
    require(rel==expected,f"durable artifact census mismatch: {sorted(rel)}")
    require(all(len(p.relative_to(durable_dir).parts)==1 for p in files),"nested durable artifact file forbidden")

    receipt=load(durable_dir/RECEIPT_NAME)
    require(receipt.get("schema")=="symthaea.rel.qualification-evidence-persistence-receipt.v1","persistence receipt schema mismatch")
    require(receipt.get("authority")=="QualificationEvidencePersistenceOnly","persistence receipt authority mismatch")
    require(receipt.get("source_pipeline_head")==contract["source_pipeline_head"],"source pipeline head mismatch")
    require(receipt.get("source_pipeline_tree")==contract["source_pipeline_tree"],"source pipeline tree mismatch")
    require(receipt.get("payload_file_count")==len(payload),"payload count mismatch")
    require(receipt.get("scientific_payload_parsed") is False,"persistence receipt parsed science")
    require(receipt.get("scientific_result_interpreted") is False,"persistence receipt interpreted science")
    require(receipt.get("transport_identity_scientific_authority") is False,"transport promoted to scientific authority")
    require(not any(receipt.get("claims",{}).values()),"persistence receipt makes scientific/runtime claims")

    entries=receipt.get("payload_files")
    require(isinstance(entries,list),"payload entries missing")
    names=[e.get("basename") for e in entries]
    require(names==sorted(payload,key=lambda s:s.encode()),"payload order/census mismatch")
    require(len(names)==len(set(names)),"duplicate payload basename")
    rebuilt=[]
    for entry in entries:
        name=entry["basename"]; path=durable_dir/name
        require(type(entry.get("byte_length")) is int and entry["byte_length"]==path.stat().st_size,f"{name}: byte length mismatch")
        digest=entry.get("sha256")
        require(isinstance(digest,str) and len(digest)==64 and digest==sha256(path),f"{name}: sha256 mismatch")
        rebuilt.append({"basename":name,"byte_length":path.stat().st_size,"sha256":digest})
    commitment=hashlib.sha256(canonical(rebuilt)).hexdigest()
    require(receipt.get("payload_set_commitment_sha256")==commitment,"payload-set commitment mismatch")

    return {
        "schema":"symthaea.rel.qualification-evidence-persistence-verification-receipt.v1",
        "authority":"QualificationEvidencePersistenceVerificationOnly",
        "source_pipeline_head":contract["source_pipeline_head"],
        "payload_file_count":len(payload),
        "payload_set_commitment_sha256":commitment,
        "exact_file_census_verified":True,
        "byte_lengths_verified":True,
        "sha256_commitments_verified":True,
        "scientific_payload_parsed":False,
        "scientific_result_interpreted":False,
        "claims":{"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}
    }


def self_test()->dict[str,Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp); durable=root/"durable"; durable.mkdir(); contract=root/"contract.json"
        names=["predicate-contract-receipt.json","execution-v3-receipt.json","observation-seal-v3.json","comparison-only-qualification-receipt.json","qualification-input-manifest.json","qualification-assembly-receipt.json","qualification-only.json","post-qualification-firewall-receipt.json"]
        c={"schema":CONTRACT_SCHEMA,"authority":"QualificationDurabilityContractOnly","source_pipeline_head":"46"*20,"source_pipeline_tree":"d8"*20,"durable_qualification_evidence":{"payload_files":names,"persistence_receipt":RECEIPT_NAME,"artifact_file_count":9,"persistence_receipt_verification_required_before_packaging":True,"packaging_may_not_substitute_transport_digest_for_inner_verification":True}}
        contract.write_text(json.dumps(c)+"\n")
        entries=[]
        for i,name in enumerate(sorted(names,key=lambda s:s.encode())):
            p=durable/name; p.write_bytes(f"{name}\nsynthetic-{i}\n".encode()); entries.append({"basename":name,"byte_length":p.stat().st_size,"sha256":sha256(p)})
        commitment=hashlib.sha256(canonical(entries)).hexdigest()
        (durable/RECEIPT_NAME).write_text(json.dumps({"schema":"symthaea.rel.qualification-evidence-persistence-receipt.v1","authority":"QualificationEvidencePersistenceOnly","source_pipeline_head":c["source_pipeline_head"],"source_pipeline_tree":c["source_pipeline_tree"],"payload_file_count":8,"payload_files":entries,"payload_set_commitment_sha256":commitment,"scientific_payload_parsed":False,"scientific_result_interpreted":False,"transport_identity_scientific_authority":False,"claims":{"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}},sort_keys=True)+"\n")
        r=verify(contract,durable); require(r["payload_set_commitment_sha256"]==commitment,"verification commitment mismatch")
        victim=durable/names[0]; victim.write_bytes(victim.read_bytes()+b"tamper")
        try: verify(contract,durable)
        except ValueError as exc: require("sha256 mismatch" in str(exc) or "byte length mismatch" in str(exc),"tamper rejection failed for wrong reason")
        else: raise ValueError("tampered payload was accepted")
    return {"schema":"symthaea.rel.qualification-evidence-persistence-verification-self-test.v1","authority":"QualificationEvidencePersistenceVerificationOnly","valid_artifact_accepted":True,"tampered_payload_rejected":True,"scientific_payload_parsed":False,"scientific_result_interpreted":False}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=pathlib.Path); p.add_argument("--durable-dir",type=pathlib.Path); p.add_argument("--self-test",action="store_true"); args=p.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True)); return
    require(args.contract is not None and args.durable_dir is not None,"--contract and --durable-dir are required")
    print(json.dumps(verify(args.contract,args.durable_dir),indent=2,sort_keys=True))

if __name__=="__main__": main()
