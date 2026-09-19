#!/usr/bin/env python3
"""Build a content-addressed REL-005A durable qualification evidence receipt.

Authority: QualificationEvidencePersistenceOnly.
This layer hashes already-validated qualification evidence bytes. It does not
parse or reinterpret scientific results, predicates, metrics, or thresholds.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, tempfile
from typing import Any

SCHEMA = "symthaea.rel.qualification-durability-contract.v3"
AUTHORITY = "QualificationEvidencePersistenceOnly"
RECEIPT = "qualification-evidence-persistence-receipt.json"


def require(condition: bool, message: str) -> None:
    if not condition: raise ValueError(message)
def load(path: pathlib.Path) -> dict[str, Any]:
    value=json.loads(path.read_text()); require(isinstance(value,dict),f"{path}: expected JSON object"); return value
def sha256(path: pathlib.Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(value: Any) -> bytes: return json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True).encode()


def build(contract_path:pathlib.Path,input_dir:pathlib.Path,output:pathlib.Path)->dict[str,Any]:
    contract=load(contract_path)
    require(contract.get("schema")==SCHEMA,"contract schema mismatch")
    require(contract.get("authority")=="QualificationDurabilityContractOnly","contract authority mismatch")
    require(contract.get("selected_comparison_result") is None,"durability contract selected an outcome")
    require(contract.get("outcome_independent") is True,"durability contract is outcome-dependent")
    require(not any(contract.get("claims",{}).values()),"durability contract makes runtime/scientific claims")
    spec=contract["durable_qualification_evidence"]
    expected=set(spec["payload_files"])
    require(spec["persistence_receipt"]==RECEIPT,"persistence receipt name mismatch")
    require(spec["artifact_file_count"]==len(expected)+1,"artifact file count mismatch")
    require(spec["persistence_receipt_must_not_interpret_scientific_result"] is True,"persistence authority broadened")
    require(spec["persistence_receipt_verification_required_before_packaging"] is True,"verification-before-packaging missing")
    files=[p for p in input_dir.rglob("*") if p.is_file()]
    rel={p.relative_to(input_dir).as_posix() for p in files}
    require(rel==expected,f"payload census mismatch: {sorted(rel)}")
    require(all(len(p.relative_to(input_dir).parts)==1 for p in files),"nested payload file forbidden")
    entries=[]
    for name in sorted(expected,key=lambda s:s.encode()):
        path=input_dir/name
        entries.append({"basename":name,"byte_length":path.stat().st_size,"sha256":sha256(path)})
    commitment=hashlib.sha256(canonical(entries)).hexdigest()
    receipt={
        "schema":"symthaea.rel.qualification-evidence-persistence-receipt.v1",
        "authority":AUTHORITY,
        "relation":"REL-005A",
        "source_pipeline_head":contract["source_pipeline_head"],
        "source_pipeline_tree":contract["source_pipeline_tree"],
        "payload_file_count":len(entries),
        "payload_files":entries,
        "payload_set_commitment_sha256":commitment,
        "scientific_payload_parsed":False,
        "scientific_result_interpreted":False,
        "transport_identity_scientific_authority":False,
        "claims":{"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}
    }
    output.write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
    return receipt


def self_test()->dict[str,Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root=pathlib.Path(tmp); contract=root/"contract.json"; payload=root/"payload"; payload.mkdir()
        names=["predicate-contract-receipt.json","execution-v3-receipt.json","observation-seal-v3.json","comparison-only-qualification-receipt.json","qualification-input-manifest.json","qualification-assembly-receipt.json","qualification-only.json","post-qualification-firewall-receipt.json"]
        contract.write_text(json.dumps({
            "schema":SCHEMA,"authority":"QualificationDurabilityContractOnly","relation":"REL-005A","source_pipeline_head":"46"*20,"source_pipeline_tree":"d8"*20,
            "allowed_comparison_results":["ALL_PREDICATES_PASS","PREDICATE_FAILURES"],"selected_comparison_result":None,"outcome_independent":True,
            "durable_qualification_evidence":{"payload_files":names,"persistence_receipt":RECEIPT,"artifact_file_count":9,"persistence_receipt_must_not_interpret_scientific_result":True,"persistence_receipt_verification_required_before_packaging":True},
            "claims":{"durability_implemented":False,"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}
        },indent=2)+"\n")
        for i,name in enumerate(names): (payload/name).write_bytes(f"{name}\nsynthetic-{i}\n".encode())
        a=build(contract,payload,root/"a.json"); b=build(contract,payload,root/"b.json")
        require(a["payload_set_commitment_sha256"]==b["payload_set_commitment_sha256"],"commitment is nondeterministic")
        require(a["scientific_payload_parsed"] is False and a["scientific_result_interpreted"] is False,"persistence exceeded authority")
        (payload/names[-1]).unlink()
        try: build(contract,payload,root/"missing.json")
        except ValueError as exc: require("payload census mismatch" in str(exc),"missing-file rejection failed for wrong reason")
        else: raise ValueError("missing payload file was accepted")
    return {"schema":"symthaea.rel.qualification-evidence-persistence-self-test.v1","authority":AUTHORITY,"deterministic_commitment":True,"exact_payload_census":True,"missing_payload_rejected":True,"scientific_payload_parsed":False,"scientific_result_interpreted":False}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=pathlib.Path); p.add_argument("--input-dir",type=pathlib.Path); p.add_argument("--output",type=pathlib.Path); p.add_argument("--self-test",action="store_true"); args=p.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),indent=2,sort_keys=True)); return
    require(args.contract is not None and args.input_dir is not None and args.output is not None,"--contract, --input-dir, and --output are required")
    print(json.dumps(build(args.contract,args.input_dir,args.output),indent=2,sort_keys=True))

if __name__=="__main__": main()
