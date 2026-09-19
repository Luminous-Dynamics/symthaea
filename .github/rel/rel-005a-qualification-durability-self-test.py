#!/usr/bin/env python3
"""Static self-test for REL-005A QualificationDurabilityContractOnly v3."""
from __future__ import annotations
import argparse, json, pathlib
from typing import Any

SCHEMA="symthaea.rel.qualification-durability-contract.v3"
AUTHORITY="QualificationDurabilityContractOnly"
ALLOWED=["ALL_PREDICATES_PASS","PREDICATE_FAILURES"]
ORDER=["QualificationInputFirewallOnly","QualificationOnly","PostQualificationFirewallOnly","QualificationEvidencePersistenceOnly","QualificationEvidencePersistenceVerificationOnly","QualificationCapsulePackagingOnly","AttestationOnly"]
PAYLOAD={"predicate-contract-receipt.json","execution-v3-receipt.json","observation-seal-v3.json","comparison-only-qualification-receipt.json","qualification-input-manifest.json","qualification-assembly-receipt.json","qualification-only.json","post-qualification-firewall-receipt.json"}
RECEIPT="qualification-evidence-persistence-receipt.json"


def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:pathlib.Path)->dict[str,Any]:
    v=json.loads(p.read_text()); require(isinstance(v,dict),"contract must be a JSON object"); return v


def validate(c:dict[str,Any])->dict[str,Any]:
    require(c.get("schema")==SCHEMA,"schema mismatch")
    require(c.get("authority")==AUTHORITY,"authority mismatch")
    require(c.get("relation")=="REL-005A","relation mismatch")
    require(c.get("allowed_comparison_results")==ALLOWED,"allowed outcomes changed")
    require(c.get("selected_comparison_result") is None,"durability contract selected an outcome")
    require(c.get("outcome_independent") is True,"durability policy is not outcome-independent")
    require("qualification_mapping" not in c,"durability layer must not contain scientific mapping")
    require(c.get("authority_order")==ORDER,"authority order mismatch")
    jobs=c["job_boundaries"]
    require(jobs["qualification_job_ends_after_durable_evidence"] is True,"qualification job durability boundary missing")
    require(jobs["capsule_packaging_runs_in_fresh_downstream_job"] is True,"capsule job boundary missing")
    require(jobs["capsule_job_may_receive_only_durable_qualification_evidence"] is True,"capsule input boundary missing")
    require(jobs["capsule_job_must_verify_persistence_receipt_before_build"] is True,"persistence verification gate missing")
    require(jobs["attestation_runs_in_fresh_downstream_job"] is True and jobs["attestation_job_may_receive_only_final_capsule"] is True,"attestation boundary mismatch")
    e=c["durable_qualification_evidence"]
    require(e["exact_file_census"] is True,"exact census missing")
    require(set(e["payload_files"])==PAYLOAD and len(e["payload_files"])==len(PAYLOAD),"payload census mismatch")
    require(e["persistence_receipt"]==RECEIPT and e["artifact_file_count"]==9,"durable artifact census mismatch")
    for key in ("raw_observation_allowed","execution_logs_allowed","detailed_predicate_values_allowed","scientific_thresholds_allowed","transport_identity_is_scientific_authority"):
        require(e[key] is False,f"forbidden property enabled: {key}")
    for key in ("inner_content_commitments_are_authoritative","persistence_receipt_must_hash_every_payload_file","persistence_receipt_must_record_every_payload_byte_length","persistence_receipt_must_not_interpret_scientific_result","persistence_receipt_verification_required_before_packaging","packaging_may_not_substitute_transport_digest_for_inner_verification","upload_must_complete_before_capsule_packaging"):
        require(e[key] is True,f"required durability property missing: {key}")
    f=c["failure_semantics"]
    require(f["persistence_failure_before_durable_artifact"]=="QUALIFICATION_EVIDENCE_NOT_DURABLE","persistence failure semantics mismatch")
    require(f["persistence_verification_failure"]=="DURABLE_EVIDENCE_VERIFICATION_FAILED","verification failure semantics mismatch")
    require(f["capsule_failure_after_durable_artifact_mutates_qualification_result"] is False,"capsule failure may mutate qualification")
    require(f["attestation_failure_after_durable_artifact_mutates_qualification_result"] is False,"attestation failure may mutate qualification")
    require(f["workflow_conclusion_is_scientific_authority"] is False,"workflow conclusion promoted to scientific authority")
    require(f["downstream_packaging_may_not_rewrite_qualification_only_json"] is True,"packaging may rewrite qualification output")
    require(c["forbidden_result_dependent_durability_policy"] is True,"result-dependent durability not forbidden")
    require(not any(c["claims"].values()),"contract makes runtime/scientific claims")
    require(len({tuple(ORDER) for _ in ALLOWED})==1,"result-dependent authority topology")
    require(len({frozenset(PAYLOAD) for _ in ALLOWED})==1,"result-dependent payload census")
    return {"schema":"symthaea.rel.qualification-durability-contract-static-receipt.v3","authority":AUTHORITY,"self_tests_passed":True,"outcome_blind":True,"allowed_result_count":2,"payload_file_count":8,"durable_artifact_file_count":9,"qualification_mapping_present":False,"content_addressed_persistence_required":True,"independent_persistence_verification_required":True,"persistence_precedes_packaging":True,"packaging_failure_mutates_qualification":False,"attestation_failure_mutates_qualification":False,"claims":{"durability_implemented":False,"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}}


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=pathlib.Path,required=True); a=p.parse_args(); print(json.dumps(validate(load(a.contract)),indent=2,sort_keys=True))
if __name__=="__main__": main()
