#!/usr/bin/env python3
"""Adversarial static self-test for REL-005A QualificationDurabilityContractOnly v5."""
from __future__ import annotations
import argparse, copy, json, pathlib
from typing import Any

SCHEMA="symthaea.rel.qualification-durability-contract.v5"
AUTHORITY="QualificationDurabilityContractOnly"
ALLOWED=["ALL_PREDICATES_PASS","PREDICATE_FAILURES"]
ORDER=["QualificationInputFirewallOnly","QualificationOnly","PostQualificationFirewallOnly","QualificationEvidencePersistenceOnly","QualificationEvidencePersistenceVerificationOnly","QualificationCapsulePackagingOnly","AttestationOnly"]
PAYLOAD={"predicate-contract-receipt.json","execution-v3-receipt.json","observation-seal-v3.json","comparison-only-qualification-receipt.json","qualification-input-manifest.json","qualification-assembly-receipt.json","qualification-only.json","post-qualification-firewall-receipt.json"}
RECEIPT="qualification-evidence-persistence-receipt.json"
DOMAIN="symthaea.rel.qualification-evidence-payload-set.v1"
FORBIDDEN_RESULT_FIELDS=("qualification_mapping","per_result_policy","result_specific_artifact_policy","selected_result_policy")
CANON={"sort_keys":True,"separators":[",",":"],"ensure_ascii":True}

def require(c:bool,m:str)->None:
    if not c: raise ValueError(m)
def load(p:pathlib.Path)->dict[str,Any]:
    v=json.loads(p.read_text()); require(isinstance(v,dict),"contract must be object"); return v

def validate(c:dict[str,Any])->dict[str,Any]:
    require(c.get("schema")==SCHEMA,"schema mismatch"); require(c.get("authority")==AUTHORITY,"authority mismatch"); require(c.get("relation")=="REL-005A","relation mismatch")
    require(c.get("allowed_comparison_results")==ALLOWED,"allowed outcomes changed"); require(c.get("selected_comparison_result") is None,"durability contract selected an outcome")
    require(c.get("outcome_independent") is True,"durability policy not outcome-independent"); require(c.get("authority_order")==ORDER,"authority order mismatch")
    forbidden=c.get("forbidden_result_dependent_fields")
    require(tuple(forbidden or [])==FORBIDDEN_RESULT_FIELDS,"forbidden result-dependent fields changed")
    for key in FORBIDDEN_RESULT_FIELDS: require(key not in c,f"forbidden result-dependent field present: {key}")
    j=c["job_boundaries"]
    for key in ("qualification_job_ends_after_durable_evidence","capsule_packaging_runs_in_fresh_downstream_job","capsule_job_may_receive_only_durable_qualification_evidence","capsule_job_must_verify_persistence_receipt_before_build","attestation_runs_in_fresh_downstream_job","attestation_job_may_receive_only_final_capsule"):
        require(j.get(key) is True,f"job boundary missing: {key}")
    e=c["durable_qualification_evidence"]
    require(e["exact_file_census"] is True,"exact census missing"); require(set(e["payload_files"])==PAYLOAD and len(e["payload_files"])==len(PAYLOAD),"payload census mismatch")
    require(e["persistence_receipt"]==RECEIPT and e["artifact_file_count"]==9,"durable artifact census mismatch")
    for key in ("raw_observation_allowed","execution_logs_allowed","detailed_predicate_values_allowed","scientific_thresholds_allowed","transport_identity_is_scientific_authority"):
        require(e[key] is False,f"forbidden property enabled: {key}")
    for key in ("inner_content_commitments_are_authoritative","persistence_receipt_must_hash_every_payload_file","persistence_receipt_must_record_every_payload_byte_length","persistence_receipt_must_not_interpret_scientific_result","persistence_receipt_verification_required_before_packaging","packaging_may_not_substitute_transport_digest_for_inner_verification","upload_must_complete_before_capsule_packaging"):
        require(e[key] is True,f"required durability property missing: {key}")
    pc=e["payload_commitment"]
    require(pc=={"schema":DOMAIN,"algorithm":"sha256","domain_separator":DOMAIN,"canonical_json":CANON,"bind_relation":True,"bind_source_pipeline_head":True,"bind_source_pipeline_tree":True,"bind_ordered_payload_entries":True},"payload commitment contract mismatch")
    require(e.get("runtime_tools_must_hardcode_forbidden_result_fields") is True,"runtime forbidden-field policy missing")
    require(e.get("persistence_receipt_must_self_describe_commitment") is True,"receipt self-description requirement missing")
    f=c["failure_semantics"]
    require(f["persistence_failure_before_durable_artifact"]=="QUALIFICATION_EVIDENCE_NOT_DURABLE","persistence failure semantics mismatch")
    require(f["persistence_verification_failure"]=="DURABLE_EVIDENCE_VERIFICATION_FAILED","verification failure semantics mismatch")
    require(f["capsule_failure_after_durable_artifact_mutates_qualification_result"] is False and f["attestation_failure_after_durable_artifact_mutates_qualification_result"] is False,"downstream failure mutates qualification")
    require(f["workflow_conclusion_is_scientific_authority"] is False,"workflow conclusion promoted"); require(f["downstream_packaging_may_not_rewrite_qualification_only_json"] is True,"packaging may rewrite qualification")
    require(c["forbidden_result_dependent_durability_policy"] is True,"result-dependent durability not forbidden"); require(not any(c["claims"].values()),"contract makes runtime/scientific claims")
    return {"schema":"symthaea.rel.qualification-durability-contract-static-receipt.v5","authority":AUTHORITY,"self_tests_passed":True,"outcome_blind":True,"allowed_result_count":2,"payload_file_count":8,"durable_artifact_file_count":9,"qualification_mapping_present":False,"domain_separated_content_commitment_required":True,"runtime_policy_hardcoded":True,"persistence_receipt_self_describing":True,"content_addressed_persistence_required":True,"independent_persistence_verification_required":True,"persistence_precedes_packaging":True,"packaging_failure_mutates_qualification":False,"attestation_failure_mutates_qualification":False,"claims":{"durability_implemented":False,"qualification_completed":False,"rel_005a_qualified":False,"scientific_pass":False,"scientific_fail":False}}

def must_reject(base:dict[str,Any], mutate, expected:str)->None:
    x=copy.deepcopy(base); mutate(x)
    try: validate(x)
    except ValueError as e: require(expected in str(e),f"mutation rejected for wrong reason: {e}")
    else: raise ValueError(f"mutation accepted: {expected}")

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--contract",type=pathlib.Path,required=True); a=p.parse_args(); c=load(a.contract); receipt=validate(c)
    must_reject(c,lambda x:x.__setitem__("selected_comparison_result","ALL_PREDICATES_PASS"),"selected an outcome")
    must_reject(c,lambda x:x.__setitem__("allowed_comparison_results",["ALL_PREDICATES_PASS"]),"allowed outcomes changed")
    must_reject(c,lambda x:x.__setitem__("qualification_mapping",{"ALL_PREDICATES_PASS":True}),"forbidden result-dependent field present")
    must_reject(c,lambda x:x.__setitem__("per_result_policy",{"PREDICATE_FAILURES":{"retain":False}}),"forbidden result-dependent field present")
    must_reject(c,lambda x:x["durable_qualification_evidence"]["payload_commitment"].__setitem__("bind_source_pipeline_head",False),"payload commitment contract mismatch")
    must_reject(c,lambda x:x["job_boundaries"].__setitem__("capsule_job_must_verify_persistence_receipt_before_build",False),"job boundary missing")
    must_reject(c,lambda x:x.__setitem__("forbidden_result_dependent_fields",["per_result_policy"]),"forbidden result-dependent fields changed")
    must_reject(c,lambda x:x["durable_qualification_evidence"]["payload_commitment"]["canonical_json"].__setitem__("sort_keys",False),"payload commitment contract mismatch")
    receipt["adversarial_mutations_rejected"]=8
    print(json.dumps(receipt,indent=2,sort_keys=True))
if __name__=="__main__":main()
