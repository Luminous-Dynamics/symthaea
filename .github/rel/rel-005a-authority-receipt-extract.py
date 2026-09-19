#!/usr/bin/env python3
"""REL-005A AuthorityReceiptExtractionOnly helper.

May materialize sealed source bytes to verify commitments. It never parses
scientific observation fields and emits only authority receipts downstream.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, shutil
from typing import Any

PREDICATE_HEAD='43bf1d588447f602ce0f1549986bb558a839762c'
EXECUTION_HEAD='6931639e53060809f9d459196a329c4fe983be7b'
SEAL_HEAD='0a94ed976926fbdcd6b752f94e174df21f915fc8'
OBS_SHA='e271a5c2e6b51fda15cd6c3209e52859296f44ecd790b9c5e72f769d4aec1c4d'
SEAL_CHAIN='a3a288caa9221ec5859b5ac81fcc55f61b4ff43d5e52ad096021c99ea6bf31ab'
ASSEMBLY_HEAD='3dc01663ff8b2f9f6eb0568f8649ccdd999ddf29'
FIREWALL_HEAD='952ca3d7754a043f03da0875cf63b8665e63bf91'
QUALIFICATION_HEAD='452f0324a561a2872c00b6197237d4db46189f8d'
ATTESTATION_HEAD='2c81facb5b5ffb4e66428871b806f68622c9c188'
COMPARISON_HEAD='a47efccf80fa66bacbdfa9930f59c8439be7ee38'
PROJECTION_HEAD='07af7c4bdfdda377aa8175efe7a3233ce90e889e'


def require(c: bool, m: str) -> None:
    if not c: raise ValueError(m)

def load(p: pathlib.Path) -> dict[str, Any]:
    v=json.loads(p.read_text()); require(isinstance(v,dict),f'{p}: expected object'); return v

def sha(p: pathlib.Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(v: Any) -> bytes: return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()
def unique(root:pathlib.Path,name:str)->pathlib.Path:
    found=[p for p in root.rglob(name) if p.is_file()]; require(len(found)==1,f'{name}: census={len(found)}'); return found[0]

def validate_source(s: dict[str,Any]) -> None:
    require(s['schema']=='symthaea.rel.qualification-pipeline-source.v2','source schema')
    require(s['authority']=='PipelineOrchestrationOnly' and s['relation']=='REL-005A','source authority')
    require(s['comparison']['subject_head']==COMPARISON_HEAD and s['comparison']['source_run_id']==35349750595,'comparison binding')
    require(s['projection_contract']['subject_head']==PROJECTION_HEAD and s['projection_contract']['source_run_id']==35324485955,'projection binding')
    require(s['predicate_source']['subject_head']==PREDICATE_HEAD and s['predicate_source']['source_run_id']==35278498517,'predicate binding')
    require(s['seal_source']['subject_head']==SEAL_HEAD and s['seal_source']['source_run_id']==35284772522 and s['seal_source']['source_job_id']==105414583527,'seal binding')
    require(s['seal_source']['observation_sha256']==OBS_SHA and s['seal_source']['chain_commitment_sha256']==SEAL_CHAIN,'seal content binding')
    require(s['contracts']=={'assembly_head':ASSEMBLY_HEAD,'firewall_head':FIREWALL_HEAD,'qualification_head':QUALIFICATION_HEAD,'attestation_head':ATTESTATION_HEAD},'contract heads')
    require(s['allowed_comparison_results']==['ALL_PREDICATES_PASS','PREDICATE_FAILURES'] and s['selected_comparison_result'] is None,'outcome blindness')
    require(s['transport_identity_scientific_authority'] is False,'transport authority')
    require(not any(s['claims'].values()),'pipeline source claims')
    vis=s['job_visibility']
    require(vis['AuthorityReceiptExtractionOnly']=={'may_materialize_sealed_source_capsule':True,'may_receive_detailed_comparison':False,'output_raw_observation':False,'output_execution_logs':False,'scientific_observation_fields_parsed':False},'extractor visibility')
    require(vis['EvidenceProjectionOnly']=={'may_receive_authority_receipts':True,'may_receive_detailed_comparison':True,'may_materialize_sealed_source_capsule':False},'projection visibility')
    require(vis['QualificationAssemblyOnly']=={'may_receive_detailed_comparison':False,'may_receive_projected_metric_free_bundle':True},'assembly visibility')
    require(vis['QualificationOnly']=={'fresh_job':True,'may_receive_only_sanitized_six_file_bundle':True},'qualification visibility')
    require(vis['AttestationOnly']=={'may_receive_only_final_capsule':True},'attestation visibility')


def main()->None:
    ap=argparse.ArgumentParser(); ap.add_argument('--source',type=pathlib.Path,required=True); ap.add_argument('--predicate-root',type=pathlib.Path,required=True); ap.add_argument('--seal-root',type=pathlib.Path,required=True); ap.add_argument('--output-dir',type=pathlib.Path,required=True); a=ap.parse_args()
    source=load(a.source); validate_source(source)
    pp=unique(a.predicate_root,'predicate-contract-receipt.json'); ep=unique(a.seal_root,'execution-v3-receipt.json'); sp=unique(a.seal_root,'observation-seal-v3.json'); op=unique(a.seal_root,'rel-005a-execution-v3-observation.json'); mp=unique(a.seal_root,'execution-artifact-manifest.json')
    p,e,s=load(pp),load(ep),load(sp)
    require(p['schema']=='symthaea.rel.predicate-contract-static-receipt.v1' and p['authority']=='PredicateContractOnly','predicate schema/authority'); require(p['subject_head']==PREDICATE_HEAD and p['predicate_count']==41 and p['source_grounded'] is True and p['predicate_contract_static_valid'] is True,'predicate admissibility'); require(not any(p['claims'].values()),'predicate claims')
    require(e['schema']=='symthaea.rel.execution-only-receipt.v3' and e['authority']=='ExecutionOnly','execution schema/authority'); require(e['subject_head']==EXECUTION_HEAD and e['predicate_contract_parent']==PREDICATE_HEAD,'execution identity'); require(e['result']=='EXECUTION_OK' and str(e['measurement_exit_code'])=='0' and e['observation_present'] is True,'execution result'); require(not any(e['claims'].values()),'execution claims')
    obs=sha(op); require(obs==OBS_SHA and e['observation_sha256']==obs,'observation bytes')
    require(s['schema']=='symthaea.rel.observation-seal.v3' and s['authority']=='ObservationSeal','seal schema/authority'); require(s['observation_status']=='sealed' and s['adjudication']=='not_run' and s['scientific_result']=='not_run','seal boundary'); require(s['execution_subject_head']==EXECUTION_HEAD and s['predicate_contract_head']==PREDICATE_HEAD and s['observation_sha256']==obs,'seal identity'); require(s['chain_commitment_sha256']==SEAL_CHAIN,'seal chain frozen')
    require(s['claims']['observation_sealed'] is True,'seal claim'); require(all(s['claims'][k] is False for k in ('comparison_only_adjudicated','rel_005a_qualified','scientific_pass','scientific_fail')),'seal overclaim')
    chain=hashlib.sha256(canonical({'artifact_manifest_sha256':sha(mp),'execution_receipt_sha256':sha(ep),'execution_subject_head':EXECUTION_HEAD,'observation_sha256':obs})).hexdigest(); require(chain==SEAL_CHAIN,'recomputed seal chain')
    a.output_dir.mkdir(parents=True,exist_ok=True); outs={'predicate-contract-receipt.json':pp,'execution-v3-receipt.json':ep,'observation-seal-v3.json':sp}
    for n,path in outs.items(): shutil.copy2(path,a.output_dir/n)
    require({p.name for p in a.output_dir.iterdir() if p.is_file()}==set(outs),'output census')
    r={'schema':'symthaea.rel.authority-receipt-extraction-receipt.v2','authority':'AuthorityReceiptExtractionOnly','pipeline_source_sha256':sha(a.source),'observation_sha256_verified':obs,'seal_chain_commitment_sha256_verified':chain,'predicate_receipt_sha256':sha(a.output_dir/'predicate-contract-receipt.json'),'execution_receipt_sha256':sha(a.output_dir/'execution-v3-receipt.json'),'seal_receipt_sha256':sha(a.output_dir/'observation-seal-v3.json'),'raw_observation_output':False,'execution_logs_output':False,'scientific_observation_fields_parsed':False,'claims':{'comparison_only_adjudicated':False,'qualification_completed':False,'rel_005a_qualified':False,'scientific_pass':False,'scientific_fail':False}}
    print(json.dumps(r,indent=2,sort_keys=True))
if __name__=='__main__': main()
