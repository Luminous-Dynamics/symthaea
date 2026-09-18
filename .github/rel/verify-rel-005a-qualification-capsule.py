#!/usr/bin/env python3
"""Offline verifier for transparent REL-005A qualification capsule v2.

No network dependency. Verifies deterministic tar structure, manifest bytes,
metric-free authority-chain consistency, audit receipt hash links, and final
PASS/FAIL mapping. It does not independently reproduce the experiment.
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, re, tarfile
from typing import Any

SHA=re.compile(r'^[0-9a-f]{64}$'); GIT=re.compile(r'^[0-9a-f]{40}$')
INPUT={
 'predicate-contract-receipt.json','execution-v3-receipt.json','observation-seal-v3.json',
 'comparison-only-qualification-receipt.json','qualification-input-manifest.json',
 'qualification-assembly-receipt.json','qualification-only.json',
 'sealed-manifest-replay-receipt.json','authority-receipt-extraction.json'}
MANIFEST='qualification-capsule-manifest.json'; ALL=INPUT|{MANIFEST}
FORBIDDEN={'predicates','threshold_table','observed','expected'}


def req(c:bool,m:str)->None:
    if not c: raise ValueError(m)

def nodup(pairs:list[tuple[str,Any]])->dict[str,Any]:
    d={}
    for k,v in pairs: req(k not in d,f'duplicate JSON key: {k}'); d[k]=v
    return d

def parse(data:bytes,name:str)->dict[str,Any]:
    v=json.loads(data.decode(),object_pairs_hook=nodup); req(isinstance(v,dict),f'{name}: expected object'); return v

def digest(data:bytes)->str:return hashlib.sha256(data).hexdigest()
def keys(v:Any)->set[str]:
    out=set()
    if isinstance(v,dict):
        for k,x in v.items(): out.add(k); out|=keys(x)
    elif isinstance(v,list):
        for x in v: out|=keys(x)
    return out

def sha(v:Any)->bool:return isinstance(v,str) and SHA.fullmatch(v) is not None
def git(v:Any)->bool:return isinstance(v,str) and GIT.fullmatch(v) is not None


def read_tar(path:pathlib.Path)->dict[str,bytes]:
    req(path.is_file(),'capsule missing')
    out={}
    with tarfile.open(path,'r:') as tf:
        members=tf.getmembers(); names=[m.name for m in members]
        req(len(names)==len(set(names)),'duplicate tar member')
        req(set(names)==ALL,f'tar census mismatch: {sorted(names)}')
        req(names==sorted(names,key=lambda s:s.encode()),'tar member order not canonical')
        for m in members:
            req(m.isfile(),f'{m.name}: non-file member')
            req(m.mtime==0 and m.uid==0 and m.gid==0 and m.uname=='' and m.gname=='',f'{m.name}: metadata mismatch')
            req(m.mode==0o644,f'{m.name}: mode mismatch')
            f=tf.extractfile(m); req(f is not None,f'{m.name}: unreadable'); out[m.name]=f.read()
    return out


def verify(path:pathlib.Path)->dict[str,Any]:
    raw=read_tar(path); docs={n:parse(raw[n],n) for n in ALL}
    m=docs[MANIFEST]
    req(m.get('schema')=='symthaea.rel.qualification-capsule-manifest.v2','capsule manifest schema')
    req(m.get('authority')=='AttestationInputOnly','capsule manifest authority')
    req(m.get('audit_receipts_included')==['sealed-manifest-replay-receipt.json','authority-receipt-extraction.json'],'audit census declaration')
    req(m.get('raw_observation_included') is False and m.get('execution_logs_included') is False and m.get('detailed_predicate_values_included') is False,'capsule leakage declaration')
    entries=m.get('files'); req(isinstance(entries,list),'manifest files invalid')
    req([x.get('basename') for x in entries]==sorted(INPUT,key=lambda s:s.encode()),'manifest member order/census')
    for e in entries:
        n=e['basename']; req(e.get('byte_length')==len(raw[n]),f'{n}: length mismatch'); req(e.get('sha256')==digest(raw[n]),f'{n}: hash mismatch')

    p=docs['predicate-contract-receipt.json']; e=docs['execution-v3-receipt.json']; s=docs['observation-seal-v3.json']; c=docs['comparison-only-qualification-receipt.json']; a=docs['qualification-assembly-receipt.json']; q=docs['qualification-only.json']; r=docs['sealed-manifest-replay-receipt.json']; x=docs['authority-receipt-extraction.json']
    req(p.get('authority')=='PredicateContractOnly' and p.get('predicate_count')==41,'predicate receipt')
    req(e.get('authority')=='ExecutionOnly' and e.get('result')=='EXECUTION_OK' and str(e.get('measurement_exit_code'))=='0','execution receipt')
    req(s.get('authority')=='ObservationSeal' and s.get('observation_status')=='sealed','seal receipt')
    req(c.get('authority')=='ComparisonOnly' and c.get('predicate_count')==41,'comparison receipt')
    req(a.get('authority')=='QualificationAssemblyOnly','assembly receipt')
    req(q.get('authority')=='QualificationOnly' and q.get('qualification_completed') is True,'qualification output')
    req(r.get('schema')=='symthaea.rel.sealed-manifest-replay-receipt.v1' and r.get('authority')=='SealedManifestReplayOnly','manifest replay receipt')
    req(r.get('manifest_entry_count')==10 and r.get('exact_file_census_verified') is True and r.get('byte_lengths_verified') is True and r.get('sha256_commitments_verified') is True,'manifest replay incomplete')
    req(r.get('scientific_observation_fields_parsed') is False and r.get('execution_logs_parsed') is False,'manifest replay exceeded authority')
    req(x.get('schema')=='symthaea.rel.authority-receipt-extraction-receipt.v2' and x.get('authority')=='AuthorityReceiptExtractionOnly','extraction receipt')
    req(x.get('raw_observation_output') is False and x.get('execution_logs_output') is False and x.get('scientific_observation_fields_parsed') is False,'extraction exceeded authority')

    ph=p.get('subject_head'); eh=e.get('subject_head'); obs=e.get('observation_sha256'); req(git(ph) and git(eh) and sha(obs),'identity format')
    req(e.get('predicate_contract_parent')==ph,'execution predicate binding')
    req(s.get('predicate_contract_head')==ph and s.get('execution_subject_head')==eh and s.get('observation_sha256')==obs,'seal bindings')
    req(c.get('predicate_contract_head')==ph and c.get('execution_subject_head')==eh and c.get('observation_sha256')==obs,'comparison bindings')
    req(c.get('seal_chain_commitment_sha256')==s.get('chain_commitment_sha256'),'comparison/seal commitment')
    req(x.get('observation_sha256_verified')==obs and x.get('seal_chain_commitment_sha256_verified')==s.get('chain_commitment_sha256'),'extraction content bindings')
    req(a.get('authority_extraction_receipt_sha256')==digest(raw['authority-receipt-extraction.json']),'assembly extraction hash')
    req(a.get('sealed_manifest_replay_receipt_sha256')==digest(raw['sealed-manifest-replay-receipt.json']),'assembly replay hash')
    req(a.get('inner_content_commitments_verified') is True and a.get('transport_identity_scientific_authority') is False,'assembly semantics')

    leaked=set()
    for n in INPUT: leaked |= FORBIDDEN & keys(docs[n])
    req(not leaked,f'forbidden detailed keys leaked: {sorted(leaked)}')

    count=c.get('predicate_count'); passed=c.get('passed_count'); failed=c.get('failed_count'); ids=c.get('failed_predicate_ids'); result=c.get('comparison_result')
    req(type(count) is int and count==41 and type(passed) is int and type(failed) is int and passed+failed==41,'comparison counts')
    req(isinstance(ids,list) and len(ids)==failed and len(ids)==len(set(ids)),'failed IDs')
    if result=='ALL_PREDICATES_PASS': req(passed==41 and failed==0,'all-pass counts')
    elif result=='PREDICATE_FAILURES': req(failed>0,'failure without failed predicates')
    else: raise ValueError('unsupported comparison result')
    for k in ('comparison_result','predicate_count','passed_count','failed_count','failed_predicate_ids'): req(q.get(k)==c.get(k),f'qualification {k} mismatch')
    sp=q.get('scientific_pass') is True; sf=q.get('scientific_fail') is True; req(sp ^ sf,'scientific PASS/FAIL not XOR')
    if sp: req(result=='ALL_PREDICATES_PASS' and q.get('rel_005a_qualified') is True,'PASS mapping invalid')
    else: req(result=='PREDICATE_FAILURES' and q.get('rel_005a_qualified') is False,'FAIL mapping invalid')

    return {
      'schema':'symthaea.rel.offline-qualification-capsule-verification.v2',
      'authority':'OfflineVerificationOnly',
      'capsule_sha256':digest(path.read_bytes()),
      'member_count':len(ALL),
      'manifest_hashes_verified':True,
      'audit_receipt_hash_links_verified':True,
      'metric_leakage_absent':True,
      'authority_chain_consistent':True,
      'comparison_result':result,
      'qualification_mapping_consistent':True,
      'does_not_establish':['independent replication','predicate sufficiency','scientific truth beyond the represented authority chain','Sigstore provenance'],
    }


def main()->None:
    p=argparse.ArgumentParser(); p.add_argument('capsule',type=pathlib.Path); a=p.parse_args(); print(json.dumps(verify(a.capsule),indent=2,sort_keys=True))
if __name__=='__main__': main()
