#!/usr/bin/env python3
from __future__ import annotations
import base64,copy,importlib.util,json
from pathlib import Path
HERE=Path(__file__).resolve().parent

def load(name,file):
 s=importlib.util.spec_from_file_location(name,HERE/file); assert s and s.loader
 m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
v=load('build_validator','validate-math-retrieval-index-build.py')
cov=load('coverage_validator','validate-math-retrieval-source-coverage.py')
ixv=load('index_validator','validate-math-retrieval-index.py')
def d(n):return f'sha256:{n:064x}'
def fixture():
 coverage,candidate,experiment=cov.fixture();target=next(t for t in coverage['targets'] if t['target_id']=='S');items=[]
 for i,row in enumerate(coverage['rows']):
  payload=f'canonical-sparse-fixture-{i}\n'.encode();h=v.digest_bytes(payload);r=next(x for x in row['representations'] if x['target_id']=='S');r['representation_object_sha256']=h;r['serialized_bytes']=len(payload)
  items.append({'source_object_sha256':row['source_object_sha256'],'representation_object_sha256':h,'serialized_bytes':len(payload),'payload_base64':base64.b64encode(payload).decode()})
 candidate_raw=v.canonical_bytes(candidate);coverage['candidate_universe']['candidate_set_sha256']=v.digest_bytes(candidate_raw);coverage_raw=v.canonical_bytes(coverage)
 artifact={'version':v.ARTIFACT_VERSION,'index_id':'fixture-index','authority':v.AUTHORITY,'candidate_set_sha256':v.digest_bytes(candidate_raw),'target_id':'S','representation_sha256':target['representation_sha256'],'item_serialization_sha256':target['item_serialization_sha256'],'payload_encoding':'Base64','item_order':'SourceObjectDigestAscending','item_count':len(items),'items':items};artifact_raw=v.canonical_bytes(artifact)
 index=ixv.fixture();index['index_id']=artifact['index_id'];cu=index['candidate_universe'];cu.update({'corpus_snapshot_sha256':coverage['candidate_universe']['corpus_snapshot_sha256'],'knowledge_boundary_sha256':coverage['candidate_universe']['knowledge_boundary_sha256'],'candidate_eligibility_policy_sha256':coverage['candidate_universe']['candidate_eligibility_policy_sha256'],'candidate_set_sha256':v.digest_bytes(candidate_raw),'candidate_count':len(items)})
 rep=index['representation'];rep.update({'channel':target['channel'],'representation_family':target['representation_family'],'representation_sha256':target['representation_sha256'],'item_serialization_sha256':target['item_serialization_sha256'],'max_serialized_item_bytes':target['max_serialized_item_bytes'],'oversize_policy':'RejectItem','control_transform':'None'})
 for k in ('normalization_contract_sha256','normalization_implementation_sha256','truncation_policy_sha256','control_seed','control_artifact_sha256'):rep.pop(k,None)
 index['index']['index_artifact_sha256']=v.digest_bytes(artifact_raw);index['index']['top_k_supported']=len(items);index['index']['search_mode']='ExactDeterministic';index['index'].pop('approximation_validation_policy_sha256',None);index['index'].pop('minimum_exact_recall_at_k',None);index_raw=v.canonical_bytes(index)
 receipt={'version':v.RECEIPT_VERSION,'receipt_id':'fixture-receipt','authority':v.AUTHORITY,'coverage_sha256':v.digest_bytes(coverage_raw),'candidate_set_sha256':v.digest_bytes(candidate_raw),'index_manifest_sha256':v.digest_bytes(index_raw),'index_artifact_sha256':v.digest_bytes(artifact_raw),'target_id':'S','index_build_policy_sha256':index['index']['index_build_policy_sha256'],'builder_implementation_sha256':d(7000),'toolchain_manifest_sha256':experiment['shared_contract']['toolchain_manifest_sha256'],'index_seed':index['index']['index_seed'],'input_set_sha256':v.input_set_sha(items),'build_mode':'ExactDeterministicMaterializedScan'}
 return receipt,coverage,candidate,experiment,index,artifact

def check(t):
 r,c,ca,e,ix,a=t;rr,cr,car,ir,ar=map(v.canonical_bytes,(r,c,ca,ix,a));return v.validate_bound(r,rr,c,cr,ca,car,e,ix,ir,a,ar)
base=fixture();check(base)
def mutate_payload(t):t[5]['items'][0]['payload_base64']=base64.b64encode(b'tampered\n').decode()
def omit(t):t[5]['items'].pop()
def reverse(t):t[5]['items'].reverse()
def rep_digest(t):t[5]['items'][0]['representation_object_sha256']=d(9990)
def coverage_sha(t):t[0]['coverage_sha256']=d(9991)
def target(t):t[0]['target_id']='H'
def policy(t):t[0]['index_build_policy_sha256']=d(9992)
def approximate(t):t[4]['index'].update(search_mode='ApproximateDeterministic',approximation_validation_policy_sha256=d(9993),minimum_exact_recall_at_k=.99)
def input_set(t):t[0]['input_set_sha256']=d(9994)
for attack in (mutate_payload,omit,reverse,rep_digest,coverage_sha,target,policy,approximate,input_set):
 t=[copy.deepcopy(x) for x in base];attack(t)
 try:check(t)
 except (v.ValidationError,ValueError):continue
 raise AssertionError(f'adversarial index-build self-test unexpectedly passed: {attack.__name__}')
print('math-retrieval index-build receipt v1 self-test: PASS')
