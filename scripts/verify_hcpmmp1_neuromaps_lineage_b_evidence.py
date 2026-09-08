#!/usr/bin/env python3
"""Strict archival verifier for HCP-MMP1 neuromaps Lineage-B evidence bundles."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
from derive_hcpmmp1_neuromaps_lineage_b import (
    EVIDENCE_KEYS,EVIDENCE_SCHEMA,GENERATOR_ID,GENERATOR_VERSION,OUTPUT_KEYS,OUTPUT_SCHEMA,SOURCE_KEYS,VERTICES,commitment,validate_generator_implementation,
)
from hcpmmp_neuromaps_common import ContractError,canonical_json_bytes,digest_bytes,digest_file,exact,load_area_order,load_json,load_method,load_run,sha256
VERIFIER_PROFILE='symthaea-hcpmmp1-neuromaps-evidence-verifier-v1'
OUTPUT_DIGEST_KEYS={'left_semantic_sha256','right_semantic_sha256'};WORKBENCH_KEYS={'sha256','version_output_sha256'}
INDEPENDENCE_EXTRA={'independence_established':False,'status':'requires_external_provenance_review'}

def _validate_semantic_output(doc:Any,hemisphere:str,areas:list[str],evidence:dict[str,Any],method:dict[str,Any])->None:
 doc=exact(doc,OUTPUT_KEYS,f'{hemisphere} semantic output')
 if doc['schema']!=OUTPUT_SCHEMA or doc['space']!='fsaverage5' or doc['hemisphere']!=hemisphere or doc['vertex_count']!=VERTICES or not isinstance(doc['labels'],list) or len(doc['labels'])!=VERTICES:raise ContractError(f'{hemisphere}: semantic output identity mismatch')
 prefix='L_' if hemisphere=='left' else 'R_';allowed=set(areas);seen=set()
 for vertex,label in enumerate(doc['labels']):
  if label is None:continue
  if not isinstance(label,str) or not label.startswith(prefix):raise ContractError(f'{hemisphere}: invalid label at vertex {vertex}: {label!r}')
  base=label[len(prefix):]
  if base not in allowed:raise ContractError(f'{hemisphere}: unknown canonical area at vertex {vertex}: {label!r}')
  seen.add(base)
 if seen!=allowed:raise ContractError(f'{hemisphere}: incomplete canonical area coverage')
 source=exact(doc['source'],SOURCE_KEYS,f'{hemisphere} source')
 expected={'source_id':f"{method['lineage_id']}:{hemisphere}",'source_version':'v1','source_digest':evidence['scientific_input_commitment'],'generator_id':GENERATOR_ID,'generator_version':GENERATOR_VERSION,'generator_implementation_digest':evidence['generator_implementation']['digest'],'terms_reference':method['terms_reference']}
 if source!=expected:raise ContractError(f'{hemisphere}: source/evidence root mismatch')

def verify_bundle(output_dir:Path,method_manifest:Path,run_manifest:Path,area_order:Path,expected_content_digest:str)->dict[str,Any]:
 expected_content_digest=sha256(expected_content_digest,'expected external evidence root');method=load_method(method_manifest);run=load_run(run_manifest,method,method_manifest);areas=load_area_order(area_order)
 evidence=exact(load_json(output_dir/'derivation-evidence.json'),EVIDENCE_KEYS,'derivation evidence')
 if evidence['schema']!=EVIDENCE_SCHEMA:raise ContractError('evidence: wrong schema')
 if evidence['content_digest']!=expected_content_digest:raise ContractError('evidence: external retained root mismatch')
 stored=sha256(evidence['content_digest'],'content digest');payload=dict(evidence);del payload['content_digest']
 if stored!=digest_bytes(canonical_json_bytes(payload)):raise ContractError('evidence: content digest mismatch')
 expected_identity={'lineage_id':method['lineage_id'],'execution_id':run['execution_id'],'authorization_reference':run['authorization_reference'],'method_manifest_digest':digest_file(method_manifest),'run_manifest_digest':digest_file(run_manifest),'area_order_digest':digest_file(area_order)}
 for key,value in expected_identity.items():
  if evidence[key]!=value:raise ContractError(f'evidence: {key} mismatch')
 for key in ('method_manifest_digest','run_manifest_digest','area_order_digest','scientific_input_commitment'):sha256(evidence[key],f'evidence {key}')
 generator=validate_generator_implementation(evidence['generator_implementation'])
 expected_commitment=commitment(method_manifest,run,area_order,run['workbench']['version_output_sha256'],generator['digest'])
 if evidence['scientific_input_commitment']!=expected_commitment:raise ContractError('evidence: scientific input commitment mismatch')
 workbench=exact(evidence['workbench'],WORKBENCH_KEYS,'evidence workbench');expected_workbench={'sha256':run['workbench']['sha256'],'version_output_sha256':run['workbench']['version_output_sha256']}
 if workbench!=expected_workbench:raise ContractError('evidence: workbench root mismatch')
 outputs=exact(evidence['outputs'],OUTPUT_DIGEST_KEYS,'evidence outputs');left_path=output_dir/'left.semantic.json';right_path=output_dir/'right.semantic.json'
 if digest_file(left_path)!=outputs['left_semantic_sha256']:raise ContractError('evidence: left output digest mismatch')
 if digest_file(right_path)!=outputs['right_semantic_sha256']:raise ContractError('evidence: right output digest mismatch')
 _validate_semantic_output(load_json(left_path),'left',areas,evidence,method);_validate_semantic_output(load_json(right_path),'right',areas,evidence,method)
 expected_independence={**method['independence_contract'],**INDEPENDENCE_EXTRA};independence=exact(evidence['independence'],set(expected_independence),'evidence independence')
 if independence!=expected_independence:raise ContractError('evidence: independence authority mismatch')
 return evidence

def main(argv:list[str]|None=None)->int:
 p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--method-manifest',required=True,type=Path);p.add_argument('--run-manifest',required=True,type=Path);p.add_argument('--area-order',required=True,type=Path);p.add_argument('--expected-content-digest',required=True);a=p.parse_args(argv)
 try:r=verify_bundle(a.output_dir,a.method_manifest,a.run_manifest,a.area_order,a.expected_content_digest)
 except (ContractError,OSError,json.JSONDecodeError) as e:print(f'ERROR: {e}',file=sys.stderr);return 2
 print(json.dumps(r,sort_keys=True,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
