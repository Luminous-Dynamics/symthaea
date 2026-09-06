#!/usr/bin/env python3
"""Network-free candidate neuromaps-method HCP-MMP1 Lineage-B derivation."""
from __future__ import annotations
import argparse,json,subprocess,sys,tempfile
from pathlib import Path
from typing import Any
import hcpmmp_neuromaps_common as common_impl
import hcpmmp_neuromaps_gifti as gifti_impl
from hcpmmp_neuromaps_common import (
    AREA_SCHEMA,
    RUN_SCHEMA,
    VERTICES as VERTICES_PER_HEMISPHERE,
    ContractError,
    canonical_json_bytes,
    digest_bytes,
    digest_file,
    exact,
    load_area_order,
    load_json,
    load_method,
    load_run,
    sha256,
    verify_inputs,
)
from hcpmmp_neuromaps_gifti import normalize as normalize_label, parse_label_gifti, semantic

DerivationError = ContractError
load_method_manifest = load_method
load_run_manifest = load_run

OUTPUT_SCHEMA='symthaea-semantic-surface-labels-v1'; EVIDENCE_SCHEMA='symthaea-hcpmmp1-neuromaps-derivation-evidence-v1'; GENERATOR_ID='symthaea-hcpmmp1-neuromaps-lineage-b'; GENERATOR_VERSION='v1'; VERTICES=10242
OUTPUT_KEYS={'schema','space','hemisphere','vertex_count','labels','source'}
SOURCE_KEYS={'source_id','source_version','source_digest','generator_id','generator_version','generator_implementation_digest','terms_reference'}
EVIDENCE_KEYS={'schema','lineage_id','execution_id','authorization_reference','method_manifest_digest','run_manifest_digest','area_order_digest','scientific_input_commitment','generator_implementation','workbench','outputs','independence','content_digest'}
GENERATOR_IMPLEMENTATION_KEYS={'digest','files'}; GENERATOR_FILE_KEYS={'common','gifti','derive'}

def run_wb(wb:Path,args:list[str])->None:
 try:subprocess.run([str(wb),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 except subprocess.CalledProcessError as e:raise ContractError(f"Workbench failed: {' '.join(args[:2])}") from e

def generator_implementation()->dict[str,Any]:
 files={
  'common':digest_file(Path(common_impl.__file__).resolve()),
  'gifti':digest_file(Path(gifti_impl.__file__).resolve()),
  'derive':digest_file(Path(__file__).resolve()),
 }
 return {'files':files,'digest':digest_bytes(canonical_json_bytes(files))}

def validate_generator_implementation(value:Any)->dict[str,Any]:
 d=exact(value,GENERATOR_IMPLEMENTATION_KEYS,'generator implementation')
 files=exact(d['files'],GENERATOR_FILE_KEYS,'generator implementation files')
 for name in sorted(files):sha256(files[name],f'generator {name} sha')
 sha256(d['digest'],'generator implementation digest')
 if d['digest']!=digest_bytes(canonical_json_bytes(files)):raise ContractError('generator implementation: aggregate digest mismatch')
 return d

def commitment(method_path:Path,run:dict[str,Any],area_path:Path,version_digest:str,generator_digest:str|None=None)->str:
 if generator_digest is None:generator_digest=generator_implementation()['digest']
 sha256(generator_digest,'generator implementation digest')
 return digest_bytes(canonical_json_bytes({'method_manifest_digest':digest_file(method_path),'area_order_digest':digest_file(area_path),'generator_implementation_digest':generator_digest,'workbench_sha256':run['workbench']['sha256'],'workbench_version_output_sha256':version_digest,'inputs':{r:run['inputs'][r]['sha256'] for r in sorted(run['inputs'])}}))

def derive(method_path:Path,run_path:Path,area_path:Path,outdir:Path)->dict[str,Any]:
 method=load_method(method_path);run=load_run(run_path,method,method_path);wb,inputs,vd=verify_inputs(run);areas=load_area_order(area_path);impl=generator_implementation();root=commitment(method_path,run,area_path,vd,impl['digest']);outdir.mkdir(parents=True,exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='symthaea-hcpmmp-lineage-b-') as t:
  td=Path(t);l32,r32=td/'left.32k.label.gii',td/'right.32k.label.gii';l10,r10=td/'left.10k.label.gii',td/'right.10k.label.gii'
  run_wb(wb,['-cifti-separate',str(inputs['hcp_left_dlabel']),'COLUMN','-label','CORTEX_LEFT',str(l32)]);run_wb(wb,['-cifti-separate',str(inputs['hcp_right_dlabel']),'COLUMN','-label','CORTEX_RIGHT',str(r32)])
  for h,src,dst in (('left',l32,l10),('right',r32,r10)):
   run_wb(wb,['-label-resample',str(src),str(inputs[f'fslr32k_{h}_sphere_to_fsaverage']),str(inputs[f'fsaverage10k_{h}_sphere']),'ADAP_BARY_AREA',str(dst),'-area-metrics',str(inputs[f'fslr32k_{h}_vaavg']),str(inputs[f'fsaverage10k_{h}_vaavg']),'-current-roi',str(inputs[f'fslr32k_{h}_medialwall_roi'])])
  left=semantic(l10,inputs['fsaverage10k_left_medialwall_roi'],'left',areas,root,f"{method['lineage_id']}:left",method['terms_reference'],GENERATOR_ID,GENERATOR_VERSION,impl['digest']);right=semantic(r10,inputs['fsaverage10k_right_medialwall_roi'],'right',areas,root,f"{method['lineage_id']}:right",method['terms_reference'],GENERATOR_ID,GENERATOR_VERSION,impl['digest'])
  _,_,vd_after=verify_inputs(run)
  if vd_after!=vd:raise ContractError('execution inputs: Workbench version changed during derivation')
  if generator_implementation()!=impl:raise ContractError('generator implementation changed during derivation')
  lp,rp=outdir/'left.semantic.json',outdir/'right.semantic.json';lp.write_bytes(canonical_json_bytes(left)+b'\n');rp.write_bytes(canonical_json_bytes(right)+b'\n')
 ev={'schema':EVIDENCE_SCHEMA,'lineage_id':method['lineage_id'],'execution_id':run['execution_id'],'authorization_reference':run['authorization_reference'],'method_manifest_digest':digest_file(method_path),'run_manifest_digest':digest_file(run_path),'area_order_digest':digest_file(area_path),'scientific_input_commitment':root,'generator_implementation':impl,'workbench':{'sha256':run['workbench']['sha256'],'version_output_sha256':vd},'outputs':{'left_semantic_sha256':digest_file(lp),'right_semantic_sha256':digest_file(rp)},'independence':{**method['independence_contract'],'independence_established':False,'status':'requires_external_provenance_review'}};ev['content_digest']=digest_bytes(canonical_json_bytes(ev));(outdir/'derivation-evidence.json').write_bytes(canonical_json_bytes(ev)+b'\n');return ev

def _validate_output(d:Any,h:str,evidence:dict[str,Any])->None:
 d=exact(d,OUTPUT_KEYS,f'{h} output')
 if d['schema']!=OUTPUT_SCHEMA or d['space']!='fsaverage5' or d['hemisphere']!=h or d['vertex_count']!=VERTICES or not isinstance(d['labels'],list) or len(d['labels'])!=VERTICES:raise ContractError(f'{h}: output identity mismatch')
 s=exact(d['source'],SOURCE_KEYS,f'{h} source');sha256(s['source_digest'],f'{h} source digest');sha256(s['generator_implementation_digest'],f'{h} generator digest')
 if s['source_id']!=f"{evidence['lineage_id']}:{h}" or s['source_version']!='v1' or s['source_digest']!=evidence['scientific_input_commitment'] or s['generator_id']!=GENERATOR_ID or s['generator_version']!=GENERATOR_VERSION or s['generator_implementation_digest']!=evidence['generator_implementation']['digest'] or not isinstance(s['terms_reference'],str) or not s['terms_reference']:raise ContractError(f'{h}: output source provenance mismatch')

def validate_evidence(outdir:Path)->dict[str,Any]:
 d=exact(load_json(outdir/'derivation-evidence.json'),EVIDENCE_KEYS,'derivation evidence')
 if d['schema']!=EVIDENCE_SCHEMA:raise ContractError('evidence: wrong schema')
 for k in ('method_manifest_digest','run_manifest_digest','area_order_digest','scientific_input_commitment'):sha256(d[k],f'evidence {k}')
 validate_generator_implementation(d['generator_implementation'])
 if set(d['workbench'])!={'sha256','version_output_sha256'} or set(d['outputs'])!={'left_semantic_sha256','right_semantic_sha256'}:raise ContractError('evidence: nested schema mismatch')
 lp,rp=outdir/'left.semantic.json',outdir/'right.semantic.json'
 if digest_file(lp)!=d['outputs']['left_semantic_sha256'] or digest_file(rp)!=d['outputs']['right_semantic_sha256']:raise ContractError('evidence: output digest mismatch')
 _validate_output(load_json(lp),'left',d);_validate_output(load_json(rp),'right',d);i=d['independence']
 if not isinstance(i,dict) or i.get('independence_established') is not False or i.get('status')!='requires_external_provenance_review':raise ContractError('evidence: independence authority escalation')
 stored=sha256(d['content_digest'],'content digest');payload=dict(d);del payload['content_digest']
 if stored!=digest_bytes(canonical_json_bytes(payload)):raise ContractError('evidence: content digest mismatch')
 return d

def main(argv:list[str]|None=None)->int:
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True);d=sub.add_parser('derive')
 for name in ('method-manifest','run-manifest','area-order','output-dir'):d.add_argument('--'+name,required=True,type=Path)
 v=sub.add_parser('verify-evidence');v.add_argument('--output-dir',required=True,type=Path);a=p.parse_args(argv)
 try:r=derive(a.method_manifest,a.run_manifest,a.area_order,a.output_dir) if a.cmd=='derive' else validate_evidence(a.output_dir)
 except (ContractError,OSError,json.JSONDecodeError) as e:print(f'ERROR: {e}',file=sys.stderr);return 2
 print(json.dumps(r,sort_keys=True,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
