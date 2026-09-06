"""Closed-world provenance and manifest helpers for HCP-MMP1 Lineage B."""
from __future__ import annotations
import hashlib, json, re, subprocess
from pathlib import Path
from typing import Any

METHOD_SCHEMA='symthaea-hcpmmp1-neuromaps-method-v1'; RUN_SCHEMA='symthaea-hcpmmp1-neuromaps-run-v1'
AREA_SCHEMA='symthaea-hcp-mmp1-area-order-v1'; VERTICES=10242; AREAS=180
SHA256_RE=re.compile(r'^sha256:[0-9a-f]{64}$'); GIT_RE=re.compile(r'^[0-9a-f]{40}$'); MD5_RE=re.compile(r'^[0-9a-f]{32}$')
METHOD_KEYS={'schema','lineage_id','atlas','source_space','target_space','target_vertices_per_hemisphere','terms_reference','source_atlas','method_provenance','template_bundles','required_inputs','independence_contract'}
SOURCE_KEYS={'provider','scene_id','study_id','source_bytes_status','automatic_acquisition_permitted','hemisphere_pair_required','left_file_id','right_file_id','left_filename','right_filename'}
PROV_KEYS={'repository','commit','transforms_blob_sha','registry_blob_sha','atlas_fetcher_blob_sha','label_resample_method','area_correction','source_roi_required','target_mask_profile','implementation_family','license','citation_doi'}
BUNDLE_KEYS={'osf_project_id','osf_file_id','md5'}
INDEP_KEYS={'same_atlas_root_required','execution_independence_requires_external_proof','transform_method_distinct_from_mills','transform_implementation_family_independent','semantic_normalizer_independent','independence_established_by_this_manifest','external_provenance_review_required'}
RUN_KEYS={'schema','method_manifest_digest','execution_id','authorization_reference','workbench','inputs'}; WB_KEYS={'path','sha256','version_output_sha256'}; INPUT_KEYS={'path','sha256'}
AREA_KEYS={'schema','atlas','hemisphere_area_count','source','areas'}
REQUIRED_INPUT_ROLES={
 'hcp_left_dlabel','hcp_right_dlabel',
 'fslr32k_left_medialwall_roi','fslr32k_left_sphere_to_fsaverage','fslr32k_left_vaavg',
 'fslr32k_right_medialwall_roi','fslr32k_right_sphere_to_fsaverage','fslr32k_right_vaavg',
 'fsaverage10k_left_medialwall_roi','fsaverage10k_left_sphere','fsaverage10k_left_vaavg',
 'fsaverage10k_right_medialwall_roi','fsaverage10k_right_sphere','fsaverage10k_right_vaavg',
}
EXPECTED_SOURCE={
 'provider':'BALSA/Human Connectome Project','scene_id':'WN56','study_id':'RVVG',
 'source_bytes_status':'operator_pinned_required','automatic_acquisition_permitted':False,'hemisphere_pair_required':True,
 'left_file_id':'npz0','right_file_id':'pkN9',
 'left_filename':'Q1-Q6_RelatedParcellation210.L.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii',
 'right_filename':'Q1-Q6_RelatedParcellation210.R.CorticalAreas_dil_Colors.32k_fs_LR.dlabel.nii',
}

class ContractError(ValueError): pass

def canonical_json_bytes(v:Any)->bytes: return json.dumps(v,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest_bytes(b:bytes)->str: return 'sha256:'+hashlib.sha256(b).hexdigest()
def digest_file(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return 'sha256:'+h.hexdigest()
def load_json(p:Path)->Any:
 with p.open(encoding='utf-8') as f: return json.load(f)
def exact(v:Any,keys:set[str],ctx:str)->dict[str,Any]:
 if not isinstance(v,dict): raise ContractError(f'{ctx}: expected object')
 miss=sorted(keys-set(v)); unk=sorted(set(v)-keys)
 if miss: raise ContractError(f"{ctx}: missing fields: {', '.join(miss)}")
 if unk: raise ContractError(f"{ctx}: unknown fields: {', '.join(unk)}")
 return v
def nonempty(v:Any,ctx:str)->str:
 if not isinstance(v,str) or not v.strip(): raise ContractError(f'{ctx}: expected non-empty string')
 return v
def sha256(v:Any,ctx:str)->str:
 if not isinstance(v,str) or not SHA256_RE.fullmatch(v): raise ContractError(f'{ctx}: expected sha256:<64 lowercase hex>')
 return v

def load_method(p:Path)->dict[str,Any]:
 d=exact(load_json(p),METHOD_KEYS,'method manifest')
 if d['schema']!=METHOD_SCHEMA or d['atlas']!='HCP-MMP1.0/Glasser360': raise ContractError('method manifest: identity mismatch')
 if d['source_space']!='fsLR32k' or d['target_space']!='fsaverage5' or d['target_vertices_per_hemisphere']!=VERTICES: raise ContractError('method manifest: coordinate contract mismatch')
 nonempty(d['lineage_id'],'lineage_id'); nonempty(d['terms_reference'],'terms_reference')
 s=exact(d['source_atlas'],SOURCE_KEYS,'source_atlas')
 if s!=EXPECTED_SOURCE: raise ContractError('source_atlas: WN56/RVVG hemisphere-pair contract changed')
 q=exact(d['method_provenance'],PROV_KEYS,'method_provenance')
 for k in ('repository','label_resample_method','area_correction','target_mask_profile','implementation_family','license','citation_doi'): nonempty(q[k],f'provenance {k}')
 for k in ('commit','transforms_blob_sha','registry_blob_sha','atlas_fetcher_blob_sha'):
  if not isinstance(q[k],str) or not GIT_RE.fullmatch(q[k]): raise ContractError(f'provenance {k}: invalid git SHA')
 if q['label_resample_method']!='ADAP_BARY_AREA' or q['area_correction']!='average_vertex_area_metrics' or q['source_roi_required'] is not True or q['target_mask_profile']!='symthaea-positive-label-mask-v1': raise ContractError('method provenance: transform contract changed')
 for name,e in exact(d['template_bundles'],{'fsLR32k','fsaverage10k'},'template_bundles').items():
  e=exact(e,BUNDLE_KEYS,f'bundle {name}'); nonempty(e['osf_project_id'],'project'); nonempty(e['osf_file_id'],'file')
  if not isinstance(e['md5'],str) or not MD5_RE.fullmatch(e['md5']): raise ContractError(f'bundle {name}: invalid MD5')
 if not isinstance(d['required_inputs'],dict) or set(d['required_inputs'])!=REQUIRED_INPUT_ROLES: raise ContractError('required_inputs: exact v1 role set required')
 for r,desc in d['required_inputs'].items(): nonempty(r,'input role'); nonempty(desc,f'input {r}')
 want={'same_atlas_root_required':True,'execution_independence_requires_external_proof':True,'transform_method_distinct_from_mills':True,'transform_implementation_family_independent':False,'semantic_normalizer_independent':False,'independence_established_by_this_manifest':False,'external_provenance_review_required':True}
 if exact(d['independence_contract'],INDEP_KEYS,'independence_contract')!=want: raise ContractError('independence authority boundary changed')
 return d

def load_run(p:Path,method:dict[str,Any],method_path:Path)->dict[str,Any]:
 d=exact(load_json(p),RUN_KEYS,'run manifest')
 if d['schema']!=RUN_SCHEMA or d['method_manifest_digest']!=digest_file(method_path): raise ContractError('run manifest: identity mismatch')
 nonempty(d['execution_id'],'execution_id'); nonempty(d['authorization_reference'],'authorization_reference')
 w=exact(d['workbench'],WB_KEYS,'workbench'); nonempty(w['path'],'workbench path'); sha256(w['sha256'],'workbench sha'); sha256(w['version_output_sha256'],'version sha')
 if not isinstance(d['inputs'],dict) or set(d['inputs'])!=REQUIRED_INPUT_ROLES: raise ContractError('run inputs: exact v1 roles required')
 for r,e in d['inputs'].items(): e=exact(e,INPUT_KEYS,f'input {r}'); nonempty(e['path'],f'{r} path'); sha256(e['sha256'],f'{r} sha')
 if d['inputs']['hcp_left_dlabel']['sha256']==d['inputs']['hcp_right_dlabel']['sha256']: raise ContractError('run inputs: WN56 left/right source bytes must be distinct')
 return d

def verify_inputs(run:dict[str,Any])->tuple[Path,dict[str,Path],str]:
 wb=Path(run['workbench']['path']).expanduser().resolve(strict=True)
 if not wb.is_file() or digest_file(wb)!=run['workbench']['sha256']: raise ContractError('workbench byte identity mismatch')
 try: v=subprocess.run([str(wb),'-version'],check=True,capture_output=True)
 except (OSError,subprocess.CalledProcessError) as e: raise ContractError('failed pinned Workbench -version') from e
 vd=digest_bytes(v.stdout+v.stderr)
 if vd!=run['workbench']['version_output_sha256']: raise ContractError('workbench version-output digest mismatch')
 paths={}
 for r,e in run['inputs'].items():
  p=Path(e['path']).expanduser().resolve(strict=True)
  if not p.is_file() or digest_file(p)!=e['sha256']: raise ContractError(f'{r}: input byte identity mismatch')
  paths[r]=p
 return wb,paths,vd

def load_area_order(p:Path)->list[str]:
 d=exact(load_json(p),AREA_KEYS,'area order')
 if d['schema']!=AREA_SCHEMA or d['atlas']!='HCP-MMP1.0/Glasser360' or d['hemisphere_area_count']!=AREAS: raise ContractError('area order: identity mismatch')
 a=d['areas']
 if not isinstance(a,list) or len(a)!=AREAS or len(set(a))!=AREAS or any(not isinstance(x,str) or not x or x.startswith(('L_','R_')) for x in a): raise ContractError('area order: invalid namespace')
 return a
