#!/usr/bin/env python3
from __future__ import annotations
import argparse,importlib.metadata,os,platform,ssl,sys,tempfile
from pathlib import Path
from ll009ag_common import CE,E,cb,hf,lj,vp,wj,_env_capsule

def library_version(name:str)->str:
 if name=='openssl':return ssl.OPENSSL_VERSION
 if name in {'gdal','proj'}:
  try:import rasterio
  except Exception as e:raise CE(f'cannot probe {name}: rasterio import failed: {e}') from e
  attr='__gdal_version__' if name=='gdal' else '__proj_version__';value=getattr(rasterio,attr,None)
  if not isinstance(value,str) or not value:raise CE(f'rasterio does not expose {attr}')
  return value
 raise CE(f'no deterministic library probe registered for {name!r}')

def package_version(name:str)->str:
 try:value=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError as e:raise CE(f'required package not installed: {name}') from e
 if not value:raise CE(f'empty package version: {name}')
 return value

def capture(policy:dict,profile:str)->dict:
 contracts=policy['environment_contracts']
 if profile not in contracts:raise CE(f'profile not declared by policy: {profile}')
 c=contracts[profile];exe=Path(sys.executable).resolve(strict=True)
 runtime={'python':{'implementation':platform.python_implementation().lower(),'version':platform.python_version(),'executable_path':str(exe),'executable_sha256':hf(exe)},'platform':{'system':platform.system(),'release':platform.release(),'machine':platform.machine()},'packages':{name:package_version(name) for name in c['required_packages']},'libraries':{name:library_version(name) for name in c['required_libraries']},'environment':{name:os.environ.get(name) for name in c['required_environment_variables']}}
 out={'schema_version':E,'profile':profile,'runtime':runtime};_env_capsule(out,profile,c);return out

def write_once(path:Path,value:dict)->None:
 data=cb(value)
 if path.exists():
  if path.is_symlink() or not path.is_file():raise CE(f'output is not an ordinary file: {path}')
  if path.read_bytes()!=data:raise CE(f'refusing to overwrite differing environment capsule: {path}')
  return
 wj(path,value)

def selftest()->None:
 policy=vp({'schema_version':'ll009ag.site01-campaign-policy.v1','study_id':'synthetic','required_environment_profiles':['acquisition','gis','analysis'],'environment_contracts':{'acquisition':{'required_packages':[],'required_libraries':['openssl'],'required_environment_variables':['LL009AG_SYNTH_UNSET']},'gis':{'required_packages':[],'required_libraries':[],'required_environment_variables':[]},'analysis':{'required_packages':[],'required_libraries':[],'required_environment_variables':[]}},'protected_files':[],'stage_plan':[],'network_authorized_stage_ids':[],'classification_order':['descriptive_geometry'],'classification_ceiling':'descriptive_geometry','semantic_rules':{'descriptive_geometry':{'enabled':True,'requires_all_artifact_ids':[]}}})
 old=os.environ.pop('LL009AG_SYNTH_UNSET',None)
 try:
  a=capture(policy,'acquisition');assert a['runtime']['environment']['LL009AG_SYNTH_UNSET'] is None;assert a['runtime']['libraries']['openssl']==ssl.OPENSSL_VERSION;assert len(a['runtime']['python']['executable_sha256'])==64
  with tempfile.TemporaryDirectory(prefix='ll009ag-env-') as td:
   p=Path(td)/'capsule.json';write_once(p,a);write_once(p,a);changed=dict(a);changed['profile']='gis'
   try:write_once(p,changed);raise AssertionError('differing overwrite accepted')
   except CE:pass
 finally:
  if old is not None:os.environ['LL009AG_SYNTH_UNSET']=old
 print('LL-009AG environment capture self-test: PASS')

def main()->int:
 ap=argparse.ArgumentParser(description='Capture exact LL-009AG runtime identity; performs no network access');ap.add_argument('--self-test',action='store_true');ap.add_argument('--policy',type=Path);ap.add_argument('--profile',choices=['acquisition','gis','analysis']);ap.add_argument('--output',type=Path);a=ap.parse_args()
 try:
  if a.self_test:selftest();return 0
  if not a.policy or not a.profile or not a.output:ap.error('--policy, --profile and --output are required unless --self-test')
  policy=vp(lj(a.policy));out=capture(policy,a.profile);write_once(a.output,out);print('CAPTURED',a.profile,a.output);return 0
 except CE as e:print('LL-009AG ENV FAIL:',e,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
