from __future__ import annotations
import hashlib,json,os,stat,tempfile
from pathlib import Path
from typing import Any
P='ll009ag.site01-campaign-policy.v1';E='ll009ag.environment-capsule.v1';M='ll009ag.evidence-manifest.v1';A='ll009ag.semantic-assertion.v1'
PRE='ll009ag.campaign-preparation-receipt.v1';FRZ='ll009ag.campaign-freeze-receipt.v1';FIN='ll009ag.campaign-finalization-receipt.v1'
MODES={'none','canonical_receipt_sha256'}
class CE(RuntimeError):pass
def cb(v:Any)->bytes:return (json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False)+'\n').encode()
def hb(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def hf(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for c in iter(lambda:f.read(1<<20),b''):h.update(c)
 return h.hexdigest()
def lj(p:Path)->Any:
 try:return json.loads(p.read_text(encoding='utf-8'))
 except Exception as e:raise CE(f'cannot read JSON {p}: {e}') from e
def wj(p:Path,v:Any)->None:
 p.parent.mkdir(parents=True,exist_ok=True);data=cb(v)
 with tempfile.NamedTemporaryFile(dir=p.parent,prefix='.'+p.name+'.',delete=False) as f:t=Path(f.name);f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
def receipt(v:dict[str,Any])->dict[str,Any]:
 if 'receipt_sha256' in v:raise CE('payload already has receipt_sha256')
 o=dict(v);o['receipt_sha256']=hb(cb(v));return o
def vr(v:Any,schema:str|None=None)->str:
 if not isinstance(v,dict):raise CE('receipt must be object')
 if schema and v.get('schema_version')!=schema:raise CE(f'expected {schema}, got {v.get("schema_version")!r}')
 got=v.get('receipt_sha256');u=dict(v);u.pop('receipt_sha256',None);want=hb(cb(u))
 if got!=want:raise CE(f'receipt self-hash mismatch: expected {want}, got {got}')
 return want
def req(v:Any,label:str)->str:
 if not isinstance(v,str) or not v:raise CE(f'{label} must be non-empty string')
 return v
def sha(v:Any,label:str)->str:
 v=req(v,label)
 if len(v)!=64:raise CE(f'{label} must be SHA-256')
 try:int(v,16)
 except ValueError as e:raise CE(f'{label} must be hex') from e
 return v.lower()
def rel(v:Any,label:str)->str:
 v=req(v,label)
 if '\\' in v or v.startswith('/') or any(x in ('','.','..') for x in v.split('/')):raise CE(f'unsafe {label}: {v!r}')
 return v
def file(root:Path,r:str,label:str)->Path:
 r=rel(r,label);root=root.resolve();p=root
 for x in r.split('/'):
  p=p/x
  if p.is_symlink():raise CE(f'{label} traverses symlink: {p}')
 try:q=p.resolve(strict=True)
 except FileNotFoundError as e:raise CE(f'{label} missing: {p}') from e
 try:q.relative_to(root)
 except ValueError as e:raise CE(f'{label} escapes root') from e
 if not stat.S_ISREG(q.stat().st_mode):raise CE(f'{label} not regular file')
 return q
def closed(root:Path)->set[str]:
 root=root.resolve(strict=True)
 if root.is_symlink() or not root.is_dir():raise CE('evidence root must be ordinary directory')
 out=set()
 for dp,dn,fn in os.walk(root,followlinks=False):
  d=Path(dp)
  for n in dn:
   p=d/n
   if p.is_symlink() or not stat.S_ISDIR(p.stat().st_mode):raise CE(f'symlink/special directory: {p}')
  for n in fn:
   p=d/n
   if p.is_symlink() or not stat.S_ISREG(p.stat().st_mode):raise CE(f'symlink/special evidence: {p}')
   out.add(p.relative_to(root).as_posix())
 return out
def vp(v:Any)->dict[str,Any]:
 if not isinstance(v,dict) or v.get('schema_version')!=P:raise CE(f'policy schema must be {P}')
 req(v.get('study_id'),'study_id');prof=v.get('required_environment_profiles')
 if prof!=['acquisition','gis','analysis']:raise CE('environment profiles must be exactly acquisition,gis,analysis')
 pf=v.get('protected_files')
 if not isinstance(pf,list) or len(pf)!=len(set(pf)):raise CE('protected_files must be unique list')
 [rel(x,'protected file') for x in pf]
 stages=v.get('stage_plan');auth=v.get('network_authorized_stage_ids')
 if not isinstance(stages,list) or not isinstance(auth,list):raise CE('stage plan/network authority must be lists')
 ids=[]
 for x in stages:
  if not isinstance(x,dict):raise CE('stage must be object')
  i=req(x.get('id'),'stage id');ids.append(i)
  if x.get('environment_profile') not in prof or not isinstance(x.get('may_access_network'),bool):raise CE(f'invalid stage {i}')
  if x['may_access_network']!=(i in auth):raise CE(f'network authority mismatch for {i}')
 if len(ids)!=len(set(ids)):raise CE('duplicate stage id')
 order=v.get('classification_order');ceil=v.get('classification_ceiling');rules=v.get('semantic_rules')
 if not isinstance(order,list) or not order or len(order)!=len(set(order)) or ceil not in order or not isinstance(rules,dict) or set(rules)!=set(order):raise CE('invalid classification policy')
 for c in order:
  r=rules[c]
  if not isinstance(r,dict) or not isinstance(r.get('enabled'),bool) or not isinstance(r.get('requires_all_artifact_ids'),list):raise CE(f'invalid semantic rule {c}')
  z=[req(x,f'{c} required artifact') for x in r['requires_all_artifact_ids']]
  if len(z)!=len(set(z)):raise CE(f'duplicate required artifact in {c}')
 for c in order[order.index(ceil)+1:]:
  if rules[c]['enabled']:raise CE(f'class above ceiling enabled: {c}')
 return v
def envmap(vals:list[str])->dict[str,str]:
 out={}
 for x in vals:
  if '=' not in x:raise CE('--env must be PROFILE=relative/path.json')
  k,p=x.split('=',1);k=req(k,'profile');p=rel(p,'environment path')
  if k in out:raise CE(f'duplicate environment profile {k}')
  out[k]=p
 return out
def envbind(policy:dict[str,Any],root:Path,m:dict[str,str])->list[dict[str,Any]]:
 r=policy['required_environment_profiles']
 if set(m)!=set(r):raise CE(f'environment bindings must exactly be {r}')
 out=[]
 for k in r:
  p=file(root,m[k],f'{k} environment');v=lj(p)
  if not isinstance(v,dict) or v.get('schema_version')!=E or v.get('profile')!=k or not isinstance(v.get('runtime'),dict) or not v['runtime']:raise CE(f'invalid {k} environment capsule')
  out.append({'profile':k,'path':m[k],'sha256':hf(p),'byte_count':p.stat().st_size,'canonical_payload_sha256':hb(cb(v))})
 return out
def pbind(policy:dict[str,Any],root:Path)->list[dict[str,Any]]:
 return [{'path':r,'sha256':hf(p:=file(root,r,f'protected {r}')),'byte_count':p.stat().st_size} for r in policy['protected_files']]
