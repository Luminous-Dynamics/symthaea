from __future__ import annotations
from pathlib import Path
from typing import Any
from ll009ag_common import *
def prepare(pp:Path,repo:Path,head:str,rt:Path,em:dict[str,str])->dict[str,Any]:
 p=vp(lj(pp));head=sha(head,'repo head')
 return receipt({'schema_version':PRE,'state':'PREPARED','study_id':p['study_id'],'repo_head':head,'policy':{'sha256':hf(pp),'byte_count':pp.stat().st_size},'protected_files':pbind(p,repo),'environment_capsules':envbind(p,rt,em),'stage_plan_sha256':hb(cb(p['stage_plan'])),'network_authorized_stage_ids':list(p['network_authorized_stage_ids']),'drift_rule':'drift_before_evidence_requires_reprepare;post_freeze_drift_refuses_mixed_lineage'})
def verify_pre(pre:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path)->dict[str,Any]:
 vr(pre,PRE);p=vp(lj(pp))
 if pre.get('state')!='PREPARED' or pre.get('study_id')!=p['study_id'] or pre.get('repo_head')!=sha(head,'repo head'):raise CE('PREPARED identity/head mismatch')
 if pre.get('policy')!={'sha256':hf(pp),'byte_count':pp.stat().st_size}:raise CE('policy drift since PREPARED')
 now={x['path']:x for x in pbind(p,repo)};old={x.get('path'):x for x in pre.get('protected_files',[]) if isinstance(x,dict)}
 if now!=old:raise CE('protected file drift since PREPARED')
 olde={x.get('profile'):x for x in pre.get('environment_capsules',[]) if isinstance(x,dict)}
 if set(olde)!=set(p['required_environment_profiles']):raise CE('environment profile set drift')
 now={x['profile']:x for x in envbind(p,rt,{k:rel(olde[k].get('path'),f'{k} env path') for k in olde})}
 if now!=olde:raise CE('environment capsule drift since PREPARED')
 if pre.get('stage_plan_sha256')!=hb(cb(p['stage_plan'])):raise CE('stage-plan drift')
 return p
def vm(v:Any,study:str)->dict[str,dict[str,Any]]:
 if not isinstance(v,dict) or v.get('schema_version')!=M or v.get('study_id')!=study or not isinstance(v.get('artifacts'),list) or not v['artifacts']:raise CE('invalid evidence manifest')
 out={};paths=set()
 for n,x in enumerate(v['artifacts']):
  if not isinstance(x,dict):raise CE('artifact must be object')
  i=req(x.get('id'),f'artifact {n} id');p=rel(x.get('path'),f'artifact {i} path');d=x.get('dependencies');mode=x.get('self_hash_mode')
  if i in out or p in paths or not isinstance(d,list) or mode not in MODES:raise CE(f'invalid/duplicate artifact {i}')
  d=[req(y,f'{i} dependency') for y in d]
  if len(d)!=len(set(d)):raise CE(f'duplicate dependency in {i}')
  out[i]={'id':i,'path':p,'dependencies':d,'self_hash_mode':mode};paths.add(p)
 for i,x in out.items():
  for d in x['dependencies']:
   if d not in out or d==i:raise CE(f'bad dependency {i}->{d}')
 seen=set();active=set()
 def walk(i:str):
  if i in seen:return
  if i in active:raise CE(f'dependency cycle at {i}')
  active.add(i)
  for d in out[i]['dependencies']:walk(d)
  active.remove(i);seen.add(i)
 for i in out:walk(i)
 return out
def abind(x:dict[str,Any],root:Path)->dict[str,Any]:
 p=file(root,x['path'],f'artifact {x["id"]}');o={'id':x['id'],'path':x['path'],'sha256':hf(p),'byte_count':p.stat().st_size,'dependencies':list(x['dependencies']),'self_hash_mode':x['self_hash_mode']}
 if x['self_hash_mode']=='canonical_receipt_sha256':
  v=lj(p);o['verified_receipt_sha256']=vr(v);o['verified_receipt_schema']=req(v.get('schema_version'),f'{x["id"]} schema')
 return o
def freeze(pre:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path,mp:Path,er:Path)->dict[str,Any]:
 p=verify_pre(pre,pp,repo,head,rt);mv=lj(mp);a=vm(mv,p['study_id']);actual=closed(er);decl={x['path'] for x in a.values()}
 if actual!=decl:raise CE(f'evidence-root closure failure missing={sorted(decl-actual)} extra={sorted(actual-decl)}')
 b=[abind(a[i],er) for i in sorted(a)]
 return receipt({'schema_version':FRZ,'state':'FROZEN','study_id':p['study_id'],'preparation_receipt_sha256':pre['receipt_sha256'],'manifest':{'sha256':hf(mp),'byte_count':mp.stat().st_size,'canonical_payload_sha256':hb(cb(mv))},'artifact_count':len(b),'artifacts':b,'exact_file_set':sorted(actual),'lineage_rule':'post_freeze_substitution_or_drift_is_fatal'})
def verify_frz(frz:dict[str,Any],pre:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path,mp:Path,er:Path):
 vr(frz,FRZ);vr(pre,PRE)
 if frz.get('state')!='FROZEN' or frz.get('preparation_receipt_sha256')!=pre['receipt_sha256']:raise CE('FROZEN predecessor mismatch')
 p=verify_pre(pre,pp,repo,head,rt);mv=lj(mp);a=vm(mv,p['study_id'])
 if frz.get('manifest')!={'sha256':hf(mp),'byte_count':mp.stat().st_size,'canonical_payload_sha256':hb(cb(mv))}:raise CE('manifest drift after FROZEN')
 actual=closed(er);decl={x['path'] for x in a.values()}
 if actual!=decl or frz.get('exact_file_set')!=sorted(actual):raise CE('evidence file-set drift after FROZEN')
 old={x.get('id'):x for x in frz.get('artifacts',[]) if isinstance(x,dict)}
 if set(old)!=set(a):raise CE('frozen artifact set mismatch')
 for i in a:
  if abind(a[i],er)!=old[i]:raise CE(f'artifact substitution/drift after FROZEN: {i}')
 if frz.get('artifact_count')!=len(a):raise CE('artifact count mismatch')
 return p,a
def va(v:Any,study:str)->dict[str,Any]:
 if not isinstance(v,dict) or v.get('schema_version')!=A or v.get('study_id')!=study or not isinstance(v.get('basis_artifact_ids'),list) or not v['basis_artifact_ids']:raise CE('invalid semantic assertion')
 req(v.get('evidence_class'),'evidence class');ids=[req(x,'basis artifact') for x in v['basis_artifact_ids']]
 if len(ids)!=len(set(ids)):raise CE('duplicate basis artifact')
 return v
def classify(p:dict[str,Any],a:dict[str,Any],ids:set[str])->dict[str,Any]:
 c=a['evidence_class'];order=p['classification_order'];ceil=p['classification_ceiling']
 if c not in order or order.index(c)>order.index(ceil):raise CE(f'evidence class {c!r} exceeds/escapes policy ceiling {ceil!r}')
 r=p['semantic_rules'][c]
 if not r['enabled']:raise CE(f'evidence class disabled: {c}')
 basis=set(a['basis_artifact_ids']);need=set(r['requires_all_artifact_ids'])
 if basis-ids:raise CE(f'unfrozen basis artifacts: {sorted(basis-ids)}')
 if need-basis or need-ids:raise CE(f'missing required provenance for {c}: {sorted((need-basis)|(need-ids))}')
 return {'evidence_class':c,'policy_ceiling':ceil,'basis_artifact_ids':sorted(basis),'required_artifact_ids':sorted(need)}
def finalize(pre:dict[str,Any],frz:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path,mp:Path,er:Path,ap:Path)->dict[str,Any]:
 p,a=verify_frz(frz,pre,pp,repo,head,rt,mp,er);av=va(lj(ap),p['study_id']);cl=classify(p,av,set(a))
 return receipt({'schema_version':FIN,'state':'FINALIZED','study_id':p['study_id'],'preparation_receipt_sha256':pre['receipt_sha256'],'freeze_receipt_sha256':frz['receipt_sha256'],'semantic_assertion':{'sha256':hf(ap),'byte_count':ap.stat().st_size,'canonical_payload_sha256':hb(cb(av))},'classification':cl,'artifact_count':len(a),'numerical_recomputation_performed':False,'source_download_performed':False,'promotion_semantics':'policy-bounded explicit provenance only; no probability or physical-completeness upgrade'})
