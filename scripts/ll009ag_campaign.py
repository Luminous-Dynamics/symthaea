from __future__ import annotations
from pathlib import Path
from typing import Any
from ll009ag_common import *
REQS={'required','optional_diagnostic','not_yet_available'}
IDS={'sha256','receipt_sha256'}

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

def _field_path(v:Any,path:list[str],label:str)->Any:
 cur=v
 for k in path:
  if not isinstance(cur,dict) or k not in cur:raise CE(f'{label} missing field path {path!r}')
  cur=cur[k]
 return cur

def vm(v:Any,study:str)->dict[str,dict[str,Any]]:
 if not isinstance(v,dict) or v.get('schema_version')!=M or v.get('study_id')!=study or not isinstance(v.get('artifacts'),list) or not v['artifacts']:raise CE('invalid evidence manifest')
 out={};paths=set()
 for n,x in enumerate(v['artifacts']):
  if not isinstance(x,dict):raise CE('artifact must be object')
  i=req(x.get('id'),f'artifact {n} id');requirement=x.get('requirement','required')
  if requirement not in REQS:raise CE(f'invalid requirement for {i}: {requirement!r}')
  deps=x.get('dependencies',[]);mode=x.get('self_hash_mode','none');bindings=x.get('dependency_bindings',[])
  if i in out or not isinstance(deps,list) or mode not in MODES or not isinstance(bindings,list):raise CE(f'invalid/duplicate artifact {i}')
  deps=[req(y,f'{i} dependency') for y in deps]
  if len(deps)!=len(set(deps)):raise CE(f'duplicate dependency in {i}')
  if requirement=='not_yet_available':
   if x.get('path') not in (None,'') or mode!='none' or bindings:raise CE(f'not_yet_available artifact {i} must have no path/self-hash/bindings')
   p=None
  else:
   p=rel(x.get('path'),f'artifact {i} path')
   if p in paths:raise CE(f'duplicate artifact path {p}')
   paths.add(p)
  bs=[];bdeps=[]
  for j,b in enumerate(bindings):
   if not isinstance(b,dict):raise CE(f'{i} dependency binding {j} must be object')
   dep=req(b.get('dependency_id'),f'{i} binding dependency');identity=b.get('identity');fp=b.get('field_path')
   if identity not in IDS or not isinstance(fp,list) or not fp:raise CE(f'invalid dependency binding in {i}')
   fp=[req(k,f'{i} binding field') for k in fp]
   if dep in bdeps:raise CE(f'duplicate dependency binding {i}->{dep}')
   bdeps.append(dep);bs.append({'dependency_id':dep,'field_path':fp,'identity':identity})
  if requirement!='not_yet_available' and set(bdeps)!=set(deps):raise CE(f'present artifact {i} must bind every dependency exactly once')
  out[i]={'id':i,'path':p,'requirement':requirement,'dependencies':deps,'dependency_bindings':bs,'self_hash_mode':mode}
 for i,x in out.items():
  for dep in x['dependencies']:
   if dep not in out or dep==i:raise CE(f'bad dependency {i}->{dep}')
 seen=set();active=set()
 def walk(i:str):
  if i in seen:return
  if i in active:raise CE(f'dependency cycle at {i}')
  active.add(i)
  for dep in out[i]['dependencies']:walk(dep)
  active.remove(i);seen.add(i)
 for i in out:walk(i)
 return out

def _optional_file(root:Path,r:str,label:str)->Path|None:
 r=rel(r,label);root=root.resolve(strict=True);p=root
 for x in r.split('/'):
  p=p/x
  if p.is_symlink():raise CE(f'{label} traverses symlink: {p}')
 if not p.exists():return None
 return file(root,r,label)

def abind(x:dict[str,Any],root:Path)->dict[str,Any]:
 requirement=x['requirement']
 if requirement=='not_yet_available':return {'id':x['id'],'requirement':requirement,'availability':'not_yet_available','dependencies':list(x['dependencies'])}
 p=_optional_file(root,x['path'],f'artifact {x["id"]}')
 if p is None:
  if requirement=='required':raise CE(f'required artifact missing: {x["id"]}')
  return {'id':x['id'],'path':x['path'],'requirement':requirement,'availability':'absent_optional_diagnostic','dependencies':list(x['dependencies']),'self_hash_mode':x['self_hash_mode']}
 o={'id':x['id'],'path':x['path'],'requirement':requirement,'availability':'present','sha256':hf(p),'byte_count':p.stat().st_size,'dependencies':list(x['dependencies']),'self_hash_mode':x['self_hash_mode']}
 if x['self_hash_mode'] in {'ag_compact_receipt_sha256','ll009_indent2_receipt_sha256'}:
  v=lj(p);o['verified_receipt_sha256']=vr(v,mode=x['self_hash_mode']);o['verified_receipt_schema']=req(v.get('schema_version'),f'{x["id"]} schema')
 return o

def _verify_dependency_bindings(a:dict[str,dict[str,Any]],bound:dict[str,dict[str,Any]],root:Path)->None:
 for i,x in a.items():
  child=bound[i]
  if child['availability']!='present':continue
  for dep in x['dependencies']:
   if bound[dep]['availability']!='present':raise CE(f'present artifact {i} depends on unavailable artifact {dep}')
  if not x['dependencies']:continue
  child_json=lj(file(root,x['path'],f'artifact {i}'))
  verified=[]
  for b in x['dependency_bindings']:
   dep=b['dependency_id'];parent=bound[dep]
   expected=parent.get('verified_receipt_sha256') if b['identity']=='receipt_sha256' else parent.get('sha256')
   expected=sha(expected,f'{i}->{dep} expected {b["identity"]}')
   got=sha(_field_path(child_json,b['field_path'],f'{i}->{dep}'),f'{i}->{dep} observed {b["identity"]}')
   if got!=expected:raise CE(f'dependency binding mismatch {i}->{dep}: expected {expected}, got {got!r}')
   verified.append({'dependency_id':dep,'field_path':b['field_path'],'identity':b['identity'],'verified_value':expected})
  child['verified_dependency_bindings']=verified

def _bind_all(a:dict[str,dict[str,Any]],root:Path)->tuple[list[dict[str,Any]],set[str],dict[str,Any]]:
 actual=closed(root);decl_paths={x['path'] for x in a.values() if x['path'] is not None}
 if actual-decl_paths:raise CE(f'evidence-root closure failure extra={sorted(actual-decl_paths)}')
 bound={i:abind(x,root) for i,x in a.items()}
 present_paths={x['path'] for x in bound.values() if x.get('availability')=='present'}
 if actual!=present_paths:raise CE(f'evidence-root closure failure missing={sorted(present_paths-actual)} extra={sorted(actual-present_paths)}')
 _verify_dependency_bindings(a,bound,root)
 unavailable=sorted(i for i,x in bound.items() if x['availability']=='not_yet_available')
 optional_absent=sorted(i for i,x in bound.items() if x['availability']=='absent_optional_diagnostic')
 summary={'required_complete':True,'campaign_completeness':'complete' if not unavailable else 'incomplete_declared','promotion_eligible':not unavailable,'not_yet_available_ids':unavailable,'absent_optional_diagnostic_ids':optional_absent}
 return [bound[i] for i in sorted(bound)],present_paths,summary

def freeze(pre:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path,mp:Path,er:Path)->dict[str,Any]:
 p=verify_pre(pre,pp,repo,head,rt);mv=lj(mp);a=vm(mv,p['study_id']);b,actual,summary=_bind_all(a,er)
 return receipt({'schema_version':FRZ,'state':'FROZEN','study_id':p['study_id'],'preparation_receipt_sha256':pre['receipt_sha256'],'manifest':{'sha256':hf(mp),'byte_count':mp.stat().st_size,'canonical_payload_sha256':hb(cb(mv))},'artifact_count':len(b),'present_artifact_count':sum(x['availability']=='present' for x in b),'artifacts':b,'exact_file_set':sorted(actual),'completeness':summary,'lineage_rule':'post_freeze_substitution_or_drift_is_fatal'})

def verify_frz(frz:dict[str,Any],pre:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path,mp:Path,er:Path):
 vr(frz,FRZ);vr(pre,PRE)
 if frz.get('state')!='FROZEN' or frz.get('preparation_receipt_sha256')!=pre['receipt_sha256']:raise CE('FROZEN predecessor mismatch')
 p=verify_pre(pre,pp,repo,head,rt);mv=lj(mp);a=vm(mv,p['study_id'])
 if frz.get('manifest')!={'sha256':hf(mp),'byte_count':mp.stat().st_size,'canonical_payload_sha256':hb(cb(mv))}:raise CE('manifest drift after FROZEN')
 b,actual,summary=_bind_all(a,er);old={x.get('id'):x for x in frz.get('artifacts',[]) if isinstance(x,dict)};new={x['id']:x for x in b}
 if old!=new or frz.get('exact_file_set')!=sorted(actual) or frz.get('completeness')!=summary:raise CE('artifact/file/completeness drift after FROZEN')
 if frz.get('artifact_count')!=len(a) or frz.get('present_artifact_count')!=sum(x['availability']=='present' for x in b):raise CE('artifact count mismatch')
 return p,a,new,summary

def va(v:Any,study:str)->dict[str,Any]:
 if not isinstance(v,dict) or v.get('schema_version')!=A or v.get('study_id')!=study or not isinstance(v.get('basis_artifact_ids'),list) or not v['basis_artifact_ids']:raise CE('invalid semantic assertion')
 req(v.get('evidence_class'),'evidence class');ids=[req(x,'basis artifact') for x in v['basis_artifact_ids']]
 if len(ids)!=len(set(ids)):raise CE('duplicate basis artifact')
 return v

def classify(p:dict[str,Any],assertion:dict[str,Any],present_ids:set[str])->dict[str,Any]:
 c=assertion['evidence_class'];order=p['classification_order'];ceil=p['classification_ceiling']
 if c not in order or order.index(c)>order.index(ceil):raise CE(f'evidence class {c!r} exceeds/escapes policy ceiling {ceil!r}')
 rr=p['semantic_rules'][c]
 if not rr['enabled']:raise CE(f'evidence class disabled: {c}')
 basis=set(assertion['basis_artifact_ids']);need=set(rr['requires_all_artifact_ids'])
 if basis-present_ids:raise CE(f'unavailable basis artifacts: {sorted(basis-present_ids)}')
 if need-basis or need-present_ids:raise CE(f'missing required provenance for {c}: {sorted((need-basis)|(need-present_ids))}')
 return {'evidence_class':c,'policy_ceiling':ceil,'basis_artifact_ids':sorted(basis),'required_artifact_ids':sorted(need)}

def finalize(pre:dict[str,Any],frz:dict[str,Any],pp:Path,repo:Path,head:str,rt:Path,mp:Path,er:Path,ap:Path)->dict[str,Any]:
 p,a,bound,summary=verify_frz(frz,pre,pp,repo,head,rt,mp,er);av=va(lj(ap),p['study_id']);present={i for i,x in bound.items() if x['availability']=='present'};cl=classify(p,av,present)
 cl=dict(cl);cl['promotion_status']='eligible' if summary['promotion_eligible'] else 'blocked_incomplete_declared';cl['effective_evidence_class']=cl['evidence_class'] if summary['promotion_eligible'] else None
 return receipt({'schema_version':FIN,'state':'FINALIZED','study_id':p['study_id'],'preparation_receipt_sha256':pre['receipt_sha256'],'freeze_receipt_sha256':frz['receipt_sha256'],'semantic_assertion':{'sha256':hf(ap),'byte_count':ap.stat().st_size,'canonical_payload_sha256':hb(cb(av))},'classification':cl,'campaign_completeness':summary['campaign_completeness'],'promotion_eligible':summary['promotion_eligible'],'not_yet_available_ids':summary['not_yet_available_ids'],'absent_optional_diagnostic_ids':summary['absent_optional_diagnostic_ids'],'artifact_count':len(a),'present_artifact_count':len(present),'numerical_recomputation_performed':False,'source_download_performed':False,'promotion_semantics':'policy-bounded explicit provenance only; incomplete_declared roots are integrity snapshots and are not promotable'})
