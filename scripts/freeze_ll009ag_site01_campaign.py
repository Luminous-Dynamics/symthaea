#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys,tempfile
from pathlib import Path
from ll009ag_common import *
from ll009ag_campaign import prepare,verify_pre,vm,abind,freeze,verify_frz,va,classify,finalize

def expect(label,fn):
 try:fn()
 except CE:return
 raise CE(f'expected failure did not occur: {label}')

def native_receipt(v:dict)->dict:
 if 'receipt_sha256' in v:raise CE('native receipt payload already hashed')
 o=dict(v);o['receipt_sha256']=hb((json.dumps(v,sort_keys=True,indent=2,separators=(',',': '))+'\n').encode());return o

def selftest(policy_path:Path):
 cp=vp(lj(policy_path))
 if cp['classification_ceiling']!='hybrid_scenario_sampled_visibility':raise CE('checked policy ceiling changed unexpectedly')
 with tempfile.TemporaryDirectory(prefix='ll009ag-') as td:
  r=Path(td);repo=r/'repo';rt=r/'rt';er=r/'evidence';repo.mkdir();rt.mkdir();er.mkdir()
  pol={'schema_version':P,'study_id':'synthetic','required_environment_profiles':['acquisition','gis','analysis'],'protected_files':['tool.py'],
   'stage_plan':[{'id':'acquire','environment_profile':'acquisition','may_access_network':True},{'id':'gis','environment_profile':'gis','may_access_network':False},{'id':'analysis','environment_profile':'analysis','may_access_network':False}],
   'network_authorized_stage_ids':['acquire'],'environment_contracts':{k:{'required_packages':['numpy'] if k in ('gis','analysis') else [],'required_libraries':['openssl'] if k=='acquisition' else (['gdal','proj'] if k=='gis' else []),'required_environment_variables':['PYTHONHASHSEED','TZ']} for k in ['acquisition','gis','analysis']},'classification_order':['descriptive_geometry','empirical_sampled_visibility','hybrid_scenario_sampled_visibility','risk_qualified_visibility','deterministic_visibility'],
   'classification_ceiling':'hybrid_scenario_sampled_visibility','semantic_rules':{'descriptive_geometry':{'enabled':True,'requires_all_artifact_ids':['q']},'empirical_sampled_visibility':{'enabled':True,'requires_all_artifact_ids':['q']},'hybrid_scenario_sampled_visibility':{'enabled':True,'requires_all_artifact_ids':['z']},'risk_qualified_visibility':{'enabled':False,'requires_all_artifact_ids':[]},'deterministic_visibility':{'enabled':False,'requires_all_artifact_ids':[]}}}
  pp=r/'policy.json';wj(pp,pol);(repo/'tool.py').write_text('fixed\n')
  em={}
  for k in ['acquisition','gis','analysis']:
   em[k]=k+'.json';wj(rt/em[k],{'schema_version':E,'profile':k,'runtime':{'python':{'implementation':'cpython','version':'3.13.0','executable_sha256':'a'*64},'platform':{'system':'Linux','machine':'x86_64'},'packages':({'numpy':'1.26.4'} if k in ('gis','analysis') else {}),'libraries':({'openssl':'OpenSSL synthetic'} if k=='acquisition' else ({'gdal':'3.9.0','proj':'9.4.0'} if k=='gis' else {})),'environment':{'PYTHONHASHSEED':None,'TZ':'UTC'}}})
  head='1'*64;pre=prepare(pp,repo,head,rt,em)
  if pre!=prepare(pp,repo,head,rt,em):raise CE('PREPARED not deterministic')

  q=native_receipt({'schema_version':'q.v1','value':1})
  z=native_receipt({'schema_version':'z.v1','upstream':q['receipt_sha256']})
  wj(er/'q.json',q);wj(er/'z.json',z)
  mv={'schema_version':M,'study_id':'synthetic','artifacts':[
   {'id':'q','path':'q.json','requirement':'required','dependencies':[],'dependency_bindings':[],'self_hash_mode':'ll009_indent2_receipt_sha256'},
   {'id':'z','path':'z.json','requirement':'required','dependencies':['q'],'dependency_bindings':[{'dependency_id':'q','field_path':['upstream'],'identity':'receipt_sha256'}],'self_hash_mode':'ll009_indent2_receipt_sha256'},
   {'id':'diag','path':'diag.json','requirement':'optional_diagnostic','dependencies':['q'],'dependency_bindings':[{'dependency_id':'q','field_path':['q_file_sha256'],'identity':'sha256'}],'self_hash_mode':'ll009_indent2_receipt_sha256'},
   {'id':'future','requirement':'not_yet_available','dependencies':[],'dependency_bindings':[],'self_hash_mode':'none'}]}
  mp=r/'manifest.json';wj(mp,mv)
  fr=freeze(pre,pp,repo,head,rt,mp,er)
  if fr!=freeze(pre,pp,repo,head,rt,mp,er):raise CE('FROZEN not deterministic')
  if fr['completeness']['campaign_completeness']!='incomplete_declared' or fr['completeness']['promotion_eligible'] is not False:raise CE('declared-unavailable lane did not block promotion')
  if fr['completeness']['absent_optional_diagnostic_ids']!=['diag'] or fr['completeness']['not_yet_available_ids']!=['future']:raise CE('availability summary drift')
  av={'schema_version':A,'study_id':'synthetic','evidence_class':'hybrid_scenario_sampled_visibility','basis_artifact_ids':['z']};ap=r/'assertion.json';wj(ap,av)
  fi=finalize(pre,fr,pp,repo,head,rt,mp,er,ap)
  if fi!=finalize(pre,fr,pp,repo,head,rt,mp,er,ap):raise CE('FINALIZED not deterministic')
  if fi['promotion_eligible'] is not False or fi['campaign_completeness']!='incomplete_declared' or fi['classification']['effective_evidence_class'] is not None:raise CE('FINALIZED incorrectly promoted incomplete campaign')

  old=(rt/'analysis.json').read_bytes();wj(rt/'analysis.json',{'schema_version':E,'profile':'analysis','runtime':{'python':{'implementation':'cpython','version':'drift','executable_sha256':'a'*64},'platform':{'system':'Linux','machine':'x86_64'},'packages':{'numpy':'1.26.4'},'libraries':{},'environment':{'PYTHONHASHSEED':None,'TZ':'UTC'}}});expect('environment drift before freeze',lambda:verify_pre(pre,pp,repo,head,rt));(rt/'analysis.json').write_bytes(old)
  oldt=(repo/'tool.py').read_bytes();(repo/'tool.py').write_text('drift\n');expect('tool drift before freeze',lambda:verify_pre(pre,pp,repo,head,rt));(repo/'tool.py').write_bytes(oldt)

  oldz=(er/'z.json').read_bytes();wj(er/'z.json',native_receipt({'schema_version':'z.v1','upstream':q['receipt_sha256'],'changed':True}));expect('artifact substitution after freeze',lambda:verify_frz(fr,pre,pp,repo,head,rt,mp,er));(er/'z.json').write_bytes(oldz)
  (er/'extra').write_text('x');expect('extra evidence file',lambda:verify_frz(fr,pre,pp,repo,head,rt,mp,er));(er/'extra').unlink()

  wrong_up='2'*64;wj(er/'z.json',native_receipt({'schema_version':'z.v1','upstream':wrong_up}));expect('valid child with wrong upstream binding',lambda:freeze(pre,pp,repo,head,rt,mp,er));(er/'z.json').write_bytes(oldz)

  oldq=(er/'q.json').read_bytes();(er/'q.json').unlink();expect('required artifact missing',lambda:freeze(pre,pp,repo,head,rt,mp,er));(er/'q.json').write_bytes(oldq)
  fr_optional_absent=freeze(pre,pp,repo,head,rt,mp,er)
  if fr_optional_absent['completeness']['absent_optional_diagnostic_ids']!=['diag']:raise CE('missing optional diagnostic not recorded')

  diag=native_receipt({'schema_version':'diag.v1','q_file_sha256':hf(er/'q.json')});wj(er/'diag.json',diag)
  fr_with_diag=freeze(pre,pp,repo,head,rt,mp,er)
  if fr_with_diag['completeness']['absent_optional_diagnostic_ids']:raise CE('present optional diagnostic marked absent')
  bad_diag=native_receipt({'schema_version':'diag.v1','q_file_sha256':'3'*64});wj(er/'diag.json',bad_diag);expect('optional diagnostic wrong dependency binding',lambda:freeze(pre,pp,repo,head,rt,mp,er));(er/'diag.json').unlink()

  badq=lj(er/'q.json');badq['value']=999;wj(er/'q.json',badq);expect('inner self-hash tamper',lambda:abind(vm(mv,'synthetic')['q'],er));(er/'q.json').write_bytes(oldq)
  wrong_mode={'schema_version':M,'study_id':'synthetic','artifacts':[{'id':'q','path':'q.json','requirement':'required','dependencies':[],'dependency_bindings':[],'self_hash_mode':'ag_compact_receipt_sha256'}]};expect('wrong native self-hash contract',lambda:abind(vm(wrong_mode,'synthetic')['q'],er))
  missing_binding={'schema_version':M,'study_id':'synthetic','artifacts':[{'id':'q','path':'q.json','requirement':'required','dependencies':[],'dependency_bindings':[],'self_hash_mode':'ll009_indent2_receipt_sha256'},{'id':'z','path':'z.json','requirement':'required','dependencies':['q'],'dependency_bindings':[],'self_hash_mode':'ll009_indent2_receipt_sha256'}]};expect('missing dependency binding',lambda:vm(missing_binding,'synthetic'))
  cyc={'schema_version':M,'study_id':'synthetic','artifacts':[
   {'id':'q','path':'q.json','requirement':'required','dependencies':['z'],'dependency_bindings':[{'dependency_id':'z','field_path':['upstream'],'identity':'receipt_sha256'}],'self_hash_mode':'ll009_indent2_receipt_sha256'},
   {'id':'z','path':'z.json','requirement':'required','dependencies':['q'],'dependency_bindings':[{'dependency_id':'q','field_path':['upstream'],'identity':'receipt_sha256'}],'self_hash_mode':'ll009_indent2_receipt_sha256'}]};expect('dependency cycle',lambda:vm(cyc,'synthetic'))

  bad={'schema_version':A,'study_id':'synthetic','evidence_class':'risk_qualified_visibility','basis_artifact_ids':['z']};expect('over-promotion',lambda:classify(pol,va(bad,'synthetic'),{'q','z'}))
  wrong={'schema_version':A,'study_id':'synthetic','evidence_class':'hybrid_scenario_sampled_visibility','basis_artifact_ids':['q']};expect('missing provenance',lambda:classify(pol,va(wrong,'synthetic'),{'q','z'}))
  future_basis={'schema_version':A,'study_id':'synthetic','evidence_class':'descriptive_geometry','basis_artifact_ids':['future']};expect('unavailable semantic basis',lambda:classify(pol,va(future_basis,'synthetic'),{'q','z'}))
 print('LL-009AG self-test PASS: deterministic replay; native receipt hashes; exact dependency bindings; required/optional/unavailable handling; environment contracts; drift, substitution, closure, self-hash tamper, cycles, and semantic over-promotion fail closed')

def rr(p:Path,sch:str):v=lj(p);vr(v,sch);return v
def common(x):x.add_argument('--policy',type=Path,required=True);x.add_argument('--repo-root',type=Path,required=True);x.add_argument('--repo-head',required=True);x.add_argument('--runtime-root',type=Path,required=True)
def parser():
 p=argparse.ArgumentParser(description='LL-009AG frozen Site01 campaign root; no downloads or scientific recomputation');sp=p.add_subparsers(dest='cmd',required=True)
 x=sp.add_parser('prepare');common(x);x.add_argument('--env',action='append',default=[]);x.add_argument('--output',type=Path,required=True)
 x=sp.add_parser('freeze');common(x);x.add_argument('--preparation',type=Path,required=True);x.add_argument('--manifest',type=Path,required=True);x.add_argument('--evidence-root',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
 x=sp.add_parser('finalize');common(x);x.add_argument('--preparation',type=Path,required=True);x.add_argument('--freeze',type=Path,required=True);x.add_argument('--manifest',type=Path,required=True);x.add_argument('--evidence-root',type=Path,required=True);x.add_argument('--assertion',type=Path,required=True);x.add_argument('--output',type=Path,required=True)
 x=sp.add_parser('verify');common(x);x.add_argument('--preparation',type=Path,required=True);x.add_argument('--freeze',type=Path,required=True);x.add_argument('--manifest',type=Path,required=True);x.add_argument('--evidence-root',type=Path,required=True);x.add_argument('--assertion',type=Path,required=True);x.add_argument('--finalization',type=Path,required=True)
 x=sp.add_parser('self-test');x.add_argument('--policy',type=Path,required=True);return p

def main()->int:
 a=parser().parse_args()
 try:
  if a.cmd=='self-test':selftest(a.policy);return 0
  if a.cmd=='prepare':o=prepare(a.policy,a.repo_root,a.repo_head,a.runtime_root,envmap(a.env));wj(a.output,o);print('PREPARED',o['receipt_sha256']);return 0
  pre=rr(a.preparation,PRE)
  if a.cmd=='freeze':o=freeze(pre,a.policy,a.repo_root,a.repo_head,a.runtime_root,a.manifest,a.evidence_root);wj(a.output,o);print('FROZEN',o['receipt_sha256'],o['completeness']['campaign_completeness']);return 0
  fr=rr(a.freeze,FRZ)
  if a.cmd=='finalize':o=finalize(pre,fr,a.policy,a.repo_root,a.repo_head,a.runtime_root,a.manifest,a.evidence_root,a.assertion);wj(a.output,o);print('FINALIZED',o['receipt_sha256'],o['classification']['evidence_class'],'promotion_eligible='+str(o['promotion_eligible']).lower());return 0
  fi=rr(a.finalization,FIN);want=finalize(pre,fr,a.policy,a.repo_root,a.repo_head,a.runtime_root,a.manifest,a.evidence_root,a.assertion)
  if fi!=want:raise CE('FINALIZED receipt does not replay exactly')
  print('VERIFY PASS',fi['receipt_sha256'],fi['classification']['evidence_class'],'promotion_eligible='+str(fi['promotion_eligible']).lower());return 0
 except CE as e:print('LL-009AG FAIL:',e,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
