#!/usr/bin/env python3
from __future__ import annotations
import argparse,sys,tempfile
from pathlib import Path
from ll009ag_common import *
from ll009ag_campaign import prepare,verify_pre,vm,abind,freeze,verify_frz,va,classify,finalize

def expect(label,fn):
 try:fn()
 except CE:return
 raise CE(f'expected failure did not occur: {label}')
def selftest(policy_path:Path):
 cp=vp(lj(policy_path))
 if cp['classification_ceiling']!='hybrid_scenario_sampled_visibility':raise CE('checked policy ceiling changed unexpectedly')
 with tempfile.TemporaryDirectory(prefix='ll009ag-') as td:
  r=Path(td);repo=r/'repo';rt=r/'rt';er=r/'evidence';repo.mkdir();rt.mkdir();er.mkdir()
  pol={'schema_version':P,'study_id':'synthetic','required_environment_profiles':['acquisition','gis','analysis'],'protected_files':['tool.py'],
   'stage_plan':[{'id':'acquire','environment_profile':'acquisition','may_access_network':True},{'id':'gis','environment_profile':'gis','may_access_network':False},{'id':'analysis','environment_profile':'analysis','may_access_network':False}],
   'network_authorized_stage_ids':['acquire'],'classification_order':['descriptive_geometry','empirical_sampled_visibility','hybrid_scenario_sampled_visibility','risk_qualified_visibility','deterministic_visibility'],
   'classification_ceiling':'hybrid_scenario_sampled_visibility','semantic_rules':{'descriptive_geometry':{'enabled':True,'requires_all_artifact_ids':['q']},'empirical_sampled_visibility':{'enabled':True,'requires_all_artifact_ids':['q']},'hybrid_scenario_sampled_visibility':{'enabled':True,'requires_all_artifact_ids':['z']},'risk_qualified_visibility':{'enabled':False,'requires_all_artifact_ids':[]},'deterministic_visibility':{'enabled':False,'requires_all_artifact_ids':[]}}}
  pp=r/'policy.json';wj(pp,pol);(repo/'tool.py').write_text('fixed\n')
  em={}
  for k in ['acquisition','gis','analysis']:
   em[k]=k+'.json';wj(rt/em[k],{'schema_version':E,'profile':k,'runtime':{'python':'3.13','platform':'synthetic'}})
  head='1'*64;pre=prepare(pp,repo,head,rt,em)
  if pre!=prepare(pp,repo,head,rt,em):raise CE('PREPARED not deterministic')
  q=receipt({'schema_version':'q.v1','value':1});z=receipt({'schema_version':'z.v1','upstream':q['receipt_sha256']});wj(er/'q.json',q);wj(er/'z.json',z)
  mv={'schema_version':M,'study_id':'synthetic','artifacts':[{'id':'q','path':'q.json','dependencies':[],'self_hash_mode':'canonical_receipt_sha256'},{'id':'z','path':'z.json','dependencies':['q'],'self_hash_mode':'canonical_receipt_sha256'}]};mp=r/'manifest.json';wj(mp,mv)
  fr=freeze(pre,pp,repo,head,rt,mp,er)
  if fr!=freeze(pre,pp,repo,head,rt,mp,er):raise CE('FROZEN not deterministic')
  av={'schema_version':A,'study_id':'synthetic','evidence_class':'hybrid_scenario_sampled_visibility','basis_artifact_ids':['z']};ap=r/'assertion.json';wj(ap,av);fi=finalize(pre,fr,pp,repo,head,rt,mp,er,ap)
  if fi!=finalize(pre,fr,pp,repo,head,rt,mp,er,ap):raise CE('FINALIZED not deterministic')
  old=(rt/'analysis.json').read_bytes();wj(rt/'analysis.json',{'schema_version':E,'profile':'analysis','runtime':{'python':'drift'}});expect('environment drift',lambda:verify_pre(pre,pp,repo,head,rt));(rt/'analysis.json').write_bytes(old)
  oldt=(repo/'tool.py').read_bytes();(repo/'tool.py').write_text('drift\n');expect('tool drift',lambda:verify_pre(pre,pp,repo,head,rt));(repo/'tool.py').write_bytes(oldt)
  oldz=(er/'z.json').read_bytes();wj(er/'z.json',receipt({'schema_version':'z.v1','changed':True}));expect('artifact substitution',lambda:verify_frz(fr,pre,pp,repo,head,rt,mp,er));(er/'z.json').write_bytes(oldz)
  (er/'extra').write_text('x');expect('extra file',lambda:verify_frz(fr,pre,pp,repo,head,rt,mp,er));(er/'extra').unlink()
  oldq=(er/'q.json').read_bytes();(er/'q.json').unlink();expect('missing file',lambda:verify_frz(fr,pre,pp,repo,head,rt,mp,er));(er/'q.json').write_bytes(oldq)
  badq=lj(er/'q.json');badq['value']=999;wj(er/'q.json',badq);expect('inner self-hash tamper',lambda:abind({'id':'q','path':'q.json','dependencies':[],'self_hash_mode':'canonical_receipt_sha256'},er));(er/'q.json').write_bytes(oldq)
  cyc={'schema_version':M,'study_id':'synthetic','artifacts':[{'id':'q','path':'q.json','dependencies':['z'],'self_hash_mode':'canonical_receipt_sha256'},{'id':'z','path':'z.json','dependencies':['q'],'self_hash_mode':'canonical_receipt_sha256'}]};expect('cycle',lambda:vm(cyc,'synthetic'))
  bad={'schema_version':A,'study_id':'synthetic','evidence_class':'risk_qualified_visibility','basis_artifact_ids':['z']};expect('over-promotion',lambda:classify(pol,va(bad,'synthetic'),{'q','z'}))
  wrong={'schema_version':A,'study_id':'synthetic','evidence_class':'hybrid_scenario_sampled_visibility','basis_artifact_ids':['q']};expect('missing provenance',lambda:classify(pol,va(wrong,'synthetic'),{'q','z'}))
 print('LL-009AG self-test PASS: deterministic state replay; drift, substitution, closure, self-hash tamper, dependency cycles, and semantic over-promotion fail closed')
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
  if a.cmd=='freeze':o=freeze(pre,a.policy,a.repo_root,a.repo_head,a.runtime_root,a.manifest,a.evidence_root);wj(a.output,o);print('FROZEN',o['receipt_sha256']);return 0
  fr=rr(a.freeze,FRZ)
  if a.cmd=='finalize':o=finalize(pre,fr,a.policy,a.repo_root,a.repo_head,a.runtime_root,a.manifest,a.evidence_root,a.assertion);wj(a.output,o);print('FINALIZED',o['receipt_sha256'],o['classification']['evidence_class']);return 0
  fi=rr(a.finalization,FIN);want=finalize(pre,fr,a.policy,a.repo_root,a.repo_head,a.runtime_root,a.manifest,a.evidence_root,a.assertion)
  if fi!=want:raise CE('FINALIZED receipt does not replay exactly')
  print('VERIFY PASS',fi['receipt_sha256'],fi['classification']['evidence_class']);return 0
 except CE as e:print('LL-009AG FAIL:',e,file=sys.stderr);return 2
if __name__=='__main__':raise SystemExit(main())
