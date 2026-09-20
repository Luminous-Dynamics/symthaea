#!/usr/bin/env python3
from __future__ import annotations
import copy, json, subprocess, sys, tempfile
from pathlib import Path
import test_scip_text_claim_blinding_witness as tv22

ROOT=Path(__file__).resolve().parents[1]
V=ROOT/'scripts/scip_text_claim_blinding_witness_set.py'
V22P=ROOT/'scripts/qualification/scip_text_claim_blinding_witness_policy_v1.json'
P=ROOT/'scripts/qualification/scip_text_claim_blinding_witness_set_policy_v1.json'
POLICY_SHA='f50117a1dbe55574f65608194943595b298e66fde6e7dea727e57a51267f1a8f'
ALL_SLOTS=('extraction-annotator-0','extraction-annotator-1','extraction-adjudicator','alignment-annotator-0','alignment-annotator-1','alignment-adjudicator')

def wset(b,slots=ALL_SLOTS):
 return {'schema':'symthaea.scip-text-claim-blinding-witness-set/v1','authority':'process-completeness-only','policy_semantic_sha256':POLICY_SHA,
         'witnesses':[tv22.witness(b,s) for s in slots]}

def run(ws,b,raw=None):
 with tempfile.TemporaryDirectory() as t:
  t=Path(t); wp=t/'w.json'; bp=t/'b.json'
  wp.write_text(raw if raw is not None else json.dumps(ws),encoding='utf-8'); bp.write_text(json.dumps(b),encoding='utf-8')
  return subprocess.run([sys.executable,'-B',str(V),str(wp),str(bp),'--v22-policy',str(V22P),'--policy',str(P)],capture_output=True,text=True)

def ok(ws,b):
 r=run(ws,b); assert r.returncode==0,r.stderr; return json.loads(r.stdout)

def reject(ws,b,raw=None):
 r=run(ws,b,raw); assert r.returncode!=0 and 'ERROR:' in r.stderr,(r.returncode,r.stdout,r.stderr)

def no_adjudicators(b):
 b=copy.deepcopy(b)
 ex=b['extraction']; ex['annotations'][1]['claim_inventory_sha256']=ex['annotations'][0]['claim_inventory_sha256']; ex['exact_agreement']=True; ex['adjudication']=None; ex['resolved_inventory_sha256']=ex['annotations'][0]['claim_inventory_sha256']
 # Rebind alignment annotations to the new frozen extraction inventory while keeping alignment disagreement resolved by no adjudicator only after exact agreement.
 for r in b['alignment']['annotations']: r['frozen_surface_inventory_sha256']=ex['resolved_inventory_sha256']
 b['alignment']['annotations'][1]['outcomes']=copy.deepcopy(b['alignment']['annotations'][0]['outcomes'])
 b['alignment']['annotations'][1]['aligned_inventory_sha256']=b['alignment']['annotations'][0]['aligned_inventory_sha256']
 b['alignment']['exact_agreement']=True; b['alignment']['adjudication']=None; b['alignment']['resolved_inventory_sha256']=b['alignment']['annotations'][0]['aligned_inventory_sha256']
 return b

def main():
 b=tv22.bundle(); ws=wset(b); out=ok(ws,b)
 assert out['role_count']==6 and out['required_role_slots']==list(ALL_SLOTS)
 assert out['process_evidence_set_structurally_complete'] is True
 assert out['case_admission_receipt_recomputed'] is False and out['evidence_authenticity_established'] is False
 assert out['surface_fidelity_established'] is False and out['confirmatory_execution_authorized'] is False
 base_receipt=out['witness_set_receipt_sha256']

 # Exact role census/order is load-bearing.
 z=copy.deepcopy(ws); z['witnesses'].pop(2); reject(z,b)
 z=copy.deepcopy(ws); z['witnesses'].append(copy.deepcopy(z['witnesses'][-1])); reject(z,b)
 z=copy.deepcopy(ws); z['witnesses'][0],z['witnesses'][1]=z['witnesses'][1],z['witnesses'][0]; reject(z,b)

 # Cross-role common bindings must agree even when each individual witness remains structurally valid.
 z=copy.deepcopy(ws); z['witnesses'][1]['case_admission_receipt_sha256']=tv22.h('other-case-admission'); reject(z,b)
 z=copy.deepcopy(ws); z['witnesses'][1]['public_claim_schema_sha256']=tv22.h('other-public-schema')
 for a in z['witnesses'][1]['allowed_artifacts']:
  if a['class']=='public-claim-schema': a['sha256']=z['witnesses'][1]['public_claim_schema_sha256']
 reject(z,b)

 # Cross-role identities/evidence roots must remain distinct.
 for field in ('session_id_sha256','challenge_nonce_sha256','session_manifest_sha256','access_control_snapshot_sha256','audit_capture_sha256'):
  z=copy.deepcopy(ws); z['witnesses'][1][field]=z['witnesses'][0][field]; reject(z,b)

 # Context change changes the composed receipt without pretending semantic fidelity.
 z=copy.deepcopy(ws)
 for w in z['witnesses']: w['case_admission_receipt_sha256']=tv22.h('new-common-case-admission')
 changed=ok(z,b); assert changed['witness_set_receipt_sha256']!=base_receipt and changed['surface_fidelity_established'] is False

 # Required roles are derived from the bundle; exact-agreement cases need no adjudicators.
 b4=no_adjudicators(b)
 slots4=('extraction-annotator-0','extraction-annotator-1','alignment-annotator-0','alignment-annotator-1')
 ws4=wset(b4,slots4); out4=ok(ws4,b4); assert out4['role_count']==4 and out4['required_role_slots']==list(slots4)
 z=copy.deepcopy(ws4); z['witnesses'].append(copy.deepcopy(z['witnesses'][-1])); reject(z,b4)

 # Unknown authority fields and duplicate JSON keys fail closed.
 z=copy.deepcopy(ws); z['surface_fidelity_established']=True; reject(z,b)
 raw=json.dumps(ws,separators=(',',':')).replace('"authority":"process-completeness-only"','"authority":"process-completeness-only","authority":"qualified"',1)
 reject(ws,b,raw=raw)

 print('PASS_BLINDING_WITNESS_SET_ADVERSARIAL')
 return 0
if __name__=='__main__': raise SystemExit(main())
