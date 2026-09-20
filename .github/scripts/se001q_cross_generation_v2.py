#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
D='symthaea.se001q.cross-generation-corroboration.v2'; P='symthaea.se001q.github-provenance.v1'; T='symthaea.se001q.artifact-transport.v1'; A={'symthaea.se001q.lock-delta-verification.v1','symthaea.se001q.lock-delta-verification.v2'}; S='symthaea.se001q.lock-delta-verification.v2'
def cj(o): return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sh(b): return 'sha256:'+hashlib.sha256(b).hexdigest()
def sf(p): return sh(Path(p).read_bytes())
def die(m): raise SystemExit(m)
def j(p):
 o=json.loads(Path(p).read_text()); rej(o); return o
def rej(v,p='$'):
 if isinstance(v,dict):
  for k,c in v.items():
   q=f'{p}.{k}'
   if k=='repair_authority' or (k=='repair_authority_claim' and c!='NONE') or (k=='qualification_claim' and c!='NONE'): die(f'authority violation at {q}')
   rej(c,q)
 elif isinstance(v,list):
  for i,c in enumerate(v): rej(c,f'{p}[{i}]')
def vm(root,m):
 if m.get('schema')!='symthaea.se001q.lock-delta-manifest.v1' or m.get('manifest_id')!=sh(cj(m['identity'])): die(f'{root}: bad manifest')
 e={x['path']:x for x in m['identity']['files']}; a={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
 if set(e)!=set(a): die(f'{root}: manifest file-set mismatch')
 for r,p in a.items():
  if e[r]['sha256']!=sf(p) or e[r]['bytes']!=p.stat().st_size: die(f'{root}: manifest mismatch: {r}')
def sl(root):
 p=[x for x in [root/'Cargo.lock.before',root/'Cargo.lock.online-before',root/'Cargo.lock.offline-before'] if x.exists()]
 if not p: die(f'{root}: no source lock')
 b=p[0].read_bytes()
 if any(x.read_bytes()!=b for x in p[1:]): die(f'{root}: source lock copies disagree')
 return p[0]
def gl(root):
 p=[x for x in [root/'Cargo.lock.offline-after',root/'Cargo.lock.online-after'] if x.exists()]
 if not p: die(f'{root}: no generated lock')
 b=p[0].read_bytes()
 if any(x.read_bytes()!=b for x in p[1:]): die(f'{root}: generated lock copies disagree')
 return p[0]
def load(rootarg,varg,parg,targ):
 r=Path(rootarg).resolve(); s,m,w,v,p,t=j(r/'summary.json'),j(r/'manifest.json'),j(r/'lock-delta-witness.json'),j(varg),j(parg),j(targ)
 if s.get('schema')!='symthaea.se001q.lock-delta-observation.v1' or s.get('observation_id')!=sh(cj(s['identity'])): die(f'{r}: bad summary')
 vm(r,m)
 if m['identity'].get('observation_id')!=s['observation_id']: die(f'{r}: manifest observation mismatch')
 if w.get('schema')!='symthaea.lock-delta-witness.v1' or w.get('witness_id')!=sh(cj(w['identity'])): die(f'{r}: bad lock witness')
 if v.get('schema') not in A or v.get('result')!='PASS' or v.get('status')!='REPRODUCIBLE_LOCK_DELTA' or v.get('observation_id')!=s['observation_id'] or v.get('manifest_id')!=m['manifest_id'] or v.get('lock_delta_witness_id')!=w['witness_id']: die(f'{r}: verification mismatch')
 if p.get('schema')!=P or p.get('provenance_witness_id')!=sh(cj(p['identity'])): die(f'{r}: bad provenance witness')
 if t.get('schema')!=T or t.get('transport_witness_id')!=sh(cj(t['identity'])): die(f'{r}: bad transport witness')
 ti=t['identity']; pi=p['identity']
 if ti.get('provenance_witness_id')!=p['provenance_witness_id'] or ti.get('run_id')!=pi['run']['id'] or ti.get('artifact_id')!=pi['artifact']['id'] or ti.get('artifact_digest')!=pi['artifact']['digest'] or ti.get('observation_id')!=s['observation_id'] or ti.get('manifest_id')!=m['manifest_id'] or ti.get('lock_delta_witness_id')!=w['witness_id'] or ti.get('verification_schema')!=v.get('schema'): die(f'{r}: transport binding mismatch')
 a,b=sl(r),gl(r)
 if sf(a)!=w['identity'].get('source_lock_sha256') or sf(b)!=w['identity'].get('generated_lock_sha256'): die(f'{r}: retained lock bytes mismatch witness')
 return {'summary':s,'manifest':m,'witness':w,'verification':v,'provenance':p,'transport':t,'source':a,'generated':b}
def rec(label,c):
 s=c['summary']['identity']; p=c['provenance']; t=c['transport']; return {'label':label,'run_id':p['identity']['run']['id'],'artifact_id':p['identity']['artifact']['id'],'artifact_digest':p['identity']['artifact']['digest'],'verifier_sha':p['identity']['run']['head_sha'],'provenance_witness_id':p['provenance_witness_id'],'transport_witness_id':t['transport_witness_id'],'verification_schema':c['verification']['schema'],'observation_id':c['summary']['observation_id'],'manifest_id':c['manifest']['manifest_id'],'lock_delta_witness_id':c['witness']['witness_id'],'experiment_id':s['experiment_id'],'toolchain':s['toolchain']}
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--contract',required=True)
 for x in ('a','b'):
  for y in ('evidence','verification','provenance','transport'): ap.add_argument(f'--{y}-{x}',required=True)
 ap.add_argument('--output',required=True); n=ap.parse_args(); c=j(n.contract)
 if c.get('schema')!='symthaea.se001q.cross-generation-corroboration-contract.v2' or c.get('domain')!=D or c.get('authority',{}).get('sufficient_for_repair_grant') is not False or set(c.get('accepted_verification_schemas',[]))!=A or c.get('strong_verification_schema')!=S: die('bad corroboration contract')
 a=load(n.evidence_a,n.verification_a,n.provenance_a,n.transport_a); b=load(n.evidence_b,n.verification_b,n.provenance_b,n.transport_b); ai,bi=a['summary']['identity'],b['summary']['identity']
 if ai.get('subject',{}).get('sha')!=bi.get('subject',{}).get('sha') or ai.get('experiment_id')!=bi.get('experiment_id') or ai.get('toolchain')!=bi.get('toolchain'): die('subject/experiment/toolchain mismatch')
 if a['source'].read_bytes()!=b['source'].read_bytes(): die('source lock mismatch')
 if a['generated'].read_bytes()!=b['generated'].read_bytes(): die('generated lock mismatch')
 ra,rb=rec('A',a),rec('B',b)
 if ra['run_id']==rb['run_id']: die('run independence failed')
 if ra['artifact_id']==rb['artifact_id']: die('artifact independence failed')
 if ra['verifier_sha']==rb['verifier_sha']: die('verifier-generation independence failed')
 if sum(x['verification_schema']==S for x in (ra,rb))<1: die('at least one v2 verifier required')
 ident={'domain':D,'contract_sha256':sf(n.contract),'subject_sha':ai['subject']['sha'],'experiment_id':ai['experiment_id'],'source_lock_sha256':sf(a['source']),'generated_lock_sha256':sf(a['generated']),'toolchain':ai['toolchain'],'runs':[ra,rb],'independence':{'distinct_run_ids':True,'distinct_artifact_ids':True,'distinct_verifier_generations':True,'strong_independent_verifier_present':True,'source_lock_bytes_identical':True,'generated_lock_bytes_identical':True,'github_provenance_witnesses_bound':True,'artifact_transport_witnesses_bound':True},'result':'CORROBORATED_GENERATED_LOCK','authority':{'meaning':'cross-generation corroboration with GitHub provenance and artifact transport binding only','sufficient_for_repair_grant':False,'qualification_claim':'NONE','repair_authority_claim':'NONE'}}
 out={'schema':D,'corroboration_id':sh(cj(ident)),'identity':ident}; rej(out); Path(n.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'schema':D,'result':ident['result'],'corroboration_id':out['corroboration_id'],'generated_lock_sha256':ident['generated_lock_sha256'],'sufficient_for_repair_grant':False,'qualification_claim':'NONE','repair_authority_claim':'NONE'},sort_keys=True))
if __name__=='__main__': main()
