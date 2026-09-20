#!/usr/bin/env python3
import argparse,hashlib,json,zipfile
from pathlib import Path,PurePosixPath
D='symthaea.se001q.artifact-transport.v1'
def cj(o): return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sh(b): return 'sha256:'+hashlib.sha256(b).hexdigest()
def sf(p): return sh(Path(p).read_bytes())
def die(m): raise SystemExit(m)
def j(p): return json.loads(Path(p).read_text())
def rej(v,p='$'):
 if isinstance(v,dict):
  for k,c in v.items():
   q=f'{p}.{k}'
   if k=='repair_authority' or (k=='repair_authority_claim' and c!='NONE') or (k=='qualification_claim' and c!='NONE'): die(f'authority violation at {q}')
   rej(c,q)
 elif isinstance(v,list):
  for i,c in enumerate(v): rej(c,f'{p}[{i}]')
def safe(n):
 p=PurePosixPath(n)
 if p.is_absolute() or '..' in p.parts or '\\' in n: die(f'unsafe ZIP path: {n}')
 return p.as_posix()
def mf(root,m):
 if m.get('schema')!='symthaea.se001q.lock-delta-manifest.v1' or m.get('manifest_id')!=sh(cj(m['identity'])): die('bad evidence manifest')
 e={x['path']:x for x in m['identity']['files']}; a={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
 if set(e)!=set(a): die('evidence manifest file-set mismatch')
 for r,p in a.items():
  if e[r]['sha256']!=sf(p) or e[r]['bytes']!=p.stat().st_size: die(f'evidence manifest mismatch: {r}')
 return a
def main():
 ap=argparse.ArgumentParser()
 for x in ('contract','provenance','artifact-zip','evidence-root','verification','output'): ap.add_argument('--'+x,required=True)
 ap.add_argument('--zip-evidence-prefix',default='lock-diagnostic'); ap.add_argument('--zip-verification-path',default='lock-diagnostic-verification/verification.json'); n=ap.parse_args()
 c,p=j(n.contract),j(n.provenance); rej(c); rej(p)
 if c.get('schema')!='symthaea.se001q.artifact-transport-contract.v1' or c.get('domain')!=D or c.get('authority',{}).get('sufficient_for_repair_grant') is not False: die('bad transport contract')
 if p.get('schema')!=c['required_provenance_schema'] or p.get('provenance_witness_id')!=sh(cj(p['identity'])): die('bad provenance witness')
 z=Path(n.artifact_zip).resolve(); zs=sf(z); ad=p['identity']['artifact'].get('digest')
 if zs!=ad: die('artifact ZIP digest mismatch with GitHub provenance')
 r=Path(n.evidence_root).resolve(); m,s,w,v=j(r/'manifest.json'),j(r/'summary.json'),j(r/'lock-delta-witness.json'),j(n.verification)
 for o in (m,s,w,v): rej(o)
 files=mf(r,m)
 if s.get('observation_id')!=sh(cj(s['identity'])) or w.get('witness_id')!=sh(cj(w['identity'])): die('evidence identity mismatch')
 if v.get('result')!='PASS' or v.get('status')!='REPRODUCIBLE_LOCK_DELTA' or v.get('observation_id')!=s['observation_id'] or v.get('manifest_id')!=m['manifest_id'] or v.get('lock_delta_witness_id')!=w['witness_id']: die('verification binding mismatch')
 pref=n.zip_evidence_prefix.strip('/'); vp=safe(n.zip_verification_path)
 if pref!=c['required_zip_evidence_prefix'] or vp!=c['required_zip_verification_path']: die('ZIP layout violates contract')
 with zipfile.ZipFile(z,'r') as q:
  names=[]; seen=set()
  for i in q.infolist():
   x=safe(i.filename)
   if i.is_dir(): continue
   if x in seen: die(f'duplicate ZIP member: {x}')
   seen.add(x); names.append(x)
  exp={f'{pref}/{x}' for x in files}|{f'{pref}/manifest.json'}; act={x for x in names if x.startswith(pref+'/')}
  if act!=exp: die('ZIP evidence member-set mismatch')
  for rel,pth in files.items():
   if q.read(f'{pref}/{rel}')!=pth.read_bytes(): die(f'ZIP/extracted evidence mismatch: {rel}')
  if q.read(f'{pref}/manifest.json')!=(r/'manifest.json').read_bytes() or vp not in seen or q.read(vp)!=Path(n.verification).read_bytes(): die('ZIP/extracted verification or manifest mismatch')
  es=[{'path':x,'bytes':len(q.read(x)),'sha256':sh(q.read(x))} for x in sorted(names)]
 ident={'domain':D,'contract_sha256':sf(n.contract),'provenance_witness_id':p['provenance_witness_id'],'run_id':p['identity']['run']['id'],'artifact_id':p['identity']['artifact']['id'],'artifact_digest':ad,'artifact_zip_sha256':zs,'zip_entry_set_sha256':sh(cj(es)),'zip_evidence_prefix':pref,'verification_zip_path':vp,'observation_id':s['observation_id'],'manifest_id':m['manifest_id'],'lock_delta_witness_id':w['witness_id'],'verification_schema':v.get('schema'),'transport_binding':{'github_artifact_digest_matches_zip_bytes':True,'zip_evidence_members_match_manifest_file_set':True,'zip_evidence_bytes_match_extracted_capsule':True,'zip_verification_bytes_match_extracted_verification':True},'authority':{'meaning':'artifact transport consistency only','sufficient_for_repair_grant':False,'qualification_claim':'NONE','repair_authority_claim':'NONE'}}
 out={'schema':D,'transport_witness_id':sh(cj(ident)),'identity':ident}; rej(out); Path(n.output).write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'schema':D,'result':'PASS','transport_witness_id':out['transport_witness_id'],'artifact_zip_sha256':zs,'manifest_id':m['manifest_id'],'lock_delta_witness_id':w['witness_id'],'sufficient_for_repair_grant':False,'qualification_claim':'NONE','repair_authority_claim':'NONE'},sort_keys=True))
if __name__=='__main__': main()
