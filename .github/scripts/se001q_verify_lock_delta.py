#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

def canonical(o): return json.dumps(o,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha_bytes(b): return 'sha256:'+hashlib.sha256(b).hexdigest()
def sha_file(p): return sha_bytes(Path(p).read_bytes())
def die(m): raise SystemExit(m)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--evidence',required=True); ns=ap.parse_args(); root=Path(ns.evidence)
    summary=json.loads((root/'summary.json').read_text()); manifest=json.loads((root/'manifest.json').read_text())
    if summary.get('schema')!='symthaea.se001q.lock-delta-observation.v1': die('bad summary schema')
    if summary.get('observation_id')!=sha_bytes(canonical(summary['identity'])): die('summary id mismatch')
    si=summary['identity']
    if si.get('qualification_claim')!='NONE' or si.get('repair_authority_claim')!='NONE': die('summary authority violation')
    if manifest.get('schema')!='symthaea.se001q.lock-delta-manifest.v1': die('bad manifest schema')
    if manifest.get('manifest_id')!=sha_bytes(canonical(manifest['identity'])): die('manifest id mismatch')
    mi=manifest['identity']
    if mi.get('observation_id')!=summary['observation_id']: die('manifest observation mismatch')
    if mi.get('qualification_claim')!='NONE' or mi.get('repair_authority_claim')!='NONE': die('manifest authority violation')
    expected={e['path']:e for e in mi['files']}
    actual={p.relative_to(root).as_posix():p for p in root.rglob('*') if p.is_file() and p.name!='manifest.json'}
    if set(expected)!=set(actual): die('manifest file set mismatch')
    for path,p in actual.items():
        e=expected[path]
        if e['sha256']!=sha_file(p) or e['bytes']!=p.stat().st_size: die(f'manifest mismatch: {path}')
    if si.get('before_lock_sha256')!='sha256:d7dcc25f8dc3807a86cea758f5380059ec8460d304618bcc5cfd6d5c9a6ce44f': die('unexpected source lock')
    if not si.get('frozen_subject_unchanged') or not si.get('source_preserved'): die('source preservation failed')
    status=si.get('status')
    witness_path=root/'lock-delta-witness.json'
    if status=='REPRODUCIBLE_LOCK_DELTA':
        if not witness_path.exists(): die('missing lock delta witness')
        if si.get('online_after_lock_sha256')!=si.get('offline_after_lock_sha256'): die('lock reproduction mismatch')
        if si.get('online_after_lock_sha256')==si.get('before_lock_sha256'): die('claimed empty lock delta')
        if si.get('online_changed_tracked_paths')!=['Cargo.lock'] or si.get('offline_changed_tracked_paths')!=['Cargo.lock']: die('unsafe tracked mutation')
        if not si.get('lock_delta_reproduced_offline') or not si.get('counterfactual_lock_stable') or not si.get('lock_boundary_cleared'): die('causal diagnostic predicates failed')
        for gate in si['counterfactual_locked_gates']:
            if not gate.get('lock_unchanged') or gate.get('lock_update_diagnostic_present'): die('counterfactual gate did not clear lock boundary')
        w=json.loads(witness_path.read_text())
        if w.get('schema')!='symthaea.lock-delta-witness.v1' or w.get('witness_id')!=sha_bytes(canonical(w['identity'])): die('lock witness id mismatch')
        wi=w['identity']
        if wi.get('qualification_claim')!='NONE' or wi.get('repair_authority_claim')!='NONE': die('lock witness authority violation')
        if wi.get('source_lock_sha256')!=si['before_lock_sha256'] or wi.get('generated_lock_sha256')!=si['offline_after_lock_sha256']: die('lock witness binding mismatch')
    elif witness_path.exists():
        die('lock witness exists without reproducible lock delta')
    print(json.dumps({'schema':'symthaea.se001q.lock-delta-verification.v1','result':'PASS','status':status,'observation_id':summary['observation_id'],'manifest_id':manifest['manifest_id'],'lock_delta_witness_id':json.loads(witness_path.read_text())['witness_id'] if witness_path.exists() else None,'qualification_claim':'NONE','repair_authority_claim':'NONE'},sort_keys=True))

if __name__=='__main__': main()
