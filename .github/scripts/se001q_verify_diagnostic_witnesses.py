#!/usr/bin/env python3
import argparse, base64, hashlib, json, re
from pathlib import Path

HEX64=re.compile(r'^sha256:[0-9a-f]{64}$')


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def die(msg):
    raise SystemExit(msg)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('path')
    ns=ap.parse_args()
    doc=json.loads(Path(ns.path).read_text(encoding='utf-8'))
    if doc.get('schema')!='symthaea.diagnostic-witness-set.v1': die('bad set schema')
    identity=doc.get('identity')
    expected='sha256:'+hashlib.sha256(canon(identity)).hexdigest()
    if doc.get('set_id')!=expected: die('set id mismatch')
    if identity.get('qualification_claim')!='NONE' or identity.get('repair_authority_claim')!='NONE': die('set authority boundary violated')
    witnesses=doc.get('witnesses')
    if not isinstance(witnesses,list) or len(witnesses)!=3: die('expected exactly three witnesses')
    ids=[]; gates=[]
    for w in witnesses:
        if w.get('schema')!='symthaea.diagnostic-witness.v1': die('bad witness schema')
        wi=w.get('identity')
        wid='sha256:'+hashlib.sha256(canon(wi)).hexdigest()
        if w.get('witness_id')!=wid: die('witness id mismatch')
        ids.append(wid)
        source=wi['source']; evidence=wi['evidence']; rule=wi['rule']; conclusion=wi['conclusion']
        gates.append(source['gate_id'])
        if source.get('original_classification_state')!='FAIL_UNCLASSIFIED': die('witness rewrites source classification')
        if conclusion.get('failure_class')!='CARGO_LOCKFILE_UPDATE_REQUIRED': die('unexpected failure class')
        if conclusion.get('confidence')!='EXACT_RETAINED_BYTE_MATCH': die('unexpected confidence')
        if conclusion.get('qualification_claim')!='NONE' or conclusion.get('repair_authority_claim')!='NONE': die('witness authority boundary violated')
        if evidence.get('exit_code')!=rule.get('required_exit_code') or evidence.get('exit_code')!=101: die('exit-code predicate mismatch')
        excerpt=base64.b64decode(evidence['excerpt_base64'], validate=True)
        if 'sha256:'+hashlib.sha256(excerpt).hexdigest()!=evidence['excerpt_sha256']: die('excerpt hash mismatch')
        text=excerpt.decode('utf-8')
        if not text.startswith('cannot update the lock file '): die('diagnostic prefix mismatch')
        if not text.endswith(rule['required_excerpt_utf8_suffix']): die('diagnostic suffix mismatch')
        br=evidence['byte_range']
        if br['end_exclusive']-br['start']!=len(excerpt): die('byte range length mismatch')
        for field in ('stream_sha256','excerpt_sha256'):
            if not HEX64.match(evidence[field]): die(f'bad digest: {field}')
        if not HEX64.match(source['observation_id']) or not HEX64.match(source['classification_id']): die('bad source id')
    if ids!=identity.get('witness_ids'): die('witness set ordering/ids mismatch')
    if gates!=['focused-all-targets','focused-tests','focused-clippy']: die('unexpected gate set/order')
    print(json.dumps({'schema':'symthaea.diagnostic-witness-verification.v1','result':'PASS','set_id':doc['set_id'],'witness_ids':ids,'qualification_claim':'NONE','repair_authority_claim':'NONE'}, sort_keys=True))

if __name__=='__main__': main()
