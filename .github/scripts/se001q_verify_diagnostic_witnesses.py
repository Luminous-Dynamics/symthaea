#!/usr/bin/env python3
import argparse, base64, hashlib, json, re
from pathlib import Path

HEX64=re.compile(r'^sha256:[0-9a-f]{64}$')
EXPECTED_SET_ID='sha256:0a7a16301a164334ef9b4097f019f7d6e8890c7a3c83d7548dd25cc6eaf09216'
EXPECTED_SUBJECT='47de7f2a306cffb66b5505786220590aa5f42e90'
EXPECTED_VERIFIER='3dd33625162ec7137ef06c8079794a8d2aecc95d'
EXPECTED_MANIFEST='sha256:fdc699087ec4d66a8b0ea5ebddba1c79f039c89c06b850b65bc5c40528b0a731'
EXPECTED_RUN='35440131926'
EXPECTED_ATTEMPT='1'
EXPECTED_ARTIFACT='10591990496'
EXPECTED_ARTIFACT_SHA='sha256:6b9f2566e762b3064b0925d4ffbc86d8b3bdb4dd81b9c0863379b2b19362d78e'
EXPECTED=[
 ('focused-all-targets','sha256:f3abb0c5d1687b7a0ba353c077b06e8df1d0987c83fc36bade0a705512a88e01','sha256:4f6407a26096030e585e9a0a65c8a8ae68a1a1324257fe9a1926e9ed9d0cfdb7','sha256:8d2e2b202860f5b09b7f261b6933d52116c7994eb49fc9718cf9307b9ea00299'),
 ('focused-tests','sha256:0d8fca3617854d36684bcfaa3874113f3f44d08231adafe26082dfd39a037295','sha256:c3fa9db6e1bece3ca65a45a3131aca090698c306920624a63d5af7976e74d180','sha256:3eb078beb9e422591a9c51e76b66a283e11095e8d12721a6875f9343598e6e19'),
 ('focused-clippy','sha256:e841297900c3da1074a06cb7fe0e59652603de992985a0e54e5198f46699890a','sha256:a2496d6ceabb8b1384a79360af85aabddc118b7948a9df494e99c83a9e9c9d16','sha256:afab2b2ad486367c552b9119f43421de7ce79baed1d0df61d154c9dd9d797c2e'),
]

def canon(obj): return json.dumps(obj,sort_keys=True,separators=(',', ':'),ensure_ascii=False).encode()
def die(msg): raise SystemExit(msg)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path'); ns=ap.parse_args()
    doc=json.loads(Path(ns.path).read_text(encoding='utf-8'))
    if doc.get('schema')!='symthaea.diagnostic-witness-set.v1': die('bad set schema')
    identity=doc.get('identity')
    if not isinstance(identity,dict): die('set identity missing')
    expected='sha256:'+hashlib.sha256(canon(identity)).hexdigest()
    if doc.get('set_id')!=expected: die('set id mismatch')
    if doc.get('set_id')!=EXPECTED_SET_ID: die('unexpected witness set id')
    if identity.get('subject_sha')!=EXPECTED_SUBJECT: die('set subject mismatch')
    if identity.get('verifier_sha')!=EXPECTED_VERIFIER: die('set verifier mismatch')
    if identity.get('source_evidence_manifest_id')!=EXPECTED_MANIFEST: die('set source manifest mismatch')
    if identity.get('witness_ids')!=[x[1] for x in EXPECTED]: die('set witness id list mismatch')
    if identity.get('qualification_claim')!='NONE' or identity.get('repair_authority_claim')!='NONE': die('set authority boundary violated')
    witnesses=doc.get('witnesses')
    if not isinstance(witnesses,list) or len(witnesses)!=len(EXPECTED): die('expected exactly three witnesses')
    ids=[]; gates=[]
    for w,expected_tuple in zip(witnesses,EXPECTED):
        gate_expected,wid_expected,obs_expected,class_expected=expected_tuple
        if w.get('schema')!='symthaea.diagnostic-witness.v1': die('bad witness schema')
        wi=w.get('identity')
        if not isinstance(wi,dict): die('witness identity missing')
        wid='sha256:'+hashlib.sha256(canon(wi)).hexdigest()
        if w.get('witness_id')!=wid: die('witness id mismatch')
        if wid!=wid_expected: die(f'unexpected witness id: {gate_expected}')
        ids.append(wid)
        source=wi.get('source'); evidence=wi.get('evidence'); rule=wi.get('rule'); conclusion=wi.get('conclusion')
        if not all(isinstance(x,dict) for x in (source,evidence,rule,conclusion)): die('malformed witness')
        gates.append(source.get('gate_id'))
        if source.get('gate_id')!=gate_expected: die('gate mismatch')
        if source.get('subject_sha')!=EXPECTED_SUBJECT or source.get('verifier_sha')!=EXPECTED_VERIFIER: die('source subject/verifier mismatch')
        if source.get('observation_id')!=obs_expected or source.get('classification_id')!=class_expected: die('source observation/classification mismatch')
        if source.get('original_classification_state')!='FAIL_UNCLASSIFIED': die('witness rewrites source classification')
        if conclusion.get('failure_class')!='CARGO_LOCKFILE_UPDATE_REQUIRED': die('unexpected failure class')
        if conclusion.get('confidence')!='EXACT_RETAINED_BYTE_MATCH': die('unexpected confidence')
        if conclusion.get('qualification_claim')!='NONE' or conclusion.get('repair_authority_claim')!='NONE': die('witness authority boundary violated')
        if evidence.get('exit_code')!=rule.get('required_exit_code') or evidence.get('exit_code')!=101: die('exit-code predicate mismatch')
        excerpt=base64.b64decode(evidence['excerpt_base64'],validate=True)
        if 'sha256:'+hashlib.sha256(excerpt).hexdigest()!=evidence['excerpt_sha256']: die('excerpt hash mismatch')
        text=excerpt.decode('utf-8')
        if not text.startswith('cannot update the lock file '): die('diagnostic prefix mismatch')
        if not text.endswith(rule['required_excerpt_utf8_suffix']): die('diagnostic suffix mismatch')
        br=evidence['byte_range']
        if br['end_exclusive']-br['start']!=len(excerpt): die('byte range length mismatch')
        for field in ('stream_sha256','excerpt_sha256'):
            if not HEX64.fullmatch(evidence[field]): die(f'bad digest: {field}')
        provenance=w.get('provenance')
        if not isinstance(provenance,dict): die('witness provenance missing')
        if str(provenance.get('github_run_id'))!=EXPECTED_RUN: die('provenance run mismatch')
        if str(provenance.get('github_run_attempt'))!=EXPECTED_ATTEMPT: die('provenance attempt mismatch')
        if str(provenance.get('artifact_id'))!=EXPECTED_ARTIFACT: die('provenance artifact mismatch')
        if provenance.get('artifact_zip_sha256')!=EXPECTED_ARTIFACT_SHA: die('provenance artifact digest mismatch')
        if provenance.get('evidence_manifest_id')!=EXPECTED_MANIFEST: die('provenance manifest mismatch')
    if ids!=identity.get('witness_ids'): die('witness set ordering/ids mismatch')
    if gates!=[x[0] for x in EXPECTED]: die('unexpected gate set/order')
    print(json.dumps({'schema':'symthaea.diagnostic-witness-verification.v1.1','result':'PASS','set_id':doc['set_id'],'witness_ids':ids,'source_run_id':EXPECTED_RUN,'source_artifact_id':EXPECTED_ARTIFACT,'qualification_claim':'NONE','repair_authority_claim':'NONE'},sort_keys=True))

if __name__=='__main__': main()
