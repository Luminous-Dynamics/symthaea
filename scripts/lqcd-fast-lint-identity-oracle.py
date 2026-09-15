#!/usr/bin/env python3
import argparse, hashlib, json, re
from pathlib import Path

EXPECTED_PROFILE='lqcd-particle-physics-focused-v2'
EXPECTED_COMMAND='cargo clippy --locked -p symthaea-particle-physics --all-targets -- -D warnings'
EXPECTED_LINT='clippy::needless_range_loop'
SCHEMA='symthaea.lqcd.fast-lint-identity.v1'

ERR_RE=re.compile(r"error: the loop variable `(?P<var>[^`]+)` is used to index `(?P<indexed>[^`]+)`\n\s*--> (?P<path>[^:\n]+):(?P<line>\d+):(?P<col>\d+)")

def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()

def sha(b):
    return hashlib.sha256(b).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--attempt', required=True)
    ap.add_argument('--diagnostic', required=True)
    ap.add_argument('--output', required=True)
    args=ap.parse_args()
    attempt_p=Path(args.attempt); diag_p=Path(args.diagnostic)
    attempt_bytes=attempt_p.read_bytes(); diag_bytes=diag_p.read_bytes()
    a=json.loads(attempt_bytes); d=json.loads(diag_bytes)

    errors=[]
    if a.get('schema_version')!='symthaea.focused-qualification-attempt.v2': errors.append('attempt_schema')
    if a.get('terminal_disposition')!='ClippyFailed': errors.append('attempt_disposition')
    if a.get('gates',{}).get('clippy')!='FAIL': errors.append('clippy_gate')
    if any(v!='PASS' for k,v in a.get('gates',{}).items() if k!='clippy'): errors.append('non_clippy_gate')
    if a.get('qualification_profile',{}).get('profile_id')!=EXPECTED_PROFILE: errors.append('profile')
    if d.get('authority')!='NON_AUTHORITATIVE_DIAGNOSTIC': errors.append('diag_authority')
    if d.get('terminal_disposition')!='ClippyFailed': errors.append('diag_disposition')
    if d.get('subject_sha')!=a.get('subject',{}).get('checked_out_commit_sha'): errors.append('subject_binding')
    if d.get('canonical_attempt_sha256')!=sha(attempt_bytes): errors.append('attempt_digest_binding')
    if EXPECTED_COMMAND not in d.get('failure',{}).get('command',''): errors.append('command')
    rustc=a.get('toolchain',{}).get('rustc','').splitlines()[0]
    clippy=a.get('toolchain',{}).get('clippy','')
    if rustc!='rustc 1.96.0 (ac68faa20 2026-05-25)': errors.append('rustc')
    if clippy!='clippy 0.1.96 (ac68faa20c 2026-05-25)': errors.append('clippy_version')
    if errors:
        raise SystemExit('binding failure: '+','.join(errors))

    identities=[]
    for m in ERR_RE.finditer(d.get('failure',{}).get('output_tail','')):
        ident={
            'lint': EXPECTED_LINT,
            'severity':'error',
            'message': f"the loop variable `{m.group('var')}` is used to index `{m.group('indexed')}`",
            'path': m.group('path'),
            'line': int(m.group('line')),
            'column': int(m.group('col')),
        }
        ident['identity_sha256']=sha(canon(ident))
        identities.append(ident)
    identities.sort(key=lambda x:(x['path'],x['line'],x['column'],x['message'],x['lint']))
    if len(identities)!=4:
        raise SystemExit(f'expected 4 lint identities, got {len(identities)}')

    result={
        'schema_version':SCHEMA,
        'authority':'INDEPENDENT_ENGINEERING_EVIDENCE',
        'attempt_sha256':sha(attempt_bytes),
        'diagnostic_sha256':sha(diag_bytes),
        'subject_sha':d['subject_sha'],
        'verifier_sha':d['verifier_sha'],
        'profile_id':EXPECTED_PROFILE,
        'recipe_semantics_sha256':a['qualification_profile']['recipe_semantics_sha256'],
        'rustc':rustc,
        'clippy':clippy,
        'command':EXPECTED_COMMAND,
        'terminal_disposition':'ClippyFailed',
        'lint_identities':identities,
        'identity_set_sha256':sha(canon(identities)),
        'repair_authorized':False,
        'attribution':'AttributionUnknown',
        'claim_ceiling':'typed lint identity only; causal attribution and Rust repair remain unauthorized',
    }
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'identity_count':len(identities),'identity_set_sha256':result['identity_set_sha256'],'repair_authorized':False},sort_keys=True))

if __name__=='__main__':
    main()
