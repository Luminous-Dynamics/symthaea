#!/usr/bin/env python3
import argparse, hashlib, json
from pathlib import Path

SCHEMA='symthaea.lqcd.fast-lint-attribution.v1'
CTX=('profile_id','recipe_semantics_sha256','rustc','clippy','command')

def canon(x): return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()
def sha(x): return hashlib.sha256(x).hexdigest()
def load(p): return json.loads(Path(p).read_text())

def classify(candidate, base):
    if base is None:
        return 'BaseComparisonUnavailable','no exact-base normalized evidence supplied'
    mismatches=[k for k in CTX if candidate.get(k)!=base.get(k)]
    if mismatches:
        return 'VerifierOrToolchainShift','context mismatch: '+','.join(mismatches)
    cids=candidate.get('lint_identities')
    bids=base.get('lint_identities')
    if not isinstance(cids,list) or not isinstance(bids,list):
        return 'BaseComparisonUnavailable','normalized lint identity set missing'
    cset=[x.get('identity_sha256') for x in cids]
    bset=[x.get('identity_sha256') for x in bids]
    if any(not x for x in cset+bset):
        return 'BaseComparisonUnavailable','identity hash missing'
    if cset==bset and candidate.get('terminal_disposition')=='ClippyFailed' and base.get('terminal_disposition')=='ClippyFailed':
        return 'InheritedBaselineFailure','exact normalized lint identity set reproduced on base'
    if not bset and base.get('terminal_disposition') in ('Pass','Qualified','AllRequiredGatesPass'):
        return 'CandidateRegression','equivalent base has no lint failures and candidate does'
    if candidate.get('terminal_disposition')=='ClippyFailed' and base.get('terminal_disposition')=='ClippyFailed' and cset!=bset:
        return 'BaselineDifferentFailure','base and candidate fail with different normalized lint identity sets'
    return 'BaseComparisonUnavailable','base evidence does not support a defined causal classification'

def synthetic_candidate():
    return {
      'profile_id':'p','recipe_semantics_sha256':'r','rustc':'rust','clippy':'clippy','command':'cmd',
      'terminal_disposition':'ClippyFailed','lint_identities':[{'identity_sha256':'a'},{'identity_sha256':'b'}]
    }

def self_test():
    c=synthetic_candidate(); cases=[]
    cases.append(('InheritedBaselineFailure',classify(c,dict(c))[0]))
    cases.append(('CandidateRegression',classify(c,{**c,'terminal_disposition':'AllRequiredGatesPass','lint_identities':[]})[0]))
    cases.append(('BaselineDifferentFailure',classify(c,{**c,'lint_identities':[{'identity_sha256':'a'},{'identity_sha256':'x'}]})[0]))
    cases.append(('VerifierOrToolchainShift',classify(c,{**c,'rustc':'other'})[0]))
    cases.append(('BaseComparisonUnavailable',classify(c,None)[0]))
    b=dict(c); b.pop('lint_identities')
    cases.append(('BaseComparisonUnavailable',classify(c,b)[0]))
    for expected,got in cases:
        assert expected==got,(expected,got)
    return {'synthetic_cases':len(cases),'pass':True}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate')
    ap.add_argument('--base')
    ap.add_argument('--output')
    ap.add_argument('--self-test',action='store_true')
    args=ap.parse_args()
    if args.self_test:
        print(json.dumps(self_test(),sort_keys=True)); return
    if not args.candidate or not args.output:
        raise SystemExit('--candidate and --output required')
    c=load(args.candidate); b=load(args.base) if args.base else None
    cls,reason=classify(c,b)
    result={
      'schema_version':SCHEMA,
      'candidate_sha256':sha(Path(args.candidate).read_bytes()),
      'base_sha256':sha(Path(args.base).read_bytes()) if args.base else None,
      'classification':cls,
      'reason':reason,
      'rust_repair_authorized': cls=='CandidateRegression',
      'baseline_maintenance_scope_required': cls=='InheritedBaselineFailure',
      'claim_ceiling':'engineering causal attribution only',
    }
    Path(args.output).write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print(json.dumps(result,sort_keys=True))

if __name__=='__main__': main()
