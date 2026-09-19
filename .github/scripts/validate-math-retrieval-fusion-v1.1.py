#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, importlib.util, json, sys
from pathlib import Path

VERSION='math-retrieval-fusion-v1.1'; PARENT='math-retrieval-fusion-v1'
EXTRA={'input_rank_interpretation','rrf_arithmetic','rrf_formula','interleave_start_channel','interleave_step_policy','duplicate_turn_policy','exhausted_channel_policy'}
class ValidationError(ValueError): pass

def load_parent():
    p=Path(__file__).with_name('validate-math-retrieval-fusion.py')
    s=importlib.util.spec_from_file_location('sym_fusion_v1_parent',p)
    if s is None or s.loader is None: raise ValidationError(f'cannot load parent validator: {p}')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def method_shape(d):
    if not isinstance(d,dict) or d.get('version')!=VERSION: raise ValidationError(f'root.version: expected {VERSION}')
    f=d.get('fusion')
    if not isinstance(f,dict): raise ValidationError('fusion: object required')
    if f.get('input_rank_interpretation')!='OneBasedAscendingIndexRank': raise ValidationError('fusion.input_rank_interpretation invariant failed')
    m=f.get('method')
    if m=='ReciprocalRankFusion':
        if f.get('rrf_arithmetic')!='ExactRational': raise ValidationError('RRF: exact rational arithmetic required')
        if f.get('rrf_formula')!='SumOneOverKPlusRank': raise ValidationError('RRF: frozen formula required')
        for k in ('interleave_start_channel','interleave_step_policy','duplicate_turn_policy','exhausted_channel_policy'):
            if k in f: raise ValidationError(f'RRF: {k} forbidden')
    elif m=='DeterministicInterleave':
        if 'rrf_k' in f or 'rrf_arithmetic' in f or 'rrf_formula' in f: raise ValidationError('Interleave: RRF fields forbidden')
        if f.get('interleave_start_channel') not in {'Syntax','ExactNormalForm'}: raise ValidationError('Interleave: start channel required')
        if f.get('interleave_step_policy')!='StrictAlternatingByChannel': raise ValidationError('Interleave: strict alternating policy required')
        if f.get('duplicate_turn_policy')!='DuplicateConsumesTurnNoBackfill': raise ValidationError('Interleave: duplicate turn policy required')
        if f.get('exhausted_channel_policy')!='ContinueOtherWithinFrozenInputQuota': raise ValidationError('Interleave: exhausted-channel policy required')
    else: raise ValidationError('fusion.method: unsupported')

def project(d):
    x=copy.deepcopy(d); x['version']=PARENT
    for k in EXTRA: x['fusion'].pop(k,None)
    return x

def validate(d):
    method_shape(d); p=load_parent(); x=project(d)
    try: p.validate(x)
    except Exception as e:
        t=getattr(p,'V',ValueError)
        if isinstance(e,t): raise ValidationError(str(e)) from e
        raise

def fixture(method='ReciprocalRankFusion'):
    d={'version':VERSION,'fusion':{'method':method,'input_rank_interpretation':'OneBasedAscendingIndexRank'}}
    if method=='ReciprocalRankFusion': d['fusion'].update(rrf_k=60,rrf_arithmetic='ExactRational',rrf_formula='SumOneOverKPlusRank')
    else: d['fusion'].update(interleave_start_channel='Syntax',interleave_step_policy='StrictAlternatingByChannel',duplicate_turn_policy='DuplicateConsumesTurnNoBackfill',exhausted_channel_policy='ContinueOtherWithinFrozenInputQuota')
    return d

def self_test():
    for m in ('ReciprocalRankFusion','DeterministicInterleave'): method_shape(fixture(m))
    attacks=[]
    x=fixture(); attacks += [lambda d:d['fusion'].__setitem__('rrf_arithmetic','Float64'),lambda d:d['fusion'].pop('rrf_formula'),lambda d:d['fusion'].__setitem__('interleave_start_channel','Syntax')]
    y=fixture('DeterministicInterleave'); attacks2=[lambda d:d['fusion'].pop('interleave_start_channel'),lambda d:d['fusion'].__setitem__('duplicate_turn_policy','SkipAndBackfill'),lambda d:d['fusion'].__setitem__('rrf_k',60)]
    for base,fs in ((x,attacks),(y,attacks2)):
        for fn in fs:
            z=copy.deepcopy(base); fn(z)
            try: method_shape(z)
            except ValidationError: continue
            raise AssertionError('fusion v1.1 method-shape attack passed')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('path',nargs='?',type=Path); ap.add_argument('--self-test',action='store_true'); a=ap.parse_args()
    if a.self_test: self_test(); print('math-retrieval fusion v1.1 method self-test: PASS'); return 0
    if a.path is None: ap.error('path required unless --self-test')
    try: validate(json.loads(a.path.read_text()))
    except (OSError,json.JSONDecodeError,ValidationError) as e: print(f'INVALID: {e}',file=sys.stderr); return 1
    print('VALID'); return 0
if __name__=='__main__': raise SystemExit(main())
