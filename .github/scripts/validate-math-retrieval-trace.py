#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, sys
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path

VERSION='math-retrieval-trace-v1'; AUTH='MeasurementOnly'
class ValidationError(ValueError): pass

def dg(b:bytes): return 'sha256:'+hashlib.sha256(b).hexdigest()
def sh(x,w):
    if not isinstance(x,str) or len(x)!=71 or not x.startswith('sha256:') or any(c not in '0123456789abcdef' for c in x[7:]): raise ValidationError(f'{w}: invalid sha256')
    return x

def load_graph_profile():
    p=Path(__file__).with_name('validate-math-retrieval-graph-v1.1.py')
    s=importlib.util.spec_from_file_location('sym_graph_profile_v11',p)
    if s is None or s.loader is None: raise ValidationError(f'cannot load graph profile: {p}')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m.load_parent()

def closed(o,keys,w):
    if not isinstance(o,dict) or set(o)!=set(keys): raise ValidationError(f'{w}: exact fields required')
def nonneg_int(x,w):
    if not isinstance(x,int) or isinstance(x,bool) or x<0: raise ValidationError(f'{w}: non-negative integer required')
    return x

def pos_int(x,w):
    if not isinstance(x,int) or isinstance(x,bool) or x<1: raise ValidationError(f'{w}: positive integer required')
    return x

def unique_digests(xs,w):
    if not isinstance(xs,list): raise ValidationError(f'{w}: list required')
    out=[]
    for i,x in enumerate(xs): out.append(sh(x,f'{w}[{i}]'))
    if len(out)!=len(set(out)): raise ValidationError(f'{w}: duplicate source object digest')
    return out

def compute_fractional_rrf(fusion, channels):
    scores={}
    k=fusion['fusion']['rrf_k']
    for ch in ('Syntax','ExactNormalForm'):
        for rank,d in enumerate(channels[ch]['ranked_source_object_digests'],1):
            scores[d]=scores.get(d,Fraction(0,1))+Fraction(1,k+rank)
    return sorted(scores,key=lambda d:(-scores[d],d))

def compute_interleave(fusion, channels):
    xs={ch:list(channels[ch]['ranked_source_object_digests']) for ch in ('Syntax','ExactNormalForm')}
    pos={'Syntax':0,'ExactNormalForm':0}; out=[]; seen=set(); cur=fusion['fusion']['interleave_start_channel']
    other=lambda c:'ExactNormalForm' if c=='Syntax' else 'Syntax'
    while pos['Syntax']<len(xs['Syntax']) or pos['ExactNormalForm']<len(xs['ExactNormalForm']):
        if pos[cur]>=len(xs[cur]):
            cur=other(cur)
            if pos[cur]>=len(xs[cur]): break
            # One channel is exhausted: continue the remaining channel.
            d=xs[cur][pos[cur]]; pos[cur]+=1
            if d not in seen: seen.add(d); out.append(d)
            continue
        d=xs[cur][pos[cur]]; pos[cur]+=1
        if d not in seen: seen.add(d); out.append(d)
        cur=other(cur)
    return out

def recompute_fusion(fusion, channels):
    if fusion['fusion']['method']=='ReciprocalRankFusion': return compute_fractional_rrf(fusion,channels)
    if fusion['fusion']['method']=='DeterministicInterleave': return compute_interleave(fusion,channels)
    raise ValidationError('fusion: unsupported method')

def trace_shape(t):
    closed(t,{'version','trace_id','authority','graph','experiment_seed','query','retrieval','packing','resources','control_bindings'},'trace')
    if t['version']!=VERSION or t['authority']!=AUTH: raise ValidationError('trace: identity/authority invariant failed')
    if not isinstance(t['trace_id'],str) or not t['trace_id'].strip(): raise ValidationError('trace_id required')
    g=t['graph']; closed(g,{'bundle_sha256','graph_report_sha256','experiment_sha256','arm_id','retrieval_binding_sha256','candidate_set_sha256','candidate_count','context_packer_sha256'},'graph')
    for k in ('bundle_sha256','graph_report_sha256','experiment_sha256','retrieval_binding_sha256','candidate_set_sha256','context_packer_sha256'): sh(g[k],f'graph.{k}')
    if not isinstance(g['arm_id'],str) or not g['arm_id']: raise ValidationError('graph.arm_id required')
    pos_int(g['candidate_count'],'graph.candidate_count')
    nonneg_int(t['experiment_seed'],'experiment_seed')
    q=t['query']; closed(q,{'query_id','query_source_object_sha256'},'query')
    if not isinstance(q['query_id'],str) or not q['query_id'].strip(): raise ValidationError('query.query_id required')
    sh(q['query_source_object_sha256'],'query.query_source_object_sha256')
    r=t['resources']; closed(r,{'retrieval_queries_used','wall_time_ms_used','normalized_compute_units_used_decimal'},'resources')
    pos_int(r['retrieval_queries_used'],'resources.retrieval_queries_used'); nonneg_int(r['wall_time_ms_used'],'resources.wall_time_ms_used')
    s=r['normalized_compute_units_used_decimal']
    if not isinstance(s,str) or not s or any(c not in '0123456789.' for c in s) or s.count('.')>1: raise ValidationError('resources.normalized_compute_units_used_decimal: canonical decimal required')
    try: v=Decimal(s)
    except InvalidOperation as e: raise ValidationError('resources.normalized_compute_units_used_decimal invalid') from e
    if v<0 or not v.is_finite(): raise ValidationError('resources.normalized_compute_units_used_decimal must be finite/non-negative')
    return v

def channel_result_shape(x,w):
    closed(x,{'channel','index_manifest_sha256','index_artifact_sha256','requested_k','input_bytes_used','ranked_source_object_digests'},w)
    if x['channel'] not in {'Syntax','ExactNormalForm'}: raise ValidationError(f'{w}.channel unsupported')
    sh(x['index_manifest_sha256'],f'{w}.index_manifest_sha256'); sh(x['index_artifact_sha256'],f'{w}.index_artifact_sha256')
    pos_int(x['requested_k'],f'{w}.requested_k'); nonneg_int(x['input_bytes_used'],f'{w}.input_bytes_used')
    unique_digests(x['ranked_source_object_digests'],f'{w}.ranked_source_object_digests')
    if len(x['ranked_source_object_digests'])>x['requested_k']: raise ValidationError(f'{w}: more returned items than requested_k')

def validate_control_bindings(trace, participating_indices):
    xs=trace['control_bindings']
    if not isinstance(xs,list) or len(xs)!=len(participating_indices): raise ValidationError('control_bindings: one entry per participating index required')
    by={}
    for i,x in enumerate(xs):
        allowed={'index_manifest_sha256','control_transform'}
        if not isinstance(x,dict) or not allowed<=set(x) or set(x)-allowed-{'control_seed','control_artifact_sha256'}: raise ValidationError(f'control_bindings[{i}]: invalid fields')
        d=sh(x['index_manifest_sha256'],f'control_bindings[{i}].index_manifest_sha256')
        if d in by: raise ValidationError('control_bindings: duplicate index')
        by[d]=x
    if set(by)!=set(participating_indices): raise ValidationError('control_bindings: index set mismatch')
    for d,ix in participating_indices.items():
        expected=ix['representation']['control_transform']; got=by[d]
        if got['control_transform']!=expected: raise ValidationError(f'control_bindings[{d}]: transform mismatch')
        if expected=='None':
            if 'control_seed' in got or 'control_artifact_sha256' in got: raise ValidationError('control_bindings: non-control index carries control evidence')
        else:
            if got.get('control_seed')!=ix['representation']['control_seed']: raise ValidationError('control_bindings: seed mismatch')
            if got.get('control_artifact_sha256')!=ix['representation']['control_artifact_sha256']: raise ValidationError('control_bindings: artifact mismatch')

def validate_packing(p, ranked, packer):
    allowed={'context_packer_sha256','input_ranked_source_object_digests','output','output_items_used','output_bytes_used','stop_reason'}
    optional={'first_nonfitting'}
    if not isinstance(p,dict) or not allowed<=set(p) or set(p)-allowed-optional: raise ValidationError('packing: invalid fields')
    sh(p['context_packer_sha256'],'packing.context_packer_sha256')
    got=unique_digests(p['input_ranked_source_object_digests'],'packing.input_ranked_source_object_digests')
    if got!=ranked: raise ValidationError('packing: input ranking differs from retrieval/fusion output')
    out=p['output']
    if not isinstance(out,list): raise ValidationError('packing.output: list required')
    ds=[]; total=0
    for i,x in enumerate(out,1):
        closed(x,{'rank','source_object_sha256','canonical_payload_bytes'},f'packing.output[{i-1}]')
        if x['rank']!=i: raise ValidationError('packing.output: ranks must be contiguous from 1')
        d=sh(x['source_object_sha256'],f'packing.output[{i-1}].source_object_sha256'); b=pos_int(x['canonical_payload_bytes'],f'packing.output[{i-1}].canonical_payload_bytes')
        if b>packer['budget']['max_output_item_bytes']: raise ValidationError('packing.output: item exceeds max_output_item_bytes')
        ds.append(d); total+=b
    if ds!=ranked[:len(ds)]: raise ValidationError('packing.output: delivered items must be exact rank prefix')
    if p['output_items_used']!=len(out) or p['output_bytes_used']!=total: raise ValidationError('packing: declared output accounting mismatch')
    if len(out)>packer['budget']['max_output_items'] or total>packer['budget']['max_output_bytes']: raise ValidationError('packing: output budget exceeded')
    reason=p['stop_reason']
    if reason=='Empty':
        if ranked or out or 'first_nonfitting' in p: raise ValidationError('packing: Empty stop inconsistent')
    elif reason=='RankedCandidatesExhausted':
        if len(out)!=len(ranked) or 'first_nonfitting' in p: raise ValidationError('packing: exhaustion stop inconsistent')
    elif reason=='ItemLimitReached':
        if len(out)!=packer['budget']['max_output_items'] or len(out)>=len(ranked) or 'first_nonfitting' in p: raise ValidationError('packing: item-limit stop inconsistent')
    elif reason=='FirstNonFittingItem':
        if 'first_nonfitting' not in p or len(out)>=len(ranked) or len(out)>=packer['budget']['max_output_items']: raise ValidationError('packing: first-nonfitting stop inconsistent')
        n=p['first_nonfitting']; closed(n,{'rank','source_object_sha256','canonical_payload_bytes'},'packing.first_nonfitting')
        if n['rank']!=len(out)+1 or n['source_object_sha256']!=ranked[len(out)]: raise ValidationError('packing.first_nonfitting: wrong next ranked item')
        b=pos_int(n['canonical_payload_bytes'],'packing.first_nonfitting.canonical_payload_bytes')
        if not (b>packer['budget']['max_output_item_bytes'] or total+b>packer['budget']['max_output_bytes']): raise ValidationError('packing.first_nonfitting: item actually fits budget')
    else: raise ValidationError('packing.stop_reason unsupported')

def validate_trace(trace, bundle_path, repo_root):
    compute_used=trace_shape(trace); gp=load_graph_profile()
    raw=bundle_path.read_bytes(); bundle=json.loads(raw.decode('utf-8')); by=gp.load_bundle_artifacts(bundle,repo_root,True); report=gp.validate_docs(bundle,by)
    report['bundle_sha256']=dg(raw); exp_item=next(x for x in by.values() if x['meta']['kind']=='ExperimentManifest'); report['experiment_sha256']=exp_item['meta']['sha256']; exp=exp_item['doc']
    report_bytes=(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n').encode()
    g=trace['graph']
    if g['bundle_sha256']!=report['bundle_sha256'] or g['experiment_sha256']!=report['experiment_sha256'] or g['graph_report_sha256']!=dg(report_bytes): raise ValidationError('graph: bundle/experiment/report identity mismatch')
    if g['candidate_set_sha256']!=report['shared_candidate_set_sha256'] or g['candidate_count']!=report['shared_candidate_count'] or g['context_packer_sha256']!=report['shared_context_packer_sha256']: raise ValidationError('graph: qualified shared retrieval identity mismatch')
    if trace['experiment_seed'] not in exp['seeds']: raise ValidationError('experiment_seed not preregistered')
    arm=next((a for a in exp['arms'] if a['arm_id']==g['arm_id']),None)
    if arm is None or arm['retriever_family']=='None': raise ValidationError('graph.arm_id: retrieval arm required')
    if arm['retrieval_binding_sha256']!=g['retrieval_binding_sha256']: raise ValidationError('graph: retrieval binding mismatch')
    binding=gp.need(by,g['retrieval_binding_sha256'],'RetrievalBinding'); packer=gp.need(by,binding['context_packer_sha256'],'ContextPacker')
    if g['context_packer_sha256']!=binding['context_packer_sha256'] or trace['packing']['context_packer_sha256']!=binding['context_packer_sha256']: raise ValidationError('context packer digest mismatch')
    qd=trace['query']['query_source_object_sha256']; r=trace['retrieval']; participating={}
    if binding['mode']=='SingleIndex':
        closed(r,{'mode','single'},'retrieval')
        if r['mode']!='SingleIndex': raise ValidationError('retrieval.mode mismatch')
        s=r['single']; closed(s,{'index_manifest_sha256','index_artifact_sha256','requested_k','ranked_source_object_digests'},'retrieval.single')
        d=sh(s['index_manifest_sha256'],'retrieval.single.index_manifest_sha256'); ix=gp.need(by,d,'RetrievalIndex'); participating[d]=ix
        if d!=binding['index_manifest_sha256'] or s['index_artifact_sha256']!=ix['index']['index_artifact_sha256']: raise ValidationError('retrieval.single: index identity mismatch')
        pos_int(s['requested_k'],'retrieval.single.requested_k'); ranked=unique_digests(s['ranked_source_object_digests'],'retrieval.single.ranked_source_object_digests')
        if len(ranked)>s['requested_k'] or s['requested_k']>exp['budget']['retrieved_items_max'] or s['requested_k']>ix['index']['top_k_supported']: raise ValidationError('retrieval.single: requested/returned k exceeds qualified limits')
        if trace['resources']['retrieval_queries_used']!=1: raise ValidationError('resources: SingleIndex requires exactly one retrieval query')
    elif binding['mode']=='Fusion':
        closed(r,{'mode','fusion'},'retrieval')
        if r['mode']!='Fusion': raise ValidationError('retrieval.mode mismatch')
        ftrace=r['fusion']; closed(ftrace,{'fusion_policy_sha256','channels','fused_ranked_source_object_digests'},'retrieval.fusion')
        fd=sh(ftrace['fusion_policy_sha256'],'retrieval.fusion.fusion_policy_sha256'); fusion=gp.need(by,fd,'FusionPolicy')
        if fd!=binding['fusion_policy_sha256']: raise ValidationError('retrieval.fusion: policy identity mismatch')
        xs=ftrace['channels']
        if not isinstance(xs,list) or len(xs)!=2: raise ValidationError('retrieval.fusion.channels: exactly two required')
        channels={}
        fins={x['channel']:x for x in fusion['inputs']}
        for i,x in enumerate(xs):
            channel_result_shape(x,f'retrieval.fusion.channels[{i}]')
            if x['channel'] in channels: raise ValidationError('retrieval.fusion.channels: duplicate channel')
            channels[x['channel']]=x
        if set(channels)!={'Syntax','ExactNormalForm'}: raise ValidationError('retrieval.fusion.channels: Syntax + ExactNormalForm required')
        for ch,d in (('Syntax',binding['syntax_index_manifest_sha256']),('ExactNormalForm',binding['normal_form_index_manifest_sha256'])):
            x=channels[ch]; ix=gp.need(by,d,'RetrievalIndex'); participating[d]=ix
            if x['index_manifest_sha256']!=d or x['index_artifact_sha256']!=ix['index']['index_artifact_sha256']: raise ValidationError(f'retrieval.fusion.{ch}: index identity mismatch')
            if x['requested_k']>fins[ch]['max_input_items'] or x['requested_k']>ix['index']['top_k_supported'] or x['input_bytes_used']>fins[ch]['max_input_bytes']: raise ValidationError(f'retrieval.fusion.{ch}: input quota exceeded')
        ranked=unique_digests(ftrace['fused_ranked_source_object_digests'],'retrieval.fusion.fused_ranked_source_object_digests')
        expected=recompute_fusion(fusion,channels)
        if ranked!=expected: raise ValidationError('retrieval.fusion: fused ranking does not replay under frozen policy')
        if trace['resources']['retrieval_queries_used']!=2: raise ValidationError('resources: Fusion requires exactly two retrieval queries')
    else: raise ValidationError('binding.mode unsupported')
    if qd in ranked: raise ValidationError('retrieval: query source item returned despite exclusion policy')
    validate_control_bindings(trace,participating); validate_packing(trace['packing'],ranked,packer)
    if trace['resources']['wall_time_ms_used']>exp['budget']['wall_time_ms_max']: raise ValidationError('resources: wall-time budget exceeded')
    if trace['resources']['retrieval_queries_used']>exp['budget']['retrieval_queries_max']: raise ValidationError('resources: retrieval-query budget exceeded')
    if compute_used>Decimal(str(exp['budget']['normalized_compute_units_max'])): raise ValidationError('resources: normalized-compute budget exceeded')
    return {'version':'math-retrieval-trace-validation-report-v1','authority':AUTH,'trace_id':trace['trace_id'],'trace_sha256':None,'graph_report_sha256':g['graph_report_sha256'],'experiment_sha256':g['experiment_sha256'],'arm_id':g['arm_id'],'experiment_seed':trace['experiment_seed'],'packed_output_items':trace['packing']['output_items_used'],'packed_output_bytes':trace['packing']['output_bytes_used'],'all_checks_passed':True}

def self_test():
    # Pure replay/packing canaries; repository graph integration is a focused gate.
    f={'fusion':{'method':'ReciprocalRankFusion','rrf_k':60}}
    ch={'Syntax':{'ranked_source_object_digests':['a','b']},'ExactNormalForm':{'ranked_source_object_digests':['b','c']}}
    if compute_fractional_rrf(f,ch)!=['b','a','c']: raise AssertionError('RRF replay failed')
    i={'fusion':{'method':'DeterministicInterleave','interleave_start_channel':'Syntax'}}
    if compute_interleave(i,ch)!=['a','b','c']: raise AssertionError('interleave replay failed')
    p={'budget':{'max_output_items':2,'max_output_bytes':10,'max_output_item_bytes':8}}
    good={'context_packer_sha256':'sha256:'+'0'*64,'input_ranked_source_object_digests':['sha256:'+'1'*64,'sha256:'+'2'*64],'output':[{'rank':1,'source_object_sha256':'sha256:'+'1'*64,'canonical_payload_bytes':6}],'output_items_used':1,'output_bytes_used':6,'stop_reason':'FirstNonFittingItem','first_nonfitting':{'rank':2,'source_object_sha256':'sha256:'+'2'*64,'canonical_payload_bytes':5}}
    validate_packing(good,good['input_ranked_source_object_digests'],p)
    bad=json.loads(json.dumps(good)); bad['output'][0]['source_object_sha256']='sha256:'+'2'*64
    try: validate_packing(bad,good['input_ranked_source_object_digests'],p)
    except ValidationError: pass
    else: raise AssertionError('non-prefix packing attack passed')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('trace',nargs='?',type=Path); ap.add_argument('bundle',nargs='?',type=Path); ap.add_argument('--repo-root',type=Path,default=Path.cwd()); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--report',type=Path); a=ap.parse_args()
    if a.self_test: self_test(); print('math-retrieval trace v1 pure self-test: PASS'); return 0
    if a.trace is None or a.bundle is None: ap.error('trace and bundle required unless --self-test')
    try:
        raw=a.trace.read_bytes(); trace=json.loads(raw.decode('utf-8')); rep=validate_trace(trace,a.bundle,a.repo_root); rep['trace_sha256']=dg(raw)
    except (OSError,json.JSONDecodeError,ValidationError) as e: print(f'INVALID: {e}',file=sys.stderr); return 1
    out=json.dumps(rep,sort_keys=True,separators=(',',':'))
    if a.report: a.report.write_text(out+'\n',encoding='utf-8')
    print(out); return 0
if __name__=='__main__': raise SystemExit(main())
