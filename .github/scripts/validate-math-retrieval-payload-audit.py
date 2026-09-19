#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, importlib.util, json, sys, tempfile
from pathlib import Path

VERSION='math-retrieval-payload-audit-v1'; AUTH='MeasurementOnly'
class ValidationError(ValueError): pass

def dg(b:bytes): return 'sha256:'+hashlib.sha256(b).hexdigest()
def sh(x,w):
    if not isinstance(x,str) or len(x)!=71 or not x.startswith('sha256:') or any(c not in '0123456789abcdef' for c in x[7:]): raise ValidationError(f'{w}: invalid sha256')
    return x

def safe_path(root,rel):
    p=Path(rel)
    if p.is_absolute() or not rel or '\x00' in rel or any(x in ('','.','..') for x in p.parts): raise ValidationError(f'unsafe payload path: {rel!r}')
    rr=root.resolve(); q=(rr/p).resolve()
    try: q.relative_to(rr)
    except ValueError as e: raise ValidationError(f'payload path escapes root: {rel}') from e
    if not q.is_file(): raise ValidationError(f'payload file missing: {rel}')
    return q

def load_trace_validator():
    p=Path(__file__).with_name('validate-math-retrieval-trace.py')
    s=importlib.util.spec_from_file_location('sym_retrieval_trace_v1',p)
    if s is None or s.loader is None: raise ValidationError(f'cannot load trace validator: {p}')
    m=importlib.util.module_from_spec(s); s.loader.exec_module(m); return m

def shape(bundle):
    if not isinstance(bundle,dict) or set(bundle)!={'version','audit_id','authority','trace_sha256','graph_report_sha256','context_packer_sha256','payload_serialization_sha256','entries'}: raise ValidationError('payload audit: exact fields required')
    if bundle['version']!=VERSION or bundle['authority']!=AUTH: raise ValidationError('payload audit: identity/authority invariant failed')
    if not isinstance(bundle['audit_id'],str) or not bundle['audit_id'].strip(): raise ValidationError('audit_id required')
    for k in ('trace_sha256','graph_report_sha256','context_packer_sha256','payload_serialization_sha256'): sh(bundle[k],k)
    xs=bundle['entries']
    if not isinstance(xs,list): raise ValidationError('entries: list required')
    seen=set()
    for i,x in enumerate(xs):
        if not isinstance(x,dict) or set(x)!={'role','rank','source_object_sha256','payload_path','payload_sha256','canonical_payload_bytes'}: raise ValidationError(f'entries[{i}]: exact fields required')
        if x['role'] not in {'Delivered','FirstNonFitting'}: raise ValidationError(f'entries[{i}].role unsupported')
        if not isinstance(x['rank'],int) or isinstance(x['rank'],bool) or x['rank']<1: raise ValidationError(f'entries[{i}].rank positive integer required')
        sh(x['source_object_sha256'],f'entries[{i}].source_object_sha256'); sh(x['payload_sha256'],f'entries[{i}].payload_sha256')
        if not isinstance(x['payload_path'],str) or not x['payload_path']: raise ValidationError(f'entries[{i}].payload_path required')
        if not isinstance(x['canonical_payload_bytes'],int) or isinstance(x['canonical_payload_bytes'],bool) or x['canonical_payload_bytes']<1: raise ValidationError(f'entries[{i}].canonical_payload_bytes positive integer required')
        key=(x['role'],x['rank'],x['source_object_sha256'])
        if key in seen: raise ValidationError('entries: duplicate audit target')
        seen.add(key)
    return xs

def trace_targets(trace):
    out=[]
    for x in trace['packing']['output']:
        out.append(('Delivered',x['rank'],x['source_object_sha256'],x['canonical_payload_bytes']))
    if trace['packing']['stop_reason']=='FirstNonFittingItem':
        x=trace['packing']['first_nonfitting']; out.append(('FirstNonFitting',x['rank'],x['source_object_sha256'],x['canonical_payload_bytes']))
    return out

def audit_payloads(bundle,trace,payload_root,packer):
    xs=shape(bundle); targets=trace_targets(trace)
    if len(xs)!=len(targets): raise ValidationError('entries: must exactly cover packed outputs plus first-nonfitting target when present')
    total_delivered=0; first_nonfit=False
    for i,(entry,target) in enumerate(zip(xs,targets)):
        role,rank,source,claimed=target
        if (entry['role'],entry['rank'],entry['source_object_sha256'])!=(role,rank,source): raise ValidationError(f'entries[{i}]: role/rank/source does not match trace target order')
        if entry['canonical_payload_bytes']!=claimed: raise ValidationError(f'entries[{i}]: byte claim differs from trace')
        p=safe_path(payload_root,entry['payload_path']); raw=p.read_bytes()
        try: raw.decode('utf-8')
        except UnicodeDecodeError as e: raise ValidationError(f'entries[{i}]: canonical payload is not UTF-8') from e
        if dg(raw)!=entry['payload_sha256']: raise ValidationError(f'entries[{i}]: payload SHA-256 mismatch')
        if len(raw)!=entry['canonical_payload_bytes']: raise ValidationError(f'entries[{i}]: actual byte length mismatch')
        if role=='Delivered': total_delivered+=len(raw)
        else: first_nonfit=True
    if total_delivered!=trace['packing']['output_bytes_used']: raise ValidationError('payload audit: delivered actual bytes do not reconcile to trace output_bytes_used')
    if len(trace['packing']['output'])!=trace['packing']['output_items_used']: raise ValidationError('payload audit: trace output item count inconsistent')
    if first_nonfit:
        n=xs[-1]['canonical_payload_bytes']; b=packer['budget']; used=trace['packing']['output_bytes_used']
        if not (n>b['max_output_item_bytes'] or used+n>b['max_output_bytes']): raise ValidationError('payload audit: actual first-nonfitting payload would fit')
    return {'audited_entry_count':len(xs),'delivered_payload_bytes':total_delivered,'first_nonfitting_audited':first_nonfit}

def validate(bundle_path,trace_path,graph_bundle_path,repo_root,payload_root):
    tv=load_trace_validator(); trace_raw=trace_path.read_bytes(); trace=json.loads(trace_raw.decode('utf-8'))
    trep=tv.validate_trace(trace,graph_bundle_path,repo_root); trep['trace_sha256']=dg(trace_raw)
    raw=bundle_path.read_bytes(); bundle=json.loads(raw.decode('utf-8')); shape(bundle)
    if bundle['trace_sha256']!=trep['trace_sha256'] or bundle['graph_report_sha256']!=trep['graph_report_sha256']: raise ValidationError('payload audit: trace/report identity mismatch')
    if bundle['context_packer_sha256']!=trace['graph']['context_packer_sha256']: raise ValidationError('payload audit: context-packer identity mismatch')
    gp=tv.load_graph_profile(); graph_raw=graph_bundle_path.read_bytes(); graph_bundle=json.loads(graph_raw.decode('utf-8')); by=gp.load_bundle_artifacts(graph_bundle,repo_root,True)
    packer=gp.need(by,bundle['context_packer_sha256'],'ContextPacker')
    if bundle['payload_serialization_sha256']!=packer['source']['payload_serialization_sha256']: raise ValidationError('payload audit: payload-serialization identity mismatch')
    r=audit_payloads(bundle,trace,payload_root,packer)
    return {'version':'math-retrieval-payload-audit-report-v1','authority':AUTH,'audit_id':bundle['audit_id'],'audit_sha256':dg(raw),'trace_sha256':trep['trace_sha256'],'graph_report_sha256':trep['graph_report_sha256'],'context_packer_sha256':bundle['context_packer_sha256'],'payload_serialization_sha256':bundle['payload_serialization_sha256'],**r,'all_checks_passed':True}

def self_test():
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); (root/'a.txt').write_bytes(b'alpha'); (root/'b.txt').write_bytes(b'123456')
        s=lambda x:'sha256:'+hashlib.sha256(x.encode()).hexdigest()
        trace={'packing':{'output':[{'rank':1,'source_object_sha256':s('src-a'),'canonical_payload_bytes':5}],'output_items_used':1,'output_bytes_used':5,'stop_reason':'FirstNonFittingItem','first_nonfitting':{'rank':2,'source_object_sha256':s('src-b'),'canonical_payload_bytes':6}}}
        packer={'budget':{'max_output_items':2,'max_output_bytes':10,'max_output_item_bytes':8}}
        bun={'version':VERSION,'audit_id':'x','authority':AUTH,'trace_sha256':s('trace'),'graph_report_sha256':s('graph'),'context_packer_sha256':s('packer'),'payload_serialization_sha256':s('ser'),'entries':[
            {'role':'Delivered','rank':1,'source_object_sha256':s('src-a'),'payload_path':'a.txt','payload_sha256':dg(b'alpha'),'canonical_payload_bytes':5},
            {'role':'FirstNonFitting','rank':2,'source_object_sha256':s('src-b'),'payload_path':'b.txt','payload_sha256':dg(b'123456'),'canonical_payload_bytes':6},
        ]}
        r=audit_payloads(bun,trace,root,packer)
        if r['delivered_payload_bytes']!=5 or not r['first_nonfitting_audited']: raise AssertionError('payload audit result wrong')
        bad=json.loads(json.dumps(bun)); bad['entries'][0]['canonical_payload_bytes']=4
        try: audit_payloads(bad,trace,root,packer)
        except ValidationError: pass
        else: raise AssertionError('wrong byte claim attack passed')
        bad=json.loads(json.dumps(bun)); bad['entries'][0]['payload_path']='../escape'
        try: audit_payloads(bad,trace,root,packer)
        except ValidationError: pass
        else: raise AssertionError('path traversal attack passed')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('audit',nargs='?',type=Path); ap.add_argument('trace',nargs='?',type=Path); ap.add_argument('graph_bundle',nargs='?',type=Path); ap.add_argument('--repo-root',type=Path,default=Path.cwd()); ap.add_argument('--payload-root',type=Path,default=Path.cwd()); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--report',type=Path); a=ap.parse_args()
    if a.self_test: self_test(); print('math-retrieval payload audit v1 pure self-test: PASS'); return 0
    if a.audit is None or a.trace is None or a.graph_bundle is None: ap.error('audit, trace, and graph_bundle required unless --self-test')
    try: r=validate(a.audit,a.trace,a.graph_bundle,a.repo_root,a.payload_root)
    except (OSError,json.JSONDecodeError,ValidationError) as e: print(f'INVALID: {e}',file=sys.stderr); return 1
    out=json.dumps(r,sort_keys=True,separators=(',',':'))
    if a.report: a.report.write_text(out+'\n',encoding='utf-8')
    print(out); return 0
if __name__=='__main__': raise SystemExit(main())
