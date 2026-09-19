#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, hashlib, importlib.util, json, sys
from pathlib import Path

VERSION='math-retrieval-graph-bundle-v1'; AUTH='MeasurementOnly'
KINDS={'ExperimentManifest','RetrievalBinding','RetrievalIndex','FusionPolicy','ContextPacker'}
VALIDATORS={
 'ExperimentManifest':('validate-math-search-experiment-v2.1.py','validate_manifest'),
 'RetrievalBinding':('validate-math-retrieval-binding.py','validate'),
 'RetrievalIndex':('validate-math-retrieval-index.py','validate'),
 'FusionPolicy':('validate-math-retrieval-fusion.py','validate'),
 'ContextPacker':('validate-math-retrieval-context-packer.py','validate'),
}
class ValidationError(ValueError): pass

def digest_bytes(b): return 'sha256:'+hashlib.sha256(b).hexdigest()
def fusion_rep_identity(s,n,f):
    return digest_bytes((f'symthaea-fusion-representation-v1\nSyntax={s}\nExactNormalForm={n}\nFusionPolicy={f}\n').encode())

def shape(bundle):
    if not isinstance(bundle,dict) or set(bundle)!={'version','bundle_id','authority','artifacts'}: raise ValidationError('bundle: exact fields required')
    if bundle['version']!=VERSION or bundle['authority']!=AUTH: raise ValidationError('bundle: identity/authority invariant failed')
    if not isinstance(bundle['bundle_id'],str) or not bundle['bundle_id'].strip(): raise ValidationError('bundle_id required')
    arts=bundle['artifacts']
    if not isinstance(arts,list) or not arts: raise ValidationError('artifacts: non-empty list required')
    ps=set(); ds=set()
    for i,a in enumerate(arts):
        if not isinstance(a,dict) or set(a)!={'kind','path','sha256'}: raise ValidationError(f'artifacts[{i}]: exact kind/path/sha256 required')
        if a['kind'] not in KINDS: raise ValidationError(f'artifacts[{i}].kind unsupported')
        d=a['sha256']
        if not isinstance(d,str) or len(d)!=71 or not d.startswith('sha256:') or any(c not in '0123456789abcdef' for c in d[7:]): raise ValidationError(f'artifacts[{i}].sha256 invalid')
        if not isinstance(a['path'],str) or not a['path']: raise ValidationError(f'artifacts[{i}].path required')
        if a['path'] in ps or d in ds: raise ValidationError('bundle: duplicate path or digest alias')
        ps.add(a['path']); ds.add(d)
    if sum(a['kind']=='ExperimentManifest' for a in arts)!=1: raise ValidationError('bundle: exactly one ExperimentManifest required')
    return arts

def safe_path(root, rel):
    p=Path(rel)
    if p.is_absolute() or not rel or '\x00' in rel or any(x in ('','.','..') for x in p.parts): raise ValidationError(f'unsafe artifact path: {rel!r}')
    rr=root.resolve(); q=(rr/p).resolve()
    try: q.relative_to(rr)
    except ValueError as e: raise ValidationError(f'artifact path escapes repo root: {rel}') from e
    if not q.is_file(): raise ValidationError(f'artifact not found: {rel}')
    return q

def load_validator(name):
    p=Path(__file__).resolve().with_name(name)
    spec=importlib.util.spec_from_file_location('sym_'+name.replace('-','_').replace('.','_'),p)
    if spec is None or spec.loader is None: raise ValidationError(f'cannot load validator {p}')
    m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

def delegate(kind,doc):
    name,fn=VALIDATORS[kind]; m=load_validator(name)
    try: getattr(m,fn)(doc)
    except Exception as e:
        ets=tuple(t for t in (getattr(m,'ValidationError',None),getattr(m,'V',None)) if isinstance(t,type) and issubclass(t,Exception))
        if ets and isinstance(e,ets): raise ValidationError(f'{kind}: {e}') from e
        raise

def load_bundle_artifacts(bundle,root,do_delegate=True):
    by={}
    for a in shape(bundle):
        p=safe_path(root,a['path']); raw=p.read_bytes()
        try: doc=json.loads(raw.decode('utf-8'))
        except Exception as e: raise ValidationError(f'{a["path"]}: invalid UTF-8 JSON: {e}') from e
        d=digest_bytes(raw)
        if d!=a['sha256']: raise ValidationError(f'{a["path"]}: digest mismatch bundle={a["sha256"]} computed={d}')
        if do_delegate: delegate(a['kind'],doc)
        by[d]={'meta':a,'doc':doc}
    return by

def need(by,d,kind):
    x=by.get(d)
    if x is None: raise ValidationError(f'unresolved {kind} digest: {d}')
    if x['meta']['kind']!=kind: raise ValidationError(f'digest {d}: expected {kind}, got {x["meta"]["kind"]}')
    return x['doc']

def csig(ix):
    c=ix['candidate_universe']; ks=('corpus_snapshot_sha256','knowledge_boundary_sha256','candidate_eligibility_policy_sha256','candidate_set_sha256','candidate_count','exclude_query_source_item','exclude_solution_artifacts','source_identity_kind','dedup_policy','canonical_candidate_order')
    return tuple(c[k] for k in ks)

def expected_transform(control): return 'None' if control in ('None','LexicalRetrieval') else control

def check_packer(p,exp):
    if p['source']['source_object_contract_sha256']!=exp['shared_contract']['source_object_contract_sha256']: raise ValidationError('context packer: source-object contract drift')
    want=(exp['budget']['retrieved_items_max'],exp['budget']['retrieval_context_bytes_max'],exp['budget']['retrieved_item_bytes_max'])
    got=(p['budget']['max_output_items'],p['budget']['max_output_bytes'],p['budget']['max_output_item_bytes'])
    if got!=want: raise ValidationError(f'context packer: budget mismatch {got} != {want}')

def check_single(ix,arm,exp):
    c=ix['candidate_universe']; s=exp['shared_contract']; r=ix['representation']
    if c['corpus_snapshot_sha256']!=s['corpus_snapshot_sha256'] or c['knowledge_boundary_sha256']!=s['knowledge_boundary_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: candidate universe source drift')
    if r['control_transform']!=expected_transform(arm['control_kind']): raise ValidationError(f'arm {arm["arm_id"]}: control transform mismatch')
    if arm['control_kind']=='RandomRetrieval':
        if arm['representation_family']!='None' or arm['representation_channels']: raise ValidationError(f'arm {arm["arm_id"]}: random control must have no representation')
    else:
        if arm['representation_family']!=r['representation_family'] or len(arm['representation_channels'])!=1 or arm['representation_channels'][0]!=r['channel'] or arm['representation_sha256']!=r['representation_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: representation binding mismatch')
    if r['channel']=='ExactNormalForm':
        if r['normalization_contract_sha256']!=s['normalization_contract_sha256'] or r['normalization_implementation_sha256']!=s['normalization_implementation_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: normalizer drift')
    if ix['index']['top_k_supported']<exp['budget']['retrieved_items_max']: raise ValidationError(f'arm {arm["arm_id"]}: index top-k cannot satisfy root item ceiling')
    return csig(ix)

def check_fusion(f,b,arm,exp,six,nix,packer):
    if set(arm['representation_channels'])!={'Syntax','ExactNormalForm'} or arm['retriever_family']!='Fusion': raise ValidationError(f'arm {arm["arm_id"]}: invalid fusion arm shape')
    if arm.get('fusion_policy_sha256')!=b['fusion_policy_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: fusion policy binding mismatch')
    ins={x['channel']:x for x in f['inputs']}
    if ins['Syntax']['index_manifest_sha256']!=b['syntax_index_manifest_sha256'] or ins['ExactNormalForm']['index_manifest_sha256']!=b['normal_form_index_manifest_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: fusion input digest mismatch')
    if six['representation']['channel']!='Syntax' or nix['representation']['channel']!='ExactNormalForm': raise ValidationError(f'arm {arm["arm_id"]}: fusion channel mismatch')
    if arm['representation_family']=='FusionHDC':
        if six['representation']['representation_family']!='HDC' or nix['representation']['representation_family']!='HDC': raise ValidationError(f'arm {arm["arm_id"]}: FusionHDC requires two HDC indices')
    elif arm['representation_family']=='FusionConventional':
        if six['representation']['representation_family'] not in {'Lexical','CanonicalSparse'} or nix['representation']['representation_family']!='CanonicalSparse': raise ValidationError(f'arm {arm["arm_id"]}: conventional fusion family mismatch')
    else: raise ValidationError(f'arm {arm["arm_id"]}: fusion binding requires fusion representation family')
    s=exp['shared_contract']; nr=nix['representation']
    if nr['normalization_contract_sha256']!=s['normalization_contract_sha256'] or nr['normalization_implementation_sha256']!=s['normalization_implementation_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: fusion normalizer drift')
    if csig(six)!=csig(nix): raise ValidationError(f'arm {arm["arm_id"]}: fusion input candidate universes differ')
    for ch,ix in (('Syntax',six),('ExactNormalForm',nix)):
        if ins[ch]['max_input_items']>ix['index']['top_k_supported']: raise ValidationError(f'arm {arm["arm_id"]}: {ch} fusion quota exceeds index top-k')
    want=(exp['budget']['retrieved_items_max'],exp['budget']['retrieval_context_bytes_max'],exp['budget']['retrieved_item_bytes_max'])
    got=(f['global_budget']['max_output_items'],f['global_budget']['max_output_bytes'],f['global_budget']['max_output_item_bytes'])
    if got!=want: raise ValidationError(f'arm {arm["arm_id"]}: fusion/root budget mismatch')
    if f['output']['payload_serialization_sha256']!=packer['source']['payload_serialization_sha256']: raise ValidationError(f'arm {arm["arm_id"]}: fusion payload serialization differs from packer')
    want_rep=fusion_rep_identity(six['representation']['representation_sha256'],nix['representation']['representation_sha256'],b['fusion_policy_sha256'])
    if arm['representation_sha256']!=want_rep: raise ValidationError(f'arm {arm["arm_id"]}: fusion representation identity mismatch')
    return csig(six)

def validate_docs(bundle,by):
    arts=shape(bundle); eis=[x for x in by.values() if x['meta']['kind']=='ExperimentManifest']
    if len(eis)!=1: raise ValidationError('graph: exactly one experiment doc required')
    exp=eis[0]['doc']; rarms=[a for a in exp['arms'] if a['retriever_family']!='None']; barms=[a for a in exp['arms'] if a['retriever_family']=='None']
    if not rarms or not barms: raise ValidationError('graph: retrieval arm(s) and baseline required')
    refs={eis[0]['meta']['sha256']}; shared_p=None; shared_c=None
    for arm in rarms:
        bd=arm['retrieval_binding_sha256']; b=need(by,bd,'RetrievalBinding'); refs.add(bd)
        if b['arm_id']!=arm['arm_id']: raise ValidationError(f'arm {arm["arm_id"]}: binding arm_id mismatch')
        pd=b['context_packer_sha256']; p=need(by,pd,'ContextPacker'); refs.add(pd); check_packer(p,exp)
        if shared_p is None: shared_p=pd
        elif pd!=shared_p: raise ValidationError('graph: retrieval arms do not share one exact context packer')
        if b['mode']=='SingleIndex':
            d=b['index_manifest_sha256']; ix=need(by,d,'RetrievalIndex'); refs.add(d); sig=check_single(ix,arm,exp)
        elif b['mode']=='Fusion':
            sd,nd,fd=b['syntax_index_manifest_sha256'],b['normal_form_index_manifest_sha256'],b['fusion_policy_sha256']; six=need(by,sd,'RetrievalIndex'); nix=need(by,nd,'RetrievalIndex'); f=need(by,fd,'FusionPolicy'); refs.update((sd,nd,fd)); sig=check_fusion(f,b,arm,exp,six,nix,p)
        else: raise ValidationError(f'arm {arm["arm_id"]}: unsupported binding mode')
        if shared_c is None: shared_c=sig
        elif sig!=shared_c: raise ValidationError('graph: retrieval arms do not share one exact candidate universe')
    for a in barms:
        if 'retrieval_binding_sha256' in a: raise ValidationError(f'baseline {a["arm_id"]}: unexpected retrieval binding')
    all_d={a['sha256'] for a in arts}; un=all_d-refs
    if un: raise ValidationError('graph: unreferenced artifacts present: '+repr([by[d]['meta']['path'] for d in sorted(un)]))
    ix=next(x['doc'] for x in by.values() if x['meta']['kind']=='RetrievalIndex' and csig(x['doc'])==shared_c); c=ix['candidate_universe']
    return {'version':'math-retrieval-graph-validation-report-v1','authority':AUTH,'experiment_id':exp['experiment_id'],'retrieval_arm_count':len(rarms),'artifact_count':len(arts),'shared_candidate_set_sha256':c['candidate_set_sha256'],'shared_candidate_count':c['candidate_count'],'shared_context_packer_sha256':shared_p,'normalization_contract_sha256':exp['shared_contract']['normalization_contract_sha256'],'normalization_implementation_sha256':exp['shared_contract']['normalization_implementation_sha256'],'budget':{'retrieved_items_max':exp['budget']['retrieved_items_max'],'retrieved_item_bytes_max':exp['budget']['retrieved_item_bytes_max'],'retrieval_context_bytes_max':exp['budget']['retrieval_context_bytes_max']},'all_checks_passed':True}

def validate_graph(bundle_path,repo_root,do_delegate=True):
    raw=bundle_path.read_bytes(); bundle=json.loads(raw.decode('utf-8')); by=load_bundle_artifacts(bundle,repo_root,do_delegate); rep=validate_docs(bundle,by); rep['bundle_sha256']=digest_bytes(raw); rep['experiment_sha256']=next(x['meta']['sha256'] for x in by.values() if x['meta']['kind']=='ExperimentManifest'); return rep

def self_test():
    dg=lambda s:'sha256:'+hashlib.sha256(s.encode()).hexdigest()
    shared={'challenge_set_sha256':dg('challenge'),'corpus_snapshot_sha256':dg('corpus'),'knowledge_boundary_sha256':dg('knowledge'),'source_object_contract_sha256':dg('source'),'normalization_contract_sha256':dg('nc'),'normalization_implementation_sha256':dg('ni'),'toolchain_manifest_sha256':dg('tool'),'human_intervention_policy_sha256':dg('human')}
    budget={'accounting_policy_sha256':dg('ba'),'retrieved_items_max':4,'retrieved_item_bytes_max':4096,'retrieval_context_bytes_max':8192,'total_context_bytes_max':16384,'normalized_compute_units_max':100.0,'wall_time_ms_max':10000,'proof_calls_max':8,'solver_calls_max':8,'search_nodes_max':1000,'candidate_count_max':100,'retrieval_queries_max':1,'unused_budget_reallocation':False}
    cand={'corpus_snapshot_sha256':shared['corpus_snapshot_sha256'],'knowledge_boundary_sha256':shared['knowledge_boundary_sha256'],'candidate_eligibility_policy_sha256':dg('elig'),'candidate_set_sha256':dg('set'),'candidate_count':100,'exclude_query_source_item':True,'exclude_solution_artifacts':True,'source_identity_kind':'SourceObjectDigest','dedup_policy':'OneEntryPerSourceObject','canonical_candidate_order':'SourceObjectDigestAscending'}
    def ix(ch,fam,rs):
        r={'channel':ch,'representation_family':fam,'representation_sha256':rs,'item_serialization_sha256':dg(ch+'ser'),'serialization_encoding':'UTF-8','max_serialized_item_bytes':2048,'oversize_policy':'RejectItem','control_transform':'None'}
        if ch=='ExactNormalForm': r.update(normalization_contract_sha256=shared['normalization_contract_sha256'],normalization_implementation_sha256=shared['normalization_implementation_sha256'])
        return {'version':'math-retrieval-index-v1','index_id':ch,'authority':AUTH,'candidate_universe':copy.deepcopy(cand),'representation':r,'index':{'search_mode':'ExactDeterministic','index_build_policy_sha256':dg(ch+'b'),'index_artifact_sha256':dg(ch+'a'),'index_seed':1,'top_k_supported':4,'scoring_metric':'Cosine','scoring_policy_sha256':dg(ch+'s'),'score_precision_policy_sha256':dg(ch+'p'),'query_normalization_policy_sha256':dg(ch+'q'),'tie_break':'SourceObjectDigestAscending','deterministic':True},'retrieval_output':{'order_policy':'ScoreDescendingThenSourceObjectDigest','source_provenance_required':True,'item_byte_accounting':'CanonicalUtf8Bytes','partial_item_policy':'RejectWholeItem'}}
    sx=ix('Syntax','HDC',dg('sr')); nx=ix('ExactNormalForm','HDC',dg('nr'))
    pack={'version':'math-retrieval-context-packer-v1','packer_id':'p','authority':AUTH,'source':{'payload_kind':'CanonicalSourceObject','source_object_contract_sha256':shared['source_object_contract_sha256'],'source_fetch_policy_sha256':dg('fetch'),'payload_serialization_sha256':dg('payload')},'visibility':{},'packing':{},'budget':{'max_output_items':4,'max_output_bytes':8192,'max_output_item_bytes':4096}}
    ds,dn,dp,df,db,de=map(dg,['sx','nx','pack','fusion','binding','exp'])
    fusion={'version':'math-retrieval-fusion-v1','fusion_id':'f','authority':AUTH,'inputs':[{'channel':'Syntax','index_manifest_sha256':ds,'max_input_items':2,'max_input_bytes':4096,'rank_weight':1},{'channel':'ExactNormalForm','index_manifest_sha256':dn,'max_input_items':2,'max_input_bytes':4096,'rank_weight':1}],'fusion':{},'global_budget':{'max_output_items':4,'max_output_bytes':8192,'max_output_item_bytes':4096},'output':{'payload_serialization_sha256':pack['source']['payload_serialization_sha256']}}
    binding={'version':'math-retrieval-binding-v1','binding_id':'b','authority':AUTH,'arm_id':'F','mode':'Fusion','context_packer_sha256':dp,'syntax_index_manifest_sha256':ds,'normal_form_index_manifest_sha256':dn,'fusion_policy_sha256':df}
    exp={'schema_version':'symthaea.math-search-experiment.v2.1','experiment_id':'e','shared_contract':shared,'budget':budget,'arms':[{'arm_id':'A','retriever_family':'None'},{'arm_id':'F','retriever_family':'Fusion','retrieval_binding_sha256':db,'representation_channels':['Syntax','ExactNormalForm'],'representation_family':'FusionHDC','representation_sha256':fusion_rep_identity(sx['representation']['representation_sha256'],nx['representation']['representation_sha256'],df),'fusion_policy_sha256':df}]}
    arts=[{'kind':'ExperimentManifest','path':'e','sha256':de},{'kind':'RetrievalBinding','path':'b','sha256':db},{'kind':'RetrievalIndex','path':'s','sha256':ds},{'kind':'RetrievalIndex','path':'n','sha256':dn},{'kind':'FusionPolicy','path':'f','sha256':df},{'kind':'ContextPacker','path':'p','sha256':dp}]
    bun={'version':VERSION,'bundle_id':'x','authority':AUTH,'artifacts':arts}; by={de:{'meta':arts[0],'doc':exp},db:{'meta':arts[1],'doc':binding},ds:{'meta':arts[2],'doc':sx},dn:{'meta':arts[3],'doc':nx},df:{'meta':arts[4],'doc':fusion},dp:{'meta':arts[5],'doc':pack}}
    validate_docs(bun,by)
    attacks=[('candidate drift',lambda x:x[dn]['doc']['candidate_universe'].__setitem__('candidate_set_sha256',dg('bad'))),('packer budget',lambda x:x[dp]['doc']['budget'].__setitem__('max_output_items',5)),('normalizer drift',lambda x:x[dn]['doc']['representation'].__setitem__('normalization_implementation_sha256',dg('bad'))),('payload drift',lambda x:x[df]['doc']['output'].__setitem__('payload_serialization_sha256',dg('bad')))]
    for name,fn in attacks:
        z=copy.deepcopy(by); fn(z)
        try: validate_docs(bun,z)
        except ValidationError: continue
        raise AssertionError('attack passed: '+name)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('bundle',nargs='?',type=Path); ap.add_argument('--repo-root',type=Path,default=Path.cwd()); ap.add_argument('--self-test',action='store_true'); ap.add_argument('--report',type=Path); a=ap.parse_args()
    if a.self_test: self_test(); print('math-retrieval graph v1 cross-contract self-test: PASS'); return 0
    if a.bundle is None: ap.error('bundle path required unless --self-test')
    try: r=validate_graph(a.bundle,a.repo_root,True)
    except (OSError,json.JSONDecodeError,ValidationError) as e: print('INVALID:',e,file=sys.stderr); return 1
    out=json.dumps(r,sort_keys=True,separators=(',',':'))
    if a.report: a.report.write_text(out+'\n',encoding='utf-8')
    print(out); return 0
if __name__=='__main__': raise SystemExit(main())
