#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math, pathlib, tempfile
from typing import Any, Iterable

SITE_SCHEMA='ll009i.site-evidence-pack.v1'
REQ_SCHEMA='ll009i.candidate-site-requirements.v1'
OUT_SCHEMA='ll009i.site-viability-receipt.v1'
BASELINE_CATEGORIES={'terrain','illumination','thermal','communications','geotechnical','protected_constraints'}
DIRECTIONS={'min','max'}

class SiteError(RuntimeError): pass

def canonical_bytes(v:Any)->bytes:
    return (json.dumps(v,sort_keys=True,indent=2,separators=(',', ': '))+'\n').encode()

def sha256_bytes(b:bytes)->str:return hashlib.sha256(b).hexdigest()
def sha256_file(p:pathlib.Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def load(p:pathlib.Path)->dict[str,Any]:
    try:v=json.loads(p.read_text())
    except Exception as e:raise SiteError(f'cannot read {p}: {e}') from e
    if not isinstance(v,dict):raise SiteError(f'{p}: expected JSON object')
    return v
def finite(x:Any)->bool:return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def safe_rel(path:str)->pathlib.PurePosixPath:
    q=pathlib.PurePosixPath(path)
    if q.is_absolute() or '..' in q.parts or not q.parts:raise SiteError(f'unsafe artifact path: {path!r}')
    return q

def validate_site(pack:dict[str,Any],root:pathlib.Path)->None:
    if pack.get('schema_version')!=SITE_SCHEMA:raise SiteError('unsupported site pack schema')
    for k in ('study_id','frame_contract_id','epoch_contract_id','site_ref'):
        if not isinstance(pack.get(k),str) or not pack[k]:raise SiteError(f'site pack missing {k}')
    sources=pack.get('sources')
    if not isinstance(sources,list) or not sources:raise SiteError('site pack requires sources')
    ids=set()
    for s in sources:
        if not isinstance(s,dict):raise SiteError('site source must be object')
        sid,path,dig=s.get('source_id'),s.get('path'),s.get('sha256')
        if not all(isinstance(x,str) and x for x in (sid,path,dig)):raise SiteError('malformed site source')
        if sid in ids:raise SiteError(f'duplicate source_id {sid}')
        ids.add(sid); p=root/pathlib.Path(*safe_rel(path).parts)
        if not p.is_file():raise SiteError(f'missing source artifact {path}')
        if sha256_file(p)!=dig:raise SiteError(f'source digest mismatch: {path}')
    metrics=pack.get('metrics')
    if not isinstance(metrics,dict) or not metrics:raise SiteError('site pack requires metrics')
    for name,m in metrics.items():
        if not isinstance(name,str) or not name or not isinstance(m,dict):raise SiteError('malformed metric')
        status=m.get('status')
        if status not in {'available','unresolved'}:raise SiteError(f'{name}: invalid status')
        cat=m.get('category')
        if cat not in BASELINE_CATEGORIES:raise SiteError(f'{name}: invalid category {cat!r}')
        if status=='available':
            vals=[m.get('low'),m.get('central'),m.get('high')]
            if not all(finite(x) for x in vals) or not vals[0]<=vals[1]<=vals[2]:raise SiteError(f'{name}: invalid interval')
            if not isinstance(m.get('unit'),str) or not m['unit']:raise SiteError(f'{name}: missing unit')
            if not isinstance(m.get('evidence_class'),str) or not m['evidence_class']:raise SiteError(f'{name}: missing evidence class')
            refs=m.get('source_refs')
            if not isinstance(refs,list) or not refs or any(r not in ids for r in refs):raise SiteError(f'{name}: invalid source refs')

def check_threshold(metrics:dict[str,Any],c:dict[str,Any])->dict[str,Any]:
    metric,cat,dirn,unit=c.get('metric'),c.get('category'),c.get('direction'),c.get('unit')
    limit=c.get('limit'); allowed=c.get('allowed_evidence_classes')
    if not isinstance(metric,str) or not metric:raise SiteError('threshold missing metric')
    if cat not in BASELINE_CATEGORIES:raise SiteError(f'{metric}: invalid category')
    if dirn not in DIRECTIONS or not finite(limit) or not isinstance(unit,str) or not unit:raise SiteError(f'{metric}: invalid threshold')
    if not isinstance(allowed,list) or not allowed or any(not isinstance(x,str) or not x for x in allowed):raise SiteError(f'{metric}: invalid evidence-class policy')
    m=metrics.get(metric)
    if not isinstance(m,dict):return {'metric':metric,'category':cat,'status':'unresolved','reason':'metric missing'}
    if m.get('category')!=cat:raise SiteError(f'{metric}: category mismatch')
    if m.get('status')!='available':return {'metric':metric,'category':cat,'status':'unresolved','reason':'site evidence unresolved'}
    if m.get('unit')!=unit:raise SiteError(f'{metric}: unit mismatch {m.get("unit")!r} vs {unit!r}')
    if m.get('evidence_class') not in allowed:return {'metric':metric,'category':cat,'status':'unresolved','reason':'evidence class not admitted'}
    lo,hi=m['low'],m['high']
    if dirn=='max':
        margin=limit-hi
        status='pass' if hi<=limit else ('fail' if lo>limit else 'unresolved')
    else:
        margin=lo-limit
        status='pass' if lo>=limit else ('fail' if hi<limit else 'unresolved')
    return {'metric':metric,'category':cat,'status':status,'direction':dirn,'limit':limit,'unit':unit,'evidence_interval':{'low':lo,'central':m['central'],'high':hi},'evidence_class':m['evidence_class'],'conservative_margin':margin}

def validate_requirement(req:dict[str,Any],pack:dict[str,Any])->None:
    if req.get('schema_version')!=REQ_SCHEMA:raise SiteError('unsupported requirements schema')
    for k in ('candidate_id','architecture','study_id','frame_contract_id','epoch_contract_id','site_ref'):
        if not isinstance(req.get(k),str) or not req[k]:raise SiteError(f'requirement missing {k}')
    for k in ('study_id','frame_contract_id','epoch_contract_id','site_ref'):
        if req[k]!=pack[k]:raise SiteError(f'{req["candidate_id"]}: {k} mismatch')
    disp=req.get('category_dispositions')
    if not isinstance(disp,dict) or set(disp)!=BASELINE_CATEGORIES:raise SiteError(f'{req["candidate_id"]}: category dispositions must cover baseline exactly')
    for cat,val in disp.items():
        if not isinstance(val,dict) or val.get('status') not in {'required','not_applicable'}:raise SiteError(f'{req["candidate_id"]}: bad disposition {cat}')
        if val['status']=='not_applicable' and (not isinstance(val.get('reason'),str) or not val['reason'].strip()):raise SiteError(f'{req["candidate_id"]}: not_applicable {cat} requires reason')
    checks=req.get('checks')
    if not isinstance(checks,list):raise SiteError(f'{req["candidate_id"]}: checks must be list')
    covered=set()
    for item in checks:
        if not isinstance(item,dict):raise SiteError('check must be object')
        if item.get('type')=='threshold':covered.add(item.get('category'))
        elif item.get('type')=='any_of':
            opts=item.get('options')
            if not isinstance(opts,list) or len(opts)<2:raise SiteError('any_of requires >=2 options')
            for opt in opts:
                if not isinstance(opt,list) or not opt:raise SiteError('any_of option must be non-empty list')
                for c in opt:
                    if not isinstance(c,dict) or c.get('type')!='threshold':raise SiteError('any_of options contain thresholds only')
                    covered.add(c.get('category'))
        else:raise SiteError('unknown check type')
    for cat,val in disp.items():
        if val['status']=='required' and cat not in covered:raise SiteError(f'{req["candidate_id"]}: required category {cat} has no check')

def eval_check(metrics:dict[str,Any],item:dict[str,Any])->dict[str,Any]:
    if item['type']=='threshold':return {'type':'threshold',**check_threshold(metrics,item)}
    options=[]
    for i,opt in enumerate(item['options']):
        rs=[check_threshold(metrics,c) for c in opt]
        st='pass' if all(r['status']=='pass' for r in rs) else ('fail' if any(r['status']=='fail' for r in rs) and all(r['status']!='unresolved' for r in rs) else 'unresolved')
        options.append({'option_index':i,'status':st,'checks':rs})
    group='pass' if any(o['status']=='pass' for o in options) else ('unresolved' if any(o['status']=='unresolved' for o in options) else 'fail')
    return {'type':'any_of','name':item.get('name','unnamed'),'status':group,'options':options}

def requirement_id(req:dict[str,Any])->str:
    return 'site-req-'+sha256_bytes(canonical_bytes(req))

def evaluate(pack_path:pathlib.Path,artifact_root:pathlib.Path,req_paths:list[pathlib.Path])->dict[str,Any]:
    pack=load(pack_path);validate_site(pack,artifact_root)
    site_pack_sha=sha256_file(pack_path);site_pack_id='site-pack-'+sha256_bytes(canonical_bytes(pack))
    seen=set();results=[]
    for rp in req_paths:
        req=load(rp);validate_requirement(req,pack)
        cid=req['candidate_id']
        if cid in seen:raise SiteError(f'duplicate candidate_id {cid}')
        seen.add(cid)
        checks=[eval_check(pack['metrics'],x) for x in req['checks']]
        disp=req['category_dispositions']
        cat_results={}
        for cat in sorted(BASELINE_CATEGORIES):
            if disp[cat]['status']=='not_applicable':cat_results[cat]={'status':'pass','disposition':'not_applicable','reason':disp[cat]['reason']};continue
            relevant=[]
            for r in checks:
                if r['type']=='threshold' and r['category']==cat:relevant.append(r['status'])
                elif r['type']=='any_of':
                    if any(c['category']==cat for o in r['options'] for c in o['checks']):relevant.append(r['status'])
            status='pass' if relevant and all(x=='pass' for x in relevant) else ('fail' if 'fail' in relevant else 'unresolved')
            cat_results[cat]={'status':status,'disposition':'required'}
        overall='pass' if all(v['status']=='pass' for v in cat_results.values()) else 'fail'
        results.append({'candidate_id':cid,'architecture':req['architecture'],'requirements_file':rp.name,'requirements_sha256':sha256_file(rp),'requirement_id':requirement_id(req),'site_viability':overall,'category_results':cat_results,'check_results':checks})
    out={'schema_version':OUT_SCHEMA,'status':'pass','study_id':pack['study_id'],'frame_contract_id':pack['frame_contract_id'],'epoch_contract_id':pack['epoch_contract_id'],'site_ref':pack['site_ref'],'site_pack_sha256':site_pack_sha,'site_pack_id':site_pack_id,'candidate_results':sorted(results,key=lambda x:x['candidate_id']),'non_claim':'Site viability is a Phase-0 evidence/comparability gate, not site selection, construction qualification, or launch authority.'}
    out['receipt_sha256']=sha256_bytes(canonical_bytes(out))
    return out

def self_test()->None:
    with tempfile.TemporaryDirectory() as td:
        root=pathlib.Path(td);art=root/'art';art.mkdir();src=art/'source.json';src.write_text('{}');sd=sha256_file(src)
        pack={'schema_version':SITE_SCHEMA,'study_id':'study-x','frame_contract_id':'frame-x','epoch_contract_id':'epoch-x','site_ref':'site-x','sources':[{'source_id':'s1','path':'source.json','sha256':sd}],
              'metrics':{
                'slope_deg':{'category':'terrain','status':'available','low':3,'central':4,'high':5,'unit':'deg','evidence_class':'observed','source_refs':['s1']},
                'solar_availability':{'category':'illumination','status':'available','low':.8,'central':.9,'high':.95,'unit':'fraction','evidence_class':'derived_verified','source_refs':['s1']},
                'temp_k':{'category':'thermal','status':'available','low':90,'central':180,'high':260,'unit':'K','evidence_class':'derived_verified','source_refs':['s1']},
                'dte_availability':{'category':'communications','status':'available','low':.4,'central':.5,'high':.6,'unit':'fraction','evidence_class':'derived_verified','source_refs':['s1']},
                'relay_availability':{'category':'communications','status':'available','low':.92,'central':.96,'high':.99,'unit':'fraction','evidence_class':'derived_verified','source_refs':['s1']},
                'bearing_kpa':{'category':'geotechnical','status':'available','low':150,'central':170,'high':190,'unit':'kPa','evidence_class':'observed','source_refs':['s1']},
                'protected_overlap':{'category':'protected_constraints','status':'available','low':0,'central':0,'high':0,'unit':'fraction','evidence_class':'observed','source_refs':['s1']},}}
        pp=root/'pack.json';pp.write_bytes(canonical_bytes(pack))
        disp={c:{'status':'required'} for c in BASELINE_CATEGORIES}
        req={'schema_version':REQ_SCHEMA,'candidate_id':'launcher','architecture':'ballistic_launcher','study_id':'study-x','frame_contract_id':'frame-x','epoch_contract_id':'epoch-x','site_ref':'site-x','category_dispositions':disp,'checks':[
            {'type':'threshold','category':'terrain','metric':'slope_deg','direction':'max','limit':6,'unit':'deg','allowed_evidence_classes':['observed']},
            {'type':'threshold','category':'illumination','metric':'solar_availability','direction':'min','limit':.75,'unit':'fraction','allowed_evidence_classes':['derived_verified']},
            {'type':'threshold','category':'thermal','metric':'temp_k','direction':'max','limit':280,'unit':'K','allowed_evidence_classes':['derived_verified']},
            {'type':'any_of','name':'communications_path','options':[
                [{'type':'threshold','category':'communications','metric':'dte_availability','direction':'min','limit':.55,'unit':'fraction','allowed_evidence_classes':['derived_verified']}],
                [{'type':'threshold','category':'communications','metric':'relay_availability','direction':'min','limit':.9,'unit':'fraction','allowed_evidence_classes':['derived_verified']}],]},
            {'type':'threshold','category':'geotechnical','metric':'bearing_kpa','direction':'min','limit':140,'unit':'kPa','allowed_evidence_classes':['observed']},
            {'type':'threshold','category':'protected_constraints','metric':'protected_overlap','direction':'max','limit':0,'unit':'fraction','allowed_evidence_classes':['observed']},]}
        rp=root/'req.json';rp.write_bytes(canonical_bytes(req));out=evaluate(pp,art,[rp]);assert out['candidate_results'][0]['site_viability']=='pass'
        req2=json.loads(json.dumps(req));req2['candidate_id']='launcher-stricter';req2['checks'][4]['limit']=170
        r2=root/'req2.json';r2.write_bytes(canonical_bytes(req2));out2=evaluate(pp,art,[r2]);assert out2['candidate_results'][0]['site_viability']=='fail';assert out2['candidate_results'][0]['category_results']['geotechnical']['status']=='unresolved'
        assert out['candidate_results'][0]['requirement_id']!=out2['candidate_results'][0]['requirement_id']
        bad=json.loads(json.dumps(req));bad['category_dispositions'].pop('thermal');bp=root/'bad.json';bp.write_bytes(canonical_bytes(bad))
        try:evaluate(pp,art,[bp]);raise AssertionError('missing baseline category not rejected')
        except SiteError:pass

def parse(argv:Iterable[str]|None=None)->argparse.Namespace:
    p=argparse.ArgumentParser();p.add_argument('--site-pack',type=pathlib.Path);p.add_argument('--artifact-root',type=pathlib.Path);p.add_argument('--requirements',type=pathlib.Path,nargs='+');p.add_argument('--output',type=pathlib.Path);p.add_argument('--self-test',action='store_true');return p.parse_args(argv)
def main(argv:Iterable[str]|None=None)->int:
    a=parse(argv)
    try:
        if a.self_test:self_test();print('LL-009I self-test: PASS');return 0
        if not all((a.site_pack,a.artifact_root,a.requirements,a.output)):raise SiteError('site-pack, artifact-root, requirements, and output are required')
        out=evaluate(a.site_pack,a.artifact_root,a.requirements);payload=canonical_bytes(out)
        if a.output.exists() and a.output.read_bytes()!=payload:raise SiteError(f'refusing to overwrite differing output: {a.output}')
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(payload);print(json.dumps(out,sort_keys=True,indent=2));return 0
    except SiteError as e:print(f'LL-009I ERROR: {e}',file=__import__('sys').stderr);return 2
if __name__=='__main__':raise SystemExit(main())
