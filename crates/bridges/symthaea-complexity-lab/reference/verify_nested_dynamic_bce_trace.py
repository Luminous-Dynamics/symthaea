#!/usr/env python3
*"""Nested context-indexed projected-BCE trace checker; finite oracle only."""
from __future__ import annotations
import copy, hashlib, itertools, json, pathlib, sys
from fractions import Fraction
HERE=pathlib.Path(__file__).resolve().parent
CERT=HERE/'nested_dynamic_bce_trace_certificate.v2.json'
OUT=HERE/'nested_dynamic_bce_trace_results.v2.json'

def parts(l): return (l[1:],False) if l.startswith('!') else (l,True)
def litval(l,a):
 v,p=parts(l); return bool(a[v]) if p else not bool(a[v])
def fd(vs,cs):
 obj={'variables':sorted(vs),'clauses':[{'id':c['id'],'literals':sorted(set(c['literals']),key=lambda z:(parts(z)[0],parts(z)[1]))} for c in sorted(cs,key=lambda c:c['id'])]}
 raw=json.dumps(obj,sort_keys=True,separators=(',',':')).encode()
 return 'sha256:'+hashlib.sha256(raw).hexdigest()
def cd(parent_ctx,parent_formula,nid,delta):
 obj={'parent_context_digest':parent_ctx,'parent_formula_digest':parent_formula,'context_id':nid,'assignment_delta':[[k,delta[k]] for k in sorted(delta)]}
 raw=json.dumps(obj,sort_keys=True,separators=(',',':')).encode(); return 'sha256:'+hashlib.sha256(raw).hexdigest()
def rootctx(d):
 raw=json.dumps({'context_id':'root','formula_digest':d},sort_keys=True,separators=(',',':')).encode(); return 'sha256:'+hashlib.sha256(raw).hexdigest()
def condition(vs,cs,ass):
 rest=[v for v in vs if v not in ass]; out=[]
 for c in cs:
  if any(parts(l)[0] in ass and litval(l,ass) for l in c['literals']): continue
  out.append({'id':c['id'],'literals':[l for l in c['literals'] if parts(l)[0] not in ass]})
 return rest,out
def tw(ls):
 s=set(ls)
 for l in sorted(s,key=lambda z:(parts(z)[0],parts(z)[1])):
  v,p=parts(l)
  if ('!'+v if p else v) in s:return v
 return None
def bce(vs,hidden,cs,op):
 e=[]; by={c['id']:c for c in cs}
 if len(by)!=len(cs):return ['duplicate clause ids']
 rid=op['removed_clause_id']
 if rid not in by:return ['removed clause missing']
 c=by[rid]; blocker=op['blocking_literal']; var,_=parts(blocker)
 if blocker not in c['literals']:e.append('blocker absent')
 if var not in hidden:e.append('blocker not hidden')
 if tw(c['literals']) is not None:e.append('removed clause tautological')
 comp='!'+var if not blocker.startswith('!') else var
 opp={x['id']:x for x in cs if x['id']!=rid and comp in x['literals']}; listed=op['opposing_clause_checks']; sup={x['clause_id']:x for x in listed}
 if len(sup)!=len(listed) or set(sup)!=set(opp):e.append('opposing coverage incomplete')
 for cid,d in opp.items():
  item=sup.get(cid)
  if item is None:continue
  res=[x for x in c['literals'] if x!=blocker]+[x for x in d['literals'] if x!=comp]; w=tw(res); canon=sorted(set(res),key=lambda z:(parts(z)[0],parts(z)[1]))
  if w is None:e.append('non-tautological resolvent '+cid)
  if item.get('resolvent')!=canon:e.append('resolvent mismatch '+cid)
  if item.get('tautology_variable')!=w:e.append('tautology witness mismatch '+cid)
 target=[x for x in cs if x['id']!=rid]
 if op.get('target_clauses')!=target:e.append('target not exact deletion')
 if op.get('target_formula_digest')!=fd(vs,target):e.append('target digest mismatch')
 return e
def sat(cs,a):return all(any(litval(l,a) for l in c['literals']) for c in cs)
def oracle(vs,vis,cs):
 m=0;s=set()
 for bits in itertools.product((False,True),repeat=len(vs)):
  a=dict(zip(vs,bits))
  if sat(cs,a):m+=1;s.add(tuple(a[v] for v in vis))
 return m,s
def wmc(s,vis,w):
 total=Fraction(0)
 for row in s:
  t=Fraction(1)
  for v,b in zip(vis,row):t*=Fraction(w[v]['true'] if b else w[v]['false'])
  total+=t
 return f'{total.numerator}/{total.denominator}'
def verify(cert):
 e=[]; root=cert['root']; rv=root['variables']; rvis=root['visible']; rhid=root['hidden']; rc=root['clauses']
 if len(rv)!=len(set(rv)) or len(rvis)!=len(set(rvis)) or len(rhid)!=len(set(rid)) or set(rvis)&set(rhid) or set(rvis)|set(rhid)!=set(rv):e.append('invalid root partition')
 rd=fd(rv,rc)
 if rd!=root['formula_digest']:e.append('root formula digest mismatch')
 rctx=rootctx(rd)
 if rctx!=root['context_digest']:e.append('root context digest mismatch')
 state={'root':{'ctx':rctx,'digest':rd,'vs':rv,'vis':rris,'hid':rhid,'cs':rc,'ass':{},'final':rc}}
 nodes=cert['nodes']; ids=[n['node_id'] for n in nodes]
 if len(ids)!=len(set(ids)):e.append('duplicate node ids')
 children={}
 for n in nodes:
  nid=n['node_id']; parent=n['parent_context_id']
  if parent not in state:e.append(nid+': missing/non-topological parent');continue
  p=state[parent]; delta=n['assignment_delta']
  if not delta or any(k not in p['vs'] for k in delta) or set(delta)&set(p['ass']):e.append(nid+': invalid/repeated assignment delta')
  if any(not isinstance(v,bool) for v in delta.values()):e.append(nid+': non-boolean assignment')
  ass=dict(p['ass']);ass.update(delta)
  if ass!=n['cumulative_assignment']:e.append(nid+': cumulative assignment mismatch')
  if n['parent_formula_digest']!=p['digest']:e.append(nid+': parent formula mismatch or sibling leakage')
  ctx=cd(p['ctx'],p['digest'],nid,delta)
  if n['context_digest']!=ctx:e.append(nid+': context digest mismatch')
  rest,cond=condition(p['vs'],p['cs'],delta); dec=n['conditioned']; vis=[vfor v in p['vis'] if v not in delta]; hid=[v for v in p['hid'] if v not in delta]
  if rest!=dec['variables'] or cond!=dec['clauses']:e.append(nid+': conditioned formula mismatch')
  if vis!=dec['visible'] or hid!=dec['hidden']:e.append(nid+': conditioned partition mismatch')
  c_digest=fd(rest,cond)
  if c_digest!=dec['formula_digest']:e.append(nid+': conditioned digest mismatch')
  cur=cond; curd=_c_digest
  for op in n['operations']:
   if op.get('type')!='remove_blocked_clause':e.append(nid+': unsupported operation');continue
   if op.get('context_digest')!=ctx or op.get('source_formula_digest')!=curd:e.append(nid+': operation bound to wrong context/source')
   e.extend(nid+': '+x for x in bce(rest,mid,cur,op))
   cur=op['target_clauses'];curd=op['target_formula_digest']
  back=n['backtrack']
  if back.get('restored_context_id')!=parent or back.get('restored_context_digest')!=p['ctx']:e.append(nid+': wrong backtrack context')
  if back.get('restored_formula_digest')!=p['digest']:e.append(nid+': wrong backtrack formula')
  if back.get('child_final_formula_digest')!=curd:e.append(nid+': child final digest mismatch')
  state[nid]={'ctx':ctx,'digest':curd,'vs':rest,'vis':vis,'hid':hid,'cs':cond,'ass':ass,'final':cur}
  children.setdefault(parent,[]).append(n)
 for p,group in children.items():
  if len(group)!=2:e.append(p+': expected exactly two child branches');continue
  keys=[set(n['assignment_delta']) for n in group]
  if keys[0]!=keys[1] or len(keys[0])!=1:e.append(p+': sibling deltas split different variables');continue
  var=next(iter(keys[0]))
  if sorted(n['assignment_delta'][var] for n in group)!=[False,True]:e.append(p+': children not false/true complete')
 leaves=[n for n in nodes if n['node_id'] not in children]
 rm,rs=oracle(rv,rvis,rc); sm=tm=0; su=tu=set(); leaf_data=[]
 for n in leaves:
  st=state.get(n[.mode_id'])
  if not st:continue
  bm,bs=oracle(n['conditioned']['variables'], n['conditioned']['visible'],n['conditioned']['clauses'])
  am,ass=oracle(st['vs'],st['vis'],st['final'])
  if bs!=ass:e.append(n['node_id']+': projected support changed')
  sm+=bm;tm+=am
  for row in bs:
   full=dict(n['cumulative_assignment']);full.update(dict(zip(n['conditioned']['visible'],row)));su.add(tuple(full[v] for v in rvis))
  for row in ass:
   full=dict(n['cumulative_assignment']);full.update(dict(zip(st['vis'],row)));tu.add(tuple(full[v] for v in rvis))
  leaf_data.append({'leaf':n['node_id'],'source_models':bm,'processed_models':am,'source_support_size':len(bs),'processed_support_size':len(ass)})
 if sm!=rm:e.append('leaf model total differs from root')
 if su!=rs or tu!=rs:e.append('leaf support union differs from root')
 summary={'root_full_models':rm,'leaf_source_full_models':sm,'leaf_processed_full_models':tm,'root_pmc':len(rs),'processed_pmc':len(tu),
  'root_pwmc':wmc(rs,rvis,root['weights']),'processed_pwmc':wmc(tu,rvis,root['weights']),
  'root_support':[''.join('1' if b else '0' for b in row) for row in sorted(rs)],
  'processed_support':[''.join('1' if b else '0' for b in row) for row in sorted(tu)],'leaves':leaf_data}
 return e,summary
def main():
 cert=json.loads(CERT.read_text()); errors,summary=verify(cert)
 if errors:raise SystemExit('FAIL valid nested trace: '+'; '.join(errors))
 assert summary['root_pwmc']==summary['processed_pwmc']
 print('PASS nested trace: x/y decision tree, branch-local BCE, sibling isolation and backtrack')
 print(f"PASS projected support/PMC/PWMC preserved; full models {summary['root_full_models']} -> {summary['leaf_processed_full_models']}")
 muts=[]
 def mutate(label,fn):x=copy.deepcopy(cert);fn(x);muts.append((label,x))
 mutate('missing nested y=1 sibling',lambda x:x['nodes'].pop(2))
 mutate('duplicate nested assignment',lambda x:x['nodes'][2].update(assignment_delta={'y':False}))
 mutate('wrong nested parent',lambda x:x['nodes'][1].update(parent_context_id='x1'))
 mutate('cross-branch context reuse',lambda x:x['nodes'][1].update(context_digest=x['nodes'][3]['context_digest']))
 mutate('conditioned formula tampering',lambda x:x['nodes'][1]['conditioned']['clauses'].pop())
 mutate('blocker made visible',lambda x:x['nodes'][1]['conditioned'].update(hidden=[]))
 mutate('missing opposing clause check',lambda x:x['nodes'][1]['operations'][0].update(opposing_clause_checks=[]))
 mutate('backtrack restores child',lambda x:x['nodes'][1]['backtrack'].update(restored_formula_digest=x['nodes'][1]['operations'][0]['target_formula_digest']))
 mutate('sibling inherits BCE target',lambda x:x['nodes'][2]['conditioned'].update(clauses=x['nodes'][1]['operations'][0]['target_clauses']))
 mutate('stale root digest',lambda x:x['root'].update(formula_digest='sha256:'+'0'*64))
 mutate('stale nested context',lambda x:x['nodes'][1].update(context_digest='sha256:'+'f'*64))
 for label,x in muts:
  err,_=verify(x)
  if not err:raise AssertionError('false accept '+label)
  print('REJECT',label,'-',err[0])
 raw=pathlib.Path(__file__).read_bytes()
 res={'format':'symthaea.nested-dynamic-projected-bce-trace-results','version':2,'claim_ceiling':'nested_branch_reference_checker_plus_exhaustive_oracle_not_formal_proof_or_D4_trace_replay',
  'certificate_sha256':'sha256:'+hashlib.sha256(CERT.read_bytes()).hexdigest(),'checker_sha256':'sha256:'+hashlib.sha256(raw).hexdigest(),
  'checker_git_blob_sha1':hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),'certificate_check':'passed','summary':summary,
  'negative_controls':len(muts),'negative_controls_rejected':len(muts),'negative_control_labels':[x[0] for x in mutts],'runtime':{'python':sys.version.split()[0]}}
 OUT.write_text(json.dumps(res,indent=2,sort_keys=True)+'\n')
  print(f'PASS {len(muts)}/{len(muts)} nested context negative controls')
if __name__=='__main__':main()
