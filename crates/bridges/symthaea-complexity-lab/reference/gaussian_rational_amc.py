#!/usr/bin/env python3
"""Exact Q(i) factor-network weighted counting reference and bucket-VE oracle."""
from __future__ import annotations
from fractions import Fraction
from itertools import product
from math import gcd
from pathlib import Path
import hashlib, json, random

Z=(Fraction(0),Fraction(0)); O=(Fraction(1),Fraction(0))
def add(a,b): return (a[0]+b[0],a[1]+b[1])
def neg(a): return (-a[0],-a[1])
def mul(a,b): return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def conj(a): return (a[0],-a[1])
def encq(q):
    q=Fraction(q); return [q.numerator,q.denominator]
def enc(z): return encq(z[0])+encq(z[1])
def decq(v):
    if not isinstance(v,list) or len(v)!=2 or any(not isinstance(x,int) or isinstance(x,bool) for x in v): raise ValueError("rational shape")
    n,d=v
    if d<=0 or gcd(abs(n),d)!=1: raise ValueError("rational must be reduced with positive denominator")
    return Fraction(n,d)
def dec(v):
    if not isinstance(v,list) or len(v)!=4: raise ValueError("Q(i) value shape")
    return (decq(v[:2]),decq(v[2:]))
def canon(v): return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True)
def sha(v): return hashlib.sha256(canon(v).encode()).hexdigest()
def validate(inst):
    if not isinstance(inst,dict) or set(inst)!={"domains","factors"}: raise ValueError("instance shape")
    d,fs=inst["domains"],inst["factors"]
    if not isinstance(d,dict) or not isinstance(fs,list): raise ValueError("instance types")
    for v,n in d.items():
        if not isinstance(v,str) or not v or not isinstance(n,int) or isinstance(n,bool) or not 1<=n<=4: raise ValueError("domain")
    ids=set()
    for f in fs:
        if not isinstance(f,dict) or set(f)!={"id","scope","values"}: raise ValueError("factor shape")
        fid,s,vs=f["id"],f["scope"],f["values"]
        if not isinstance(fid,str) or not fid or fid in ids: raise ValueError("factor id")
        ids.add(fid)
        if not isinstance(s,list) or any(not isinstance(v,str) for v in s) or s!=sorted(set(s)) or any(v not in d for v in s): raise ValueError("factor scope")
        size=1
        for v in s: size*=d[v]
        if not isinstance(vs,list) or len(vs)!=size: raise ValueError("factor table size")
        for z in vs: dec(z)
    return d,fs
def value(f,env,d):
    i=0
    for v in f["scope"]: i=i*d[v]+env[v]
    return dec(f["values"][i])
def direct(inst):
    d,fs=validate(inst); vs=sorted(d); total=Z
    for vals in product(*(range(d[v]) for v in vs)):
        env=dict(zip(vs,vals)); term=O
        for f in fs: term=mul(term,value(f,env,d))
        total=add(total,term)
    return total
def join(bucket,d):
    scope=sorted({v for f in bucket for v in f["scope"]}); table=[]
    for vals in product(*(range(d[v]) for v in scope)):
        env=dict(zip(scope,vals)); term=O
        for f in bucket: term=mul(term,value(f,env,d))
        table.append(enc(term))
    return {"id":"join","scope":scope,"values":table}
def sum_out(f,v,d):
    scope=[x for x in f["scope"] if x!=v]; table=[]
    for vals in product(*(range(d[x]) for x in scope)):
        env=dict(zip(scope,vals)); total=Z
        for x in range(d[v]):
            env[v]=x; total=add(total,value(f,env,d))
        table.append(enc(total))
    return {"id":"sum_"+v,"scope":scope,"values":table}
def ve(inst,order):
    d,fs=validate(inst); seen={v for f in fs for v in f["scope"]}; work=list(fs)
    for v in sorted(set(d)-seen):
        work.append({"id":"free_"+v,"scope":[v],"values":[enc(O) for _ in range(d[v])]})
    if len(order)!=len(d) or set(order)!=set(d): raise ValueError("order must be a permutation")
    for v in order:
        bucket=[f for f in work if v in f["scope"]]
        if not bucket: continue
        work=[f for f in work if v not in f["scope"]]
        work.append(sum_out(join(bucket,d),v,d))
    result=O
    for f in work:
        if f["scope"]: raise AssertionError("non-scalar factor remains")
        result=mul(result,dec(f["values"][0]))
    return result
def rq(r): return Fraction(r.randrange(-3,4),r.randrange(1,5))
def rc(r): return (rq(r),rq(r))
def make_factor(r,fid,d,scope):
    size=1
    for v in scope: size*=d[v]
    return {"id":fid,"scope":scope,"values":[enc(rc(r)) for _ in range(size)]}
def main():
    r=random.Random(0xE7020); ring_trials=1200; ring_checks=0
    for _ in range(ring_trials):
        a,b,c=rc(r),rc(r),rc(r)
        assert add(add(a,b),c)==add(a,add(b,c))
        assert add(a,b)==add(b,a)
        assert mul(mul(a,b),c)==mul(a,mul(b,c))
        assert mul(a,b)==mul(b,a)
        assert mul(a,add(b,c))==add(mul(a,b),mul(a,c))
        assert add(a,Z)==a and mul(a,O)==a
        assert add(a,neg(a))==Z
        assert conj(conj(a))==a
        assert mul(a,conj(a))[1]==0
        ring_checks+=9
    instances=[]
    for _ in range(180):
        n=r.randrange(1,7); d={f"x{i}":2 for i in range(n)}; fs=[]
        for j in range(r.randrange(1,10)):
            w=r.randrange(0,min(3,n)+1); scope=sorted(r.sample(list(d),w)) if w else []
            fs.append(make_factor(r,f"f{j}",d,scope))
        instances.append({"domains":d,"factors":fs})
    checks=0; order_counts={"forward":0,"reverse":0,"degree_like":0,"random":0}
    for i,inst in enumerate(instances):
        vs=sorted(inst["domains"]); expected=direct(inst)
        orders={"forward":vs,"reverse":list(reversed(vs)),
                "degree_like":sorted(vs,key=lambda v:sum(v in f["scope"] for f in inst["factors"])),
                "random":r.sample(vs,len(vs))}
        for name,order in orders.items():
            got=ve(inst,order); assert got==expected,(i,name,got,expected)
            checks+=1; order_counts[name]+=1
    cancel={"domains":{"x":2},"factors":[{"id":"w","scope":["x"],"values":[enc((Fraction(1),Fraction(0))),enc((Fraction(-1),Fraction(0)))]}]}
    imag={"domains":{"x":2},"factors":[{"id":"w","scope":["x"],"values":[enc((Fraction(0),Fraction(1))),enc((Fraction(0),Fraction(-1)))]}]}
    assert direct(cancel)==Z==ve(cancel,["x"])
    assert direct(imag)==Z==ve(imag,["x"])
    malformed=[[1,0,0,1],[2,4,0,1],[1,-2,0,1],[True,1,0,1],[0,2,0,1]]
    rejected=0
    for x in malformed:
        try: dec(x)
        except ValueError: rejected+=1
    assert rejected==len(malformed)
    results={"suite":"ETH-020 Gaussian-rational algebraic model counting reference","seed":"0xE7020",
      "coefficient_domain":"Q(i), exact pair of reduced rational numbers","ring_trials":ring_trials,
      "ring_law_checks":ring_checks,"random_factor_networks":len(instances),
      "exact_direct_vs_variable_elimination_checks":checks,"elimination_orders":order_counts,
      "exact_cancellation_controls":2,"malformed_coefficient_encodings_rejected":rejected,
      "all_passed":True,
      "claim_ceiling":"Finite exact Q(i) reference arithmetic and small factor-network VE comparisons. Not the full MC2026 field track, algebraic-number fields, projected counting, Rust/Lean proof, or performance result."}
    Path(__file__).with_name("gaussian_rational_amc_results.json").write_text(json.dumps(results,indent=2,sort_keys=True)+"\n")
    print(json.dumps(results,indent=2,sort_keys=True))
if __name__=="__main__": main()
