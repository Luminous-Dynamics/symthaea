#!/usr/bin/env python3
"""Exact output-contract-aware metamorphic tests for MC, WMC, PMC, PWMC, and Q(i) AMC."""
from dataclasses import dataclass
from fractions import Fraction as F
from itertools import product
import json, platform, random, time

Z=(F(0),F(0)); O=(F(1),F(0))
def add(a,b): return (a[0]+b[0],a[1]+b[1])
def mul(a,b): return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def fmt(a): return str(a[0]) if a[1]==0 else f"{a[0]}{a[1]:+}i"
def q(r): return F(r.randint(-3,3),r.randint(1,4))
def qi(r): return (q(r),q(r))

@dataclass
class I:
    mode:str; n:int; clauses:list; projection:set; weights:dict

def clone(x): return I(x.mode,x.n,[c[:] for c in x.clauses],set(x.projection),dict(x.weights))
def wt(x,l): return x.weights.get(l,O)
def eval_exact(x):
    models=[]
    for bits in product((False,True),repeat=x.n):
        a={i+1:bits[i] for i in range(x.n)}
        if all(any(a[abs(l)]==(l>0) for l in c) for c in x.clauses): models.append(a)
    if x.mode=="mc": return (F(len(models)),F(0))
    if x.mode=="pmc":
        ps={tuple(a[v] for v in sorted(x.projection)) for a in models}
        return (F(len(ps)),F(0))
    total=Z
    if x.mode in ("wmc","amc-complex"):
        for a in models:
            t=O
            for v in range(1,x.n+1): t=mul(t,wt(x,v if a[v] else -v))
            total=add(total,t)
    elif x.mode=="pwmc":
        ps={tuple(a[v] for v in sorted(x.projection)) for a in models}
        for bits in ps:
            t=O
            for v,b in zip(sorted(x.projection),bits): t=mul(t,wt(x,v if b else -v))
            total=add(total,t)
    else: raise ValueError(x.mode)
    return total

def gen(r,mode):
    n=r.randint(1,6); cs=[]
    for _ in range(r.randint(1,9)):
        vs=r.sample(range(1,n+1),r.randint(1,min(n,3)))
        cs.append([v if r.choice((True,False)) else -v for v in vs])
    p=set(r.sample(range(1,n+1),r.randint(1,n))) if mode in ("pmc","pwmc") else set()
    w={}
    if mode=="wmc":
        for v in range(1,n+1): w[v]=(q(r),F(0)); w[-v]=(q(r),F(0))
    elif mode=="pwmc":
        for v in p: w[v]=(q(r),F(0)); w[-v]=(q(r),F(0))
    elif mode=="amc-complex":
        for v in range(1,n+1): w[v]=qi(r); w[-v]=qi(r)
    return I(mode,n,cs,p,w)

def syntax_mutation(x,r):
    y=clone(x); r.shuffle(y.clauses)
    for c in y.clauses:r.shuffle(c)
    y.clauses += [list(r.choice(y.clauses)),[1,-1]]
    return y

def complement(x,v):
    y=clone(x); y.clauses=[[-l if abs(l)==v else l for l in c] for c in y.clauses]
    a=y.weights.pop(v,None); b=y.weights.pop(-v,None)
    if b is not None:y.weights[v]=b
    if a is not None:y.weights[-v]=a
    return y

def permute(x,r):
    y=clone(x); old=list(range(1,x.n+1)); new=old[:]; r.shuffle(new); m=dict(zip(old,new))
    y.clauses=[[m[abs(l)] if l>0 else -m[abs(l)] for l in c] for c in x.clauses]
    y.projection={m[v] for v in x.projection}
    y.weights={(m[abs(l)] if l>0 else -m[abs(l)]):a for l,a in x.weights.items()}
    return y

def free_var(x,visible,wp,wn):
    y=clone(x); z=x.n+1; y.n+=1
    if visible and x.mode in ("pmc","pwmc"):y.projection.add(z)
    if x.mode in ("wmc","amc-complex") or (x.mode=="pwmc" and visible):y.weights[z]=wp;y.weights[-z]=wn
    if x.mode=="mc":s=(F(2),F(0))
    elif x.mode in ("wmc","amc-complex"):s=add(wp,wn)
    elif x.mode=="pmc":s=(F(2 if visible else 1),F(0))
    else:s=add(wp,wn) if visible else O
    return y,s

def forced_true(x,visible,wp,wn):
    y=clone(x); z=x.n+1; y.n+=1; y.clauses.append([z])
    if visible and x.mode in ("pmc","pwmc"):y.projection.add(z)
    if x.mode in ("wmc","amc-complex") or (x.mode=="pwmc" and visible):y.weights[z]=wp;y.weights[-z]=wn
    s=O if x.mode in ("mc","pmc") or (x.mode=="pwmc" and not visible) else wp
    return y,s

def unsat(x):
    y=clone(x); z=x.n+1; y.n+=1; y.clauses += [[z],[-z]]
    if x.mode in ("pmc","pwmc"):y.projection.add(z)
    return y

def controls():
    cs=[[1,2],[1,-2]]; p=I("pmc",2,cs,{1},{})
    pw=I("pwmc",2,cs,{1},{1:(F(2,3),F(0)),-1:(F(1,3),F(0))})
    w0=I("wmc",1,[],set(),{1:(F(1),F(0)),-1:(F(-1),F(0))})
    a0=I("amc-complex",1,[],set(),{1:(F(1),F(2)),-1:(F(-1),F(-2))})
    pm=eval_exact(p); pwm=eval_exact(pw); wrong=add(wt(pw,1),wt(pw,1))
    h,_=free_var(p,False,O,O)
    assert eval_exact(I("mc",2,cs,set(),{}))==(F(2),F(0)) and pm==(F(1),F(0))
    assert pwm==(F(2,3),F(0)) and wrong==(F(4,3),F(0))
    assert eval_exact(h)==pm and eval_exact(h)!=(F(2),F(0))
    assert eval_exact(w0)==Z and eval_exact(a0)==Z
    return {"multiple_hidden_witnesses":{"#SAT":"2","PMC":"1","PWMC":"2/3","incorrect_hidden_witness_weight":"4/3"},"wrong_hidden_witness_doubling_detected":True,"exact_zero":{"WMC":"0","AMC_Qi":"0"},"all_passed":True}

def run(seed=0xE7025,per_mode=100):
    t=time.perf_counter(); r=random.Random(seed); modes=["mc","wmc","pmc","pwmc","amc-complex"]
    names=["syntax_normalization","variable_complement","variable_permutation","free_variable_scaling","forced_true_scaling","unsatisfiable_extension_zero"]
    counts={n:0 for n in names}; total=0; witnesses=0; zero_bases=0; failures=[]
    for mode in modes:
        for k in range(per_mode):
            x=gen(r,mode); b=eval_exact(x); zero_bases += (b==Z)
            cases=[("syntax_normalization",syntax_mutation(x,r),b),("variable_complement",complement(x,r.randint(1,x.n)),b),("variable_permutation",permute(x,r),b)]
            wp=qi(r) if mode=="amc-complex" else (q(r),F(0)); wn=qi(r) if mode=="amc-complex" else (q(r),F(0)); vis=r.choice((True,False))
            y,s=free_var(x,vis,wp,wn); cases.append(("free_variable_scaling",y,mul(b,s)))
            y,s=forced_true(x,vis,wp,wn); cases.append(("forced_true_scaling",y,mul(b,s)))
            cases.append(("unsatisfiable_extension_zero",unsat(x),Z))
            for name,y,e in cases:
                a=eval_exact(y); total+=1; counts[name]+=1
                if a!=e:failures.append({"mode":mode,"case":k,"transform":name,"expected":fmt(e),"actual":fmt(a)})
            if mode in ("pmc","pwmc"):
                y,_=free_var(x,False,O,O); witnesses+=1
                if eval_exact(y)!=b:failures.append({"mode":mode,"case":k,"transform":"hidden_witness_invariance","expected":fmt(b),"actual":fmt(eval_exact(y))})
    c=controls()
    return {"suite":"ETH-025 exact semantics metamorphic benchmark","python_version":platform.python_version(),"seed":hex(seed),"instances_per_mode":per_mode,"modes":modes,"metamorphic_checks":total,"checks_by_transform":counts,"hidden_witness_invariance_checks":witnesses,"base_instances_with_exact_zero":zero_bases,"failure_count":len(failures),"failures":failures[:20],"targeted_semantic_controls":c,"all_passed":not failures and c["all_passed"],"elapsed_seconds":round(time.perf_counter()-t,6),"claim_ceiling":"Executed finite exact direct-oracle/metamorphic relations on small instances. Not a differential run of Ganak or Symthaea binaries, official-format parser qualification, Rust/Lean verification, or performance result."}

if __name__=="__main__": print(json.dumps(run(),indent=2,sort_keys=True))
