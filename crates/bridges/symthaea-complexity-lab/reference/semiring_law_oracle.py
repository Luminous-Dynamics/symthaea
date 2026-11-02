#!/usr/bin/env python3
"""Deterministic property-style exact tests for MaxCount/Pareto/Profile algebras."""
from __future__ import annotations
import random
import json
from pathlib import Path

def max_add(a,b):
    if a is None:return b
    if b is None:return a
    if a[0]>b[0]:return a
    if b[0]>a[0]:return b
    return (a[0],a[1]+b[1])
def max_mul(a,b):
    if a is None or b is None:return None
    return (a[0]+b[0],a[1]*b[1])

def pareto_prune(m):
    return {p:c for p,c in m.items() if not any(q!=p and all(a>=b for a,b in zip(q,p)) and any(a>b for a,b in zip(q,p)) for q in m)}
def pareto_add(a,b):
    m=dict(a)
    for p,c in b.items():m[p]=m.get(p,0)+c
    return pareto_prune(m)
def pareto_mul(a,b):
    m={}
    for p,c in a.items():
        for q,d in b.items():
            r=tuple(x+y for x,y in zip(p,q));m[r]=m.get(r,0)+c*d
    return pareto_prune(m)

def poly_add(a,b):
    m=dict(a)
    for e,c in b.items():m[e]=m.get(e,0)+c
    return {e:c for e,c in m.items() if c}
def poly_mul(a,b,bounds=(3,3)):
    m={}
    for p,c in a.items():
        for q,d in b.items():
            r=tuple(x+y for x,y in zip(p,q))
            if all(x<n for x,n in zip(r,bounds)):m[r]=m.get(r,0)+c*d
    return {e:c for e,c in m.items() if c}

def check_maxcount(rng,trials):
    z=None;o=(0,1)
    def sample(): return None if rng.randrange(5)==0 else (rng.randrange(-4,5),rng.randrange(1,5))
    for _ in range(trials):
        a,b,c=sample(),sample(),sample()
        assert max_add(a,z)==a and max_mul(a,o)==a and max_mul(a,z)==z
        assert max_add(a,b)==max_add(b,a)
        assert max_add(max_add(a,b),c)==max_add(a,max_add(b,c))
        assert max_mul(max_mul(a,b),c)==max_mul(a,max_mul(b,c))
        assert max_mul(a,max_add(b,c))==max_add(max_mul(a,b),max_mul(a,c))
        assert max_mul(max_add(a,b),c)==max_add(max_mul(a,c),max_mul(b,c))
    return trials

def check_pareto(rng,trials):
    z={};o={(0,0):1}
    def sample():
        pts={}
        for _ in range(rng.randrange(5)):
            p=(rng.randrange(4),rng.randrange(4));pts[p]=pts.get(p,0)+rng.randrange(1,4)
        return pareto_prune(pts)
    for _ in range(trials):
        a,b,c=sample(),sample(),sample()
        assert pareto_add(a,z)==a and pareto_mul(a,o)==a and pareto_mul(a,z)==z
        assert pareto_add(a,b)==pareto_add(b,a)
        assert pareto_add(pareto_add(a,b),c)==pareto_add(a,pareto_add(b,c))
        assert pareto_mul(pareto_mul(a,b),c)==pareto_mul(a,pareto_mul(b,c))
        assert pareto_mul(a,pareto_add(b,c))==pareto_add(pareto_mul(a,b),pareto_mul(a,c))
        assert pareto_mul(pareto_add(a,b),c)==pareto_add(pareto_mul(a,c),pareto_mul(b,c))
    return trials

def check_poly(rng,trials):
    z={};o={(0,0):1}
    def sample():
        out={}
        for x in range(3):
            for y in range(3):
                c=rng.randrange(-3,4)
                if c:out[(x,y)]=c
        return out
    for _ in range(trials):
        a,b,c=sample(),sample(),sample()
        assert poly_add(a,z)==a and poly_mul(a,o)==a and poly_mul(a,z)==z
        assert poly_add(a,b)==poly_add(b,a)
        assert poly_add(poly_add(a,b),c)==poly_add(a,poly_add(b,c))
        assert poly_mul(poly_mul(a,b),c)==poly_mul(a,poly_mul(b,c))
        assert poly_mul(a,poly_add(b,c))==poly_add(poly_mul(a,b),poly_mul(a,c))
        assert poly_mul(poly_add(a,b),c)==poly_add(poly_mul(a,c),poly_mul(b,c))
    return trials

def main():
    rng=random.Random(0xE7015); trials=256
    n1=check_maxcount(rng,trials);n2=check_pareto(rng,trials);n3=check_poly(rng,trials)
    result={"suite":"Exact semiring/ring algebra laws","seed":"0xE7015","trials_per_algebra":trials,
            "maxcount_law_cases":n1,"pareto_law_cases":n2,"truncated_polynomial_ring_cases":n3,
            "passed":n1+n2+n3,"total":3*trials,
            "claim_boundary":"Deterministic finite property-style exact checks on reference Python algebra implementations; does not compile Rust or constitute formal proof."}
    Path(__file__).with_name("semiring_law_results.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(f"semiring algebra law checks: {result['passed']}/{result['total']} passed")
if __name__=="__main__":main()
