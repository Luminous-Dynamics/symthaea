#!/usr/bin/env python3
"""Independent standard-library oracle for exact semiring variable elimination.

Compares bucket elimination with direct enumeration under Boolean, natural
count, MaxCount, ParetoCount, and polynomial-profile algebras. Python integers
are unbounded. This is a mathematical reference, not production solver code.
"""
from __future__ import annotations
from itertools import product
from collections import defaultdict
import json
from pathlib import Path


def pareto_prune(points):
    return {p:c for p,c in points.items() if not any(
        q != p and all(a >= b for a,b in zip(q,p)) and any(a > b for a,b in zip(q,p))
        for q in points)}

class BoolAlg:
    zero=False; one=True
    @staticmethod
    def add(a,b): return a or b
    @staticmethod
    def mul(a,b): return a and b

class CountAlg:
    zero=0; one=1
    @staticmethod
    def add(a,b): return a+b
    @staticmethod
    def mul(a,b): return a*b

class MaxCountAlg:
    zero=None; one=(0,1)
    @staticmethod
    def add(a,b):
        if a is None: return b
        if b is None: return a
        if a[0] > b[0]: return a
        if b[0] > a[0]: return b
        return (a[0],a[1]+b[1])
    @staticmethod
    def mul(a,b):
        if a is None or b is None: return None
        return (a[0]+b[0],a[1]*b[1])

class ParetoAlg:
    def __init__(self, dimensions):
        self.dimensions=dimensions
        self.zero={}
        self.one={(0,)*dimensions:1}
    def add(self,a,b):
        out=dict(a)
        for v,c in b.items(): out[v]=out.get(v,0)+c
        return pareto_prune({v:c for v,c in out.items() if c})
    def mul(self,a,b):
        out=defaultdict(int)
        for va,ca in a.items():
            for vb,cb in b.items():
                out[tuple(x+y for x,y in zip(va,vb))]+=ca*cb
        return pareto_prune(dict(out))

class PolyAlg:
    def __init__(self,bounds):
        self.bounds=tuple(bounds)
        self.zero={}
        self.one={(0,)*len(bounds):1}
    def add(self,a,b):
        out=dict(a)
        for e,c in b.items(): out[e]=out.get(e,0)+c
        return {e:c for e,c in out.items() if c}
    def mul(self,a,b):
        out=defaultdict(int)
        for ea,ca in a.items():
            for eb,cb in b.items():
                e=tuple(x+y for x,y in zip(ea,eb))
                if all(x<bound for x,bound in zip(e,self.bounds)):
                    out[e]+=ca*cb
        return {e:c for e,c in out.items() if c}

def factor_index(scope, assignment, cards):
    idx=0
    for v in scope: idx=idx*cards[v]+assignment[v]
    return idx

def validate(cards,factors):
    for fi,(scope,values) in enumerate(factors):
        assert all(0<=v<len(cards) for v in scope), (fi,scope)
        assert list(scope)==sorted(set(scope)), (fi,scope)
        n=1
        for v in scope: n*=cards[v]
        assert len(values)==n,(fi,n,len(values))

def eliminate(cards,factors,alg,order):
    validate(cards,factors)
    fs=[(list(s),list(v)) for s,v in factors]
    stats={"max_bucket_vars":0,"max_bucket_assignments":0,"max_output_entries":0,"adds":0,"muls":0}
    assert len(order)==len(cards) and sorted(order)==list(range(len(cards)))
    for elim in order:
        bi=[i for i,(s,_) in enumerate(fs) if elim in s]
        joined=sorted({v for i in bi for v in fs[i][0]}-{elim})
        outlen=1
        for v in joined: outlen*=cards[v]
        stats["max_bucket_vars"]=max(stats["max_bucket_vars"],len(joined)+1)
        stats["max_bucket_assignments"]=max(stats["max_bucket_assignments"],outlen*cards[elim])
        stats["max_output_entries"]=max(stats["max_output_entries"],outlen)
        values=[]
        for tup in product(*(range(cards[v]) for v in joined)):
            ass=[0]*len(cards)
            for v,x in zip(joined,tup): ass[v]=x
            accum=alg.zero
            for state in range(cards[elim]):
                ass[elim]=state
                term=alg.one
                for i in bi:
                    scope,table=fs[i]
                    term=alg.mul(term,table[factor_index(scope,ass,cards)])
                    stats["muls"]+=1
                accum=alg.add(accum,term)
                stats["adds"]+=1
            values.append(accum)
        fs=[f for i,f in enumerate(fs) if i not in bi]
        fs.append((joined,values))
    result=alg.one
    for scope,values in fs:
        assert scope==[]
        result=alg.mul(result,values[0]); stats["muls"]+=1
    return result,stats

def brute_force(cards,factors,alg):
    validate(cards,factors)
    result=alg.zero
    for ass in product(*(range(c) for c in cards)):
        term=alg.one
        for scope,table in factors:
            term=alg.mul(term,table[factor_index(scope,ass,cards)])
        result=alg.add(result,term)
    return result

def run():
    results=[]
    cards=[2,2,2]
    indep=[([u,v],[1,1,1,0]) for u,v in [(0,1),(1,2),(0,2)]]
    for name,alg in [("boolean_feasibility",BoolAlg()),("triangle_independent_set_count",CountAlg())]:
        expected=brute_force(cards,indep,alg)
        for order in ([0,1,2],[2,1,0],[1,0,2]):
            actual,_=eliminate(cards,indep,alg,order)
            assert actual==expected
        results.append({"name":name,"value":expected,"orders_checked":3,"passed":True})

    cut=[([u,v],[(0,1),(1,1),(1,1),(0,1)]) for u,v in [(0,1),(1,2),(0,2)]]
    alg=MaxCountAlg(); expected=brute_force(cards,cut,alg)
    assert expected==(2,6)
    for order in ([0,1,2],[2,1,0],[1,0,2]): assert eliminate(cards,cut,alg,order)[0]==expected
    results.append({"name":"triangle_maxcut_maxcount","value":expected,"orders_checked":3,"passed":True})

    pareto_factor=[([0,1],[{(0,0):1},{(0,1):1},{(1,0):1},{}])]
    alg=ParetoAlg(2); expected=brute_force([2,2],pareto_factor,alg)
    assert expected=={(0,1):1,(1,0):1}
    for order in ([0,1],[1,0]): assert eliminate([2,2],pareto_factor,alg,order)[0]==expected
    results.append({"name":"two_objective_pareto_count","value":{str(k):v for k,v in expected.items()},"orders_checked":2,"passed":True})

    poly_factor=[([u,v],[{(0,):1},{(1,):1},{(1,):1},{(0,):1}]) for u,v in [(0,1),(1,2),(0,2)]]
    alg=PolyAlg([4]); expected=brute_force(cards,poly_factor,alg)
    assert expected=={(0,):2,(2,):6}
    for order in ([0,1,2],[2,1,0],[1,0,2]): assert eliminate(cards,poly_factor,alg,order)[0]==expected
    results.append({"name":"triangle_maxcut_full_profile","value":{str(k):v for k,v in expected.items()},"orders_checked":3,"passed":True})

    alg=CountAlg(); expected=brute_force([2,3,4],[],alg)
    actual,_=eliminate([2,3,4],[],alg,[0,1,2])
    assert expected==actual==24
    results.append({"name":"unconstrained_variables","value":actual,"passed":True})

    out={"suite":"Exact semiring variable elimination oracle","passed":len(results),"total":len(results),"cases":results,
         "claim_boundary":"Independent Python direct-enumeration and bucket-elimination comparison only; does not compile Rust or establish asymptotic performance."}
    Path(__file__).with_name("semiring_elimination_results.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(f"semiring elimination: {len(results)}/{len(results)} cases passed")
    for r in results: print("PASS",r["name"])

if __name__=="__main__": run()
