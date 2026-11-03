#!/usr/bin/env python3
"""Finite exact semantic checker for the typed tractable-circuit IR v1."""
from fractions import Fraction as F
from itertools import product
import json

SCHEMA = "symthaea.typed-tractable-circuit/v1"

class Invalid(ValueError):
    pass

def rat(x):
    if not isinstance(x, list) or len(x) != 2 or any(type(v) is not int for v in x):
        raise Invalid("rational shape")
    if x[1] <= 0:
        raise Invalid("rational denominator")
    q = F(*x)
    if [q.numerator, q.denominator] != x:
        raise Invalid("noncanonical rational")
    return q

def decode(domain, x):
    if domain == "bool":
        if type(x) is not bool:
            raise Invalid("Boolean coefficient")
        return x
    if domain == "q":
        return rat(x)
    if domain == "qi":
        if not isinstance(x, dict) or set(x) != {"re", "im"}:
            raise Invalid("Q(i) coefficient shape")
        return rat(x["re"]), rat(x["im"])
    raise Invalid("unknown coefficient domain")

def zero(d):
    return False if d == "bool" else F(0) if d == "q" else (F(0), F(0))
def one(d):
    return True if d == "bool" else F(1) if d == "q" else (F(1), F(0))
def add(d, a, b):
    if d == "bool": return a or b
    if d == "qi": return a[0] + b[0], a[1] + b[1]
    return a + b
def mul(d, a, b):
    if d == "bool": return a and b
    if d == "qi": return a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0]
    return a*b
def fmt(d, x):
    if d != "qi" or x[1] == 0: return str(x if d != "qi" else x[0])
    return f"{x[0]}{'+' if x[1] >= 0 else ''}{x[1]}i"

def evaluate(c):
    required = {"schema","contract","variables","visible","domain","nodes","root"}
    if not isinstance(c,dict) or not required <= set(c) or set(c)-required-{"visible_weights"}:
        raise Invalid("circuit header")
    if c["schema"] != SCHEMA: raise Invalid("schema")
    mode, d = c["contract"], c["domain"]
    modes = {"#sat":"bool","pmc":"bool","pwmc":"bool","wmc":"q","amc-complex":"qi"}
    if mode not in modes or modes[mode] != d: raise Invalid("contract/domain mismatch")
    vs, vis, nodes = c["variables"], c["visible"], c["nodes"]
    if not isinstance(vs,list) or any(not isinstance(v,str) for v in vs) or len(set(vs)) != len(vs):
        raise Invalid("variable universe")
    if not isinstance(vis,list) or len(set(vis)) != len(vis) or not set(vis) <= set(vs):
        raise Invalid("visible set")
    if mode not in {"pmc","pwmc"} and vis: raise Invalid("visible set on nonprojected contract")
    if mode != "pwmc" and "visible_weights" in c: raise Invalid("weights outside PWMC")
    if not isinstance(nodes,list) or not nodes: raise Invalid("empty nodes")
    by, scopes = {}, {}
    for n in nodes:
        if not isinstance(n,dict) or not isinstance(n.get("id"),str) or n["id"] in by:
            raise Invalid("node id")
        by[n["id"]] = n
    for n in nodes:
        i, k = n["id"], n.get("kind")
        if k == "indicator":
            if set(n)!={"id","kind","variable","value"} or n["variable"] not in vs or type(n["value"]) is not bool:
                raise Invalid("indicator")
            scopes[i] = {n["variable"]}
        elif k == "constant":
            if set(n)!={"id","kind","value"}: raise Invalid("constant")
            decode(d,n["value"]); scopes[i] = set()
        elif k == "product":
            if set(n)!={"id","kind","children"} or not n["children"]: raise Invalid("product")
            ch=n["children"]
            if any(x not in scopes for x in ch): raise Invalid("forward product edge")
            flat=[v for x in ch for v in scopes[x]]
            if len(flat)!=len(set(flat)): raise Invalid("nondecomposable product")
            scopes[i]=set(flat)
        elif k == "sum":
            if set(n)-{"id","kind","terms","deterministic"} or not {"id","kind","terms"}<=set(n) or not n["terms"]:
                raise Invalid("sum")
            if "deterministic" in n and type(n["deterministic"]) is not bool:
                raise Invalid("determinism flag")
            ch=[]
            for t in n["terms"]:
                if not isinstance(t,dict) or set(t)!={"child","weight"} or t["child"] not in scopes:
                    raise Invalid("sum term")
                decode(d,t["weight"]); ch.append(t["child"])
            scopes[i]=set().union(*(scopes[x] for x in ch))
        else:
            raise Invalid("unknown node kind")
    root=c["root"]
    if root not in by or not scopes[root] <= set(vs): raise Invalid("root")
    def ev(i,a):
        n=by[i]
        if n["kind"]=="indicator": return one(d) if a[n["variable"]]==n["value"] else zero(d)
        if n["kind"]=="constant": return decode(d,n["value"])
        if n["kind"]=="product":
            r=one(d)
            for x in n["children"]: r=mul(d,r,ev(x,a))
            return r
        r=zero(d)
        for t in n["terms"]: r=add(d,r,mul(d,decode(d,t["weight"]),ev(t["child"],a)))
        return r
    checked=[]
    for n in nodes:
        if n["kind"]=="sum" and n.get("deterministic",False):
            scope=sorted(scopes[n["id"]])
            for bits in product((False,True),repeat=len(scope)):
                a=dict(zip(scope,bits))
                vals=[mul(d,decode(d,t["weight"]),ev(t["child"],a)) for t in n["terms"]]
                for i in range(len(vals)):
                    for j in range(i+1,len(vals)):
                        if mul(d,vals[i],vals[j]) != zero(d): raise Invalid("false determinism claim")
            checked.append(n["id"])
    total=zero("qi" if d=="qi" else "q")
    full=nonzero=0
    projections=set()
    for bits in product((False,True),repeat=len(vs)):
        a=dict(zip(vs,bits)); v=ev(root,a)
        supported=bool(v) if d=="bool" else v != zero(d)
        if not supported: continue
        nonzero += 1
        if mode in {"#sat","pmc","pwmc"}:
            full += 1; projections.add(tuple(a[x] for x in vis))
        if mode=="#sat": total += 1
        elif mode in {"wmc","amc-complex"}: total=add(d,total,v)
    if mode=="pmc": total=F(len(projections))
    elif mode=="pwmc":
        weights=c.get("visible_weights",{})
        if not isinstance(weights,dict) or not set(weights)<=set(vis): raise Invalid("PWMC weight scope")
        for key in projections:
            a=dict(zip(vis,key)); term=F(1)
            for x in vis:
                pair=weights.get(x,[[1,1],[1,1]])
                if not isinstance(pair,list) or len(pair)!=2: raise Invalid("weight pair")
                term *= rat(pair[1 if a[x] else 0])
            total += term
    return {"contract":mode,"domain":d,"root_scope":sorted(scopes[root]),"deterministic_nodes_checked":checked,
            "satisfying_full_assignments":full if mode in {"#sat","pmc","pwmc"} else None,
            "nonzero_function_assignments":nonzero,
            "distinct_projections":len(projections) if mode in {"pmc","pwmc"} else None,
            "exact_output":fmt("qi" if d=="qi" else "q",total)}

def hidden(mode):
    c={"schema":SCHEMA,"contract":mode,"variables":["x","y"],
       "visible":["x"] if mode in {"pmc","pwmc"} else [],
       "domain":"bool","nodes":[{"id":"x1","kind":"indicator","variable":"x","value":True}],"root":"x1"}
    if mode=="pwmc": c["visible_weights"]={"x":[[1,3],[2,3]]}
    return c

def cancellation(domain,mode):
    pos,neg=([1,1],[-1,1]) if domain=="q" else (
        {"re":[1,1],"im":[2,1]},{"re":[-1,1],"im":[-2,1]})
    return {"schema":SCHEMA,"contract":mode,"variables":["x"],"visible":[],"domain":domain,
            "nodes":[{"id":"x0","kind":"indicator","variable":"x","value":False},
                     {"id":"x1","kind":"indicator","variable":"x","value":True},
                     {"id":"sum","kind":"sum","deterministic":True,
                      "terms":[{"child":"x0","weight":neg},{"child":"x1","weight":pos}]}],"root":"sum"}

def fixture_tests():
    tests=[]
    for mode,want in (("#sat","2"),("pmc","1"),("pwmc","2/3")):
        r=evaluate(hidden(mode)); assert r["exact_output"]==want and r["satisfying_full_assignments"]==2
        if mode in {"pmc","pwmc"}: assert r["distinct_projections"]==1
        tests.append({"name":mode+" hidden witness","result":r})
    for dom,mode in (("q","wmc"),("qi","amc-complex")):
        r=evaluate(cancellation(dom,mode)); assert r["exact_output"]=="0" and r["deterministic_nodes_checked"]==["sum"]
        tests.append({"name":mode+" exact cancellation","result":r})
    bad={"schema":SCHEMA,"contract":"#sat","variables":["x"],"visible":[],"domain":"bool",
         "nodes":[{"id":"x0","kind":"indicator","variable":"x","value":False},
                  {"id":"p","kind":"product","children":["x0","x0"]}],"root":"p"}
    try: evaluate(bad)
    except Invalid as e: tests.append({"name":"overlapping product rejected","error":str(e)})
    else: raise AssertionError("bad product accepted")
    bad={"schema":SCHEMA,"contract":"#sat","variables":["x"],"visible":[],"domain":"bool",
         "nodes":[{"id":"a","kind":"indicator","variable":"x","value":True},
                  {"id":"b","kind":"indicator","variable":"x","value":True},
                  {"id":"s","kind":"sum","deterministic":True,"terms":[
                      {"child":"a","weight":True},{"child":"b","weight":True}]}],"root":"s"}
    try: evaluate(bad)
    except Invalid as e: tests.append({"name":"false determinism rejected","error":str(e)})
    else: raise AssertionError("bad determinism accepted")
    try: evaluate(hidden("amc-complex"))
    except Invalid as e: tests.append({"name":"cross-domain label rejected","error":str(e)})
    else: raise AssertionError("bad domain label accepted")
    return {"suite":"ETH-026 typed circuit IR reference fixtures","checks":len(tests),
            "passed":len(tests),"all_passed":True,"fixtures":tests}

if __name__=="__main__":
    print(json.dumps(fixture_tests(),indent=2,sort_keys=True))
