#!/usr/bin/env python3
"""Exact finite reference checker for Symthaea's typed tractable-circuit IR.

Not a CPOG/VeriPB emitter, proof, or scalable production evaluator.
"""
from fractions import Fraction as F
from itertools import product
import json

SCHEMA = "symthaea.typed-tractable-circuit/v1"
class Invalid(ValueError): pass

def qi_add(a,b): return (a[0]+b[0],a[1]+b[1])
def qi_mul(a,b): return (a[0]*b[0]-a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def rat(x):
    if not isinstance(x,list) or len(x)!=2 or any(type(v) is not int for v in x): raise Invalid("rational shape")
    if x[1]<=0: raise Invalid("denominator")
    y=F(*x)
    if [y.numerator,y.denominator]!=x: raise Invalid("noncanonical rational")
    return y
def dec(d,x):
    if d=="bool":
        if type(x) is not bool: raise Invalid("Boolean coefficient")
        return x
    if d=="q": return rat(x)
    if d=="qi":
        if not isinstance(x,dict) or set(x)!={"re","im"}: raise Invalid("Q(i) shape")
        return (rat(x["re"]),rat(x["im"]))
    raise Invalid("unknown domain")
def z(d): return False if d=="bool" else F(0) if d=="q" else (F(0),F(0))
def o(d): return True if d=="bool" else F(1) if d=="q" else (F(1),F(0))
def plus(d,a,b): return (a or b) if d=="bool" else a + b
def times(d,a,b): return (a and b) if d=="bool" else a * b
def fmt(d,x): return str(x) if d!="qi" else str(x[0]) if x[1]==0 else f"{x[0]}{'+' if x[1]>=0 else ''}{x[1]}i"

def evaluate(c):
    required={"schema","contract","variables","visible","domain","nodes","root"}
    if not isinstance(c,dict) or not required<=set(c) or set(c)-required-{"visible_weights"}: raise Invalid("header")
    if c["schema"]!=SCHEMA: raise Invalid("schema")
    contract,domain=c["contract"],c["domain"]
    modes={"#sat":"bool","pmc":"bool","pwmc":"bool","wmc":"q","amc-complex":"qi"}
    if contract not in modes or modes[contract]!=domain: raise Invalid("contract/domain mismatch")
    vs,visible,nodes=c["variables"],c["visible"],c["nodes"]
    if not isinstance(vs,list) or any(not isinstance(v,str) or not v for v in vs) or len(set(vs))!=len(vs): raise Invalid("variable universe")
    u=set(vs)
    if not isinstance(visible,list) or len(set(visible))!=len(visible) or not set(visible)<=u: raise Invalid("visible set")
    if contract not in {"pmc","pwmc"} and visible: raise Invalid("visible set on nonprojected contract")
    if contract!="pwmc" and "visible_weights" in c: raise Invalid("visible weights outside PWMC")
    by,scopes={},{}
    if not isinstance(nodes,list) or not nodes: raise Invalid("empty nodes")
    for n in nodes:
        if not isinstance(n,dict) or not isinstance(n.get("id"),str) or n["id"] in by: raise Invalid("node ID")
        by[n["id"]]=n

    for n in nodes:
        i,k=n["id"],n.get("kind")
        if k=="indicator":
            if set(n)!={"id","kind","variable","value"} or n["variable"] not in u or type(n["value"]) is not bool: raise Invalid("indicator")
            scopes[i]={n["variable"]}
        elif k=="constant":
            if set(n)!={"id","kind","value"}: raise Invalid("constant")
            dec(domain,n["value"]); scopes[i]=set()
        elif k=="product":
            if set(n)!={"id","kind","children"} or not isinstance(n["children"],list) or not n["children"]: raise Invalid("product")
            ch=n["children"]
            if any(x not in scopes for x in ch): raise Invalid("forward product edge")
            flat=[v for x in ch for v in scopes[x]]
            if len(flat)!=len(set(flat)): raise Invalid("nondecomposable product")
            scopes[i]=set(flat)
        elif k=="sum":
            if set(n)-{"id","kind","terms","deterministic"} or not {"id","kind","terms"}<=set(n) or not isinstance(n["terms"],list) or not n["terms"]: raise Invalid("sum")
            if "deterministic" in n and type(n["deterministic"]) is not bool: raise Invalid("determinism flag")
            ch=[]
            for t in n["terms"]:
                if not isinstance(t,dict) or set(t)!={"child","weight"} or t["child"] not in scopes: raise Invalid("sum term/edge")
                dec(domain,t["weight"]); ch.append(t["child"])
            scopes[i]=set().union(*(scopes[x] for x in ch))
        else: raise Invalid("unknown node kind")

    root=c["root"]
    if root not in by or not scopes[root]<=u: raise Invalid("root")
    def ev(i,a):
        n=by[i]; k=n["kind"]
        if k=="indicator": return o(domain) if a[n["variable"]]==n["value"] else z(domain)
        if k=="constant": return dec(domain,n["value"])
        if k=="product":
            r=o(domain)
            for x in n["children"]: r=times(domain,r,ev(x,a))
            return r
        r=z(domain)
        for t in n["terms"]: r=plus(domain,r,times(domain,dec(domain,t["weight"]),ev(t["child"],a)))
        return r

    checked=[]
    for n in nodes:
        if n["kind"]=="sum" and n.get("deterministic",False):
            scope=sorted(scopes[n["id"]])
            for bits in product((False,True),repeat=len(scope)):
                a=dict(zip(scope,bits))
                ts=[times(domain,dec(domain,t["weight"]),ev(t["child"],a)) for t in n["terms"]]
                for j in range(len(ts)):
                    for k in range(j+1,len(ts)):
                        if times(domain,ts[j],ts[k])!=z(domain): raise Invalid("false determinism claim")
            checked.append(n["id"])

    total=z("qi" if domain=="qi" else "q"); nonzero=0; sat=0; projections=set()
    for bits in product((False,True),repeat=len(vs)):
        a=dict(zip(vs,bits)); v=ev(root,a)
        if (not v) if domain=="bool" else (v==z(domain)): continue
        nonzero+=1
        if contract in {"#sat","pmc","pwmc"}:
            sat+=1; projections.add(tuple(a[x] for x in visible))
        if contract=="#sat": total+=1
        elif contract in {"wmc","amc-complex"}: total=plus(domain,total,v)
    if contract=="pmc": total=F(len(projections))
    elif contract=="pwmc":
        weights=c.get("visible_weights",{})
        if not isinstance(weights,dict) or not set(weights)<=set(visible): raise Invalid("PWMC weight scope")
        for key in projections:
            a=dict(zip(visible,key)); w=F(1)
            for x in visible:
                pair=weights.get(x,[[1,1],[1,1]])
                if not isinstance(pair,list) or len(pair)!=2: raise Invalid("weight pair")
                w*=rat(pair[1 if a[x] else 0])
            total+=w
    return {"contract":contract,"domain":domain,"root_scope":sorted(scopes[root]),"deterministic_nodes_checked":checked,"satisfying_full_assignments":sat if contract in {"#sat","pmc","pwmc"} else None,"nonzero_function_assignments":nonzero,"distinct_projections":len(projections) if contract in {"pmc","pwmc"} else None,"exact_output":fmt("qi" if domain=="qi" else "q",total)}

def hidden(contract):
    c={"schema":SCHEMA,"contract":contract,"variables":["x","y"],"visible":["x"] if contract in {"pmc","pwmc"} else [],"domain":"bool","nodes":[{"id":"x1","kind":"indicator","variable":"x","value":True}],"root":"x1"}
    if contract=="pwmc": c["visible_weights"]={"x":[[1,3],[2,3]]}
    return c

def cancellation(domain,contract):
    a,b=([1,1],[-1,1]) if domain=="q" else ({"re":[1,1],"im":[2,1]},{"re":[-1,1],"im":[-2,1]})
    return {"schema":SCHEMA,"contract":contract,"variables":["x"],"visible":[],"domain":domain,"nodes":[{"id":"x0","kind":"indicator","variable":"x","value":False},{"id":"x1","kind":"indicator","variable":"x","value":True},{"id":"sum","kind":"sum","deterministic":True,"terms":[{"child":"x0","weight":b},{"child":"x1","weight":a}]}],"root":"sum"}

def fixture_tests():
    tests=[]
    for mode,expected in (("#sat","2"),("pmc","1"),("pwmc","2/3")):
        r=evaluate(hidden(mode)); assert r["exact_output"]==expected and r["satisfying_full_assignments"]==2
        if mode in {"pmc","pwmc"}: assert r["distinct_projections"]==1
        tests.append({"name":mode+" hidden witness", "result":r})
    for domain,mode in (("q","wmc"),("qi","amc-complex")):
        r=evaluate(cancellation(domain,mode)); assert r["exact_output"]=="0" and r["deterministic_nodes_checked"]==["sum"]
        tests.append({"name":mode+" exact cancellation", "result":r})
    bad={"schema":SCHEMA,"contract":"#sat","variables":["x"],"visible":[],"domain":"bool","nodes":[{"id":"x0","kind":"indicator","variable":"x","value":False},{"id":"p","kind":"product","children":["x0","x0"]}],"root":"p"}
    try: evaluate(bad)
    except Invalid as e: tests.append({"name":"overlapping product rejected","error":str(e)})
    else: raise AssertionError("bad product accepted")
    bad={"schema":SCHEMA,"contract":"#sat","variables":["x"],"visible":[],"domain":"bool","nodes":[{"id":"a","kind":"indicator","variable":"x","value":True},{"id":"b","kind":"indicator","variable":"x","value":True},{"id":"s","kind":"sum","deterministic":True,"terms":[{"child":"a","weight":True},{"child":"b","weight":True}]}],"root":"s"}
    try: evaluate(bad)
    except Invalid as e: tests.append({"name":"false determinism rejected","error":str(e)})
    else: raise AssertionError("bad determinism accepted")
    try: evaluate(hidden("amc-complex"))
    except Invalid as e: tests.append({"name":"cross-domain label rejected","error":str(e)})
    else: raise AssertionError("bad domain label accepted")
    return {"suite":"ETH-026 typed circuit IR reference fixtures","checks":len(tests),"passed":len(tests),"all_passed":True,"fixtures":tests}

if __name__=="__main__": print(json.dumps(fixture_tests(),indent=2,sort_keys=True))
