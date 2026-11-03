#!/usr/bin/env python3
"""Finite exact oracle for domain-qualified factor-algebra rewrites."""
from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
import hashlib, json, random, subprocess, sys

FMT = "symthaea.algebraic-rewrite-certificate/v1"
KINDS = ("bool", "nat", "maxcount", "pareto", "ipoly", "qpoly")
RULES = ("factor-independent-summand", "swap-finite-sums", "split-independent-components", "swap-multiplicands")

@dataclass(frozen=True)
class Alg:
    kind: str
    @property
    def zero(self): return {"bool":False,"nat":0,"maxcount":None,"pareto":{},"ipoly":{},"qpoly":{}}[self.kind]
    @property
    def one(self): return {"bool":True,"nat":1,"maxcount":(0,1),"pareto":{(0,0):1},"ipoly":{(0,0):1},"qpoly":{(0,0):Fraction(1)}}[self.kind]
    def add(self,a,b):
        k=self.kind
        if k=="bool": return a or b
        if k=="nat": return a+b
        if k=="maxcount":
            if a is None:return b
            if b is None:return a
            return a if a[0]>b[0] else b if b[0]>a[0] else (a[0],a[1]+b[1])
        out=dict(a)
        for p,c in b.items(): out[p]=out.get(p,0)+c
        return self.prune(out) if k=="pareto" else {p:c for p,c in out.items() if c}
    def mul(self,a,b):
        k=self.kind
        if k=="bool":return a and b
        if k=="nat":return a*b
        if k=="maxcount":return None if a is None or b is None else (a[0]+b[0],a[1]*b[1])
        out={}
        for p,c in a.items():
            for q,d in b.items():
                e=tuple(x+y for x,y in zip(p,q))
                if k not in ("ipoly","qpoly") or all(x<4 for x in e):out[e]=out.get(e,0)+c*d
        return self.prune(out) if k=="pareto" else {p:c for p,c in out.items() if c}
    @staticmethod
    def prune(m):
        return {p:c for p,c in m.items() if c and not any(q!=p and all(x>=y for x,y in zip(q,p)) and any(x>y for x,y in zip(q,p)) for q in m)}
    def neg(self,a):
        if self.kind not in ("ipoly","qpoly"):raise ValueError("not a ring")
        return {p:-c for p,c in a.items()}

def enc(a,v):
    k=a.kind
    if k in ("bool","nat"):return v
    if k=="maxcount":return None if v is None else list(v)
    if k=="pareto":return [[*p,c] for p,c in sorted(v.items())]
    if k=="ipoly":return [[*p,c] for p,c in sorted(v.items())]
    if k=="qpoly":return [[*p,c.numerator,c.denominator] for p,c in sorted(v.items())]
    raise ValueError(k)
def dec(a,v):
    k=a.kind
    if k in ("bool","nat"):return v
    if k=="maxcount":return None if v is None else tuple(v)
    if k=="pareto":return {(x,y):c for x,y,c in v}
    if k=="ipoly":return {(x,y):c for x,y,c in v}
    if k=="qpoly":return {(x,y):Fraction(n,d) for x,y,n,d in v}
    raise ValueError(k)
def randval(r,a):
    k=a.kind
    if k=="bool":return bool(r.randrange(2))
    if k=="nat":return r.randrange(4)
    if k=="maxcount":return None if r.randrange(5)==0 else (r.randrange(-2,4),r.randrange(1,4))
    if k=="pareto":
        m={}
        for _ in range(r.randrange(1,4)):
            p=(r.randrange(4),r.randrange(4));m[p]=m.get(p,0)+r.randrange(1,3)
        return a.prune(m)
    if k=="ipoly":return {(i,j):r.randrange(-2,3) for i in range(3) for j in range(3) if r.randrange(3)==0 and r.randrange(-2,3)!=0}
    if k=="qpoly":return {(i,j):Fraction(r.randrange(-2,3),r.randrange(1,4)) for i in range(3) for j in range(3) if r.randrange(3)==0 and r.randrange(-2,3)!=0}
    raise ValueError(k)
def fac(r,a,name,scope,cards):
    n=1
    for v in scope:n*=cards[v]
    return {"op":"factor","name":name,"scope":list(scope),"values":[enc(a,randval(r,a)) for _ in range(n)]}
def free(t):
    if t["op"]=="factor":return set(t["scope"])
    if t["op"] in ("add","mul"):return free(t["left"])|free(t["right"])
    if t["op"] in ("sum","neg"):return free(t["body"])-({t["variable"]} if t["op"]=="sum" else set())
    return set()
def ev(t,a,cards,env):
    op=t["op"]
    if op=="factor":
        i=0
        for v in t["scope"]:i=i*cards[v]+env[v]
        return dec(a,t["values"][i])
    if op=="zero":return a.zero
    if op=="neg":return a.neg(ev(t["body"],a,cards,env))
    if op in ("add","mul"):
        x=ev(t["left"],a,cards,env);y=ev(t["right"],a,cards,env)
        return a.add(x,y) if op=="add" else a.mul(x,y)
    if op=="sum":
        total=a.zero;old=env.get(t["variable"]);had=t["variable"] in env
        for x in range(cards[t["variable"]]):
            env[t["variable"]]=x;total=a.add(total,ev(t["body"],a,cards,env))
        if had:env[t["variable"]]=old
        else:env.pop(t["variable"],None)
        return total
    raise ValueError(op)
def apply(rule,s,a):
    op=s["op"]
    if rule=="factor-independent-summand":
        if op!="sum" or s["body"]["op"]!="mul":raise ValueError("pattern")
        x=s["variable"];l,r=s["body"]["left"],s["body"]["right"]
        if x not in free(l) or x in free(r):raise ValueError("scope")
        return {"op":"mul","left":{"op":"sum","variable":x,"body":l},"right":r}
    if rule=="swap-finite-sums":
        if op!="sum" or s["body"]["op"]!="sum":raise ValueError("pattern")
        x,y=s["variable"],s["body"]["variable"]
        if x==y:raise ValueError("bound-variable-alias")
        return {"op":"sum","variable":y,"body":{"op":"sum","variable":x,"body":s["body"]["body"]}}
    if rule=="split-independent-components":
        if op!="sum" or s["body"]["op"]!="sum" or s["body"]["body"]["op"]!="mul":raise ValueError("pattern")
        x,y=s["variable"],s["body"]["variable"];l,r=s["body"]["body"]["left"],s["body"]["body"]["right"]
        if x==y or x in free(r) or y in free(l):raise ValueError("component-scope")
        return {"op":"mul","left":{"op":"sum","variable":x,"body":l},"right":{"op":"sum","variable":y,"body":r}}
    if rule=="swap-multiplicands":
        if op!="mul" or a.kind not in KINDS:raise ValueError("pattern/domain")
        return {"op":"mul","left":s["right"],"right":s["left"]}
    if rule=="cancel-additive-inverse":
        if op!="add" or a.kind not in ("ipoly","qpoly"):raise ValueError("ring-only")
        l,r=s["left"],s["right"]
        if l.get("op")=="neg" and l["body"]==r or r.get("op")=="neg" and r["body"]==l:return {"op":"zero"}
        raise ValueError("inverse-pair")
    raise ValueError("unknown-rule")
def canon(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True)
def sha(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def cert(rule,src,a,cards):
    dst=apply(rule,src,a)
    return {"format":FMT,"algebra":a.kind,"domains":dict(sorted(cards.items())),"rule_id":rule,"source":src,"target":dst,"source_sha256":sha(src),"target_sha256":sha(dst)}
def main():
    r=random.Random(0xE7018);cards={"x":2,"y":3,"z":2};positive=[];counts={x:0 for x in RULES}
    for kind in KINDS:
        a=Alg(kind)
        for _ in range(24):
            A=fac(r,a,"A",("x","y"),cards);B=fac(r,a,"B",("y","z"),cards)
            s={"op":"sum","variable":"x","body":{"op":"mul","left":A,"right":B}}
            positive.append(cert(RULES[0],s,a,cards));counts[RULES[0]]+=1
            F=fac(r,a,"F",("x","y"),cards);s={"op":"sum","variable":"x","body":{"op":"sum","variable":"y","body":F}}
            positive.append(cert(RULES[1],s,a,cards));counts[RULES[1]]+=1
            A=fac(r,a,"A",("x","z"),cards);B=fac(r,a,"B",("y","z"),cards)
            s={"op":"sum","variable":"x","body":{"op":"sum","variable":"y","body":{"op":"mul","left":A,"right":B}}}
            positive.append(cert(RULES[2],s,a,cards));counts[RULES[2]]+=1
            A=fac(r,a,"A",("x","y"),cards);B=fac(r,a,"B",("y","z"),cards)
            positive.append(cert(RULES[3],{"op":"mul","left":A,"right":B},a,cards));counts[RULES[3]]+=1
    for kind in ("ipoly","qpoly"):
        a=Alg(kind);A=fac(r,a,"R",("z",),cards);positive.append(cert("cancel-additive-inverse",{"op":"add","left":{"op":"neg","body":A},"right":A},a,cards))
    negatives=[]
    c=json.loads(json.dumps(positive[0]));c["target_sha256"]="0"*64;negatives.append(c)
    c=json.loads(json.dumps(positive[0]));c["rule_id"]="swap-finite-sums";negatives.append(c)
    c=json.loads(json.dumps(positive[0]));c["source"]["body"]["right"]["scope"]=["x","y","z"];c["source"]["body"]["right"]["values"]*=2;c["source_sha256"]=sha(c["source"]);negatives.append(c)
    c=json.loads(json.dumps(positive[-1]));c["algebra"]="maxcount";c["source_sha256"]=sha(c["source"]);c["target_sha256"]=sha(c["target"]);negatives.append(c)
    payload=positive+negatives
    checker=Path(__file__).with_name("algebraic_rewrite_checker.py")
    p=subprocess.run([sys.executable,str(checker)],input=json.dumps(payload),text=True,capture_output=True,check=True)
    response=json.loads(p.stdout)
    assert response=={"accepted":[True]*len(positive)+[False]*len(negatives),"accepted_count":len(positive),"rejected_count":len(negatives)}
    for c in positive:
        a=Alg(c["algebra"]);d=c["domains"];src=c["source"];dst=c["target"]
        vars_=sorted(free(src)|free(dst))
        for vals in product(*(range(d[v]) for v in vars_)):
            env=dict(zip(vars_,vals));assert ev(src,a,d,env)==ev(dst,a,d,env)
    out={"suite":"domain-qualified algebraic rewrite reference","seed":"0xE7018","algebras":list(KINDS),"trials_per_algebra_and_rule":24,"positive_rule_checks":counts,"positive_rule_checks_total":len(positive)-2,"ring_cancellation_checks":2,"separate_process_checker":{"accepted_positive":response["accepted_count"],"positive_total":len(positive),"rejected_negative":response["rejected_count"],"negative_total":len(negatives)},"negative_controls_rejected":response["rejected_count"],"all_passed":True,"claim_ceiling":"Finite exhaustive semantic checks on small table factors plus local certificate replay. Not a Lean proof, cross-language verifier, e-graph search, production Rust implementation, or performance result."}
    Path(__file__).with_name("algebraic_rewrite_results.json").write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=="__main__":main()
