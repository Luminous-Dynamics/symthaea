#!/usr/bin/env python3
"""Independent finite semantic and structural checker for ETH-018 certificates."""
from __future__ import annotations
from fractions import Fraction
from itertools import product
import hashlib,json,sys
FMT="symthaea.algebraic-rewrite-certificate/v1"
KINDS={"bool","nat","maxcount","pareto","ipoly","qpoly"}
RING={"ipoly","qpoly"}
COMMUTATIVE_MUL=KINDS

def canon(v):return json.dumps(v,sort_keys=True,separators=(",",":"),ensure_ascii=True)
def sha(v):return hashlib.sha256(canon(v).encode()).hexdigest()
def free(t):
 op=t["op"]
 if op=="factor":return set(t["scope"])
 if op in ("add","mul"):return free(t["left"])|free(t["right"])
 if op=="sum":return free(t["body"])-{t["variable"]}
 if op=="neg":return free(t["body"])
 if op=="zero":return set()
 raise ValueError("unknown op")
def parseval(k,v):
 if k in ("bool","nat"):return v
 if k=="maxcount":return None if v is None else tuple(v)
 if k=="pareto":return {(x,y):c for x,y,c in v}
 if k=="ipoly":return {(x,y):c for x,y,c in v}
 if k=="qpoly":return {(x,y):Fraction(n,d) for x,y,n,d in v}
 raise ValueError(k)
def zero(k):return {"bool":False,"nat":0,"maxcount":None,"pareto":{},"ipoly":{},"qpoly":{}}[k]
def add(k,a,b):
 if k=="bool":return a or b
 if k=="nat":return a+b
 if k=="maxcount":
  if a is None:return b
  if b is None:return a
  return a if a[0]>b[0] else b if b[0]>a[0] else (a[0],a[1]+b[1])
 o=dict(a)
 for p,c in b.items():o[p]=o.get(p,0)+c
 if k=="pareto":return {p:c for p,c in o.items() if c and not any(q!=p and all(x>=y for x,y in zip(q,p)) and any(x>y for x,y in zip(q,p)) for q in o)}
 return {p:c for p,c in o.items() if c}
def mul(k,a,b):
 if k=="bool":return a and b
 if k=="nat":return a*b
 if k=="maxcount":return None if a is None or b is None else (a[0]+b[0],a[1]*b[1])
 o={}
 for p,c in a.items():
  for q,d in b.items():
   e=tuple(x+y for x,y in zip(p,q))
   if k not in ("ipoly","qpoly") or all(x<4 for x in e):o[e]=o.get(e,0)+c*d
 if k=="pareto":return {p:c for p,c in o.items() if c and not any(q!=p and all(x>=y for x,y in zip(q,p)) and any(x>y for x,y in zip(q,p)) for q in o)}
 return {p:c for p,c in o.items() if c}
def evalterm(t,k,dom,env):
 op=t["op"]
 if op=="zero":return zero(k)
 if op=="factor":
  i=0
  for v in t["scope"]:i=i*dom[v]+env[v]
  return parseval(k,t["values"][i])
 if op=="sum":
  total=zero(k);v=t["variable"];old=env.get(v);had=v in env
  for n in range(dom[v]):env[v]=n;total=add(k,total,evalterm(t["body"],k,dom,env))
  if had:env[v]=old
  else:env.pop(v,None)
  return total
 if op=="neg":
  if k not in RING:raise ValueError("not ring")
  return {p:-c for p,c in evalterm(t["body"],k,dom,env).items()}
 if op in ("add","mul"):
  a=evalterm(t["left"],k,dom,env);b=evalterm(t["right"],k,dom,env)
  return add(k,a,b) if op=="add" else mul(k,a,b)
 raise ValueError("bad op")
def validate_term(t,k,dom):
 op=t["op"]
 if op=="factor":
  s=t["scope"]
  if not isinstance(s,list) or s!=sorted(set(s)) or any(v not in dom for v in s):raise ValueError("scope")
  n=1
  for v in s:n*=dom[v]
  if not isinstance(t["values"],list) or len(t["values"])!=n:raise ValueError("table size")
  for v in t["values"]:parseval(k,v)
 elif op in ("add","mul"):
  validate_term(t["left"],k,dom);validate_term(t["right"],k,dom)
 elif op in ("sum","neg"):
  if op=="sum" and t["variable"] not in dom:raise ValueError("bound variable")
  if op=="neg" and k not in RING:raise ValueError("ring only")
  validate_term(t["body"],k,dom)
 elif op!="zero":raise ValueError("op")
def apply(rule,s,k):
 if rule=="factor-independent-summand":
  if s["op"]!="sum" or s["body"]["op"]!="mul":raise ValueError("pattern")
  x=s["variable"];a=s["body"]["left"];b=s["body"]["right"]
  if x not in free(a) or x in free(b):raise ValueError("scope")
  return {"op":"mul","left":{"op":"sum","variable":x,"body":a},"right":b}
 if rule=="swap-finite-sums":
  if s["op"]!="sum" or s["body"]["op"]!="sum" or s["variable"]==s["body"]["variable"]:raise ValueError("pattern")
  x,y=s["variable"],s["body"]["variable"]
  return {"op":"sum","variable":y,"body":{"op":"sum","variable":x,"body":s["body"]["body"]}}
 if rule=="split-independent-components":
  if s["op"]!="sum" or s["body"]["op"]!="sum" or s["body"]["body"]["op"]!="mul":raise ValueError("pattern")
  x,y=s["variable"],s["body"]["variable"];a=s["body"]["body"]["left"];b=s["body"]["body"]["right"]
  if x==y or x in free(b) or y in free(a):raise ValueError("scope")
  return {"op":"mul","left":{"op":"sum","variable":x,"body":a},"right":{"op":"sum","variable":y,"body":b}}
 if rule=="swap-multiplicands":
  if s["op"]!="mul" or k not in COMMUTATIVE_MUL:raise ValueError("domain")
  return {"op":"mul","left":s["right"],"right":s["left"]}
 if rule=="cancel-additive-inverse":
  if k not in RING or s["op"]!="add":raise ValueError("ring-only")
  a,b=s["left"],s["right"]
  if a.get("op")=="neg" and a["body"]==b or b.get("op")=="neg" and b["body"]==a:return {"op":"zero"}
  raise ValueError("inverse pair")
 raise ValueError("rule")
def check(c):
 try:
  if c.get("format")!=FMT:return False
  k=c["algebra"]
  if k not in KINDS:return False
  d=c["domains"]
  if not isinstance(d,dict) or not d or any(not isinstance(v,int) or v<1 or v>8 for v in d.values()):return False
  s=c["source"];t=c["target"];validate_term(s,k,d);validate_term(t,k,d)
  if sha(s)!=c.get("source_sha256") or sha(t)!=c.get("target_sha256"):return False
  if apply(c["rule_id"],s,k)!=t:return False
  vs=sorted(free(s)|free(t))
  for values in product(*(range(d[v]) for v in vs)):
   env=dict(zip(vs,values))
   if evalterm(s,k,d,env)!=evalterm(t,k,d,env):return False
  return True
 except (KeyError,TypeError,ValueError,IndexError,ZeroDivisionError):return False
if __name__=="__main__":
 certs=json.load(sys.stdin);accepted=[check(c) for c in certs]
 print(json.dumps({"accepted":accepted,"accepted_count":sum(accepted),"rejected_count":len(accepted)-sum(accepted)}))
