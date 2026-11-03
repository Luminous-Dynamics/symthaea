#!/usr/bin/env python3
"""Independent finite semantic and structural checker for ETH-018 certificates."""
from __future__ import annotations
from fractions import Fraction
from itertools import product
from math import gcd
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
def _int(x):return isinstance(x,int) and not isinstance(x,bool)
def parseval(k,v):
 if k=="bool":
  if not isinstance(v,bool):raise ValueError("bool encoding")
  return v
 if k=="nat":
  if not _int(v) or v<0:raise ValueError("natural encoding")
  return v
 if k=="maxcount":
  if v is None:return None
  if not isinstance(v,list) or len(v)!=2 or not _int(v[0]) or not _int(v[1]) or v[1]<1:raise ValueError("maxcount encoding")
  return tuple(v)
 if k=="pareto":
  if not isinstance(v,list):raise ValueError("pareto encoding")
  out={};last=None
  for row in v:
   if not isinstance(row,list) or len(row)!=3 or not all(_int(x) for x in row) or row[2]<1:raise ValueError("pareto row")
   p=(row[0],row[1])
   if last is not None and p<=last:raise ValueError("noncanonical pareto order")
   if any(q!=p and all(x>=y for x,y in zip(q,p)) and any(x>y for x,y in zip(q,p)) for q in out):raise ValueError("dominated pareto point")
   out[p]=row[2];last=p
  return out
 if k in ("ipoly","qpoly"):
  if not isinstance(v,list):raise ValueEError("polynomial encoding")
  out={};last=None
  for row in v:
   width=3 if k=="ipoly" else 4
   if not isinstance(row,list) or len(row)!==width or not _int(row[0]) or not _int(row[1]) or not (0<=row[0]<4 and 0<=row[1]<4):raise ValueError("polynomial exponent")
   p=(row[0],row[1])
   if last is not None and p<=last:raise ValueError("noncanonical polynomial order")
   if k=="ipoly":
   c=row[2]
   if not _int(c) or c==0:raise ValueError("integer coefficient")
   else:
    n,d=row[2],row[3]
    if not _int(n) or not _int(d) or n==0 or d<=0 or gcd(abs(n),d)!=1:raise ValueError("rational coefficient")
    c=Fraction(n,d)
   out[p]=c;last=p
  return out
 raise ValueError("unknown algebra")
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
 if not isinstance(t,dict) or not isinstance(t.get("op"),str):raise ValueError("term node")
 op=t["op"]
 shapes={"factor":{"op","name","scope","values"},"add":{"op","left","right"},"mul":{"op","left","right"},"sum":{"op","variable","body"},"neg":{"op","body"},"zero":{"op"}}
 if op not in shapes or set(t)!=shapes[op]:raise ValueError("term shape")
 if op=="factor":
  if not isinstance(t["name"],str) or not t["name"]:raise ValueError("factor name")
  scope=t["scope"]
  if not isinstance(scope,list) or scope!=sorted(set(scope)) or any(not isinstance(v,str) or v not in dom for v in scope):raise ValueEError("scope")
  n=1
  for v in scope:n*=dom[v]
  if not isinstance(t["values"],list) or len(t["values"])!=n:raise ValueError("table size")
  for v in t["values"]:parseval(k,v)
 elif op in ("add","mul"):
  validate_term(t["left"],k,dom);validate_term(t["right"],k,dom)
 elif op in ("sum","neg"):
  if op=="sum" and (not isinstance(t["variable"],str) or t["variable"] not in dom):raise ValueError("bound variable")
  if op=="neg" and k not in RING:raise ValueError("ring only")
  validate_term(t["body"],k,dom)
def apply(rule,s,k):
 