#!/usr/bin/env python3
"""Compact standalone ETH-019 reference reproducer (Python standard library)."""
from __future__ import annotations
import hashlib, hmac, itertools, json, random
from dataclasses import dataclass

SCHEMA = "symthaea.incremental-exact-elimination/v1"
CHECKER = "symthaea.reference.bucket-checker/v1"
AUTH_KEY = b"ETH-019-reference-only-cache-auth-key-do-not-use-in-production"

def canon(x): return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()
def sha(x): return hashlib.sha256(canon(x)).hexdigest()

@dataclass(frozen=True)
class M:
    score: int
    multiplicity: int
    support: int = 1

class Count:
    ident = "natural-count/unbounded-int/v1"
    zero, one = 0, 1
    @staticmethod
    def add(a,b): return a+b
    @staticmethod
    def mul(a,b): return a*b
    @staticmethod
    def enc(x): return x
    @staticmethod
    def dec(x):
        if not isinstance(x,int) or isinstance(x,bool) or x<0: raise ValueError("bad natural")
        return x

class MaxCount:
    ident = "maxcount/multiplicity-distinct-support/v2"
    zero, one = None, M(0,1,1)
    @staticmethod
    def dec(x):
        if x is None: return None
        if isinstance(x,M): return x
        if len(x)==2: return M(int(x[0]),int(x[1]),1)
        return M(int(x[0]),int(x[1]),int(x[2]))
    @staticmethod
    def enc(x): return None if x is None else [x.score,x.multiplicity,x.support]
    @staticmethod
    def add(a,b):
        if a is None: return b
        if b is None: return a
        if a.score>b.score: return a
        if b.score>a.score: return b
        return M(a.score,a.multiplicity+b.multiplicity,a.support+b.support)
    @staticmethod
    def mul(a,b):
        if a is None or b is None: return None
        return M(a.score+b.score,a.multiplicity*b.multiplicity,a.support*b.support)

@dataclass(frozen=True)
class F:
    scope: tuple[int,...]
    vals: tuple
    def encoded(self,A): return {"scope":list(self.scope),"values":[A.enc(x) for x in self.vals]}
    def digest(self,A): return sha(self.encoded(A))
    def index(self,a,cards):
        i=0
        for v in self.scope: i=i*cards[v]+a[v]
        return i

def factor(scope, vals, cards, A):
    scope=tuple(scope)
    if tuple(sorted(set(scope)))!=scope: raise ValueError("scope not sorted unique")
    n=1
    for v in scope:
        if not 0<=v<len(cards): raise ValueError("scope out of range")
        n*=cards[v]
    vals=tuple(A.dec(A.enc(A.dec(x))) for x in vals)
    if len(vals)!=n: raise ValueError("table size mismatch")
    return F(scope,vals)

class Cache:
    def __init__(self,cards,A,key=AUTH_KEY,key_id="local-ref-v1"):
        self.cards,self.A,self.key,self.key_id=tuple(cards),A,key,key_id
        self.entries={}; self.last={}
    def key_for(self,v,bucket):
        ds=sorted(f.digest(self.A) for f in bucket) # multiset: duplicates preserved
        payload={"schema":SCHEMA,"algebra":self.A.ident,"cards":list(self.cards),
                 "variable":v,"inputs":ds,"checker":CHECKER}
        return sha(payload),ds
    def scope_for(self,v,bucket): return tuple(sorted({x for f in bucket for x in f.scope if x!=v}))
    def produce(self,v,bucket):
        scope=self.scope_for(v,bucket); n=1
        for x in scope:n*=self.cards[x]
        out=[]; adds=muls=0
        for oi in range(n):
            a=[0]*len(self.cards); z=oi
            for x in reversed(scope): a[x]=z%self.cards[x]; z//=self.cards[x]
            total=self.A.zero
            for state in range(self.cards[v]):
                a[v]=state; p=self.A.one
                for f in bucket: p=self.A.mul(p,f.vals[f.index(a,self.cards)]); muls+=1
                total=self.A.add(total,p); adds+=1
            out.append(total)
        return F(scope,tuple(out)),adds,muls
    def check(self,v,bucket,candidate):
        # Separate replay path; does not invoke produce() or its index helper.
        scope=tuple(sorted({x for f in bucket for x in f.scope if x!=v}))
        if candidate.scope!=scope:return False
        n=1
        for x in scope:n*=self.cards[x]
        if len(candidate.vals)!=n:return False
        for oi in range(n):
            a=[0]*len(self.cards); z=oi
            for axis in range(len(scope)-1,-1,-1):
                x=scope[axis]; a[x]=z%self.cards[x]; z//=self.cards[x]
            acc=self.A.zero
            for state in range(self.cards[v]):
                a[v]=state; term=self.A.one
                for f in bucket:
                    idx=0
                    for x in f.scope:idx=idx*self.cards[x]+a[x]
                    term=self.A.mul(term,f.vals[idx])
                acc=self.A.add(acc,term)
            if self.A.enc(acc)!=self.A.enc(candidate.vals[oi]):return False
        return True
    def receipt(self,k,outdig):
        return {"schema":"checked-message/v1","key":k,"output":outdig,
                "algebra":self.A.ident,"checker":CHECKER,"key_id":self.key_id}
    def mac(self,r):return hmac.new(self.key,canon(r),hashlib.sha256).hexdigest()
    def lookup(self,k):
        e=self.entries.get(k)
        if e is None:return None,False
        try:
            od=sha(e["out"])
            if od!=e["digest"] or e["receipt"]!=self.receipt(k,od):return None,True
            if not hmac.compare_digest(e["mac"],self.mac(e["receipt"])):return None,True
            out=F(tuple(e["out"]["scope"]),tuple(self.A.dec(x) for x in e["out"]["values"]))
            if out.digest(self.A)!=od:return None,True
            return out,False
        except Exception:return None,True
    def store(self,k,out,v,bucket):
        if not self.check(v,bucket,out):raise AssertionError("fresh message checker failed")
        data=out.encoded(self.A); od=sha(data); r=self.receipt(k,od)
        self.entries[k]={"out":data,"digest":od,"receipt":r,"mac":self.mac(r)}
    def evaluate(self,factors,order):
        order=tuple(order)
        if len(order)!=len(self.cards) or set(order)!=set(range(len(self.cards))):raise ValueError("bad order")
        work=list(factors); stats={"hits":0,"misses":0,"rejected":0,"adds":0,"muls":0,"invalidated":0}
        keys={}
        for v in order:
            bucket=[f for f in work if v in f.scope]; k,ds=self.key_for(v,bucket)
            msg,bad=self.lookup(k)
            if msg is None:
                stats["misses"]+=1; stats["rejected"]+=int(bad)
                msg,a,m=self.produce(v,bucket); stats["adds"]+=a; stats["muls"]+=m
                self.store(k,msg,v,bucket)
            else: stats["hits"]+=1
            if v in self.last and self.last[v]!=k:stats["invalidated"]+=1
            keys[v]=k
            b_ids={id(f) for f in bucket}; work=[f for f in work if id(f) not in b_ids]; work.append(msg)
        self.last=keys
        result=self.A.one
        for f in work:
            if f.scope:raise AssertionError("non-scalar output")
            result=self.A.mul(result,f.vals[0])
        return result,stats

def direct(cards,factors,A):
    total=A.zero
    for a in itertools.product(*(range(c) for c in cards)):
        p=A.one
        for f in factors:p=A.mul(p,f.vals[f.index(a,cards)])
        total=A.add(total,p)
    return A.enc(total)

def main():
    cards=(2,2,2,2,2,2); order=tuple(range(6)); A=Count()
    fs=[factor(sc,[1,1,1,0],cards,A) for sc in [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)]]
    e=Cache(cards,A); r0,s0=e.evaluate(fs,order); r1,s1=e.evaluate(fs,order)
    assert r0==r1 and s0["misses"]==6 and s1["hits"]==6
    changed=fs[:]; changed[0]=factor((0,1),[1,0,1,0],cards,A)
    ri,si=e.evaluate(changed,order); rc,sc=Cache(cards,A).evaluate(changed,order)
    assert A.enc(ri)==A.enc(rc)==direct(cards,changed,A) and si["hits"]==3 and si["misses"]==3
    perm,sp=e.evaluate(list(reversed(fs)),order); assert sp["hits"]==6
    scope=fs[:]; scope[0]=factor((0,3),[1,0,1,0],cards,A)
    rs,ss=e.evaluate(scope,order); assert A.enc(rs)==direct(cards,scope,A)
    tam=Cache(cards,A); tam.evaluate(fs,order); k=next(iter(tam.entries)); ent=tam.entries[k]
    ent["out"]["values"][0]+=9; ent["digest"]=sha(ent["out"]); ent["receipt"]=tam.receipt(k,ent["digest"])
    _,st=tam.evaluate(fs,order); assert st["rejected"]>=1
    foreign=Cache(cards,A,key=b"wrong-key"); foreign.entries=tam.entries.copy(); _,sf=foreign.evaluate(fs,order); assert sf["rejected"]>=1
    matches=hits=misses=invalid=0; cold_add=cold_mul=inc_add=inc_mul=0
    for case in range(64):
        rng=random.Random(20261103+case); c=(2,3,2,2,3,2)
        scopes=[(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(1,4)]
        for alg in (Count(),MaxCount()):
            base=[]
            for scp in scopes:
                n=1
                for v in scp:n*=c[v]
                if isinstance(alg,Count): vals=[rng.randrange(2) for _ in range(n)]
                else: vals=[None if rng.random()<.12 else (rng.randrange(-2,5),rng.randrange(1,4)) for _ in range(n)]
                base.append(factor(scp,vals,c,alg))
            cache=Cache(c,alg); cache.evaluate(base,order)
            target=base[:]; idx=(case*7+3)%len(base); pos=(case*13+1)%len(target[idx].vals)
            old=target[idx].vals[pos]
            new=(1-old) if isinstance(alg,Count) else (None if old is not None else (case%5-2,case%3+1))
            vals=list(target[idx].vals); vals[pos]=new; target[idx]=factor(target[idx].scope,vals,c,alg)
            ir,is0=cache.evaluate(target,order); cold,cs=Cache(c,alg).evaluate(target,order)
            assert alg.enc(ir)==alg.enc(cold)==direct(c,target,alg)
            matches+=1; hits+=is0["hits"]; misses+=is0["misses"]; invalid+=is0["invalidated"]
            cold_add+=cs["adds"]; cold_mul+=cs["muls"]; inc_add+=is0["adds"]; inc_mul+=is0["muls"]
    print(json.dumps({"suite":"ETH-019 compact standalone reproducer","randomized_exact_matches":matches,
      "cache_hits":hits,"cache_misses":misses,"invalidated_steps":invalid,
      "cold_additions":cold_add,"cold_multiplications":cold_mul,
      "incremental_additions":inc_add,"incremental_multiplications":inc_mul,
      "single_edit_disconnected_hits":si["hits"],"single_edit_disconnected_misses":si["misses"],
      "tamper_rejected":st["rejected"],"wrong_key_rejected":sf["rejected"],
      "factor_permutation_hits":sp["hits"],"scope_edit_exact":True,
      "claim_boundary":"Python finite reference; no Rust/Lean/public benchmark or wall-time speedup."},
      indent=2,sort_keys=True))

if __name__=="__main__":main()
