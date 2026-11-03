#!/usr/bin/env python3
"""Standalone ETH-019 reference: incremental exact inference with checked reuse."""
from __future__ import annotations
import hashlib, hmac, itertools, json, random
from dataclasses import dataclass

SCHEMA = "symthaea.incremental-exact-elimination/v1"
CHECKER = "symthaea.reference.bucket-checker/v1"
NORMALIZER = "symthaea.same-scope-pointwise-product-and-unit-elision/v1"
AUTH_KEY = b"ETH-019-reference-only-cache-auth-key-do-not-use-in-production"

def canon(x): return json.dumps(x,sort_keys=True,separators=(",",":")).encode()
def sha(x): return hashlib.sha256(canon(x)).hexdigest()

@dataclass(frozen=True)
class M:
    score:int
    multiplicity:int
    support:int=1

class Count:
    ident="natural-count/unbounded-int/v1"
    zero,one=0,1
    @staticmethod
    def add(a,b):return a+b
    @staticmethod
    def mul(a,b):return a*b
    @staticmethod
    def enc(x):return x
    @staticmethod
    def dec(x):
        if not isinstance(x,int) or isinstance(x,bool) or x<0:raise ValueError("bad natural")
        return x

class MaxCount:
    ident="maxcount/multiplicity-distinct-support/v2"
    zero,one=None,M(0,1,1)
    @staticmethod
    def dec(x):
        if x is None:return None
        if isinstance(x,M):return x
        if len(x)==2:return M(int(x[0]),int(x[1]),1)
        return M(int(x[0]),int(x[1]),int(x[2]))
    @staticmethod
    def enc(x):return None if x is None else [x.score,x.multiplicity,x.support]
    @staticmethod
    def add(a,b):
        if a is None:return b
        if b is None:return a
        if a.score>b.score:return a
        if b.score>a.score:return b
        return M(a.score,a.multiplicity+b.multiplicity,a.support+b.support)
    @staticmethod
    def mul(a,b):
        if a is None or b is None:return None
        return M(a.score+b.score,a.multiplicity*b.multiplicity,a.support*b.support)

@dataclass(frozen=True)
class F:
    scope:tuple[int,...]
    vals:tuple
    def encoded(self,A):return {"scope":list(self.scope),"values":[A.enc(x) for x in self.vals]}
    def digest(self,A):return sha(self.encoded(A))
    def index(self,a,cards):
        i=0
        for v in self.scope:i=i*cards[v]+a[v]
        return i

def factor(scope,values,cards,A):
    scope=tuple(scope)
    if tuple(sorted(set(scope)))!=scope:raise ValueError("scope not sorted unique")
    n=1
    for v in scope:
        if not 0<=v<len(cards):raise ValueError("scope out of range")
        n*=cards[v]
    values=tuple(A.dec(A.enc(A.dec(x))) for x in values)
    if len(values)!=n:raise ValueError("table size mismatch")
    return F(scope,values)

class Cache:
    def __init__(self,cards,A,key=AUTH_KEY,key_id="local-ref-v1"):
        self.cards,self.A,self.key,self.key_id=tuple(cards),A,key,key_id
        self.entries={};self.last={}

    def normalize(self,bucket):
        """Group same-scope factors by pointwise multiplication; drop exact units."""
        groups={}
        for f in bucket:groups.setdefault(f.scope,[]).append(f)
        normalized=[]
        for scope,items in sorted(groups.items()):
            items=sorted(items,key=lambda f:f.digest(self.A))
            values=[]
            for cell in range(len(items[0].vals)):
                product=self.A.one
                for f in items:product=self.A.mul(product,f.vals[cell])
                values.append(product)
            if all(self.A.enc(x)==self.A.enc(self.A.one) for x in values):continue
            normalized.append(F(scope,tuple(values)))
        return normalized

    def check_normalization(self,raw,normalized):
        """Independent same-scope reconstruction and identity-elision check."""
        expected=[]
        for scope in sorted({f.scope for f in raw}):
            members=sorted([f for f in raw if f.scope==scope],key=lambda f:f.digest(self.A))
            n=1
            for v in scope:n*=self.cards[v]
            values=[]
            for cell in range(n):
                product=self.A.one
                for f in members:product=self.A.mul(product,f.vals[cell])
                values.append(product)
            if any(self.A.enc(x)!=self.A.enc(self.A.one) for x in values):
                expected.append(F(scope,tuple(values)))
        return [f.encoded(self.A) for f in expected]==[f.encoded(self.A) for f in normalized]

    def key_for(self,v,normalized):
        ds=sorted(f.digest(self.A) for f in normalized)
        payload={"schema":SCHEMA,"normalizer":NORMALIZER,"algebra":self.A.ident,
                 "cards":list(self.cards),"variable":v,"inputs":ds,"checker":CHECKER}
        return sha(payload),ds

    def scope_for(self,v,bucket):return tuple(sorted({x for f in bucket for x in f.scope if x!=v}))

    def produce(self,v,bucket):
        scope=self.scope_for(v,bucket);n=1
        for x in scope:n*=self.cards[x]
        out=[];adds=muls=0
        for oi in range(n):
            a=[0]*len(self.cards);z=oi
            for x in reversed(scope):a[x]=z%self.cards[x];z//=self.cards[x]
            total=self.A.zero
            for state in range(self.cards[v]):
                a[v]=state;p=self.A.one
                for f in bucket:
                    p=self.A.mul(p,f.vals[f.index(a,self.cards)]);muls+=1
                total=self.A.add(total,p);adds+=1
            out.append(total)
        return F(scope,tuple(out)),adds,muls

    def check(self,v,raw_bucket,candidate):
        """Replay raw factors; permit only axes proven irrelevant by equality."""
        raw_scope=tuple(sorted({x for f in raw_bucket for x in f.scope if x!=v}))
        if any(x not in raw_scope for x in candidate.scope):return False
        size=1
        for x in candidate.scope:size*=self.cards[x]
        if len(candidate.vals)!=size:return False
        raw_size=1
        for x in raw_scope:raw_size*=self.cards[x]
        for oi in range(raw_size):
            a=[0]*len(self.cards);z=oi
            for axis in range(len(raw_scope)-1,-1,-1):
                x=raw_scope[axis];a[x]=z%self.cards[x];z//=self.cards[x]
            acc=self.A.zero
            for state in range(self.cards[v]):
                a[v]=state;term=self.A.one
                for f in raw_bucket:
                    idx=0
                    for x in f.scope:idx=idx*self.cards[x]+a[x]
                    term=self.A.mul(term,f.vals[idx])
                acc=self.A.add(acc,term)
            ci=candidate.index(a,self.cards)
            if self.A.enc(acc)!=self.A.enc(candidate.vals[ci]):return False
        return True

    def receipt(self,k,outdig):
        return {"schema":"checked-message/v1","key":k,"output":outdig,
                "algebra":self.A.ident,"checker":CHECKER,"normalizer":NORMALIZER,"key_id":self.key_id}
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
    def store(self,k,out,v,raw_bucket):
        if not self.check(v,raw_bucket,out):raise AssertionError("raw-bucket checker failed")
        data=out.encoded(self.A);od=sha(data);r=self.receipt(k,od)
        self.entries[k]={"out":data,"digest":od,"receipt":r,"mac":self.mac(r)}

    def evaluate(self,factors,order):
        order=tuple(order)
        if len(order)!=len(self.cards) or set(order)!=set(range(len(self.cards))):raise ValueError("bad order")
        work=list(factors)
        stats={"hits":0,"misses":0,"rejected":0,"adds":0,"muls":0,"invalidated":0}
        keys={}
        for v in order:
            raw_bucket=[f for f in work if v in f.scope]
            normalized=self.normalize(raw_bucket)
            if not self.check_normalization(raw_bucket,normalized):raise AssertionError("normalizer checker failed")
            k,_=self.key_for(v,normalized)
            msg,bad=self.lookup(k)
            if msg is None:
                stats["misses"]+=1;stats["rejected"]+=int(bad)
                msg,a,m=self.produce(v,normalized);stats["adds"]+=a;stats["muls"]+=m
                self.store(k,msg,v,raw_bucket)
            else:stats["hits"]+=1
            if v in self.last and self.last[v]!=k:stats["invalidated"]+=1
            keys[v]=k
            ids={id(f) for f in raw_bucket};work=[f for f in work if id(f) not in ids];work.append(msg)
        self.last=keys
        result=self.A.one
        for f in work:
            if f.scope:raise AssertionError("non-scalar output")
            result=self.A.mul(result,f.vals[0])
        return result,stats

def direct(cards,factors,A):
    total=A.zero
    for assignment in itertools.product(*(range(c) for c in cards)):
        product=A.one
        for f in factors:product=A.mul(product,f.vals[f.index(assignment,cards)])
        total=A.add(total,product)
    return A.enc(total)

def main():
    cards=(2,2,2,2,2,2);order=tuple(range(6));A=Count()
    fs=[factor(sc,[1,1,1,0],cards,A) for sc in [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)]]
    e=Cache(cards,A);r0,s0=e.evaluate(fs,order);r1,s1=e.evaluate(fs,order)
    assert r0==r1 and s0["misses"]==6 and s1["hits"]==6
    changed=fs[:];changed[0]=factor((0,1),[1,0,1,0],cards,A)
    ri,si=e.evaluate(changed,order);rc,sc=Cache(cards,A).evaluate(changed,order)
    assert A.enc(ri)==A.enc(rc)==direct(cards,changed,A) and si["hits"]==3 and si["misses"]==3
    perm,sp=e.evaluate(list(reversed(fs)),order);assert sp["hits"]==6
    c2=(2,2);a=factor((0,1),[2,1,3,0],c2,A);b=factor((0,1),[1,3,2,4],c2,A)
    merged=factor((0,1),[2,3,6,0],c2,A);unit=factor((0,),[1,1],c2,A)
    pre=Cache(c2,A);rs,_=pre.evaluate([a,b],[0,1]);rm,ps=pre.evaluate([merged,unit],[0,1])
    assert rs==rm and ps["hits"]==2
    wrong=factor((0,1),[2,4,6,0],c2,A);assert not pre.check_normalization([a,b],[wrong])
    scope=fs[:];scope[0]=factor((0,3),[1,0,1,0],cards,A)
    rs,ss=e.evaluate(scope,order);assert A.enc(rs)==direct(cards,scope,A)
    tam=Cache(cards,A);tam.evaluate(fs,order);k=next(iter(tam.entries));entry=tam.entries[k]
    entry["out"]["values"][0]+=9;entry["digest"]=sha(entry["out"]);entry["receipt"]=tam.receipt(k,entry["digest"])
    _,st=tam.evaluate(fs,order);assert st["rejected"]>=1
    foreign=Cache(cards,A,key=b"wrong-key");foreign.entries=tam.entries.copy();_,sf=foreign.evaluate(fs,order);assert sf["rejected"]>=1
    matches=hits=misses=invalid=0;cold_add=cold_mul=inc_add=inc_mul=0
    for case in range(64):
        rng=random.Random(20261103+case);c=(2,3,2,2,3,2)
        scopes=[(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(1,4)]
        for alg in (Count(),MaxCount()):
            base=[]
            for scope0 in scopes:
                n=1
                for v in scope0:n*=c[v]
                if isinstance(alg,Count):vals=[rng.randrange(2) for _ in range(n)]
                else:vals=[None if rng.random()<.12 else (rng.randrange(-2,5),rng.randrange(1,4)) for _ in range(n)]
                base.append(factor(scope0,vals,c,alg))
            cache=Cache(c,alg);cache.evaluate(base,order)
            target=base[:];idx=(case*7+3)%len(base);pos=(case*13+1)%len(target[idx].vals);old=target[idx].vals[pos]
            new=(1-old) if isinstance(alg,Count) else (None if old is not None else (case%5-2,case%3+1))
            vals=list(target[idx].vals);vals[pos]=new;target[idx]=factor(target[idx].scope,vals,c,alg)
            ir,is0=cache.evaluate(target,order);cold,cs=Cache(c,alg).evaluate(target,order)
            assert alg.enc(ir)==alg.enc(cold)==direct(c,target,alg)
            matches+=1;hits+=is0["hits"];misses+=is0["misses"];invalid+=is0["invalidated"]
            cold_add+=cs["adds"];cold_mul+=cs["muls"];inc_add+=is0["adds"];inc_mul+=is0["muls"]
    print(json.dumps({"suite":"ETH-019 content-normalized incremental exact cache",
      "randomized_exact_matches":matches,"cache_hits":hits,"cache_misses":misses,"invalidated_steps":invalid,
      "cold_additions":cold_add,"cold_multiplications":cold_mul,"incremental_additions":inc_add,
      "incremental_multiplications":inc_mul,"single_edit_disconnected_hits":si["hits"],
      "single_edit_disconnected_misses":si["misses"],"factor_permutation_hits":sp["hits"],
      "same_scope_split_merge_hits":ps["hits"],"unit_factor_elision_hits":ps["hits"],
      "normalization_mutant_rejected":True,"scope_edit_exact":True,
      "tamper_rejected":st["rejected"],"wrong_key_rejected":sf["rejected"],
      "claim_boundary":"Python finite reference only. No Rust/Lean/public benchmark or measured wall-time speedup."},
      indent=2,sort_keys=True))

if __name__=="__main__":main()
