#!/usr/bin/env python3
"""Compact standalone ETH-019 reference reproducer (Python standard library)."""
from __future__ import annotations
import hashlib, hmac, itertools, json, random
from dataclasses import dataclass

SCHEMA = "symthaea.incremental-exact-elimination/v1"
CHECKER = "symthaea.reference.bucket-checker/v1"
NORMALIZER = "symthaea.same-scope-pointwise-product-and-unit-elision/v1"
MESSAGE_CANONICALIZER = "symthaea.exact-scope-independence-elision/v1"
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
    def normalize(self,bucket):
        groups={}
        for f in bucket: groups.setdefault(f.scope,[]).append(f)
        normalized=[]
        for scope,items in sorted(groups.items()):
            items=sorted(items,key=lambda f:f.digest(self.A))
            vals=[]
            for i in range(len(items[0].vals)):
                product=self.A.one
                for f in items: product=self.A.mul(product,f.vals[i])
                vals.append(product)
            # Remove a multiplicative-identity table, even if preprocessing
            # changed factor IDs or split/merged same-scope factors.
            if all(self.A.enc(v)==self.A.enc(self.A.one) for v in vals): continue
            normalized.append(F(scope,tuple(vals)))
        return normalized
    def check_normalization(self,raw,normalized):
        # Independent reconstruction by scope and table-cell coordinates.
        expected=[]
        scope_set=sorted({f.scope for f in raw})
        for scope in scope_set:
            members=[f for f in raw if f.scope==scope]
            members.sort(key=lambda f:f.digest(self.A))
            cell_count=1
            for v in scope: cell_count*=self.cards[v]
            cells=[]
            for cell in range(cell_count):
                value=self.A.one
                for f in members: value=self.A.mul(value,f.vals[cell])
                cells.append(value)
            if any(self.A.enc(x)!=self.A.enc(self.A.one) for x in cells): expected.append(F(scope,tuple(cells)))
        return [f.encoded(self.A) for f in expected]==[f.encoded(self.A) for f in normalized]
    def key_for(self,v,bucket):
        ds=sorted(f.digest(self.A) for f in bucket)
        payload={"schema":SCHEMA,"algebra":self.A.ident,"cards":list(self.cards),
                 "variable":v,"inputs":ds,"checker":CHECKER,"normalizer":NORMALIZER,
                 "message_canonicalizer":MESSAGE_CANONICALIZER}
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
    def minimize_message_scope(self,message):
        """Remove axes on which the exact message is constant, one at a time."""
        scope=list(message.scope)
        values=list(message.vals)
        removed=0
        changed=True
        while changed:
            changed=False
            for variable in list(scope):
                next_scope=tuple(v for v in scope if v!=variable)
                new_size=1
                for v in next_scope:new_size*=self.cards[v]
                reduced=[]
                independent=True
                for index in range(new_size):
                    assignment=[0]*len(self.cards); remainder=index
                    for v in reversed(next_scope):
                        assignment[v]=remainder%self.cards[v]; remainder//=self.cards[v]
                    observed=[]
                    for state in range(self.cards[variable]):
                        assignment[variable]=state
                        cell=0
                        for v in scope:cell=cell*self.cards[v]+assignment[v]
                        observed.append(values[cell])
                    if any(self.A.enc(x)!=self.A.enc(observed[0]) for x in observed[1:]):
                        independent=False
                        break
                    reduced.append(observed[0])
                if independent:
                    scope=list(next_scope)
                    values=reduced
                    removed+=1
                    changed=True
                    break
        return F(tuple(scope),tuple(values)),removed

    def check(self,v,bucket,candidate):
        # Separate replay path. Candidate may omit raw-bucket separator axes
        # only when the exact result is constant along those axes.
        raw_scope=tuple(sorted({x for f in bucket for x in f.scope if x!=v}))
        if any(x not in raw_scope for x in candidate.scope):return False
        expected=1
        for x in candidate.scope:expected*=self.cards[x]
        if len(candidate.vals)!=expected:return False
        raw_entries=1
        for x in raw_scope:raw_entries*=self.cards[x]
        for oi in range(raw_entries):
            a=[0]*len(self.cards); z=oi
            for axis in range(len(raw_scope)-1,-1,-1):
                x=raw_scope[axis]; a[x]=z%self.cards[x]; z//=self.cards[x]
            acc=self.A.zero
            for state in range(self.cards[v]):
                a[v]=state; term=self.A.one
                for f in bucket:
                    idx=0
                    for x in f.scope:idx=idx*self.cards[x]+a[x]
                    term=self.A.mul(term,f.vals[idx])
                acc=self.A.add(acc,term)
            ci=candidate.index(a,self.cards)
            if self.A.enc(acc)!=self.A.enc(candidate.vals[ci]):return False
        return True
    def receipt(self,k,outdig):
        return {"schema":"checked-message/v1","key":k,"output":outdig,
                "algebra":self.A.ident,"checker":CHECKER,"normalizer":NORMALIZER,
                "message_canonicalizer":MESSAGE_CANONICALIZER,"key_id":self.key_id}
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
        work=list(factors); stats={"hits":0,"misses":0,"rejected":0,"adds":0,"muls":0,"invalidated":0,"scope_axes_elided":0}
        keys={}
        for v in order:
            bucket=[f for f in work if v in f.scope]
            normalized=self.normalize(bucket)
            if not self.check_normalization(bucket,normalized):raise AssertionError("normalizer checker failed")
            k,ds=self.key_for(v,normalized)
            msg,bad=self.lookup(k)
            if msg is None:
                stats["misses"]+=1; stats["rejected"]+=int(bad)
                msg,a,m=self.produce(v,normalized); stats["adds"]+=a; stats["muls"]+=m
                msg,elided=self.minimize_message_scope(msg); stats["scope_axes_elided"]+=elided
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
    # Split/merge same-scope factors and insertion/removal of unit factors
    # preserve the normalized bucket key and every downstream message.
    c2=(2,2); split_a=factor((0,1),[2,1,3,0],c2,A); split_b=factor((0,1),[1,3,2,4],c2,A)
    merged=factor((0,1),[2,3,6,0],c2,A)
    unit=factor((0,),[1,1],c2,A)
    pre=Cache(c2,A); rsplit,ssplit=pre.evaluate([split_a,split_b],[0,1])
    rmerged,smerged=pre.evaluate([merged,unit],[0,1])
    assert rsplit==rmerged and smerged["hits"]==2
    wrong=factor((0,1),[2,4,6,0],c2,A)
    assert not pre.check_normalization([split_a,split_b],[wrong])
    MA=MaxCount(); ma=factor((0,1),[(2,2),None,(1,3),(0,2)],c2,MA)
    mb=factor((0,1),[(1,4),(0,1),None,(2,2)],c2,MA)
    merged_vals=[MA.mul(x,y) for x,y in zip(ma.vals,mb.vals)]
    mm=factor((0,1),merged_vals,c2,MA); unit_m=factor((0,),[MA.one,MA.one],c2,MA)
    mp=Cache(c2,MA); msplit,_=mp.evaluate([ma,mb],[0,1]); mmerged,mps=mp.evaluate([mm,unit_m],[0,1])
    assert msplit==mmerged and mps["hits"]==2
    # A computed message can be independent of a separator axis even though
    # neither raw factor is a unit. The exact minimizer drops that axis, and a
    # later graph variant can reuse downstream messages when this output stays
    # byte-identical.
    c3=(2,2,2)
    fxy=factor((0,1),[1,2,1,2],c3,A)
    gxz=factor((0,2),[1,2,1,0],c3,A)
    msgcache=Cache(c3,A)
    original_result,original_stats=msgcache.evaluate([fxy,gxz],[0,1,2])
    assert original_result==12
    v0_entries=[entry for entry in msgcache.entries.values() if entry["receipt"]["key"]]
    assert any(entry["out"]["scope"]==[1] for entry in v0_entries)
    gxz_variant=factor((0,2),[0,2,2,0],c3,A)
    variant_result,variant_stats=msgcache.evaluate([fxy,gxz_variant],[0,1,2])
    assert variant_result==original_result==direct(c3,[fxy,gxz],A)
    assert variant_stats["misses"]==1 and variant_stats["hits"]==2
    # The same scope minimizer works for MaxCount's complete value, not only
    # for natural counts: score, multiplicity and distinct support must all
    # be constant over the removed axis.
    fxy_m=factor((0,1),[(0,1),(2,1),(1,1),(2,1)],c3,MA)
    gxz_m=factor((0,2),[(3,1),(3,1),(0,1),(1,1)],c3,MA)
    mscope_cache=Cache(c3,MA)
    mscope_result,mscope_stats=mscope_cache.evaluate([fxy_m,gxz_m],[0,1,2])
    assert mscope_result==M(5,2,2)
    assert any(entry["out"]["scope"]==[1] for entry in mscope_cache.entries.values())
    gxz_m_variant=factor((0,2),[(3,1),(3,1),(-2,1),(-1,1)],c3,MA)
    mvariant,mvariant_stats=mscope_cache.evaluate([fxy_m,gxz_m_variant],[0,1,2])
    assert mvariant==mscope_result and mvariant_stats["misses"]==1 and mvariant_stats["hits"]==2
    scope=fs[:]; scope[0]=factor((0,3),[1,0,1,0],cards,A)
    rs,ss=e.evaluate(scope,order); assert A.enc(rs)==direct(cards,scope,A)
    tam=Cache(cards,A); tam.evaluate(fs,order); k=next(iter(tam.entries)); ent=tam.entries[k]
    ent["out"]["values"][0]+=9; ent["digest"]=sha(ent["out"]); ent["receipt"]=tam.receipt(k,ent["digest"])
    _,st=tam.evaluate(fs,order); assert st["rejected"]>=1
    foreign=Cache(cards,A,key=b"wrong-key"); foreign.entries=tam.entries.copy(); _,sf=foreign.evaluate(fs,order); assert sf["rejected"]>=1
    rnd_matches=rnd_hits=rnd_misses=rnd_invalid=0
    cold_add=cold_mul=inc_add=inc_mul=0
    for case in range(64):
        rng=random.Random(20261103+case)
        c=(2,3,2,2,3,2); scopes=[(0,1),(1,2),(0,2),(3,4),(4,5),(3,5),(1,4)]
        for alg in (Count(),MaxCount()):
            base=[]
            for scope0 in scopes:
                n=1
                for v in scope0:n*=c[v]
                if isinstance(alg,Count): vals=[rng.randrange(2) for _ in range(n)]
                else: vals=[None if rng.random()<.12 else (rng.randrange(-2,5),rng.randrange(1,4)) for _ in range(n)]
                base.append(factor(scope0,vals,c,alg))
            cache=Cache(c,alg); cache.evaluate(base,order)
            target=base[:]; idx=(case*7+3)%len(base); pos=(case*13+1)%len(target[idx].vals)
            old=target[idx].vals[pos]
            new=(1-old) if isinstance(alg,Count) else (None if old is not None else (case%5-2,case%3+1))
            vv=list(target[idx].vals); vv[pos]=new; target[idx]=factor(target[idx].scope,vv,c,alg)
            ir,is0=cache.evaluate(target,order); cold,cs=Cache(c,alg).evaluate(target,order)
            assert alg.enc(ir)==alg.enc(cold)==direct(c,target,alg)
            rnd_matches+=1; rnd_hits+=is0["hits"]; rnd_misses+=is0["misses"]; rnd_invalid+=is0["invalidated"]
            cold_add+=cs["adds"]; cold_mul+=cs["muls"]; inc_add+=is0["adds"]; inc_mul+=is0["muls"]
    print(json.dumps({"suite":"ETH-019 compact standalone reproducer","randomized_exact_matches":rnd_matches,
      "cache_hits":rnd_hits,"cache_misses":rnd_misses,"invalidated_steps":rnd_invalid,
      "cold_additions":cold_add,"cold_multiplications":cold_mul,"incremental_additions":inc_add,
      "incremental_multiplications":inc_mul,"single_edit_disconnected_hits":si["hits"],
      "single_edit_disconnected_misses":si["misses"],"tamper_rejected":st["rejected"],
      "wrong_key_rejected":sf["rejected"],"factor_permutation_hits":sp["hits"],
      "same_scope_split_merge_hits":smerged["hits"],"unit_factor_elision_hits":smerged["hits"],
      "normalization_mutant_rejected":True,"maxcount_same_scope_split_merge_hits":mps["hits"],
      "message_scope_axes_elided_natural_count":original_stats["scope_axes_elided"],
      "same_result_upstream_change_natural_count_hits":variant_stats["hits"],
      "same_result_upstream_change_natural_count_misses":variant_stats["misses"],
      "message_scope_axes_elided_maxcount":mscope_stats["scope_axes_elided"],
      "same_result_upstream_change_maxcount_hits":mvariant_stats["hits"],
      "same_result_upstream_change_maxcount_misses":mvariant_stats["misses"],
      "scope_edit_exact":True,
      "claim_boundary":"Python finite reference; no Rust/Lean/public benchmark or wall-time speedup."},indent=2,sort_keys=True))

if __name__=="__main__":main()
