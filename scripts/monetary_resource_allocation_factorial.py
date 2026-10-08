#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from itertools import product
from pathlib import Path
import json, math, statistics, sys

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/"../docs/research/monetary/monetary-resource-allocation-v1.json").read_text())
PAIR=tuple(M["factors"]["monetary_pair"]); TOP=tuple(M["factors"]["topology"])
AD=tuple(M["factors"]["adapter"]); SH=tuple(M["factors"]["shock"]); SEED=tuple(M["factors"]["seed"])
RR=tuple(M["factors"]["resource_regime"]); POL=tuple(M["factors"]["allocation_policy"])
AMOUNTS=[15,5,10]; CRITICALITY=[2,1,3]
DEMAND_MULT={"redeem_reissue":1.0,"escrowed_atomic_swap":1.0,"multilateral_net_settlement":0.7,"absent":0.0}
TECH={"redeem_reissue":3,"escrowed_atomic_swap":2,"multilateral_net_settlement":2,"absent":None}
CAP={"unrestricted":30,"bounded_shared_settlement_liquidity":15,"bounded_operator_capacity":30,"both_bounded":15}
OPCAP={"unrestricted":3,"bounded_shared_settlement_liquidity":3,"bounded_operator_capacity":1,"both_bounded":1}
TOPO_OVERHEAD={"full_mesh":0,"routing_only_hub":1,"redundant_two_hub":1}

def world_jitter(shock,seed):
    return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2],"big")%3

def manual_hit(shock,seed,index):
    return int.from_bytes(sha256(f"world:{shock}:{seed}:obligation:{index}".encode()).digest()[:8],"big")%1_000_000 < 300_000

def policy_order(policy,demands):
    idx=[0,1,2]
    if policy=="fifo": return idx
    if policy=="criticality_priority": return sorted(idx,key=lambda i:(-CRITICALITY[i],i))
    if policy=="minimum_liquidity_demand": return sorted(idx,key=lambda i:(demands[i],i))
    raise ValueError(policy)

@dataclass(frozen=True)
class Run:
    topology:str; pair:str; adapter:str; shock:str; seed:int; resource_regime:str; allocation_policy:str
    allocation_policy_digest:str; obligation_order:tuple; world_jitter:int; exogenous_random_namespace:str
    completion_rate:float; mean_completion_time:float|None; weighted_completion_time:float|None
    max_liquidity_queue:int; max_operator_queue:int; liquidity_utilization:float
    resource_blocked:int; critical_obligation_failed:int; reconciliation_backlog:int; trace_digest:str

def run_one(topology,pair,adapter,shock,seed,resource_regime,allocation_policy):
    capacity=CAP[resource_regime]; opcap=OPCAP[resource_regime]; wj=world_jitter(shock,seed)
    shock_increment=5 if shock=="liquidity_shock" else 0
    demands=[0 if adapter=="absent" else max(0,math.ceil(AMOUNTS[i]*DEMAND_MULT[adapter]+shock_increment)) for i in range(3)]
    order=policy_order(allocation_policy,demands)
    available=capacity; current=2+2+1+TOPO_OVERHEAD[topology]+wj
    active=[]; pending=order.copy(); records=[]; max_liq_queue=max_op_queue=0; blocked=backlog=0
    while pending or active:
        if active:
            next_finish=min(x[0] for x in active); current=max(current,next_finish)
            done=[x for x in active if x[0]<=current]; active=[x for x in active if x[0]>current]
            available += sum(x[1] for x in done)
        started=False
        while pending:
            i=pending[0]; d=demands[i]
            if d==0:
                pending.pop(0); records.append((i,"rejected",None,None,0)); continue
            if d>capacity:
                pending.pop(0); blocked+=1; records.append((i,"unresolved",None,None,d)); continue
            if d<=available:
                pending.pop(0); available-=d; manual=manual_hit(shock,seed,i)
                manual_delay=2 if manual and opcap>0 else 0
                finish=current+TECH[adapter]+(2 if shock=="liquidity_shock" and adapter=="redeem_reissue" else 0)+TOPO_OVERHEAD[topology]+manual_delay+2
                active.append((finish,d,i)); records.append((i,"running",current,finish,d))
                max_liq_queue=max(max_liq_queue,len(pending)); started=True
            else:
                break
        if pending and not active:
            for i in pending:
                blocked+=1; records.append((i,"unresolved",None,None,demands[i]))
            pending=[]
        elif pending and not started:
            max_liq_queue=max(max_liq_queue,len(pending))
    finish_map={i:fin for i,status,start,fin,d in records if status=="running"}
    final=[]
    for i,status,start,fin,d in records:
        state="externally_finalized" if status=="running" else status
        if state=="externally_finalized" and shock=="bridge_failure": state="unresolved"
        elif state=="externally_finalized" and shock=="issuer_default" and adapter=="redeem_reissue": state="unresolved"
        elif state=="externally_finalized" and shock=="stale_quote": state="unresolved"
        elif state=="externally_finalized" and shock=="network_partition" and topology=="routing_only_hub":
            state="unresolved" if adapter=="redeem_reissue" else "rejected"
        comp=finish_map.get(i) if state=="externally_finalized" else None
        final.append({"index":i,"state":state,"completion":comp,"amount":AMOUNTS[i],"criticality":CRITICALITY[i],"demand":d,"manual":manual_hit(shock,seed,i)})
    done=[x for x in final if x["state"]=="externally_finalized"]
    mean_ct=statistics.mean([x["completion"] for x in done]) if done else None
    weighted=(sum(x["completion"]*x["criticality"] for x in done)/sum(x["criticality"] for x in done)) if done else None
    opqs=[]
    for i in range(3):
        if manual_hit(shock,seed,i):
            pos=sum(1 for j in order if order.index(j)<=order.index(i) and manual_hit(shock,seed,j))
            opqs.append(max(0,pos-opcap))
    max_op_queue=max(opqs,default=0)
    util=sum(x["demand"] for x in done)/capacity if capacity else 0
    critical_failed=sum(1 for x in final if x["criticality"]>=3 and x["state"]!="externally_finalized")
    trace=sha256(json.dumps(final,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return Run(topology,pair,adapter,shock,seed,resource_regime,allocation_policy,
               M["allocation_policies"][allocation_policy]["digest"],tuple(order),wj,
               "world:{shock}:{seed}:obligation:{index}",len(done)/3,mean_ct,weighted,max_liq_queue,max_op_queue,util,
               blocked,critical_failed,backlog,trace)

def main(out):
    runs=[run_one(*x) for x in product(TOP,PAIR,AD,SH,SEED,RR,POL)]
    assert len(runs)==12960 and len({(r.topology,r.pair,r.adapter,r.shock,r.seed,r.resource_regime,r.allocation_policy) for r in runs})==12960
    for shock,seed,i in product(SH,SEED,range(3)):
        assert len({manual_hit(shock,seed,i) for _ in POL})==1
    payload=[asdict(r) for r in runs]
    digest=sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    def meanv(xs): return sum(xs)/len(xs) if xs else None
    summary={
      "schema_version":"monetary-resource-allocation-v1-execution",
      "run_count":len(runs),"obligation_count":len(runs)*3,"common_random_number_cells":90,
      "exogenous_random_namespace":"world:{shock}:{seed}:obligation:{index}","trace_set_digest":digest,
      "completion_rate_by_policy":{p:meanv([r.completion_rate for r in runs if r.allocation_policy==p]) for p in POL},
      "mean_completion_time_by_policy":{p:meanv([r.mean_completion_time for r in runs if r.allocation_policy==p and r.mean_completion_time is not None]) for p in POL},
      "weighted_critical_completion_time_by_policy":{p:meanv([r.weighted_completion_time for r in runs if r.allocation_policy==p and r.weighted_completion_time is not None]) for p in POL},
      "critical_obligation_failures_by_policy":{p:sum(r.critical_obligation_failed for r in runs if r.allocation_policy==p) for p in POL},
      "resource_blocked_obligations_by_regime":{rr:sum(r.resource_blocked for r in runs if r.resource_regime==rr) for rr in RR},
      "claim_ceiling":"Local deterministic allocation-policy reference execution only; treatment-independent CRN namespace; no empirical causal or live-system claim."
    }
    out.mkdir(parents=True,exist_ok=True); (out/"monetary-resource-allocation-v1.execution.generated.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__": main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("."))
