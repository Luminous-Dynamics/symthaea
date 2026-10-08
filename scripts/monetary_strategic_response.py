#!/usr/bin/env python3
from __future__ import annotations
from hashlib import sha256
from itertools import product
import json, math, statistics, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/"../docs/research/monetary/monetary-strategic-response-v1.json").read_text())
PAIRS=tuple(M["factors"]["monetary_pair"]); AD=tuple(M["factors"]["adapter"]); SH=tuple(M["factors"]["shock"]); SEED=tuple(M["factors"]["seed"]); POL=tuple(M["factors"]["participant_policy"])
AMOUNTS=[15,5,10]; CRITICALITY=[2,1,3]; CAP=15
DEMAND_MULT={"redeem_reissue":1.0,"escrowed_atomic_swap":1.0,"multilateral_net_settlement":0.7,"absent":0.0}
TECH={"redeem_reissue":3,"escrowed_atomic_swap":2,"multilateral_net_settlement":2,"absent":None}

def jitter(shock,seed): return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2],"big")%3
def manual_hit(shock,seed,index): return int.from_bytes(sha256(f"world:{shock}:{seed}:obligation:{index}".encode()).digest()[:8],"big")%1_000_000 < 300000
def submission_times(policy,adapter,shock):
    shock_inc=5 if shock=="liquidity_shock" else 0
    demands=[0 if adapter=="absent" else math.ceil(AMOUNTS[i]*DEMAND_MULT[adapter]+shock_inc) for i in range(3)]
    if policy=="fixed_submission": return [0,0,0]
    if policy=="liquidity_aware_stagger": return [1 if d>=10 else 0 for d in demands]
    if policy=="critical_payment_acceleration": return [0 if CRITICALITY[i]>=3 else 1 for i in range(3)]
    raise ValueError(policy)

def run_one(pair,adapter,shock,seed,participant_policy):
    submit=submission_times(participant_policy,adapter,shock); wj=jitter(shock,seed)
    demands=[0 if adapter=="absent" else math.ceil(AMOUNTS[i]*DEMAND_MULT[adapter]+(5 if shock=="liquidity_shock" else 0)) for i in range(3)]
    records={i:{"submit":submit[i],"demand":demands[i],"criticality":CRITICALITY[i],"status":"pending","start":None,"finish":None} for i in range(3)}
    active=[]; available=CAP; t=0
    while any(r["status"]=="pending" for r in records.values()) or active:
        if active:
            nt=min(x["finish"] for x in active); t=max(t,nt)
            done=[x for x in active if x["finish"]<=t]; active=[x for x in active if x["finish"]>t]
            for x in done:
                available+=x["demand"]; records[x["i"]]["status"]="complete_candidate"
        pending=sorted([i for i,r in records.items() if r["status"]=="pending" and r["submit"]<=t],key=lambda i:(records[i]["submit"],i))
        progressed=False
        for i in pending:
            r=records[i]; d=r["demand"]
            if adapter=="absent": r["status"]="rejected"; progressed=True; continue
            if d>CAP: r["status"]="resource_rejected"; progressed=True; continue
            if d<=available:
                available-=d; r["start"]=t
                md=2 if manual_hit(shock,seed,i) else 0
                finish=t+TECH[adapter]+(2 if shock=="liquidity_shock" and adapter=="redeem_reissue" else 0)+md+2
                active.append({"i":i,"demand":d,"finish":finish}); r["finish"]=finish; r["status"]="running"; progressed=True
            else: break
        if pending and not progressed and not active:
            future=[r["submit"] for r in records.values() if r["status"]=="pending"]
            if future: t=min(future)
        elif not active:
            future=[r["submit"] for r in records.values() if r["status"]=="pending"]
            if future and min(future)>t: t=min(future)
        elif pending and not progressed:
            continue
    for i,r in records.items():
        if r["status"] in ("running","complete_candidate"):
            final="externally_finalized"
            if shock=="issuer_default" and adapter=="redeem_reissue": final="unresolved"
            elif shock=="bridge_failure": final="unresolved"
            elif shock=="stale_quote": final="unresolved"
            r["status"]=final
    done=[r for r in records.values() if r["status"]=="externally_finalized"]
    comp=[r["finish"] for r in done if r["finish"] is not None]
    crit=[r["finish"]-r["submit"] for r in done if r["finish"] is not None and r["criticality"]>=3]
    return {
      "pair":pair,"adapter":adapter,"shock":shock,"seed":seed,"participant_policy":participant_policy,
      "submission_times":tuple(submit),"world_jitter":wj,
      "completion_rate":len(done)/3,
      "system_completion_time":max(comp) if comp else None,
      "mean_submit_time":statistics.mean([r["submit"] for r in records.values()]),
      "critical_submission_delay":statistics.mean([r["submit"] for r in records.values() if r["criticality"]>=3]),
      "mean_obligation_latency":statistics.mean([r["finish"]-r["submit"] for r in done if r["finish"] is not None]) if done else None,
      "critical_obligation_latency":statistics.mean(crit) if crit else None,
      "resource_utilization":sum(r["demand"] for r in done)/CAP,
      "resource_blocked":sum(1 for r in records.values() if r["status"]=="resource_rejected"),
      "unresolved":sum(1 for r in records.values() if r["status"]=="unresolved"),
      "records":records
    }

def meanv(xs): return sum(xs)/len(xs) if xs else None

def main(out):
    runs=[run_one(*x) for x in product(PAIRS,AD,SH,SEED,POL)]
    assert len(runs)==1080 and len({(r["pair"],r["adapter"],r["shock"],r["seed"],r["participant_policy"]) for r in runs})==1080
    for shock,seed,index in product(SH,SEED,range(3)):
        assert len({manual_hit(shock,seed,index) for _ in POL})==1
    payload=json.dumps(runs,sort_keys=True,separators=(",",":")).encode()
    summary={
      "schema_version":"monetary-strategic-response-v1-execution",
      "run_count":1080,"obligation_count":3240,"common_random_number_cells":30,
      "exogenous_random_namespace":"world:{shock}:{seed}:obligation:{index}","trace_set_digest":sha256(payload).hexdigest(),
      "completion_rate_by_policy":{p:meanv([r["completion_rate"] for r in runs if r["participant_policy"]==p]) for p in POL},
      "system_completion_time_by_policy":{p:meanv([r["system_completion_time"] for r in runs if r["participant_policy"]==p and r["system_completion_time"] is not None]) for p in POL},
      "critical_submission_delay_by_policy":{p:meanv([r["critical_submission_delay"] for r in runs if r["participant_policy"]==p]) for p in POL},
      "critical_obligation_latency_by_policy":{p:meanv([r["critical_obligation_latency"] for r in runs if r["participant_policy"]==p and r["critical_obligation_latency"] is not None]) for p in POL},
      "mean_obligation_latency_by_policy":{p:meanv([r["mean_obligation_latency"] for r in runs if r["participant_policy"]==p and r["mean_obligation_latency"] is not None]) for p in POL},
      "claim_ceiling":"Local deterministic strategic-response reference execution only; no empirical behavioral or equilibrium claim."
    }
    out.mkdir(parents=True,exist_ok=True)
    (out/"monetary-strategic-response-v1.execution.generated.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))

if __name__=="__main__":
    main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("."))
