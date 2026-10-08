#!/usr/bin/env python3
"""Balanced synthetic factorial for monetary interoperability."""
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from itertools import combinations, product
import csv, json, sys
from pathlib import Path
from monetary_operational_workflow_reference import OperationalPolicy, OperationalWorkflow, SettlementOutcome, WorkflowRequest

ROOT=Path(__file__).resolve().parent
MANIFEST=ROOT/"../docs/research/monetary/monetary-causal-factorial-v1.json"
ADAPTERS=("redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent")
SHOCKS=("normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote")
SEEDS=(11,23,47,89,131)
METRICS=("completion_time","technical_settlement_time","operational_waiting_time","manual_intervention_rate","fallback_activation_rate","reconciliation_backlog","unresolved_rate","completed","operator_capacity_consumed")

@dataclass(frozen=True)
class Run:
    pair:str; adapter:str; shock:str; seed:int; scenario_id:str; common_random_number_id:str; world_jitter:int
    source_profile_id:str; source_profile_digest:str; target_profile_id:str; target_profile_digest:str; edge_digest:str
    topology_digest:str; routing_policy_digest:str; composition_generation:str; composition_digest:str; policy_digest:str; settlement_receipt_digest:str|None
    final_state:str; completion_time:int|None; technical_settlement_time:int|None; operational_waiting_time:int|None
    manual_intervention_rate:float; fallback_activation_rate:float; reconciliation_backlog:int; unresolved_rate:float; completed:float
    operator_capacity_consumed:int; trace_digest:str

M=json.loads(MANIFEST.read_text())
assert M["schema_version"]=="monetary-causal-factorial-v1" and M["factorial_size"]==360
PAIRS=M["pairs"]; PROFILE_DIGESTS=M["profile_digests"]

def jitter(shock,seed):
    return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2],"big")%3

def policy_for(shock,world_jitter):
    kw=asdict(OperationalPolicy(asynchronous_delivery_delay=world_jitter))
    if shock=="network_partition": kw.update(external_system_available=False,asynchronous_delivery_delay=world_jitter+1)
    elif shock=="issuer_default": kw.update(target_issuer_available=False)
    elif shock=="stale_quote": kw.update(quote_current=False)
    return OperationalPolicy(**kw)

def outcome_for(pair,adapter,shock):
    if adapter=="absent": return SettlementOutcome("rejected",None,False,"no_interoperability_adapter",f"synthetic:{pair}:absent:{shock}")
    cfg=PAIRS[pair]; dur=cfg["technical_settlement_duration"][adapter]; redeem=cfg["requires_redemption_by_adapter"][adapter]; tag=f"synthetic:{pair}:{adapter}:{shock}"
    if shock=="normal": return SettlementOutcome("queued" if adapter=="multilateral_net_settlement" else "settled",dur,redeem,"",tag)
    if shock=="liquidity_shock":
        if adapter=="multilateral_net_settlement": return SettlementOutcome("queued",dur+2,False,"",tag)
        if adapter=="redeem_reissue": return SettlementOutcome("settled",dur+2,True,"",tag)
        return SettlementOutcome("rejected",None,False,"adapter_liquidity_limit",tag)
    if shock=="network_partition":
        if adapter=="multilateral_net_settlement": return SettlementOutcome("queued",dur+2,False,"",tag)
        if adapter=="redeem_reissue": return SettlementOutcome("stranded",None,True,"destination_unavailable_after_source_commit",tag)
        return SettlementOutcome("rejected",None,False,"atomic_target_unavailable",tag)
    if shock=="issuer_default":
        if adapter=="redeem_reissue": return SettlementOutcome("stranded",None,True,"destination_unavailable_after_source_commit",tag)
        return SettlementOutcome("rejected",None,False,"atomic_target_unavailable" if adapter=="escrowed_atomic_swap" else "source_insufficient",tag)
    if shock=="bridge_failure": return SettlementOutcome("rejected",None,False,"bridge_unavailable",tag)
    if shock=="stale_quote": return SettlementOutcome("settled",dur,redeem,"",tag)
    raise AssertionError(shock)

def run_one(pair,adapter,shock,seed):
    cfg=PAIRS[pair]; scenario=f"factorial-{shock}-{seed}"; wj=jitter(shock,seed); source,target=cfg["source"],cfg["target"]
    assert PROFILE_DIGESTS[source] and PROFILE_DIGESTS[target] and len(cfg["edge_digest"])==64
    result=OperationalWorkflow(policy_for(shock,wj)).run(
        WorkflowRequest(scenario,0,M["fixed_dimensions"]["amount"],source,target,adapter),
        outcome_for(pair,adapter,shock), seed=seed)
    return Run(pair,adapter,shock,seed,scenario,scenario,wj,source,PROFILE_DIGESTS[source],target,PROFILE_DIGESTS[target],
               cfg["edge_digest"],M["fixed_dimensions"]["topology_digest"],M["fixed_dimensions"]["routing_policy_digest"],
               M["fixed_dimensions"]["composition_generation"],M["fixed_dimensions"]["composition_digest"],result.policy_digest,
               result.settlement_receipt_digest,result.final_state.value,result.end_to_end_completion_time,result.technical_settlement_time,
               result.operational_waiting_time,result.manual_intervention_rate,result.fallback_activation_rate,result.reconciliation_backlog,
               float(result.unresolved),float(result.final_state.value=="externally_finalized"),result.operator_capacity_consumed,result.trace_digest)

def run_all():
    runs=[run_one(*x) for x in product(PAIRS,ADAPTERS,SHOCKS,SEEDS)]
    assert len(runs)==360 and len({(r.pair,r.adapter,r.shock,r.seed) for r in runs})==360
    for shock,seed in product(SHOCKS,SEEDS):
        cells=[r for r in runs if r.shock==shock and r.seed==seed]
        assert len({r.common_random_number_id for r in cells})==1 and len({r.world_jitter for r in cells})==1
    return runs

def avg(rs,m):
    xs=[float(getattr(r,m)) for r in rs if getattr(r,m) is not None]
    return sum(xs)/len(xs) if xs else None

def group(rs,**f): return [r for r in rs if all(getattr(r,k)==v for k,v in f.items())]

def contrasts(rs):
    out={"adapter_vs_absent":[],"shock_vs_normal":[],"adapter_shock_difference_in_difference":[],"pair_vs_pair":[]}
    for pair in PAIRS:
        for shock in SHOCKS:
            for a in ADAPTERS[:-1]:
                for m in METRICS:
                    x,y=avg(group(rs,pair=pair,adapter=a,shock=shock),m),avg(group(rs,pair=pair,adapter="absent",shock=shock),m)
                    if x is not None and y is not None: out["adapter_vs_absent"].append({"pair":pair,"adapter":a,"shock":shock,"metric":m,"delta":x-y})
            for a in ADAPTERS:
                for shock2 in SHOCKS[1:]:
                    for m in METRICS:
                        x,y=avg(group(rs,pair=pair,adapter=a,shock=shock2),m),avg(group(rs,pair=pair,adapter=a,shock="normal"),m)
                        if x is not None and y is not None: out["shock_vs_normal"].append({"pair":pair,"adapter":a,"shock":shock2,"metric":m,"delta":x-y})
    for pair in PAIRS:
        for a in ADAPTERS[:-1]:
            for shock in SHOCKS[1:]:
                for m in METRICS:
                    x1,y1=avg(group(rs,pair=pair,adapter=a,shock=shock),m),avg(group(rs,pair=pair,adapter="absent",shock=shock),m)
                    x0,y0=avg(group(rs,pair=pair,adapter=a,shock="normal"),m),avg(group(rs,pair=pair,adapter="absent",shock="normal"),m)
                    if None not in (x1,y1,x0,y0): out["adapter_shock_difference_in_difference"].append({"pair":pair,"adapter":a,"shock":shock,"metric":m,"delta_in_delta":(x1-y1)-(x0-y0)})
    for a in ADAPTERS:
        for shock in SHOCKS:
            for left,right in combinations(PAIRS,2):
                for m in METRICS:
                    x,y=avg(group(rs,pair=left,adapter=a,shock=shock),m),avg(group(rs,pair=right,adapter=a,shock=shock),m)
                    if x is not None and y is not None: out["pair_vs_pair"].append({"left_pair":left,"right_pair":right,"adapter":a,"shock":shock,"metric":m,"delta":x-y})
    return out

def main(out_dir=Path(".")):
    rs=run_all(); out_dir.mkdir(parents=True,exist_ok=True); rows=[asdict(r) for r in rs]
    with (out_dir/"monetary-causal-factorial-v1.results.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    summary={"run_count":len(rs),"factor_levels":{"pairs":list(PAIRS),"adapters":list(ADAPTERS),"shocks":list(SHOCKS),"seeds":list(SEEDS)},
             "completion_rate_by_adapter":{a:avg(group(rs,adapter=a),"completed") for a in ADAPTERS},
             "completion_rate_by_shock":{s:avg(group(rs,shock=s),"completed") for s in SHOCKS},
             "metric_means":{m:avg(rs,m) for m in METRICS},
             "trace_set_digest":sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),
             "common_random_number_cells":30,
             "claim_ceiling":"Synthetic balanced factorial only; no empirical identification, policy recommendation, macro forecasting or live-system safety claim."}
    (out_dir/"monetary-causal-factorial-v1.summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (out_dir/"monetary-causal-factorial-v1.contrasts.json").write_text(json.dumps(contrasts(rs),indent=2,sort_keys=True)+"\n")
    print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("."))
