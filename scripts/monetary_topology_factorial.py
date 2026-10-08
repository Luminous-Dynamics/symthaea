#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from itertools import product
import csv,json,sys
from pathlib import Path
from monetary_operational_workflow_reference import OperationalPolicy,OperationalWorkflow,SettlementOutcome,WorkflowRequest

ROOT=Path(__file__).resolve().parent; MANIFEST=ROOT/"../docs/research/monetary-causal-factorial-v1.json"
TOPOLOGIES=("full_mesh","routing_only_hub","redundant_two_hub"); ADAPTERS=("redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent")
SHOCKS=("normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"); SEEDS=(11,23,47,89,131)
M=json.loads(MANIFEST.read_text()); PAIRS=M["pairs"]; DIG=M["profile_digests"]
TOPO_META={
"full_mesh":{"digest":M["fixed_dimensions"]["topology_digest"],"generation":"t1","route_overhead":0,"centrality":0.20,"partition_recovery":0,"financial_powers":[]},
"routing_only_hub":{"digest":"90d10f6025f7200a3c4ff2863e755e29b7d40bbc437729410ef4ff06ba80d0e3","generation":"t2","route_overhead":1,"centrality":1.0,"partition_recovery":0,"financial_powers":[]},
"redundant_two_hub":{"digest":"400c93eabda6ceb9d224fa4944f458ab410233c1c13404297c64ab53ad077a47","generation":"t3","route_overhead":1,"centrality":0.50,"partition_recovery":1,"financial_powers":[]}}
@dataclass(frozen=True)
class Run:
 topology:str; pair:str; adapter:str; shock:str; seed:int; scenario_id:str; common_random_number_id:str; exogenous_random_namespace:str; world_jitter:int
 source_profile_id:str; source_profile_digest:str; target_profile_id:str; target_profile_digest:str; edge_digest:str
 topology_digest:str; topology_generation:str; routing_policy_digest:str; final_state:str
 completion_time:int|None; technical_settlement_time:int|None; operational_waiting_time:int|None
 reconciliation_backlog:int; unresolved_rate:float; completed:float; route_hops:int; topology_centrality:float; operator_capacity_consumed:int; trace_digest:str

def jitter(shock,seed): return int.from_bytes(sha256(f"world:{shock}:{seed}".encode()).digest()[:2],"big")%3
def policy_for(topology,shock,wj):
 d=wj+TOPO_META[topology]["route_overhead"]+(1 if topology=="redundant_two_hub" and shock=="network_partition" else 0)
 kw=asdict(OperationalPolicy(asynchronous_delivery_delay=d,manual_breakpoint_probability_ppm=300000))
 if topology=="routing_only_hub" and shock=="network_partition": kw["external_system_available"]=False
 if shock=="issuer_default": kw["target_issuer_available"]=False
 elif shock=="stale_quote": kw["quote_current"]=False
 return OperationalPolicy(**kw)
def outcome_for(pair,adapter,shock,topology):
 if adapter=="absent": return SettlementOutcome("rejected",None,False,"no_interoperability_adapter",f"synthetic:{pair}:absent:{shock}:{topology}")
 cfg=PAIRS[pair]; dur=cfg["technical_settlement_duration"][adapter]; redeem=cfg["requires_redemption_by_adapter"][adapter]; tag=f"synthetic:{pair}:{adapter}:{shock}:{topology}"
 if shock=="normal": return SettlementOutcome("queued" if adapter=="multilateral_net_settlement" else "settled",dur,redeem,"",tag)
 if shock=="liquidity_shock":
  if adapter=="multilateral_net_settlement": return SettlementOutcome("queued",dur+2,False,"",tag)
  if adapter=="redeem_reissue": return SettlementOutcome("settled",dur+2,True,"",tag)
  return SettlementOutcome("rejected",None,False,"adapter_liquidity_limit",tag)
 if shock=="network_partition":
  if topology=="routing_only_hub":
   if adapter=="multilateral_net_settlement": return SettlementOutcome("queued",dur+2,False,"",tag)
   if adapter=="redeem_reissue": return SettlementOutcome("stranded",None,True,"destination_unavailable_after_source_commit",tag)
   return SettlementOutcome("rejected",None,False,"atomic_target_unavailable",tag)
  return SettlementOutcome("queued" if adapter=="multilateral_net_settlement" else "settled",dur+(1 if topology=="redundant_two_hub" else 0),redeem,"",tag)
 if shock=="issuer_default":
  if adapter=="redeem_reissue": return SettlementOutcome("stranded",None,True,"destination_unavailable_after_source_commit",tag)
  return SettlementOutcome("rejected",None,False,"atomic_target_unavailable" if adapter=="escrowed_atomic_swap" else "source_insufficient",tag)
 if shock=="bridge_failure": return SettlementOutcome("rejected",None,False,"bridge_unavailable",tag)
 if shock=="stale_quote": return SettlementOutcome("settled",dur,redeem,"",tag)
 raise AssertionError(shock)
def run_one(topology,pair,adapter,shock,seed):
 cfg=PAIRS[pair]; meta=TOPO_META[topology]; scenario=f"topology-factorial-{topology}-{shock}-{seed}"; wj=jitter(shock,seed); rng=f"world:{shock}:{seed}:obligation:0"
 r=OperationalWorkflow(policy_for(topology,shock,wj)).run(WorkflowRequest(scenario,0,10,cfg["source"],cfg["target"],adapter),outcome_for(pair,adapter,shock,topology),seed=seed,exogenous_random_namespace=rng)
 return Run(topology,pair,adapter,shock,seed,scenario,f"world-{shock}-{seed}",rng,wj,cfg["source"],DIG[cfg["source"]],cfg["target"],DIG[cfg["target"]],cfg["edge_digest"],meta["digest"],meta["generation"],M["fixed_dimensions"]["routing_policy_digest"],r.final_state.value,r.end_to_end_completion_time,r.technical_settlement_time,r.operational_waiting_time,r.reconciliation_backlog,float(r.unresolved),float(r.final_state.value=="externally_finalized"),1 if topology=="full_mesh" else 2,meta["centrality"],r.operator_capacity_consumed,r.trace_digest)
def main(out=Path(".")):
 rs=[run_one(*x) for x in product(TOPOLOGIES,PAIRS,ADAPTERS,SHOCKS,SEEDS)]
 assert len(rs)==1080 and len({(r.topology,r.pair,r.adapter,r.shock,r.seed) for r in rs})==1080
 for shock,seed in product(SHOCKS,SEEDS):
  c=[r for r in rs if r.shock==shock and r.seed==seed]; assert len({r.common_random_number_id for r in c})==1 and len({r.world_jitter for r in c})==1 and len({r.exogenous_random_namespace for r in c})==1
 rows=[asdict(r) for r in rs]; out.mkdir(parents=True,exist_ok=True)
 with (out/"monetary-topology-factorial-v1.results.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
 avg=lambda xs,k:(sum(float(getattr(x,k)) for x in xs if getattr(x,k) is not None)/len([x for x in xs if getattr(x,k) is not None])) if any(getattr(x,k) is not None for x in xs) else None
 summary={"run_count":1080,"common_random_number_cells":30,"factor_levels":{"topologies":list(TOPOLOGIES),"pairs":list(PAIRS),"adapters":list(ADAPTERS),"shocks":list(SHOCKS),"seeds":list(SEEDS)},"completion_rate_by_topology":{t:avg([r for r in rs if r.topology==t],"completed") for t in TOPOLOGIES},"completion_rate_by_topology_and_shock":{f"{t}:{s}":avg([r for r in rs if r.topology==t and r.shock==s],"completed") for t in TOPOLOGIES for s in SHOCKS},"operational_waiting_by_topology":{t:avg([r for r in rs if r.topology==t],"operational_waiting_time") for t in TOPOLOGIES},"trace_set_digest":sha256(json.dumps(rows,sort_keys=True,separators=(",",":")).encode()).hexdigest(),"claim_ceiling":"Synthetic topology factorial only; treatment-invariant CRN namespace; no empirical or systemic-safety claim."}
 (out/"monetary-topology-factorial-v1.summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n"); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=="__main__": main(Path(sys.argv[1]) if len(sys.argv)>1 else Path("."))
