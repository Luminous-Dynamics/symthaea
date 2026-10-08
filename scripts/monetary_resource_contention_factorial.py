#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass, asdict
from hashlib import sha256
from itertools import product
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
M=json.loads((ROOT/'../docs/research/monetary-resource-contention-v1.json').read_text())
C=json.loads((ROOT/'../docs/research/monetary-causal-factorial-v1.json').read_text())
TOP=('full_mesh','routing_only_hub','redundant_two_hub'); PAIR=tuple(M['factors']['monetary_pair']); AD=('redeem_reissue','escrowed_atomic_swap','multilateral_net_settlement','absent'); SH=tuple(M['factors']['shock']); SEED=tuple(M['factors']['seed']); RR=tuple(M['factors']['resource_regime'])
TOPO={'full_mesh':('6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983',0),'routing_only_hub':('90d10f6025f7200a3c4ff2863e755e29b7d40bbc437729410ef4ff06ba80d0e3',1),'redundant_two_hub':('400c93eabda6ceb9d224fa4944f458ab410233c1c13404297c64ab53ad077a47',1)}
CAP={'unrestricted':30,'bounded_shared_settlement_liquidity':15,'bounded_operator_capacity':30,'both_bounded':15}
OPCAP={'unrestricted':3,'bounded_shared_settlement_liquidity':3,'bounded_operator_capacity':1,'both_bounded':1}
DEMAND=M['fixed_dimensions']['liquidity_demand_units']; SHOCK_INC=M['fixed_dimensions']['liquidity_shock_increment']; MANUAL_PPM=M['fixed_dimensions']['manual_breakpoint_probability_ppm']

def h(s): return int.from_bytes(sha256(s.encode()).digest()[:8],'big')
def jitter(shock,seed): return int.from_bytes(sha256(f'world:{shock}:{seed}'.encode()).digest()[:2],'big')%3
def manual(shock,seed,i): return h(f'world:{shock}:{seed}:obligation:{i}')%1000000 < MANUAL_PPM

@dataclass(frozen=True)
class Cell:
 topology:str; pair:str; adapter:str; shock:str; seed:int; resource_regime:str; resource_digest_set:tuple
 completion_rate:float; completion_time:float|None; technical_time:float|None; operational_wait:float|None
 max_liquidity_queue:int; max_operator_queue:int; liquidity_utilization:float; resource_blocked:int; reconciliation_backlog:int; blast_radius:int; trace_digest:str

def run_cell(topo,pair,adapter,shock,seed,rr):
 cap=CAP[rr]; opcap=OPCAP[rr]; wj=jitter(shock,seed); meta=TOPO[topo]; tech=0 if adapter=='absent' else C['pairs'][pair]['technical_settlement_duration'][adapter]
 records=[]; mlq=moq=liqused=blocked=backlog=0
 for i in range(3):
  final='rejected'; comp=ow=None; rb=0; rec=0
  if adapter=='absent' or shock=='bridge_failure': pass
  elif shock=='issuer_default':
   if adapter=='redeem_reissue': final='unresolved'; rec=1
  elif shock=='stale_quote': final='unresolved'; rec=1
  elif shock=='network_partition' and topo=='routing_only_hub':
   final='unresolved' if adapter=='redeem_reissue' else 'rejected'; rec=int(final=='unresolved')
  else:
   demand=DEMAND[adapter]+(SHOCK_INC if shock=='liquidity_shock' else 0)
   if demand>cap:
    final='unresolved' if rr in ('bounded_shared_settlement_liquidity','both_bounded') else 'rejected'; rec=int(final=='unresolved'); rb=int(rr in ('bounded_shared_settlement_liquidity','both_bounded'))
   else:
    parallel=max(1,cap//demand); wave=i//parallel; mlq=max(mlq,max(0,i-parallel+1)); liqused=max(liqused,min(cap,demand*min(3,parallel)))
    lw=wave*(tech+1 if adapter!='absent' else 1); md=0
    if manual(shock,seed,i) and opcap>0: moq=max(moq,max(0,i-opcap)); md=(i//opcap+1)*2
    if manual(shock,seed,i) and opcap==0: final='unresolved'; rec=1
    if final=='rejected':
     final='externally_finalized'; po=1 if topo=='redundant_two_hub' and shock=='network_partition' else 0; operational=6+meta[1]+po+wj+lw+md+(2 if adapter=='redeem_reissue' else 0); comp=tech+operational+1; ow=operational+1
  records.append({'i':i,'final':final,'completion':comp,'technical':tech if comp is not None else None,'operational_wait':ow,'resource_blocked':rb,'reconciliation':rec}); blocked+=rb; backlog+=rec
 done=[x for x in records if x['final']=='externally_finalized']; rd=sha256(json.dumps(records,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 return Cell(topo,pair,adapter,shock,seed,rr,tuple(M['resource_regimes'][rr]),len(done)/3,sum(x['completion'] for x in done)/len(done) if done else None,sum(x['technical'] for x in done)/len(done) if done else None,sum(x['operational_wait'] for x in done)/len(done) if done else None,mlq,moq,liqused/cap,blocked,backlog,blocked,rd)

def main(out):
 runs=[run_cell(*x) for x in product(TOP,PAIR,AD,SH,SEED,RR)]
 assert len(runs)==4320 and len({(r.topology,r.pair,r.adapter,r.shock,r.seed,r.resource_regime) for r in runs})==4320
 payload=[asdict(r) for r in runs]; digest=sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 summary={'schema_version':'monetary-resource-contention-v1-execution','run_count':4320,'obligation_count':12960,'common_random_number_cells':30,'exogenous_random_namespace':M['fixed_dimensions']['exogenous_random_namespace'],'trace_set_digest':digest,'completion_rate_by_resource':{},'operational_waiting_by_resource':{},'liquidity_queue_mean_by_resource':{},'operator_queue_mean_by_resource':{},'resource_blocked_obligations_by_resource':{},'resource_exhaustion_rate_by_resource':{}}
 for rr in RR:
  g=[r for r in runs if r.resource_regime==rr]; waits=[r.operational_wait for r in g if r.operational_wait is not None]; b=sum(r.resource_blocked for r in g)
  summary['completion_rate_by_resource'][rr]=mean([r.completion_rate for r in g]); summary['operational_waiting_by_resource'][rr]=mean(waits); summary['liquidity_queue_mean_by_resource'][rr]=mean([r.max_liquidity_queue for r in g]); summary['operator_queue_mean_by_resource'][rr]=mean([r.max_operator_queue for r in g]); summary['resource_blocked_obligations_by_resource'][rr]=b; summary['resource_exhaustion_rate_by_resource'][rr]=b/(len(g)*3)
 out.mkdir(parents=True,exist_ok=True); (out/'monetary-resource-contention-v1.execution.generated.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n'); print(json.dumps(summary,indent=2,sort_keys=True))
if __name__=='__main__': main(Path(sys.argv[1]) if len(sys.argv)>1 else Path('.'))
