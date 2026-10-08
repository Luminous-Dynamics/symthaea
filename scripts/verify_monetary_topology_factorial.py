#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
TOPO={"full_mesh":"6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983","routing_only_hub":"90d10f6025f7200a3c4ff2863e755e29b7d40bbc437729410ef4ff06ba80d0e3","redundant_two_hub":"400c93eabda6ceb9d224fa4944f458ab410233c1c13404297c64ab53ad077a47"}
PAIRS={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}; ADAPTERS={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}; SHOCKS={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}; SEEDS={11,23,47,89,131}; EXPECTED={f"TOP-X{i:02d}" for i in range(1,11)}
def fail(m): raise ValueError(m)
def main():
 if len(sys.argv)!=3: print("usage: verify_monetary_topology_factorial.py MANIFEST.json NEGATIVE.json",file=sys.stderr); return 2
 try:
  m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text())
  f=m["factors"]
  if m["schema_version"]!="monetary-topology-factorial-v1" or m["factorial_size"]!=1080: fail("schema")
  if set(f["topology"])!=set(TOPO) or set(f["monetary_pair"])!=PAIRS or set(f["adapter"])!=ADAPTERS or set(f["shock"])!=SHOCKS or set(f["seed"])!=SEEDS: fail("factor set")
  if set(m["topologies"])!=set(TOPO): fail("topology set")
  for k,v in TOPO.items():
   if m["topologies"][k]["digest"]!=v or m["topologies"][k]["financial_powers"]!=[]: fail("topology identity/powers")
  fd=m["fixed_dimensions"]
  if fd["no_financial_hub_inference"] is not True or "topology change requires new topology" not in fd["composition_identity_rule"]: fail("identity guard")
  if n["schema_version"]!="monetary-topology-factorial-negative-v1" or {c["id"] for c in n["cases"]}!=EXPECTED: fail("negative fixtures")
  print("independent topology-factorial check: 1080-cell design; 3 topology levels; 10 negative fixtures"); return 0
 except (OSError,KeyError,json.JSONDecodeError,ValueError) as e: print(f"verification failed: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
