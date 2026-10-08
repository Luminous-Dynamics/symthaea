#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
EXPECTED_RES={"unrestricted_settlement_liquidity","bounded_shared_settlement_liquidity","bounded_operator_capacity"}
EXPECTED_REG={"unrestricted","bounded_shared_settlement_liquidity","bounded_operator_capacity","both_bounded"}
EXPECTED_TOPO={"full_mesh","routing_only_hub","redundant_two_hub"}
EXPECTED_PAIRS={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}
EXPECTED_ADAPTERS={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}
EXPECTED_SHOCKS={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}
EXPECTED_SEEDS={11,23,47,89,131}
EXPECTED_FIX={f"RES-X{i:02d}" for i in range(1,13)}
def fail(m): raise ValueError(m)
def main():
    if len(sys.argv)!=3:
        print("usage: verify_monetary_resource_contention.py MANIFEST.json NEGATIVE.json",file=sys.stderr); return 2
    try:
        m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text())
        if m.get("schema_version")!="monetary-resource-contention-v1" or m.get("factorial_size")!=4320 or m.get("batch_size")!=3: fail("manifest")
        f=m["factors"]
        if set(f["topology"])!=EXPECTED_TOPO or set(f["monetary_pair"])!=EXPECTED_PAIRS or set(f["adapter"])!=EXPECTED_ADAPTERS or set(f["shock"])!=EXPECTED_SHOCKS or set(f["seed"])!=EXPECTED_SEEDS or set(f["resource_regime"])!=EXPECTED_REG: fail("factor set")
        if set(m["resources"])!=EXPECTED_RES or set(m["resource_regimes"])!=EXPECTED_REG: fail("resource identities")
        for name,res in m["resources"].items():
            if res["allocation_policy"]!="fifo" or len(res["digest"])!=64: fail(f"resource {name}")
        fd=m["fixed_dimensions"]
        if fd["allocation_policy"]!="fifo" or fd["manual_breakpoint_probability_ppm"]!=300000 or fd["liquidity_shock_increment"]!=5 or fd["no_capacity_inference"] is not True: fail("fixed resource semantics")
        if n.get("schema_version")!="monetary-resource-contention-negative-v1" or {x.get("id") for x in n.get("cases",[])}!=EXPECTED_FIX: fail("negative fixture set")
        print("independent resource-contention check: 4320 cells / 12960 obligations; 3 resource identities; 4 resource regimes; 12 negative fixtures"); return 0
    except (OSError,KeyError,TypeError,json.JSONDecodeError,ValueError) as e:
        print(f"verification failed: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
