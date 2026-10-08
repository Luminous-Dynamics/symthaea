#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
EXPECTED_POL={"fifo","criticality_priority","minimum_liquidity_demand"}
EXPECTED_TOPO={"full_mesh","routing_only_hub","redundant_two_hub"}
EXPECTED_PAIRS={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}
EXPECTED_AD={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}
EXPECTED_SH={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}
EXPECTED_SEED={11,23,47,89,131}; EXPECTED_RR={"unrestricted","bounded_shared_settlement_liquidity","bounded_operator_capacity","both_bounded"}; EXPECTED_FIX={f"ALLOC-X{i:02d}" for i in range(1,13)}
def fail(m): raise ValueError(m)
def main():
    if len(sys.argv)!=3:
        print("usage: verify_monetary_resource_allocation.py MANIFEST.json NEGATIVE.json",file=sys.stderr); return 2
    try:
        m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text()); f=m["factors"]
        if m["schema_version"]!="monetary-resource-allocation-v1" or m["factorial_size"]!=12960 or m["batch_size"]!=3: fail("manifest")
        if set(f["topology"])!=EXPECTED_TOPO or set(f["monetary_pair"])!=EXPECTED_PAIRS or set(f["adapter"])!=EXPECTED_AD or set(f["shock"])!=EXPECTED_SH or set(f["seed"])!=EXPECTED_SEED or set(f["resource_regime"])!=EXPECTED_RR or set(f["allocation_policy"])!=EXPECTED_POL: fail("factor set")
        if set(m["allocation_policies"])!=EXPECTED_POL: fail("policy identities")
        for p in EXPECTED_POL:
            if len(m["allocation_policies"][p]["digest"])!=64: fail(f"{p}: digest")
        fixed=m["fixed_dimensions"]
        if fixed["exogenous_random_namespace"]!="world:{shock}:{seed}:obligation:{index}": fail("CRN namespace")
        if fixed["no_policy_mutation_of_capacity"] is not True: fail("capacity mutation")
        if fixed["policy_invariant"].startswith("allocation policy changes admission order only") is False: fail("policy invariant")
        if n["schema_version"]!="monetary-resource-allocation-negative-v1" or {x["id"] for x in n["cases"]}!=EXPECTED_FIX: fail("negative fixtures")
        print("independent allocation-policy check: 12960 cells / 38880 obligations; 3 allocation policies; treatment-independent CRN; 12 negative fixtures"); return 0
    except (OSError,KeyError,TypeError,json.JSONDecodeError,ValueError) as e:
        print(f"verification failed: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
