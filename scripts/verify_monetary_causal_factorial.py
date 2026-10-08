#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
EXPECTED_PAIRS={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}; EXPECTED_ADAPTERS={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}; EXPECTED_SHOCKS={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}; EXPECTED_SEEDS={11,23,47,89,131}; EXPECTED_FIXTURES={f"FAC-X{i:02d}" for i in range(1,14)}
def fail(msg): raise ValueError(msg)
def main():
    if len(sys.argv)!=3: print("usage: verify_monetary_causal_factorial.py MANIFEST.json NEGATIVE.json",file=sys.stderr); return 2
    try:
        m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text()); f=m.get("factors",{}); fd=m.get("fixed_dimensions",{})
        if m.get("schema_version")!="monetary-causal-factorial-v1": fail("schema version")
        if set(f.get("monetary_pair",[]))!=EXPECTED_PAIRS or set(f.get("adapter",[]))!=EXPECTED_ADAPTERS or set(f.get("shock",[]))!=EXPECTED_SHOCKS or set(f.get("seed",[]))!=EXPECTED_SEEDS: fail("factor set")
        expected=360
        if m.get("factorial_size")!=expected or m.get("seeds")!=sorted(EXPECTED_SEEDS): fail("factorial size/seeds")
        if not fd.get("fixed_topology") or not fd.get("fixed_routing") or not fd.get("same_profile_and_edge_digests") or not fd.get("no_cross_asset_scalar_conservation"): fail("fixed dimensions")
        if fd.get("exogenous_random_namespace")!="world:{shock}:{seed}:obligation:{index}": fail("CRN namespace")
        ep=m.get("execution_policy",{})
        if not ep.get("balanced_design") or not ep.get("common_random_numbers") or not ep.get("no_adapter_control"): fail("execution policy")
        if set(m.get("measurement_semantics",{}))!={"completion_time","technical_settlement_time","operational_waiting_time","reconciliation_backlog","trace_binding","seed_binding"}: fail("measurement semantics")
        if n.get("schema_version")!="monetary-causal-factorial-negative-v1" or {c.get("id") for c in n.get("cases",[])}!=EXPECTED_FIXTURES: fail("negative fixture set")
        print("independent causal-factorial check: 360 balanced runs; treatment-independent CRN namespace; 13 negative fixtures"); return 0
    except (OSError,json.JSONDecodeError,ValueError) as exc: print(f"verification failed: {exc}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
