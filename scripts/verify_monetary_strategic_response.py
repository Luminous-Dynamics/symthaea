#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
EXPECTED_POL={"fixed_submission","liquidity_aware_stagger","critical_payment_acceleration"}
EXPECTED_PAIR={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}; EXPECTED_AD={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}
EXPECTED_SH={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}; EXPECTED_SEED={11,23,47,89,131}; EXPECTED_FIX={f"STRAT-X{i:02d}" for i in range(1,13)}
def fail(m): raise ValueError(m)
def main():
    if len(sys.argv)!=3:
        print("usage: verify_monetary_strategic_response.py MANIFEST.json NEGATIVE.json",file=sys.stderr); return 2
    try:
        m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text()); f=m["factors"]; fd=m["fixed_dimensions"]
        if m["schema_version"]!="monetary-strategic-response-v1" or m["factorial_size"]!=1080 or m["batch_size"]!=3: fail("manifest")
        if set(f["monetary_pair"])!=EXPECTED_PAIR or set(f["adapter"])!=EXPECTED_AD or set(f["shock"])!=EXPECTED_SH or set(f["seed"])!=EXPECTED_SEED or set(f["participant_policy"])!=EXPECTED_POL: fail("factor set")
        if set(m["participant_policies"])!=EXPECTED_POL: fail("participant policy set")
        if fd["exogenous_random_namespace"]!="world:{shock}:{seed}:obligation:{index}": fail("CRN namespace")
        if fd["behavioral_boundary"].startswith("participant policy may change submission time only") is False: fail("behavioral boundary")
        if fd["resource_regime"]!="bounded_shared_settlement_liquidity": fail("resource regime")
        if n["schema_version"]!="monetary-strategic-response-negative-v1" or {x["id"] for x in n["cases"]}!=EXPECTED_FIX: fail("negative fixtures")
        print("independent strategic-response check: 1080 cells / 3240 obligations; 3 participant heuristics; treatment-independent CRN; 12 negative fixtures"); return 0
    except (OSError,KeyError,TypeError,json.JSONDecodeError,ValueError) as e:
        print(f"verification failed: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
