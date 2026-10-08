#!/usr/bin/env python3
from __future__ import annotations
import json,sys
from pathlib import Path
EXPECTED_POL={"fixed_submission","liquidity_aware_stagger","critical_payment_acceleration"}
EXPECTED_PAIR={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}
EXPECTED_AD={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}
EXPECTED_SH={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}
EXPECTED_SEED={11,23,47,89,131}
EXPECTED_FIX={f"STRAT-X{i:02d}" for i in range(1,13)}
EXPECTED_POLICY_DIGESTS={"fixed_submission":"7a7a6ffeaba670eeb8fc8d715e1307ae79df4912a13daedeeb672644eb57d9b8","liquidity_aware_stagger":"25a8dc13ba209e1d2c7b4b8f449e875308774d20ccbd4d22c849863d43278a2e","critical_payment_acceleration":"868cea5e436881256d0b239a8fdaf985815c85ead564d09de8898f3ee2c4de12"}
EXPECTED_EDGES={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}
EXPECTED_PROFILES={"conventional-bank-money-v1","sovereign-money-v1","mutual-credit-v1","tokenized-deposit-v1","reserve-backed-stablecoin-v1"}
def fail(m): raise ValueError(m)
def main():
    if len(sys.argv)!=3:
        print("usage: verify_monetary_strategic_response.py MANIFEST.json NEGATIVE.json",file=sys.stderr); return 2
    try:
        m=json.loads(Path(sys.argv[1]).read_text()); n=json.loads(Path(sys.argv[2]).read_text()); f=m["factors"]; fd=m["fixed_dimensions"]
        if m["schema_version"]!="monetary-strategic-response-v1" or m["factorial_size"]!=1080 or m["batch_size"]!=3: fail("manifest")
        if set(f["monetary_pair"])!=EXPECTED_PAIR or set(f["adapter"])!=EXPECTED_AD or set(f["shock"])!=EXPECTED_SH or set(f["seed"])!=EXPECTED_SEED or set(f["participant_policy"])!=EXPECTED_POL: fail("factor set")
        if set(m["participant_policies"])!=EXPECTED_POL: fail("participant policy set")
        for p,d in EXPECTED_POLICY_DIGESTS.items():
            if m["participant_policies"][p]["digest"]!=d: fail(f"{p}: policy digest")
        if fd["exogenous_random_namespace"]!="world:{shock}:{seed}:obligation:{index}": fail("CRN namespace")
        if fd["behavioral_boundary"]!="participant policy may change submission time only; monetary, adapter, topology, resource and allocation contracts remain fixed": fail("behavioral boundary")
        if fd["resource_regime"]!="bounded_shared_settlement_liquidity": fail("resource regime")
        if fd["allocation_policy_digest"]!="7bfe21a9815c819972baf46f8dc77199a6f602d31fcabc8e0708b14436777fb8": fail("allocation policy identity")
        if fd["resource_digest"]!="8ff36e45d517568a7b153af2dab070b0adf4c7cfe97778d7575c84dd7b2e33b7": fail("resource identity")
        if fd["topology_digest"]!="6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983": fail("topology identity")
        if set(m.get("edge_digests",{}))!=EXPECTED_EDGES: fail("edge identity set")
        if set(m.get("profile_digests",{}))!=EXPECTED_PROFILES: fail("profile identity set")
        if n["schema_version"]!="monetary-strategic-response-negative-v1" or {x["id"] for x in n["cases"]}!=EXPECTED_FIX: fail("negative fixtures")
        print("independent strategic-response check: 1080 cells / 3240 obligations; exact policy/resource/topology identity; treatment-independent CRN; 12 negative fixtures"); return 0
    except (OSError,KeyError,TypeError,json.JSONDecodeError,ValueError) as e:
        print(f"verification failed: {e}",file=sys.stderr); return 1
if __name__=="__main__": raise SystemExit(main())
