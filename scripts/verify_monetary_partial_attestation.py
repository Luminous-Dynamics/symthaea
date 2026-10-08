#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_PAIRS={"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}
EXPECTED_AD={"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}
EXPECTED_SH={"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}
EXPECTED_SEED={11,23,47,89,131}
EXPECTED_ALLO={"fifo","criticality_priority","minimum_liquidity_demand"}
EXPECTED_FOCAL={0,1,2}
EXPECTED_REPORT={"truthful","criticality_inflation","liquidity_demand_underreport","dual_misreport"}
EXPECTED_COV={"none","focal_only","nonfocal_only","full"}
EXPECTED_FALL={"claim_fallback","neutral_fallback","reject_unattested"}
EXPECTED_FIX={f"ATTEST-X{i:02d}" for i in range(1,13)}
EXPECTED_CRN="world:{shock}:{seed}:obligation:{index}"
EXPECTED_TOPOLOGY="6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983"
EXPECTED_RESOURCE="8ff36e45d517568a7b153af2dab070b0adf4c7cfe97778d7575c84dd7b2e33b7"
EXPECTED_ALLOCATION={
    "fifo":"7bfe21a9815c819972baf46f8dc77199a6f602d31fcabc8e0708b14436777fb8",
    "criticality_priority":"89fb3406736d4d36c7a29c8094e3be198b3135090e7c8f78a13bfec0d2f2d234",
    "minimum_liquidity_demand":"fe1ae1b40106ac7c5343f5aa254ce3e0bd454df49fbdca99b4ae40519e47c900",
}
EXPECTED_FALL_DIGESTS={
    "claim_fallback":"f13f6e997eae84285eee9e21c14141c103269ee02e38d87a1e104d581348b7e9",
    "neutral_fallback":"9a9b69caf22c749f91b2ba5baeb5f8603f57c2b10b40127c8d4187f783706efc",
    "reject_unattested":"d3ff59528f88a233303ad1378f074bfdefb06a51fcd7439806542b52e7f605cc",
}
EXPECTED_COVERAGE_DIGESTS={
    "none":"53e818c198526ef4b2e02b1dc77742892e472898d17d0356fea6af2a9b25d172",
    "focal_only":"b0bc36f6d0f89c754ee8568c63dd36b6ba0623d8d6260ee37041f744d18ea7ec",
    "nonfocal_only":"a3f49fb6bf2c201132ffe8ef0a608dac2dca95a668d9bbdc77259603c7befbb8",
    "full":"4706b9e35a5ade465ce4889e33a83de84d6aed0566dccaedb2b9ad19c8cf3ec5",
}

def fail(message): raise ValueError(message)

def main():
    if len(sys.argv)!=3:
        print("usage: verify_monetary_partial_attestation.py MANIFEST.json NEGATIVE.json",file=sys.stderr)
        return 2
    try:
        m=json.loads(Path(sys.argv[1]).read_text())
        n=json.loads(Path(sys.argv[2]).read_text())
        f=m["factors"]; fixed=m["fixed_dimensions"]
        if len(f) != 9: fail("factor axis count")
        expected=(
            len(f["monetary_pair"])*len(f["adapter"])*len(f["shock"])*len(f["seed"])*
            len(f["allocation_policy"])*len(f["focal_obligation"])*len(f["reporting_policy"])*
            len(f["coverage_profile"])*len(f["fallback_policy"])
        )
        if m["schema_version"]!="monetary-partial-attestation-v1" or m["factorial_size"]!=155520 or m["batch_size"]!=3: fail("manifest")
        if expected!=155520: fail("factorial cardinality")
        if set(f["monetary_pair"])!=EXPECTED_PAIRS or set(f["adapter"])!=EXPECTED_AD or set(f["shock"])!=EXPECTED_SH: fail("base factors")
        if set(f["seed"])!=EXPECTED_SEED or set(f["allocation_policy"])!=EXPECTED_ALLO or set(f["focal_obligation"])!=EXPECTED_FOCAL: fail("control factors")
        if set(f["reporting_policy"])!=EXPECTED_REPORT or set(f["coverage_profile"])!=EXPECTED_COV or set(f["fallback_policy"])!=EXPECTED_FALL: fail("attestation factors")
        if set(m["coverage_profiles"])!=EXPECTED_COV or {k:v["digest"] for k,v in m["coverage_profiles"].items()}!=EXPECTED_COVERAGE_DIGESTS: fail("coverage identities")
        if set(m["fallback_policies"])!=EXPECTED_FALL or {k:v["digest"] for k,v in m["fallback_policies"].items()}!=EXPECTED_FALL_DIGESTS: fail("fallback identities")
        if fixed["topology"]!="full_mesh" or fixed["topology_digest"]!=EXPECTED_TOPOLOGY: fail("topology identity")
        if fixed["resource_regime"]!="bounded_shared_settlement_liquidity" or fixed["resource_capacity"]!=15 or fixed["resource_digest"]!=EXPECTED_RESOURCE: fail("resource identity")
        if fixed["allocation_policy_digests"]!=EXPECTED_ALLOCATION: fail("allocation identity")
        if fixed["exogenous_random_namespace"]!=EXPECTED_CRN: fail("CRN namespace")
        if fixed["reservation_boundary"]!="resource reservation always consumes immutable true liquidity demand; declared demand is never a resource quantity": fail("reservation boundary")
        if n["schema_version"]!="monetary-partial-attestation-negative-v1" or {x["id"] for x in n["cases"]}!=EXPECTED_FIX: fail("negative fixtures")
        print("independent partial-attestation check: 155520 cells / 466560 obligations; exact coverage/fallback identities; immutable true/report boundary; exact topology/resource/allocation provenance; treatment-independent CRN; 12 negative fixtures")
        return 0
    except (OSError,KeyError,TypeError,json.JSONDecodeError,ValueError) as exc:
        print(f"verification failed: {exc}",file=sys.stderr)
        return 1

if __name__=="__main__": raise SystemExit(main())
