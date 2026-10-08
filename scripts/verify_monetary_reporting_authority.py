#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

EXPECTED_PAIRS = {"bank_mutual","bank_stablecoin","tokenized_deposit_stablecoin"}
EXPECTED_AD = {"redeem_reissue","escrowed_atomic_swap","multilateral_net_settlement","absent"}
EXPECTED_SH = {"normal","liquidity_shock","network_partition","issuer_default","bridge_failure","stale_quote"}
EXPECTED_SEEDS = {11,23,47,89,131}
EXPECTED_ALLO = {"fifo","criticality_priority","minimum_liquidity_demand"}
EXPECTED_FOCAL = {0,1,2}
EXPECTED_REPORT = {"truthful","criticality_inflation","liquidity_demand_underreport","dual_misreport"}
EXPECTED_AUTH = {"claim_driven","verified_truth","mismatch_rejected"}
EXPECTED_FIX = {f"AUTH-X{i:02d}" for i in range(1,13)}
EXPECTED_CRN = "world:{shock}:{seed}:obligation:{index}"
EXPECTED_TOPOLOGY_DIGEST = "6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983"
EXPECTED_RESOURCE_DIGEST = "8ff36e45d517568a7b153af2dab070b0adf4c7cfe97778d7575c84dd7b2e33b7"
EXPECTED_ALLOCATION_DIGESTS = {
    "fifo":"7bfe21a9815c819972baf46f8dc77199a6f602d31fcabc8e0708b14436777fb8",
    "criticality_priority":"89fb3406736d4d36c7a29c8094e3be198b3135090e7c8f78a13bfec0d2f2d234",
    "minimum_liquidity_demand":"fe1ae1b40106ac7c5343f5aa254ce3e0bd454df49fbdca99b4ae40519e47c900",
}
EXPECTED_REPORT_DIGESTS = {
    "truthful":"429a35ed7ab43e231454b9bb1700c67fdc14da1f4d25116709eb074bbd5651c3",
    "criticality_inflation":"c266caba2ea2f1e0f64541e828ba2319073499d04faa4446e5ebcb25a4d0073f",
    "liquidity_demand_underreport":"ef2711b330dfdcc0b17577b95041641f5b2d14adcdb99e727ee8b9034e454170",
    "dual_misreport":"c2df69e738326df89cf6cb1133f939779f36c8dd3d11f9b6fc2b7c1d71f8954c",
}
EXPECTED_AUTH_DIGESTS = {
    "claim_driven":"d7d276c3cad620b3eda18d9b536bbbbb2808cf57dfcd07fd18196183ea87d5f0",
    "verified_truth":"aa319856bc3fd947c444030aae9ee62da1511d92fb17a8060d167fa2e1379264",
    "mismatch_rejected":"6240c36d9f4ad5874b0207301ce296df91ec6a521a1e0f90be66456fb4b019d6",
}

def fail(message):
    raise ValueError(message)

def main():
    if len(sys.argv) != 3:
        print("usage: verify_monetary_reporting_authority.py MANIFEST.json NEGATIVE.json", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(Path(sys.argv[1]).read_text())
        negative = json.loads(Path(sys.argv[2]).read_text())
        f = manifest["factors"]
        fixed = manifest["fixed_dimensions"]

        expected = len(f["monetary_pair"]) * len(f["adapter"]) * len(f["shock"]) * len(f["seed"]) * len(f["allocation_policy"]) * len(f["focal_obligation"]) * len(f["reporting_policy"]) * len(f["authority_mode"])
        if manifest["schema_version"] != "monetary-reporting-authority-v1" or manifest["factorial_size"] != 38880 or manifest["batch_size"] != 3:
            fail("manifest")
        if expected != 38880:
            fail("factorial cardinality")
        if set(f["monetary_pair"]) != EXPECTED_PAIRS or set(f["adapter"]) != EXPECTED_AD or set(f["shock"]) != EXPECTED_SH:
            fail("base factor set")
        if set(f["seed"]) != EXPECTED_SEEDS or set(f["allocation_policy"]) != EXPECTED_ALLO or set(f["focal_obligation"]) != EXPECTED_FOCAL:
            fail("control factor set")
        if set(f["reporting_policy"]) != EXPECTED_REPORT or set(f["authority_mode"]) != EXPECTED_AUTH:
            fail("reporting/authority factor set")
        if fixed["topology"] != "full_mesh" or fixed["topology_digest"] != EXPECTED_TOPOLOGY_DIGEST:
            fail("topology identity")
        if fixed["resource_regime"] != "bounded_shared_settlement_liquidity" or fixed["resource_capacity"] != 15 or fixed["resource_digest"] != EXPECTED_RESOURCE_DIGEST:
            fail("resource identity")
        if fixed["allocation_policy_digests"] != EXPECTED_ALLOCATION_DIGESTS:
            fail("allocation identity")
        if {k:v["digest"] for k,v in manifest["reporting_policies"].items()} != EXPECTED_REPORT_DIGESTS:
            fail("reporting identity")
        if {k:v["digest"] for k,v in manifest["authority_modes"].items()} != EXPECTED_AUTH_DIGESTS:
            fail("authority identity")
        if fixed["exogenous_random_namespace"] != EXPECTED_CRN:
            fail("CRN namespace")
        if fixed["reservation_boundary"] != "resource reservation always consumes immutable true liquidity demand; declared demand is never a resource quantity":
            fail("reservation boundary")
        if negative["schema_version"] != "monetary-reporting-authority-negative-v1":
            fail("negative schema")
        if {x["id"] for x in negative["cases"]} != EXPECTED_FIX:
            fail("negative fixtures")
        print("independent reporting-authority check: 38880 cells / 116640 obligations; exact authority/report digests; immutable truth boundary; exact topology/resource/allocation identities; treatment-independent CRN; 12 negative fixtures")
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
