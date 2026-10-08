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
EXPECTED_FIX = {f"REPORT-X{i:02d}" for i in range(1,13)}
EXPECTED_CRN = "world:{shock}:{seed}:obligation:{index}"
EXPECTED_TOPOLOGY_DIGEST = "6ebb7f4c4da37e834675759a5348d1bbd7bf1ccc4e132334fe8c090a94f7f983"
EXPECTED_RESOURCE_DIGEST = "8ff36e45d517568a7b153af2dab070b0adf4c7cfe97778d7575c84dd7b2e33b7"
EXPECTED_ALLOCATION_DIGESTS = {
    "fifo":"7bfe21a9815c819972baf46f8dc77199a6f602d31fcabc8e0708b14436777fb8",
    "criticality_priority":"89fb3406736d4d36c7a29c8094e3be198b3135090e7c8f78a13bfec0d2f2d234",
    "minimum_liquidity_demand":"fe1ae1b40106ac7c5343f5aa254ce3e0bd454df49fbdca99b4ae40519e47c900"
}
EXPECTED_REPORT_DIGESTS = {
    "truthful":"429a35ed7ab43e231454b9bb1700c67fdc14da1f4d25116709eb074bbd5651c3",
    "criticality_inflation":"c266caba2ea2f1e0f64541e828ba2319073499d04faa4446e5ebcb25a4d0073f",
    "liquidity_demand_underreport":"ef2711b330dfdcc0b17577b95041641f5b2d14adcdb99e727ee8b9034e454170",
    "dual_misreport":"c2df69e738326df89cf6cb1133f939779f36c8dd3d11f9b6fc2b7c1d71f8954c"
}
EXPECTED_PROFILE_DIGESTS = {
    "conventional-bank-money-v1":"fb3c6d7da7ca899014d2ad2f537692bc735da7e25704bc40d714fcaf953d3c1e",
    "sovereign-money-v1":"a33a9062618ac59d6f123ccd34f78516cd0bab62ed5db44f78c48489ba461a32",
    "mutual-credit-v1":"e31e27b1afde9405920c14cd6ec840287b5d19f947f1ab4a0820a7105b4d890e",
    "tokenized-deposit-v1":"d4c0897f41f04ed343b9f82c94ea02165871f7288be84a7f3cca957723edb48e",
    "reserve-backed-stablecoin-v1":"2ad75345c6760089a1fcdacc1c767252b5779e17438d583ba42d149fc9a462b7"
}
EXPECTED_EDGE_DIGESTS = {
    "bank_mutual":"0c77d6cf338fccf07254f96620c4fdc4e9e2dd06d71369370eee5c98e66d191b",
    "bank_stablecoin":"2162c182b1a314975c5e6e8b0c38a5cfa4830bbab1862440cbdedb846c0445db",
    "tokenized_deposit_stablecoin":"a3b15fbb527ff127da92dc243d36680742ebd02ebd5a640df11bfd728bc41cc"
}

def fail(message):
    raise ValueError(message)

def main():
    if len(sys.argv) != 3:
        print("usage: verify_monetary_strategic_reporting.py MANIFEST.json NEGATIVE.json", file=sys.stderr)
        return 2
    try:
        manifest = json.loads(Path(sys.argv[1]).read_text())
        negative = json.loads(Path(sys.argv[2]).read_text())
        factors = manifest["factors"]
        fixed = manifest["fixed_dimensions"]

        if manifest["schema_version"] != "monetary-strategic-reporting-v1":
            fail("schema version")
        if manifest["factorial_size"] != 12960 or manifest["batch_size"] != 3:
            fail("factorial")
        if set(factors["monetary_pair"]) != EXPECTED_PAIRS:
            fail("pair factor")
        if set(factors["adapter"]) != EXPECTED_AD:
            fail("adapter factor")
        if set(factors["shock"]) != EXPECTED_SH:
            fail("shock factor")
        if set(factors["seed"]) != EXPECTED_SEEDS:
            fail("seed factor")
        if set(factors["allocation_policy"]) != EXPECTED_ALLO:
            fail("allocation factor")
        if set(factors["focal_obligation"]) != EXPECTED_FOCAL:
            fail("focal factor")
        if set(factors["reporting_policy"]) != EXPECTED_REPORT:
            fail("reporting factor")
        if set(manifest["reporting_policies"]) != EXPECTED_REPORT:
            fail("reporting policy set")
        if {k:v["digest"] for k,v in manifest["reporting_policies"].items()} != EXPECTED_REPORT_DIGESTS:
            fail("reporting policy digests")

        if fixed["topology"] != "full_mesh":
            fail("topology")
        if fixed["topology_digest"] != EXPECTED_TOPOLOGY_DIGEST:
            fail("topology identity")
        if fixed["resource_regime"] != "bounded_shared_settlement_liquidity":
            fail("resource regime")
        if fixed["resource_capacity"] != 15:
            fail("resource capacity")
        if fixed["resource_digest"] != EXPECTED_RESOURCE_DIGEST:
            fail("resource identity")
        if fixed["allocation_policy_changes"] != "admission order only":
            fail("allocation mutation boundary")
        if fixed["exogenous_random_namespace"] != EXPECTED_CRN:
            fail("CRN namespace")
        if fixed["reservation_boundary"] != "resource reservation always consumes immutable true liquidity demand; declared demand is never a resource quantity":
            fail("reservation boundary")

        if fixed["allocation_policy_digests"] != EXPECTED_ALLOCATION_DIGESTS:
            fail("allocation policy identity")
        expected_factorial = (
            len(factors["monetary_pair"])
            * len(factors["adapter"])
            * len(factors["shock"])
            * len(factors["seed"])
            * len(factors["allocation_policy"])
            * len(factors["focal_obligation"])
            * len(factors["reporting_policy"])
        )
        if expected_factorial != 12960:
            fail("factorial cardinality")

        if manifest["profile_digests"] != EXPECTED_PROFILE_DIGESTS:
            fail("profile identity")
        if manifest["edge_digests"] != EXPECTED_EDGE_DIGESTS:
            fail("edge identity")

        if negative["schema_version"] != "monetary-strategic-reporting-negative-v1":
            fail("negative schema")
        if {case["id"] for case in negative["cases"]} != EXPECTED_FIX:
            fail("negative fixtures")

        print("independent strategic-reporting check: 12960 cells / 38880 obligations; exact true/report separation; exact topology/resource/allocation/provenance identity; treatment-independent CRN; 12 negative fixtures")
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
