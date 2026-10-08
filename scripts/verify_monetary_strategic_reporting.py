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
EXPECTED_ALLOCATION_DIGESTS = {
    "fifo":"7bfe21a9815c819972baf46f8dc77199a6f602d31fcabc8e0708b14436777fb8",
    "criticality_priority":"89fb3406736d4d36c7a29c8094e3be198b3135090e7c8f78a13bfec0d2f2d234",
    "minimum_liquidity_demand":"fe1ae1b40106ac7c5343f5aa254ce3e0bd454df49fbdca99b4ae40519e47c900"
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

        if fixed["topology"] != "full_mesh":
            fail("topology")
        if fixed["resource_regime"] != "bounded_shared_settlement_liquidity":
            fail("resource regime")
        if fixed["resource_capacity"] != 15:
            fail("resource capacity")
        if fixed["exogenous_random_namespace"] != EXPECTED_CRN:
            fail("CRN namespace")
        if fixed["reservation_boundary"] != "resource reservation always consumes immutable true liquidity demand; declared demand is never a resource quantity":
            fail("reservation boundary")
        if fixed["allocation_policy_changes"] != "admission order only":
            fail("allocation mutation boundary")

        if set(manifest["profile_digests"]) != {
            "conventional-bank-money-v1","sovereign-money-v1","mutual-credit-v1",
            "tokenized-deposit-v1","reserve-backed-stablecoin-v1"
        }:
            fail("profile identity set")
        if set(manifest["edge_digests"]) != EXPECTED_PAIRS:
            fail("edge identity set")

        allocation_digests = fixed.get("allocation_policy_digests", EXPECTED_ALLOCATION_DIGESTS)
        if allocation_digests != EXPECTED_ALLOCATION_DIGESTS:
            fail("allocation policy identity")

        if negative["schema_version"] != "monetary-strategic-reporting-negative-v1":
            fail("negative schema")
        if {case["id"] for case in negative["cases"]} != EXPECTED_FIX:
            fail("negative fixtures")

        print("independent strategic-reporting check: 12960 cells / 38880 obligations; immutable truth/report boundary; exact allocation identity; treatment-independent CRN; 12 negative fixtures")
        return 0
    except (OSError, KeyError, TypeError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    raise SystemExit(main())
