#!/usr/bin/env python3
"""Independent semantic checker for monetary interoperability v1.

The checker is deliberately separate from production economic code. It verifies
that the composition matrix covers the five seed profiles in both directions,
that profile digests are pinned, and that adversarial fixtures target explicit
interoperability failure boundaries.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROFILES = {
    "conventional-bank-money-v1": "fb3c6d7da7ca899014d2ad2f537692bc735da7e25704bc40d714fcaf953d3c1e",
    "sovereign-money-v1": "a33a9062618ac59d6f123ccd34f78516cd0bab62ed5db44f78c48489ba461a32",
    "mutual-credit-v1": "e31e27b1afde9405920c14cd6ec840287b5d19f947f1ab4a0820a7105b4d890e",
    "tokenized-deposit-v1": "d4c0897f41f04ed343b9f82c94ea02165871f7288be84a7f3cca957723edb48e",
    "reserve-backed-stablecoin-v1": "2ad75345c6760089a1fcdacc1c767252b5779e17438d583ba42d149fc9a462b7",
}

MODES = {
    "native_common_settlement_asset",
    "redeem_reissue",
    "escrowed_atomic_swap",
    "correspondent_bridge",
    "multilateral_net_settlement",
    "message_only",
    "absent",
}

EXPECTED_FIXTURES = {
    "IO-X01", "IO-X02", "IO-X03", "IO-X04", "IO-X05", "IO-X06",
    "IO-X07", "IO-X08", "IO-X09", "IO-X10", "IO-X11", "IO-X12",
    "IO-X13", "IO-X14", "IO-X15", "IO-X16",
}

def fail(message: str) -> None:
    raise ValueError(message)

def check_matrix(matrix: dict) -> None:
    if matrix.get("schema_version") != "monetary-interoperability-matrix-v1":
        fail("matrix: schema version")
    if matrix.get("profile_count") != 5:
        fail("matrix: expected five profiles")
    if matrix.get("directed_edge_count") != 25:
        fail("matrix: expected 25 directed/self edges")
    edges = matrix.get("edges")
    if not isinstance(edges, list) or len(edges) != 25:
        fail("matrix: edge count")
    seen = set()
    cross = 0
    self_edges = 0
    for edge in edges:
        required = {
            "edge_id", "source_profile_id", "source_profile_digest",
            "target_profile_id", "target_profile_digest", "directionality",
            "test_status", "adapter_modes_to_test", "paired_scenario_policy",
        }
        if set(edge) != required:
            fail(f"{edge.get('edge_id','<unknown>')}: key set")
        source = edge["source_profile_id"]
        target = edge["target_profile_id"]
        if source not in PROFILES or target not in PROFILES:
            fail(f"{edge['edge_id']}: unknown profile")
        if edge["source_profile_digest"] != PROFILES[source]:
            fail(f"{edge['edge_id']}: source digest mismatch")
        if edge["target_profile_digest"] != PROFILES[target]:
            fail(f"{edge['edge_id']}: target digest mismatch")
        if len(set(edge["adapter_modes_to_test"])) != len(MODES):
            fail(f"{edge['edge_id']}: adapter mode duplication")
        if set(edge["adapter_modes_to_test"]) != MODES:
            fail(f"{edge['edge_id']}: adapter mode set")
        if source == target:
            self_edges += 1
            if edge["directionality"] != "bidirectional_explicitly_symmetric":
                fail(f"{edge['edge_id']}: self-edge directionality")
            if edge["test_status"] != "control":
                fail(f"{edge['edge_id']}: self-edge must be a control")
        else:
            cross += 1
            if edge["directionality"] != "directed":
                fail(f"{edge['edge_id']}: cross-architecture edge must be directed")
            if edge["test_status"] != "planned":
                fail(f"{edge['edge_id']}: cross-architecture status")
        pair = (source, target)
        if pair in seen:
            fail(f"duplicate edge: {source}->{target}")
        seen.add(pair)
    if cross != 20 or self_edges != 5:
        fail(f"matrix: expected 20 cross edges + 5 self controls, got {cross} + {self_edges}")
    for source in PROFILES:
        for target in PROFILES:
            if (source, target) not in seen:
                fail(f"matrix: missing edge {source}->{target}")

def check_fixtures(fixtures: dict) -> None:
    if fixtures.get("schema_version") != "monetary-interoperability-negative-v1":
        fail("fixtures: schema version")
    cases = fixtures.get("cases")
    if not isinstance(cases, list):
        fail("fixtures: cases")
    ids = {case.get("id") for case in cases}
    if ids != EXPECTED_FIXTURES:
        fail("fixtures: missing or unexpected fixture IDs")
    for case in cases:
        cid = case["id"]
        expected = case.get("expected")
        if not isinstance(expected, str) or not expected:
            fail(f"{cid}: missing expected disposition")
        if case.get("kind") == "implicit_conversion_rate_inheritance":
            if case.get("conversion_rule") != "inherit_source_profile":
                fail(f"{cid}: inheritance adversary malformed")
        if case.get("kind") == "hidden_backstop_inheritance":
            if case.get("backstop") != "inherit_target_public_backstop":
                fail(f"{cid}: backstop inheritance adversary malformed")
        if case.get("kind") == "double_redemption":
            event_ids = case.get("event_ids")
            if not isinstance(event_ids, list) or len(event_ids) != 2 or event_ids[0] != event_ids[1]:
                fail(f"{cid}: replay fixture malformed")
        if case.get("kind") == "triangular_round_trip_leakage":
            path = case.get("path")
            if not isinstance(path, list) or len(path) < 3:
                fail(f"{cid}: path fixture malformed")

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: verify_monetary_interoperability.py MATRIX.json NEGATIVE.json", file=sys.stderr)
        return 2
    try:
        matrix = json.loads(Path(sys.argv[1]).read_text())
        fixtures = json.loads(Path(sys.argv[2]).read_text())
        check_matrix(matrix)
        check_fixtures(fixtures)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    print("independent monetary interoperability check: 25 edges (20 cross-architecture + 5 self-controls); 16 negative fixtures")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
