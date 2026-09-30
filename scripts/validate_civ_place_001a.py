#!/usr/bin/env python3
"""Independent stdlib-only oracle for CIV-PLACE-001A.

This script intentionally does not import Symthaea/Mycelix production code.
It validates the frozen source fixture and independently derives its hostile
case dispositions.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "docs" / "engineering" / "fixtures" / "civ-place-001a.json"
DIGEST_FILE = ROOT / "docs" / "engineering" / "fixtures" / "civ-place-001a.sha256"

EXPECTED_DIGEST = "380d4ed23eeaa360cf0b80d445771edd0fbf46c35421ab6ab184c1560a371667"

def canonical(obj: object) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")

def fail(message: str) -> None:
    raise SystemExit(f"CIV-PLACE-001A oracle FAIL: {message}")

def place_identity(place: dict[str, object]) -> str:
    identity = {
        "profile": "CIV-PLACE-001A",
        "place_id": place["place_id"],
        "level": place["level"],
        "configuration": place["configuration"],
        "frame": place["frame"],
    }
    return hashlib.sha256(canonical(identity)).hexdigest()

def derive(case: dict[str, object]) -> str:
    kind = case["kind"]
    value = case["input"]

    if kind == "identity":
        return "DistinctIdentity"
    if kind == "spatial_proximity":
        return "NoDependency" if value["connected"] is False else "DependencyDeclared"
    if kind == "common_mode":
        # Absence of a discovered common upstream is not positive proof of independence.
        # The graph may be incomplete, partitioned, or missing a hidden dependency.
        if value["shared_upstream"] is None:
            return "IndependenceUnknown"
        if value["shared_upstream"] is False:
            return "IndependentWitnessed" if value.get("independence_witness") else "IndependenceUnknown"
        # A pair of redundant components cannot be independent when they
        # share one upstream source; a larger declared group retains the
        # common-mode relation as the claim-bearing disposition.
        return "NotIndependent" if len(value["members"]) == 2 else "CommonModeRetained"
    if kind == "service":
        return "ServiceUnresolved" if value["quality"] == "Unknown" else "ServiceResolved"
    if kind == "currentness":
        return "CurrentnessBlocked" if value["currentness"] != "CURRENT" else "Current"
    if kind == "source_class":
        return "PromotionRejected" if (
            value["source_class"] == "InferredState"
            and value["presented_as"] == "DirectPhysicalObservation"
        ) else "SourceClassCompatible"
    if kind == "fallback":
        if value["primary"] == "Unavailable" and value["fallback"] == "Available":
            return (
                "FullServiceByFallback"
                if value["fallback_capacity"] == "Sufficient"
                else "DegradedService"
            )
        return "FallbackUnresolved"
    if kind == "governance":
        return (
            "GovernanceOnlyChanged"
            if value["engineering_identity"]
            and value["stewardship_ref"] != value["new_stewardship_ref"]
            else "EngineeringChanged"
        )
    if kind == "operator":
        return (
            "OperatorOnlyChanged"
            if value["configuration"] and value["operator"] != value["new_operator"]
            else "ConfigurationChanged"
        )
    if kind == "change_impact":
        return (
            "SelectiveReopen"
            if value["changed"] and value["affected"] and value["unaffected"]
            else "ImpactUndeclared"
        )
    if kind == "authority":
        if "dispatch_permit" in value:
            return "NoDispatchAuthority" if value["dispatch_permit"] is False else "PermitPresent"
        return "DispatchRejected" if value["exact_cp_authority"] is False else "DispatchReviewRequired"
    if kind == "interop":
        return "ProjectionIncomplete" if value["exact_semantics"] is False else "ProjectionComplete"
    if kind == "publication":
        return "PublicationInvalid" if (
            value["restricted_dependency"] == "present"
            and value["public_representation"] == "omitted"
        ) else "PublicationValid"
    if kind == "transport":
        return "NonAuthoritativeObservation" if value["authenticated"] is False else "AuthenticatedObservation"
    if kind == "service_currentness":
        return "StaleCurrentness" if value["material_outage"] else "CurrentnessUnchanged"
    if kind == "dependence":
        return "DependenceRetained" if value["feed_a_source"] == value["feed_b_source"] else "IndependentSources"
    if kind == "claim_ceiling":
        return "NoRealAuthority" if value["synthetic_pass"] and not value["real_authority"] else "AuthorityReviewRequired"

    fail(f"unknown hostile-case kind: {kind!r}")

def main() -> int:
    raw = FIXTURE.read_bytes()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"invalid fixture JSON: {exc}")

    if raw != canonical(data):
        fail("fixture is not byte-canonical JSON")

    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != EXPECTED_DIGEST:
        fail(f"fixture digest changed: {actual_digest} != {EXPECTED_DIGEST}")

    recorded = DIGEST_FILE.read_text(encoding="utf-8").strip().split()[0]
    if recorded != EXPECTED_DIGEST:
        fail("digest sidecar does not match frozen digest")

    required = {
        "schema_version", "profile", "place_levels", "degradation_states",
        "dependency_classes", "places", "hostile_cases",
    }
    if set(data) != required:
        fail(f"top-level schema drift: {sorted(data)}")

    if data["profile"] != "CIV-PLACE-001A":
        fail("wrong profile")
    if data["schema_version"] != "civ-place-001a-v1":
        fail("wrong schema version")

    place_ids = set()
    global_member_ids = set()
    for place in data["places"]:
        if place["place_id"] in place_ids:
            fail(f"duplicate place id: {place['place_id']}")
        place_ids.add(place["place_id"])
        global_member_ids.update(place["members"])

        if len(place["members"]) != len(set(place["members"])):
            fail(f"duplicate member id in {place['place_id']}")

        interface_ids = [item["id"] for item in place["interfaces"]]
        if len(interface_ids) != len(set(interface_ids)):
            fail(f"duplicate interface id in {place['place_id']}")

        for interface in place["interfaces"]:
            if interface["source"] not in global_member_ids and interface["source"] not in place_ids:
                # A source may be declared in a later place; defer full reference check.
                pass
            if interface["target"] == interface["source"]:
                fail(f"self-interface: {interface['id']}")

        service_ids = [item["id"] for item in place["services"]]
        if len(service_ids) != len(set(service_ids)):
            fail(f"duplicate service id in {place['place_id']}")

        for group in place["common_mode_groups"]:
            if len(group["members"]) != len(set(group["members"])):
                fail(f"duplicate common-mode member in {group['id']}")
            if not set(group["members"]).issubset(set(place["members"])):
                fail(f"common-mode member outside place: {group['id']}")

    known_ids = place_ids | global_member_ids
    for place in data["places"]:
        for interface in place["interfaces"]:
            if interface["source"] not in known_ids or interface["target"] not in known_ids:
                fail(f"unresolved interface reference: {interface['id']}")

    if len(data["hostile_cases"]) != 21:
        fail("hostile corpus count changed")
    case_ids = [case["id"] for case in data["hostile_cases"]]
    if case_ids != [f"H{i:02d}" for i in range(1, 22)]:
        fail("hostile case ordering/identity changed")

    # Independent identity mutation check.
    home = next(p for p in data["places"] if p["place_id"] == "place:p0:home-001")
    original_identity = place_identity(home)
    mutated = dict(home)
    mutated["configuration"] = "cfg:home-001:g2"
    if original_identity == place_identity(mutated):
        fail("configuration mutation did not change place identity")

    for case in data["hostile_cases"]:
        actual = derive(case)
        if actual != case["expected"]:
            fail(f"{case['id']}: derived {actual!r}, expected {case['expected']!r}")

    summary = {
        "profile": data["profile"],
        "fixture_sha256": actual_digest,
        "places": len(data["places"]),
        "hostile_cases": len(data["hostile_cases"]),
        "identity_mutation_checked": True,
        "authority_claim": "none",
        "result": "PASS",
    }
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0

if __name__ == "__main__":
    sys.exit(main())
