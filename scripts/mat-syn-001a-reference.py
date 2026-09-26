#!/usr/bin/env python3
"""Independent known-answer validator for MAT-SYN-001A corpus v2.

This script intentionally imports no Symthaea production code. It derives the
expected synthesis-feasibility vector directly from the synthetic declared
facts, then compares that result with the frozen known answers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

CORPUS = Path("docs/release/evidence/mat-syn-001a-synthetic-corpus-v2.json")
EXPECTED_SHA256 = "b659efbfa59acb16e272490ba475ae7b6a9ac7eb3337ee8b3ad410f1638b4b2a"
EXPECTED_GIT_BLOB_SHA1 = "79e9b6c0b261a00ec41f75939f1ddca89328ed0b"
EXPECTED_SCHEMA = "mat-syn-001a-synthetic-corpus-v2"
EXPECTED_AUTHORITY = "representation_only_no_execution_authority"
EXPECTED_ARCHITECTURE_HEAD = "c17ce53dcf3eac7c79d116df02c7e5d5d1b5a0e7"
EXPECTED_COUNT = 28
VECTOR_KEYS = (
    "formation_energy_support",
    "phase_competition_support",
    "dynamical_stability_support",
    "local_reproduction_supported",
    "route_represented",
    "current_capability_supported",
    "resource_availability_supported",
    "physical_authorization_supported",
    "physical_trial_observed",
    "target_phase_observed",
    "target_property_observed",
    "microstructure_bound",
    "independent_repeatability",
    "planning_preconditions_supported",
    "requalification_required",
    "manufacturability_supported",
)


def fail(message: str) -> None:
    raise SystemExit(f"MAT-SYN reference failure: {message}")


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def derive(facts: dict[str, object]) -> set[str]:
    result: set[str] = set()

    formation = facts.get("formation_energy") in {"supported", "negative"}
    phase_competition = facts.get("phase_competition") in {
        "supported",
        "supported_low_hull_distance",
    }
    dynamical = facts.get("dynamical_stability") == "supported"

    if formation:
        result.add("formation_energy_support")
    if phase_competition:
        result.add("phase_competition_support")
    if dynamical:
        result.add("dynamical_stability_support")

    computational_support = formation or phase_competition or dynamical
    if facts.get("calculation_origin") == "local_reproduction" and computational_support:
        result.add("local_reproduction_supported")

    route_represented = facts.get("route") == "represented"
    process_plan_represented = facts.get("process_plan") == "represented"
    capability_current = facts.get("capability") == "current_sufficient"
    availability_current = facts.get("resource_availability") == "current"
    safety_current = facts.get("safety_review") == "current"
    route_identity_current = not bool(facts.get("route_identity_changed", False))
    requalification_required = facts.get("currentness") == "stale"

    if route_represented:
        result.add("route_represented")
    if capability_current:
        result.add("current_capability_supported")
    if availability_current:
        result.add("resource_availability_supported")
    if requalification_required:
        result.add("requalification_required")

    authorization_supported = (
        facts.get("authorization") == "current_matching"
        and route_represented
        and process_plan_represented
        and capability_current
        and availability_current
        and safety_current
        and route_identity_current
        and not requalification_required
    )
    if authorization_supported:
        result.add("physical_authorization_supported")

    planning_supported = (
        route_represented
        and process_plan_represented
        and capability_current
        and availability_current
        and safety_current
        and route_identity_current
        and not requalification_required
    )
    if planning_supported:
        result.add("planning_preconditions_supported")

    lineage_valid = not (
        facts.get("precursor_lot_changed") is True
        and facts.get("sample_identity_reused") is True
    )
    physical_trial = facts.get("physical_trial") == "executed" and lineage_valid
    if physical_trial:
        result.add("physical_trial_observed")

    phase_observed = physical_trial and facts.get("phase_characterization") in {
        "target_phase_observed",
        "target_phase_observed_both",
    }
    if phase_observed:
        result.add("target_phase_observed")

    property_observed = physical_trial and facts.get("property_characterization") in {
        "target_property_observed",
        "target_property_observed_both",
    }
    if property_observed:
        result.add("target_property_observed")

    if facts.get("microstructure") == "exact_context_bound" and physical_trial:
        result.add("microstructure_bound")

    independent_repeatability = (
        physical_trial
        and isinstance(facts.get("root_batches"), int)
        and int(facts["root_batches"]) >= 2
        and facts.get("independence") == "independent_roots_supported"
        and phase_observed
        and property_observed
    )
    if independent_repeatability:
        result.add("independent_repeatability")

    # Deliberate non-derivation: no synthetic input in this corpus can establish
    # manufacturability. Coupon success, scores, plans, and repeatability are all
    # insufficient by construction.
    return result


def main() -> None:
    raw = CORPUS.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_SHA256:
        fail(f"corpus SHA-256 drift: expected {EXPECTED_SHA256}, observed {digest}")

    blob_identity = git_blob_sha1(raw)
    if blob_identity != EXPECTED_GIT_BLOB_SHA1:
        fail(
            "corpus Git-blob drift: "
            f"expected {EXPECTED_GIT_BLOB_SHA1}, observed {blob_identity}"
        )

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError as exc:
        fail(f"invalid JSON: {exc}")

    if doc.get("schema") != EXPECTED_SCHEMA:
        fail("schema mismatch")
    if doc.get("authority") != EXPECTED_AUTHORITY:
        fail("authority mismatch")
    if doc.get("architecture_head") != EXPECTED_ARCHITECTURE_HEAD:
        fail("architecture-head mismatch")

    keys = doc.get("vector_keys")
    if keys != list(VECTOR_KEYS):
        fail("vector key set/order drift")
    if len(set(keys)) != len(VECTOR_KEYS):
        fail("duplicate vector key")

    cases = doc.get("cases")
    if not isinstance(cases, list) or len(cases) != EXPECTED_COUNT:
        fail(f"expected {EXPECTED_COUNT} cases")

    expected_ids = [f"MAT-SYN-001A-{index:02d}" for index in range(1, EXPECTED_COUNT + 1)]
    observed_ids = [case.get("id") for case in cases]
    if observed_ids != expected_ids:
        fail("case identity/order drift")
    if len(set(observed_ids)) != EXPECTED_COUNT:
        fail("duplicate case identity")

    vector_set = set(VECTOR_KEYS)
    for case in cases:
        case_id = case["id"]
        facts = case.get("facts")
        expected_true = case.get("expected_true")
        limits = case.get("limits")

        if not isinstance(facts, dict):
            fail(f"{case_id}: facts must be an object")
        if not isinstance(expected_true, list):
            fail(f"{case_id}: expected_true must be a list")
        if len(expected_true) != len(set(expected_true)):
            fail(f"{case_id}: duplicate expected vector entry")
        unknown = set(expected_true) - vector_set
        if unknown:
            fail(f"{case_id}: unknown vector entries {sorted(unknown)}")
        if not isinstance(limits, list) or not limits or not all(
            isinstance(item, str) and item.strip() for item in limits
        ):
            fail(f"{case_id}: non-empty limits are required")

        derived = derive(facts)
        expected = set(expected_true)
        if derived != expected:
            fail(
                f"{case_id}: vector mismatch; "
                f"expected_true={sorted(expected)}, derived_true={sorted(derived)}"
            )

        if "manufacturability_supported" in expected:
            fail(f"{case_id}: synthetic corpus must not establish manufacturability")

    # Explicit semantic sentinels for boundaries that are easy to accidentally
    # collapse during future refactors.
    by_id = {case["id"]: case for case in cases}
    sentinels = {
        "MAT-SYN-001A-06": ({"formation_energy_support"}, {"phase_competition_support", "route_represented"}),
        "MAT-SYN-001A-07": ({"phase_competition_support"}, {"formation_energy_support", "route_represented"}),
        "MAT-SYN-001A-11": ({"current_capability_supported"}, {"resource_availability_supported", "planning_preconditions_supported"}),
        "MAT-SYN-001A-22": ({"formation_energy_support"}, {"local_reproduction_supported"}),
        "MAT-SYN-001A-27": ({"target_property_observed", "microstructure_bound"}, {"independent_repeatability", "manufacturability_supported"}),
        "MAT-SYN-001A-28": ({"physical_authorization_supported"}, {"physical_trial_observed"}),
    }
    for case_id, (must_have, must_not_have) in sentinels.items():
        expected = set(by_id[case_id]["expected_true"])
        if not must_have <= expected or expected & must_not_have:
            fail(f"{case_id}: semantic sentinel violated")

    print(
        f"ok cases={EXPECTED_COUNT} sha256={digest} git_blob={blob_identity}"
    )


if __name__ == "__main__":
    main()
