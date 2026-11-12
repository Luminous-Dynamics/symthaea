#!/usr/bin/env python3
"""Independent integer oracle for the post-fault sustained-supply corpus."""

from __future__ import annotations

import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-post-fault-supply-v0.1.json"

IDENTITY_FIELDS = (
    "configuration_digest",
    "operating_point_digest",
    "post_fault_topology_digest",
    "fault_event_digest",
    "critical_load_manifest_digest",
    "source_capability_digest",
    "storage_state_digest",
)


def validate_case(case, identity):
    for field in IDENTITY_FIELDS:
        if not identity.get(field, "").strip():
            return {"validation_error": "empty_identity", "field": field}

    ids = case["critical_load_ids"]
    if not ids:
        return {"validation_error": "no_critical_loads"}
    if ids != sorted(set(ids)):
        return {"validation_error": "noncanonical_critical_load_manifest"}

    usable_energy = case["usable_storage_energy_lower_j"]
    if usable_energy < 0:
        return {"validation_error": "negative_usable_energy"}

    expected_start = 0
    seen_segments = set()
    cumulative_energy = 0
    maximum_storage_power = 0

    for segment in case["segments"]:
        sid = segment["segment_id"]
        if not sid:
            return {"validation_error": "empty_segment_id"}
        if sid in seen_segments:
            return {"validation_error": "duplicate_segment_id"}
        seen_segments.add(sid)
        if segment["duration_seconds"] <= 0:
            return {"validation_error": "zero_duration"}
        if segment["start_seconds"] != expected_start:
            return {"validation_error": "non_contiguous_timeline"}

        loads = segment["critical_loads"]
        load_ids = [load["load_id"] for load in loads]
        if len(load_ids) != len(set(load_ids)):
            return {"validation_error": "duplicate_load_in_segment"}
        if load_ids != sorted(load_ids):
            return {"validation_error": "noncanonical_segment_load_order"}
        if load_ids != ids:
            return {"validation_error": "load_manifest_mismatch"}

        total_load = 0
        for load in loads:
            if load["demand_upper_w"] < 0:
                return {"validation_error": "negative_load_demand"}
            path = load["path"]
            status = path["status"]
            if status == "unknown":
                return {
                    "disposition": "not_evaluable_path",
                    "segment_id": sid,
                    "load_id": load["load_id"],
                }
            if status == "confirmed_unavailable":
                digest = path.get("evidence_digest", "")
                if not digest.strip():
                    return {"validation_error": "invalid_path_evidence"}
                return {
                    "disposition": "not_sustained_path",
                    "segment_id": sid,
                    "load_id": load["load_id"],
                    "evidence_digest": digest,
                }
            if status != "confirmed_available" or not path.get("path_digest", "").strip():
                return {"validation_error": "invalid_path_evidence"}
            total_load += load["demand_upper_w"]

        source_lower = segment["non_storage_source_capacity_lower_w"]
        storage_power_lower = segment["storage_discharge_power_lower_w"]
        if source_lower < 0 or storage_power_lower < 0:
            return {"validation_error": "negative_power_bound"}

        required_storage_power = max(0, total_load - source_lower)
        maximum_storage_power = max(maximum_storage_power, required_storage_power)
        if required_storage_power > storage_power_lower:
            return {
                "disposition": "not_sustained_power_limit",
                "segment_id": sid,
                "required_storage_power_w": required_storage_power,
                "available_storage_power_lower_w": storage_power_lower,
                "unmet_power_w": required_storage_power - storage_power_lower,
            }

        cumulative_energy += required_storage_power * segment["duration_seconds"]
        expected_start += segment["duration_seconds"]
        if cumulative_energy > usable_energy:
            prior_energy = cumulative_energy - (
                required_storage_power * segment["duration_seconds"]
            )
            remaining_before_segment = max(0, usable_energy - prior_energy)
            depletion_offset = (
                0 if required_storage_power == 0
                else (remaining_before_segment + required_storage_power - 1)
                     // required_storage_power
            )
            return {
                "disposition": "not_sustained_energy_limit",
                "segment_id": sid,
                "required_storage_energy_j": cumulative_energy,
                "usable_storage_energy_lower_j": usable_energy,
                "deficit_energy_j": cumulative_energy - usable_energy,
                "reserve_depletion_time_seconds": (
                    segment["start_seconds"] + depletion_offset
                ),
            }

    return {
        "disposition": "sustained_electrical_supply",
        "total_duration_seconds": expected_start,
        "maximum_storage_power_required_w": maximum_storage_power,
        "required_storage_energy_j": cumulative_energy,
        "remaining_storage_energy_lower_j": usable_energy - cumulative_energy,
    }


def main():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_integer_reference"
    identity = data["identity_template"]
    assert all(identity.get(field, "").strip() for field in IDENTITY_FIELDS)

    for case in data["cases"]:
        observed = validate_case(case, identity)
        assert observed == case["expected"], (case["id"], observed, case["expected"])
        print(f"PASS {case['id']}: {observed['disposition']}")

    # Execute the frozen structure/provenance negative controls against copies
    # of a valid input instead of trusting a list of expected labels.
    controls = data["validation_negative_controls"]
    by_id = {control["id"]: control["expected"] for control in controls}

    bad_identity = copy.deepcopy(identity)
    bad_identity["configuration_digest"] = ""
    result = validate_case(copy.deepcopy(data["cases"][0]), bad_identity)
    assert result["validation_error"] == by_id["empty-configuration-digest"]

    bad_manifest = copy.deepcopy(data["cases"][0])
    bad_manifest["segments"][0]["critical_loads"][0]["load_id"] = "undeclared-load"
    result = validate_case(bad_manifest, identity)
    assert result["validation_error"] == by_id["load-manifest-mismatch"]

    bad_timeline = copy.deepcopy(data["cases"][0])
    bad_timeline["segments"][1]["start_seconds"] += 1
    result = validate_case(bad_timeline, identity)
    assert result["validation_error"] == by_id["noncontiguous-segments"]

    reversed_segments = copy.deepcopy(data["cases"][0])
    reversed_segments["segments"].reverse()
    result = validate_case(reversed_segments, identity)
    assert result["validation_error"] == by_id["segment-ordering"]

    duplicate_segment = copy.deepcopy(data["cases"][0])
    duplicate_segment["segments"][1]["segment_id"] = (
        duplicate_segment["segments"][0]["segment_id"]
    )
    result = validate_case(duplicate_segment, identity)
    assert result["validation_error"] == by_id["duplicate-segment-id"]

    duplicate_load = copy.deepcopy(data["cases"][0])
    duplicate_load["segments"][0]["critical_loads"].append(
        copy.deepcopy(duplicate_load["segments"][0]["critical_loads"][0])
    )
    result = validate_case(duplicate_load, identity)
    assert result["validation_error"] == by_id["duplicate-load"]

    print("PASS all six executable profile/provenance negative controls")


if __name__ == "__main__":
    main()
