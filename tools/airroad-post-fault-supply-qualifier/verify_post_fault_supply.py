#!/usr/bin/env python3
"""Independent integer oracle for the post-fault sustained-supply corpus."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/research/fixtures/airroad-post-fault-supply-v0.1.json"


def validate_case(case):
    ids = case["critical_load_ids"]
    if not ids or ids != sorted(set(ids)):
        return {"validation_error": "noncanonical_critical_load_manifest"}
    expected_start = 0
    seen_segments = set()
    cumulative_energy = 0
    maximum_storage_power = 0

    for segment in case["segments"]:
        sid = segment["segment_id"]
        if sid in seen_segments:
            return {"validation_error": "duplicate_segment_id"}
        seen_segments.add(sid)
        if segment["duration_seconds"] <= 0:
            return {"validation_error": "zero_duration"}
        if segment["start_seconds"] != expected_start:
            return {"validation_error": "non_contiguous_timeline"}

        loads = segment["critical_loads"]
        load_ids = [load["load_id"] for load in loads]
        if load_ids != sorted(set(load_ids)):
            return {"validation_error": "duplicate_or_noncanonical_segment_load"}
        if load_ids != ids:
            return {"validation_error": "load_manifest_mismatch"}

        total_load = 0
        for load in loads:
            path = load["path"]
            status = path["status"]
            if status == "unknown":
                return {
                    "disposition": "not_evaluable_path",
                    "segment_id": sid,
                    "load_id": load["load_id"],
                }
            if status == "confirmed_unavailable":
                return {
                    "disposition": "not_sustained_path",
                    "segment_id": sid,
                    "load_id": load["load_id"],
                    "evidence_digest": path["evidence_digest"],
                }
            if status != "confirmed_available" or not path.get("path_digest", "").strip():
                return {"validation_error": "invalid_path_evidence"}
            total_load += load["demand_upper_w"]

        required_storage_power = max(
            0, total_load - segment["non_storage_source_capacity_lower_w"]
        )
        maximum_storage_power = max(maximum_storage_power, required_storage_power)
        storage_power_lower = segment["storage_discharge_power_lower_w"]
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
        reserve = case["usable_storage_energy_lower_j"]
        if cumulative_energy > reserve:
            prior_energy = cumulative_energy - (
                required_storage_power * segment["duration_seconds"]
            )
            remaining_before_segment = max(0, reserve - prior_energy)
            depletion_offset = (
                0 if required_storage_power == 0
                else (remaining_before_segment + required_storage_power - 1)
                     // required_storage_power
            )
            return {
                "disposition": "not_sustained_energy_limit",
                "segment_id": sid,
                "required_storage_energy_j": cumulative_energy,
                "usable_storage_energy_lower_j": reserve,
                "deficit_energy_j": cumulative_energy - reserve,
                "reserve_depletion_time_seconds": (
                    segment["start_seconds"] + depletion_offset
                ),
            }

    return {
        "disposition": "sustained_electrical_supply",
        "total_duration_seconds": expected_start,
        "maximum_storage_power_required_w": maximum_storage_power,
        "required_storage_energy_j": cumulative_energy,
        "remaining_storage_energy_lower_j": (
            case["usable_storage_energy_lower_j"] - cumulative_energy
        ),
    }


def main():
    data = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert data["schema_version"] == "0.1.0"
    assert data["evidence_class"] == "synthetic_exact_integer_reference"

    for case in data["cases"]:
        observed = validate_case(case)
        assert observed == case["expected"], (case["id"], observed, case["expected"])
        print(f"PASS {case['id']}: {observed['disposition']}")

    controls = {item["id"]: item["expected"] for item in data["validation_negative_controls"]}
    assert controls["empty-configuration-digest"] == "empty_identity"
    assert controls["load-manifest-mismatch"] == "load_manifest_mismatch"
    assert controls["noncontiguous-segments"] == "non_contiguous_timeline"
    assert controls["segment-ordering"] == "non_contiguous_timeline"
    assert controls["duplicate-segment-id"] == "duplicate_segment_id"
    assert controls["duplicate-load"] == "duplicate_load_in_segment"
    print("PASS validation negative-control contract")


if __name__ == "__main__":
    main()
