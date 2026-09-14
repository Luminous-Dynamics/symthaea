#!/usr/bin/env python3
"""Independent staged Beta6CampaignSubject oracle for LQCD-021J."""

import hashlib
import json
import math
import struct

ORACLE_ID = "beta6_campaign_subject_transition_oracle_v1"
TAG = b"symthaea.lqcd.beta6-campaign-subject.v1\0"
STAGES = {
    "Template": 1,
    "Pilot": 2,
    "Final": 3,
    "SealedAnalysis": 4,
    "Comparison": 5,
}


def unbound(reason):
    return {"$unbound": reason}


def is_unbound(value):
    return (
        isinstance(value, dict)
        and set(value) == {"$unbound"}
        and isinstance(value["$unbound"], str)
        and value["$unbound"]
    )


def u32(value):
    if not 0 <= value < (1 << 32):
        raise ValueError("u32 range")
    return value.to_bytes(4, "big")


def u64(value):
    if not 0 <= value < (1 << 64):
        raise ValueError("u64 range")
    return value.to_bytes(8, "big")


def encode_string(value):
    raw = value.encode("utf-8")
    return u32(len(raw)) + raw


def encode_value(value):
    """Language-neutral typed canonical encoding; no JSON float identity."""
    if is_unbound(value):
        return b"U" + encode_string(value["$unbound"])
    if value is None:
        return b"0"
    if isinstance(value, bool):
        return b"B" + (b"\x01" if value else b"\x00")
    if isinstance(value, int):
        if value < 0:
            raise ValueError("negative integer unsupported by v1")
        return b"I" + u64(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite float")
        return b"F" + struct.pack(">Q", struct.unpack(">Q", struct.pack(">d", value))[0])
    if isinstance(value, str):
        return b"S" + encode_string(value)
    if isinstance(value, list):
        return b"L" + u32(len(value)) + b"".join(encode_value(item) for item in value)
    if isinstance(value, dict):
        items = sorted(value.items(), key=lambda item: item[0].encode("utf-8"))
        if any(not isinstance(key, str) for key, _ in items):
            raise TypeError("map keys must be strings")
        return (
            b"M"
            + u32(len(items))
            + b"".join(encode_string(key) + encode_value(item) for key, item in items)
        )
    raise TypeError("unsupported canonical type: " + type(value).__name__)


def subject_bytes(stage, predecessor_digest, payload):
    if stage not in STAGES:
        raise ValueError("unknown stage")
    if len(predecessor_digest) != 32:
        raise ValueError("predecessor digest length")
    return TAG + bytes([STAGES[stage]]) + predecessor_digest + encode_value(payload)


def subject_digest(stage, predecessor_digest, payload):
    return hashlib.sha256(subject_bytes(stage, predecessor_digest, payload)).digest()


def require_unbound(value, label):
    if not is_unbound(value):
        raise ValueError(label + " must remain typed Unbound")


def require_bound(value, label):
    if is_unbound(value):
        raise ValueError(label + " must be bound")


def validate_template(payload):
    if payload["target_retained_configurations"] != 4000:
        raise ValueError("template target must remain 4000")
    for key in (
        "pilot_execution_budget",
        "pilot_chain_count",
        "final_chain_allocations",
        "final_burn_in",
        "final_stride",
        "final_seed_commitments",
        "final_block_policy",
        "final_plateau_windows",
        "final_r_ranges",
    ):
        require_unbound(payload[key], "template." + key)


def validate_pilot(payload):
    for key in (
        "pilot_execution_budget",
        "pilot_chain_count",
        "pilot_stream_commitments",
        "diagnostic_decision_policy",
        "operator_bridge_plan",
    ):
        require_bound(payload[key], "pilot." + key)
    for key in (
        "final_chain_allocations",
        "final_burn_in",
        "final_stride",
        "final_seed_commitments",
        "final_block_policy",
        "final_plateau_windows",
        "final_r_ranges",
    ):
        require_unbound(payload[key], "pilot." + key)


def derive_schedule(allocations):
    slots = []
    seen_chains = set()
    for allocation in allocations:
        chain_id = allocation["chain_id"]
        if chain_id in seen_chains:
            raise ValueError("duplicate chain id")
        seen_chains.add(chain_id)
        count = allocation["retained_count"]
        first = allocation["first_retained_update"]
        stride = allocation["stride"]
        if not isinstance(count, int) or count <= 0:
            raise ValueError("invalid retained count")
        if not isinstance(first, int) or first < 0 or not isinstance(stride, int) or stride <= 0:
            raise ValueError("invalid schedule")
        for ordinal in range(count):
            slots.append(
                {
                    "chain_id": chain_id,
                    "retained_ordinal": ordinal,
                    "update_ordinal": first + ordinal * stride,
                }
            )
    return slots


def validate_final(payload):
    for key in (
        "final_chain_allocations",
        "final_burn_in",
        "final_stride",
        "final_seed_commitments",
        "final_block_policy",
        "final_plateau_windows",
        "final_r_ranges",
        "fit_family_policy",
        "final_admission_policy",
        "analysis_subject",
    ):
        require_bound(payload[key], "final." + key)
    slots = derive_schedule(payload["final_chain_allocations"])
    if len(slots) != 4000:
        raise ValueError("final retained schedule must contain exactly 4000 slots")
    if payload["expected_retained_configurations"] != 4000:
        raise ValueError("final expected retained count mismatch")
    return slots


def validate_sealed_analysis(payload):
    for key in (
        "final_campaign_digest",
        "admitted_configuration_set_commitment",
        "complete_measurement_set_commitment",
        "analysis_subject",
        "sealed_result_digest",
    ):
        require_bound(payload[key], "sealed." + key)
    if "ehk_targets" in payload or "benchmark_delta" in payload:
        raise ValueError("target leakage into sealed analysis subject")


def validate_comparison(payload):
    for key in (
        "sealed_analysis_digest",
        "benchmark_digest",
        "systematic_ledger_digest",
        "comparison_semantics",
        "comparison_receipt_digest",
    ):
        require_bound(payload[key], "comparison." + key)


VALIDATORS = {
    "Template": validate_template,
    "Pilot": validate_pilot,
    "Final": validate_final,
    "SealedAnalysis": validate_sealed_analysis,
    "Comparison": validate_comparison,
}


def validate_transition(parent_stage, parent_digest, child_stage, child_predecessor, child_payload):
    if STAGES[child_stage] != STAGES[parent_stage] + 1:
        raise ValueError("stage skip or regression")
    if child_predecessor != parent_digest:
        raise ValueError("stale/substituted predecessor")
    return VALIDATORS[child_stage](child_payload)


def main():
    template = {
        "schema": "beta6_campaign_template_v1",
        "benchmark_lineage": "LQCD-020Q#2488",
        "measurement_design": "LQCD-021A#2528",
        "historical_fidelity_ledger": "1dd5a6db9cc68290ea80cffb3ecbc5a47fcaf01e2a1663a6e7950463224e8fc1",
        "action": "SU3_Wilson_plaquette",
        "beta": 6.0,
        "geometry": [16, 16, 16, 32],
        "target_retained_configurations": 4000,
        "sampler": "cm_heatbath_or_v1:force=staple:or_sweeps=3:max_attempts=256",
        "ape_subject": "spatial_ape_ehk_polar_v1",
        "wilson_subject": "generalized_bresenham_cubic_ape_spatial_unsmeared_temporal_wilson_v1",
        "numerical_profile_subject": "synthetic-qualified-profile-fixture",
        "pilot_decision_policy": "synthetic-frozen-pilot-decision-fixture",
        "pilot_execution_budget": unbound("throughput-dependent"),
        "pilot_chain_count": unbound("pilot authorization"),
        "final_chain_allocations": unbound("pilot-derived"),
        "final_burn_in": unbound("pilot-derived"),
        "final_stride": unbound("pilot-derived"),
        "final_seed_commitments": unbound("freeze after pilot before final"),
        "final_block_policy": unbound("pilot-derived"),
        "final_plateau_windows": unbound("pilot-derived"),
        "final_r_ranges": unbound("pilot-derived"),
    }
    validate_template(template)
    zero = bytes(32)
    template_digest = subject_digest("Template", zero, template)

    template_reordered = dict(reversed(list(template.items())))
    if subject_digest("Template", zero, template_reordered) != template_digest:
        raise AssertionError("map ordering changed subject digest")

    pilot = {
        "schema": "beta6_pilot_subject_v1",
        "pilot_execution_budget": {"cycles_per_start": 120, "wallclock_budget_class": "fixture"},
        "pilot_chain_count": 4,
        "pilot_stream_commitments": ["pilot-stream-a", "pilot-stream-b", "pilot-stream-c", "pilot-stream-d"],
        "diagnostic_decision_policy": "synthetic-frozen-pilot-decision-fixture",
        "operator_bridge_plan": "paired-exhaustive-vs-bresenham-tractable-domain",
        "final_chain_allocations": unbound("pilot evidence not yet complete"),
        "final_burn_in": unbound("pilot evidence not yet complete"),
        "final_stride": unbound("pilot evidence not yet complete"),
        "final_seed_commitments": unbound("freeze after pilot"),
        "final_block_policy": unbound("pilot evidence not yet complete"),
        "final_plateau_windows": unbound("pilot evidence not yet complete"),
        "final_r_ranges": unbound("pilot evidence not yet complete"),
    }
    validate_pilot(pilot)
    pilot_digest = subject_digest("Pilot", template_digest, pilot)
    validate_transition("Template", template_digest, "Pilot", template_digest, pilot)

    allocations = [
        {"chain_id": f"final-chain-{index}", "retained_count": 1000, "first_retained_update": 10000, "stride": 20}
        for index in range(4)
    ]
    final = {
        "schema": "beta6_final_campaign_subject_v1",
        "final_chain_allocations": allocations,
        "final_burn_in": 8000,
        "final_stride": 20,
        "final_seed_commitments": [f"synthetic-seed-commitment-{index}" for index in range(4)],
        "final_block_policy": {"block_size": 20, "policy_id": "synthetic"},
        "final_plateau_windows": "synthetic-frozen-windows",
        "final_r_ranges": "synthetic-frozen-ranges",
        "fit_family_policy": "synthetic-frozen-three-family-policy",
        "final_admission_policy": "LQCD-021G-synthetic-fixture",
        "analysis_subject": "LQCD-021F-synthetic-fixture",
        "expected_retained_configurations": 4000,
    }
    schedule = validate_final(final)
    schedule_digest = hashlib.sha256(encode_value(schedule)).digest()
    final_digest = subject_digest("Final", pilot_digest, final)
    validate_transition("Pilot", pilot_digest, "Final", pilot_digest, final)

    final_changed = json.loads(json.dumps(final))
    final_changed["final_burn_in"] = 8001
    changed_digest = subject_digest("Final", pilot_digest, final_changed)
    if changed_digest == final_digest:
        raise AssertionError("burn-in mutation did not change campaign identity")

    invalid_final = json.loads(json.dumps(final))
    invalid_final["final_chain_allocations"][-1]["retained_count"] = 999
    try:
        validate_final(invalid_final)
        raise AssertionError("3999-slot final campaign accepted")
    except ValueError as exc:
        if "exactly 4000" not in str(exc):
            raise

    sealed = {
        "schema": "beta6_sealed_analysis_subject_v1",
        "final_campaign_digest": final_digest.hex(),
        "admitted_configuration_set_commitment": "synthetic-config-set",
        "complete_measurement_set_commitment": "synthetic-measurement-set",
        "analysis_subject": "LQCD-021F-synthetic-fixture",
        "sealed_result_digest": "synthetic-target-isolated-result",
    }
    validate_sealed_analysis(sealed)
    sealed_digest = subject_digest("SealedAnalysis", final_digest, sealed)
    validate_transition("Final", final_digest, "SealedAnalysis", final_digest, sealed)

    comparison = {
        "schema": "beta6_comparison_subject_v1",
        "sealed_analysis_digest": sealed_digest.hex(),
        "benchmark_digest": "frozen-ehk-benchmark-v1",
        "systematic_ledger_digest": "synthetic-systematic-ledger",
        "comparison_semantics": "LQCD-021I-synthetic-fixture",
        "comparison_receipt_digest": "synthetic-comparison-receipt",
    }
    validate_comparison(comparison)
    comparison_digest = subject_digest("Comparison", sealed_digest, comparison)
    validate_transition("SealedAnalysis", sealed_digest, "Comparison", sealed_digest, comparison)

    alternate_comparison = dict(comparison)
    alternate_comparison["benchmark_digest"] = "later-benchmark-v2"
    alternate_comparison_digest = subject_digest("Comparison", sealed_digest, alternate_comparison)
    if alternate_comparison_digest == comparison_digest:
        raise AssertionError("benchmark mutation did not change comparison identity")
    if subject_digest("SealedAnalysis", final_digest, sealed) != sealed_digest:
        raise AssertionError("sealed analysis changed during comparison mutation")

    leaking = dict(sealed)
    leaking["ehk_targets"] = [0.2189, 5.369, 8.831, 10.89]
    try:
        validate_sealed_analysis(leaking)
        raise AssertionError("target leakage accepted")
    except ValueError as exc:
        if "target leakage" not in str(exc):
            raise

    try:
        validate_transition("Template", template_digest, "Pilot", bytes(32), pilot)
        raise AssertionError("stale predecessor accepted")
    except ValueError as exc:
        if "predecessor" not in str(exc):
            raise
    try:
        validate_transition("Template", template_digest, "Final", template_digest, final)
        raise AssertionError("stage skip accepted")
    except ValueError as exc:
        if "stage skip" not in str(exc):
            raise

    result = {
        "oracle_id": ORACLE_ID,
        "canonical_encoding": "typed_tlv_v1",
        "stage_chain": list(STAGES),
        "template_digest": template_digest.hex(),
        "pilot_digest": pilot_digest.hex(),
        "final_digest": final_digest.hex(),
        "sealed_analysis_digest": sealed_digest.hex(),
        "comparison_digest": comparison_digest.hex(),
        "alternate_comparison_digest": alternate_comparison_digest.hex(),
        "exact_retained_schedule_count": len(schedule),
        "retained_schedule_digest": schedule_digest.hex(),
        "map_order_invariant": True,
        "burn_in_change_changes_final_identity": True,
        "3999_slots_rejected": True,
        "stale_predecessor_rejected": True,
        "stage_skip_rejected": True,
        "target_leakage_into_sealed_analysis_rejected": True,
        "comparison_target_change_does_not_mutate_sealed_analysis": True,
        "claim_boundary": {
            "synthetic_transition_semantics_established": True,
            "real_beta6_campaign_authorized": False,
        },
    }
    text = json.dumps(result, sort_keys=True, separators=(",", ":"))
    print("ok")
    print("result_sha256=" + hashlib.sha256(text.encode()).hexdigest())
    print(text)


if __name__ == "__main__":
    main()
