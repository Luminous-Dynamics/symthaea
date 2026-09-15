#!/usr/bin/env python3
"""Negative-control self-test for SPINE-000D campaign preflight."""

from __future__ import annotations

import copy

from validate_spine_000d_campaign import ManifestError, validate

ZERO40 = "0" * 40
ZERO64 = "0" * 64


def base_manifest() -> dict:
    return {
        "schema": "symthaea.spine.000d.campaign-manifest.v1",
        "authority": "measurement-only",
        "campaign_id": "selftest-campaign",
        "subject": {
            "git_head": ZERO40,
            "git_tree": ZERO40,
            "subject_file_hashes": {"subject.rs": ZERO64},
            "cargo_toml_sha256": ZERO64,
            "cargo_lock_sha256": ZERO64,
            "rust_toolchain": "rustc selftest",
        },
        "target_subsystem": "selftest_manager",
        "static_overlap_refs": [],
        "runtime_influence_evidence_refs": [],
        "intervention_arms": ["FULL", "OUTPUT_SHAM", "DISABLED"],
        "arm_order_policy": {"kind": "COUNTERBALANCED"},
        "rng_alignment_policy": {"kind": "MATCHED_STREAMS"},
        "workload_rule": {"kind": "FROZEN_EXPLICIT", "workload_ids": ["w1", "w2"]},
        "initial_state_rule": {"kind": "FRESH_GENESIS"},
        "scheduler_state_rule": "fresh identical scheduler state",
        "primary_observables": [
            {
                "id": "prediction_error",
                "kind": "CONTINUOUS",
                "direction": "LOWER_BETTER",
                "equivalence_rule": {"kind": "ABSOLUTE_SESOI", "bound": 0.01},
            }
        ],
        "secondary_observables": [],
        "safety_hard_gates": [{"id": "authority", "rule": "violations == 0"}],
        "multiplicity_policy": {"kind": "SMALL_PRIMARY_FAMILY"},
        "confidence_interval_policy": {"kind": "PAIRED_BOOTSTRAP", "level": 0.95},
        "interaction_followup_rule": {"triggers": ["STATIC_OVERLAP"]},
        "stopping_rule": {"kind": "FIXED_N", "n_pairs": 8},
        "classification_scope_fields": [
            "subject",
            "workload",
            "observable",
            "intervention",
            "comparison",
            "evidence_lineage",
        ],
    }


def expect_fail(label: str, mutate) -> None:
    manifest = copy.deepcopy(base_manifest())
    mutate(manifest)
    try:
        validate(manifest)
    except ManifestError:
        return
    raise AssertionError(f"negative control unexpectedly passed: {label}")


def main() -> int:
    validate(base_manifest())

    expect_fail(
        "missing_output_sham",
        lambda m: m.__setitem__("intervention_arms", ["FULL", "DISABLED"]),
    )
    expect_fail(
        "duplicate_primary_id",
        lambda m: m["primary_observables"].append(copy.deepcopy(m["primary_observables"][0])),
    )
    expect_fail(
        "exact_with_bound",
        lambda m: m["primary_observables"][0].__setitem__(
            "equivalence_rule", {"kind": "EXACT", "bound": 0.0}
        ),
    )
    expect_fail(
        "bounded_without_bound",
        lambda m: m["primary_observables"][0].__setitem__(
            "equivalence_rule", {"kind": "ABSOLUTE_SESOI"}
        ),
    )
    expect_fail(
        "explicit_plus_generator_hashes",
        lambda m: m["workload_rule"].__setitem__("generator_subject_sha256", ZERO64),
    )
    expect_fail(
        "generator_plus_workload_ids",
        lambda m: m.__setitem__(
            "workload_rule",
            {
                "kind": "FROZEN_GENERATOR",
                "workload_ids": ["w1"],
                "generator_subject_sha256": ZERO64,
                "materialized_workload_sha256": ZERO64,
            },
        ),
    )
    expect_fail(
        "fresh_genesis_with_checkpoint",
        lambda m: m["initial_state_rule"].__setitem__("checkpoint_sha256", ZERO64),
    )
    expect_fail(
        "random_order_without_seed",
        lambda m: m.__setitem__("arm_order_policy", {"kind": "RANDOMIZED_FROZEN_SEED"}),
    )
    expect_fail(
        "declared_rng_divergence_without_details",
        lambda m: m.__setitem__("rng_alignment_policy", {"kind": "DECLARED_DIVERGENCE"}),
    )
    expect_fail(
        "fixed_n_without_n",
        lambda m: m.__setitem__("stopping_rule", {"kind": "FIXED_N"}),
    )
    expect_fail(
        "scope_tuple_mutated",
        lambda m: m.__setitem__("classification_scope_fields", ["subject", "observable"]),
    )

    print("SPINE-000D campaign validator self-test: PASS")
    print("positive_controls=1")
    print("negative_controls=11")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
