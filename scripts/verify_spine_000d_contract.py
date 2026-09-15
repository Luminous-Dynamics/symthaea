#!/usr/bin/env python3
"""Static verifier for SPINE-000D causal-ablation preregistration."""

import json
from pathlib import Path

CONTRACT = Path("docs/research/SPINE_000D_CAUSAL_ABLATION_CONTRACT.md")
SCHEMA = Path("docs/research/SPINE_000D_CAMPAIGN_MANIFEST.schema.json")


def fail(msg: str) -> None:
    raise SystemExit(f"SPINE-000D CONTRACT FAIL: {msg}")


def require(condition: bool, msg: str) -> None:
    if not condition:
        fail(msg)


def has_conditional_required(node: dict, discriminator: str, value: str, field: str) -> bool:
    for clause in node.get("allOf", []):
        if_node = clause.get("if", {})
        props = if_node.get("properties", {})
        disc = props.get(discriminator, {})
        if disc.get("const") != value:
            continue
        if field in clause.get("then", {}).get("required", []):
            return True
    return False


def main() -> int:
    contract = CONTRACT.read_text(encoding="utf-8")
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))

    require(
        schema.get("$schema") == "https://json-schema.org/draft/2020-12/schema",
        "unexpected JSON Schema dialect",
    )
    require(
        schema.get("additionalProperties") is False,
        "campaign manifest must fail closed on unknown top-level fields",
    )

    required = set(schema.get("required", []))
    for field in (
        "subject",
        "target_subsystem",
        "intervention_arms",
        "arm_order_policy",
        "rng_alignment_policy",
        "workload_rule",
        "initial_state_rule",
        "scheduler_state_rule",
        "primary_observables",
        "safety_hard_gates",
        "multiplicity_policy",
        "confidence_interval_policy",
        "interaction_followup_rule",
        "stopping_rule",
        "classification_scope_fields",
    ):
        require(field in required, f"manifest no longer requires {field}")

    arms = schema["properties"]["intervention_arms"]
    arm_items = arms["items"]["enum"]
    for arm in ("FULL", "OUTPUT_SHAM", "DISABLED", "STATE_FROZEN"):
        require(arm in arm_items, f"missing intervention arm {arm}")
    require(arms.get("minItems", 0) >= 3, "three-arm base design is no longer mandatory")
    mandatory_contains = {
        clause.get("contains", {}).get("const") for clause in arms.get("allOf", [])
    }
    require(
        {"FULL", "OUTPUT_SHAM", "DISABLED"}.issubset(mandatory_contains),
        "FULL + OUTPUT_SHAM + DISABLED are not all structurally mandatory",
    )

    workload = schema["properties"]["workload_rule"]
    require(
        has_conditional_required(workload, "kind", "FROZEN_EXPLICIT", "workload_ids"),
        "FROZEN_EXPLICIT does not require workload_ids",
    )
    require(
        has_conditional_required(
            workload, "kind", "FROZEN_GENERATOR", "generator_subject_sha256"
        ),
        "FROZEN_GENERATOR does not require generator hash",
    )
    require(
        has_conditional_required(
            workload, "kind", "FROZEN_GENERATOR", "materialized_workload_sha256"
        ),
        "FROZEN_GENERATOR does not require materialized workload hash",
    )

    initial = schema["properties"]["initial_state_rule"]
    require(
        has_conditional_required(
            initial, "kind", "FROZEN_CHECKPOINT", "checkpoint_sha256"
        ),
        "FROZEN_CHECKPOINT does not require checkpoint hash",
    )

    stopping = schema["properties"]["stopping_rule"]
    require(
        has_conditional_required(stopping, "kind", "FIXED_N", "n_pairs"),
        "FIXED_N does not require n_pairs",
    )
    require(
        has_conditional_required(stopping, "kind", "FROZEN_SEQUENTIAL", "details"),
        "FROZEN_SEQUENTIAL does not require frozen details",
    )

    order = schema["properties"]["arm_order_policy"]
    require(
        has_conditional_required(order, "kind", "RANDOMIZED_FROZEN_SEED", "seed"),
        "randomized arm order does not require a frozen seed",
    )
    require(
        has_conditional_required(order, "kind", "FIXED_JUSTIFIED", "details"),
        "fixed arm order does not require justification",
    )

    rng = schema["properties"]["rng_alignment_policy"]
    require(
        has_conditional_required(rng, "kind", "DECLARED_DIVERGENCE", "details"),
        "declared RNG divergence does not require details",
    )

    equiv = schema["$defs"]["observable"]["properties"]["equivalence_rule"]
    bounded_kinds = {"ABSOLUTE_SESOI", "RELATIVE_SESOI", "NONINFERIORITY"}
    conditional_bounded = False
    exact_forbids_bound = False
    for clause in equiv.get("allOf", []):
        props = clause.get("if", {}).get("properties", {})
        kind_rule = props.get("kind", {})
        enum_values = set(kind_rule.get("enum", []))
        if bounded_kinds.issubset(enum_values) and "bound" in clause.get("then", {}).get("required", []):
            conditional_bounded = True
        if kind_rule.get("const") == "EXACT" and clause.get("then", {}).get("not", {}).get("required") == ["bound"]:
            exact_forbids_bound = True
    require(conditional_bounded, "bounded equivalence modes do not require numeric bound")
    require(exact_forbids_bound, "EXACT equivalence does not forbid an arbitrary numeric bound")

    scope_const = schema["properties"]["classification_scope_fields"].get("const")
    require(
        scope_const
        == [
            "subject",
            "workload",
            "observable",
            "intervention",
            "comparison",
            "evidence_lineage",
        ],
        "classification scope tuple changed",
    )

    for phrase in (
        "No global load-bearing label",
        "The three base arms are mandatory",
        "Arm-order policy is mandatory",
        "RNG alignment policy is mandatory",
        "Null claims require equivalence evidence",
        "Difference is not benefit",
        "Redundancy and synergy require interaction follow-up",
        "Legacy inline paths can mask manager ablations",
        "Safety and authority gates remain separate",
        "LOAD_BEARING_POSITIVE",
        "EQUIVALENT_WITHIN_BOUND",
        "MASKED_BY_OVERLAP",
        "INTERACTION_SUSPECTED",
    ):
        require(phrase in contract, f"contract missing required concept: {phrase}")

    print("SPINE-000D contract/schema verifier: PASS")
    print("base_arms=FULL,OUTPUT_SHAM,DISABLED")
    print("arm_order_policy=required")
    print("rng_alignment_policy=required")
    print("authority=measurement-only")
    print("causal_results_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
