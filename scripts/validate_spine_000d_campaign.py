#!/usr/bin/env python3
"""Fail-closed semantic preflight for a SPINE-000D campaign manifest.

This complements the JSON Schema with cross-field checks that are awkward to
express declaratively. It does not run an ablation or produce a causal result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

SHA256 = re.compile(r"^[0-9a-f]{64}$")
GITSHA = re.compile(r"^[0-9a-f]{40}$")
BASE_ARMS = {"FULL", "OUTPUT_SHAM", "DISABLED"}
SCOPE_FIELDS = [
    "subject",
    "workload",
    "observable",
    "intervention",
    "comparison",
    "evidence_lineage",
]


class ManifestError(ValueError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def unique_ids(items: list[dict], label: str) -> None:
    ids = [item.get("id") for item in items]
    require(all(isinstance(i, str) and i for i in ids), f"{label}: every item needs non-empty id")
    require(len(ids) == len(set(ids)), f"{label}: duplicate ids are forbidden")


def validate(manifest: dict) -> None:
    require(manifest.get("schema") == "symthaea.spine.000d.campaign-manifest.v1", "wrong schema")
    require(manifest.get("authority") == "measurement-only", "authority must be measurement-only")
    require(isinstance(manifest.get("campaign_id"), str) and manifest["campaign_id"], "campaign_id missing")

    subject = manifest.get("subject")
    require(isinstance(subject, dict), "subject missing")
    require(bool(GITSHA.fullmatch(str(subject.get("git_head", "")))), "invalid subject.git_head")
    require(bool(GITSHA.fullmatch(str(subject.get("git_tree", "")))), "invalid subject.git_tree")
    for field in ("cargo_toml_sha256", "cargo_lock_sha256"):
        require(bool(SHA256.fullmatch(str(subject.get(field, "")))), f"invalid subject.{field}")
    hashes = subject.get("subject_file_hashes")
    require(isinstance(hashes, dict) and hashes, "subject_file_hashes must be non-empty")
    require(all(SHA256.fullmatch(str(v)) for v in hashes.values()), "invalid subject file hash")
    require(isinstance(subject.get("rust_toolchain"), str) and subject["rust_toolchain"], "rust_toolchain missing")

    require(isinstance(manifest.get("target_subsystem"), str) and manifest["target_subsystem"], "target_subsystem missing")

    arms = manifest.get("intervention_arms")
    require(isinstance(arms, list), "intervention_arms must be an array")
    require(len(arms) == len(set(arms)), "duplicate intervention arms")
    require(BASE_ARMS.issubset(set(arms)), "FULL + OUTPUT_SHAM + DISABLED are mandatory")
    require(set(arms).issubset(BASE_ARMS | {"STATE_FROZEN"}), "unknown intervention arm")

    order = manifest.get("arm_order_policy")
    require(isinstance(order, dict), "arm_order_policy missing")
    order_kind = order.get("kind")
    require(order_kind in {"COUNTERBALANCED", "DETERMINISTIC_ROTATION", "RANDOMIZED_FROZEN_SEED", "FIXED_JUSTIFIED"}, "invalid arm_order_policy.kind")
    if order_kind == "RANDOMIZED_FROZEN_SEED":
        require(isinstance(order.get("seed"), int), "randomized arm order requires integer seed")
    if order_kind == "FIXED_JUSTIFIED":
        require(isinstance(order.get("details"), str) and order["details"].strip(), "fixed arm order requires justification")

    rng = manifest.get("rng_alignment_policy")
    require(isinstance(rng, dict), "rng_alignment_policy missing")
    rng_kind = rng.get("kind")
    require(rng_kind in {"MATCHED_STREAMS", "DECLARED_DIVERGENCE"}, "invalid rng_alignment_policy.kind")
    if rng_kind == "DECLARED_DIVERGENCE":
        require(isinstance(rng.get("details"), str) and rng["details"].strip(), "declared RNG divergence requires details")

    workload = manifest.get("workload_rule")
    require(isinstance(workload, dict), "workload_rule missing")
    wkind = workload.get("kind")
    if wkind == "FROZEN_EXPLICIT":
        ids = workload.get("workload_ids")
        require(isinstance(ids, list) and ids and len(ids) == len(set(ids)), "explicit workload requires unique non-empty workload_ids")
        require(not any(k in workload for k in ("generator_subject_sha256", "materialized_workload_sha256")), "explicit workload must not also carry generator hashes")
    elif wkind == "FROZEN_GENERATOR":
        require("workload_ids" not in workload, "generator workload must not also carry explicit workload_ids")
        require(bool(SHA256.fullmatch(str(workload.get("generator_subject_sha256", "")))), "invalid generator_subject_sha256")
        require(bool(SHA256.fullmatch(str(workload.get("materialized_workload_sha256", "")))), "invalid materialized_workload_sha256")
    else:
        raise ManifestError("invalid workload_rule.kind")

    initial = manifest.get("initial_state_rule")
    require(isinstance(initial, dict), "initial_state_rule missing")
    ikind = initial.get("kind")
    if ikind == "FROZEN_CHECKPOINT":
        require(bool(SHA256.fullmatch(str(initial.get("checkpoint_sha256", "")))), "frozen checkpoint requires valid hash")
    elif ikind == "FRESH_GENESIS":
        require("checkpoint_sha256" not in initial, "fresh genesis must not carry checkpoint hash")
    else:
        raise ManifestError("invalid initial_state_rule.kind")

    require(isinstance(manifest.get("scheduler_state_rule"), str) and manifest["scheduler_state_rule"].strip(), "scheduler_state_rule missing")

    primary = manifest.get("primary_observables")
    secondary = manifest.get("secondary_observables", [])
    require(isinstance(primary, list) and primary, "primary_observables must be non-empty")
    require(isinstance(secondary, list), "secondary_observables must be an array")
    unique_ids(primary, "primary_observables")
    unique_ids(secondary, "secondary_observables")
    primary_ids = {x["id"] for x in primary}
    secondary_ids = {x["id"] for x in secondary}
    require(primary_ids.isdisjoint(secondary_ids), "an observable cannot be both primary and secondary")

    for observable in primary + secondary:
        direction = observable.get("direction")
        require(direction in {"HIGHER_BETTER", "LOWER_BETTER", "NO_UTILITY_DIRECTION"}, f"{observable['id']}: invalid direction")
        rule = observable.get("equivalence_rule")
        require(isinstance(rule, dict), f"{observable['id']}: equivalence_rule missing")
        kind = rule.get("kind")
        require(kind in {"EXACT", "ABSOLUTE_SESOI", "RELATIVE_SESOI", "NONINFERIORITY"}, f"{observable['id']}: invalid equivalence rule")
        if kind == "EXACT":
            require("bound" not in rule, f"{observable['id']}: EXACT must not carry numeric bound")
        else:
            require(isinstance(rule.get("bound"), (int, float)) and not isinstance(rule.get("bound"), bool), f"{observable['id']}: bounded equivalence requires numeric bound")
            if kind in {"ABSOLUTE_SESOI", "RELATIVE_SESOI"}:
                require(rule["bound"] >= 0, f"{observable['id']}: SESOI bound must be non-negative")

    gates = manifest.get("safety_hard_gates")
    require(isinstance(gates, list) and gates, "safety_hard_gates must be non-empty")
    unique_ids(gates, "safety_hard_gates")
    require(all(isinstance(g.get("rule"), str) and g["rule"].strip() for g in gates), "every safety hard gate needs a rule")

    multiplicity = manifest.get("multiplicity_policy")
    require(isinstance(multiplicity, dict), "multiplicity_policy missing")
    mkind = multiplicity.get("kind")
    require(mkind in {"SMALL_PRIMARY_FAMILY", "HIERARCHICAL", "FDR", "FWER", "OTHER_FROZEN"}, "invalid multiplicity policy")
    if mkind == "OTHER_FROZEN":
        require(isinstance(multiplicity.get("details"), str) and multiplicity["details"].strip(), "OTHER_FROZEN multiplicity needs details")

    ci = manifest.get("confidence_interval_policy")
    require(isinstance(ci, dict), "confidence_interval_policy missing")
    cikind = ci.get("kind")
    require(cikind in {"EXACT_DETERMINISTIC", "PAIRED_BOOTSTRAP", "PAIRED_PARAMETRIC", "OTHER_FROZEN"}, "invalid confidence policy")
    if cikind in {"PAIRED_BOOTSTRAP", "PAIRED_PARAMETRIC"}:
        require(isinstance(ci.get("level"), (int, float)) and 0 < ci["level"] < 1, "paired confidence policy requires level in (0,1)")
    if cikind == "OTHER_FROZEN":
        require(isinstance(ci.get("details"), str) and ci["details"].strip(), "OTHER_FROZEN confidence policy needs details")

    interaction = manifest.get("interaction_followup_rule")
    require(isinstance(interaction, dict), "interaction_followup_rule missing")
    triggers = interaction.get("triggers")
    require(isinstance(triggers, list) and triggers and len(triggers) == len(set(triggers)), "interaction triggers must be unique/non-empty")
    if "OTHER_FROZEN" in triggers:
        require(isinstance(interaction.get("details"), str) and interaction["details"].strip(), "custom interaction trigger needs details")

    stopping = manifest.get("stopping_rule")
    require(isinstance(stopping, dict), "stopping_rule missing")
    skind = stopping.get("kind")
    if skind == "FIXED_N":
        require(isinstance(stopping.get("n_pairs"), int) and stopping["n_pairs"] >= 1, "FIXED_N requires n_pairs >= 1")
    elif skind == "FROZEN_SEQUENTIAL":
        require(isinstance(stopping.get("details"), str) and stopping["details"].strip(), "FROZEN_SEQUENTIAL requires frozen details")
    else:
        raise ManifestError("invalid stopping_rule.kind")

    require(manifest.get("classification_scope_fields") == SCOPE_FIELDS, "classification scope tuple changed")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    try:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        require(isinstance(manifest, dict), "manifest must be a JSON object")
        validate(manifest)
    except (OSError, json.JSONDecodeError, ManifestError) as exc:
        raise SystemExit(f"SPINE-000D CAMPAIGN PREFLIGHT FAIL: {exc}") from exc

    print("SPINE-000D campaign preflight: PASS")
    print(f"campaign_id={manifest['campaign_id']}")
    print(f"target_subsystem={manifest['target_subsystem']}")
    print("causal_results_claimed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
