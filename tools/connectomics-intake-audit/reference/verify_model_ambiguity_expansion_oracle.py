#!/usr/bin/env python3
"""Verify append-only model-ambiguity expansion against the exact sequential oracle.

The verifier binds exact Git blob inputs and raw SHA-256 values, requires the
policy-defining universe to remain unchanged, and recomputes both exact static
policy-tree optima with the independent #10764 oracle.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any


FROZEN_FIELDS = (
    "schema_version",
    "claim_scope",
    "query",
    "worlds",
    "actions",
    "policy_space",
    "split_manifest",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def load_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_sequential_oracle(path: Path):
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location("static_model_sequential_oracle", path)
    require(spec is not None and spec.loader is not None, "could not load sequential policy oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, raw


def validate_pair(
    base: dict[str, Any],
    expanded: dict[str, Any],
    binding: dict[str, Any],
    sequential_oracle: Any,
) -> dict[str, Any]:
    for field in FROZEN_FIELDS:
        require(base.get(field) == expanded.get(field), f"frozen field mutated by model-set expansion: {field}")

    base_models = base["model_vertices"]
    expanded_models = expanded["model_vertices"]
    base_ids = [row["model_vertex_id"] for row in base_models]
    expanded_ids = [row["model_vertex_id"] for row in expanded_models]
    require(base_ids == binding["base_model_vertex_ids"], "base model vertex sequence differs from frozen manifest")
    require(expanded_ids == binding["expanded_model_vertex_ids"], "expanded model vertex sequence differs from frozen manifest")
    require(len(expanded_models) == len(base_models) + 1, "expansion must add exactly one model vertex")
    require(expanded_ids[:len(base_ids)] == base_ids, "base model vertices were reordered, deleted or replaced")
    require(expanded_models[:len(base_models)] == base_models, "an existing model vertex was mutated")
    added = expanded_models[-1]
    require(added["model_vertex_id"] == binding["added_model_vertex_id"], "appended model vertex ID mismatch")
    require(added["model_sha256"] == binding["added_model_sha256"], "appended model vertex source/model digest mismatch")

    base_validated = sequential_oracle.validate_input(base)
    expanded_validated = sequential_oracle.validate_input(expanded)
    base_validated["max_cost"] = base["policy_space"]["max_worst_branch_cost_units"]
    expanded_validated["max_cost"] = expanded["policy_space"]["max_worst_branch_cost_units"]

    base_solution = sequential_oracle.static_policy_search(base_validated)
    expanded_solution = sequential_oracle.static_policy_search(expanded_validated)
    base_risk = float(base_solution["robust_risk"])
    expanded_risk = float(expanded_solution["robust_risk"])
    tolerance = float(binding["certificate_tolerance"])
    base_certificate = base_risk <= tolerance
    expanded_certificate = expanded_risk <= tolerance

    require(
        expanded_risk + 1e-12 >= base_risk,
        f"robust loss decreased under append-only expansion: base={base_risk}, expanded={expanded_risk}",
    )
    require(
        base_certificate or not expanded_certificate,
        "certificate became valid after model-set expansion although it was invalid before",
    )
    require(
        abs(base_risk - binding["expected_base_robust_loss"]) <= 1e-9,
        "base exact robust-loss known answer mismatch",
    )
    require(
        abs(expanded_risk - binding["expected_expanded_robust_loss"]) <= 1e-9,
        "expanded exact robust-loss known answer mismatch",
    )
    require(base_certificate is binding["expected_base_certificate"], "base certificate known answer mismatch")
    require(expanded_certificate is binding["expected_expanded_certificate"], "expanded certificate known answer mismatch")

    return {
        "base_model_vertex_count": len(base_models),
        "expanded_model_vertex_count": len(expanded_models),
        "added_model_vertex_id": added["model_vertex_id"],
        "base_static_robust_loss": base_risk,
        "expanded_static_robust_loss": expanded_risk,
        "robust_loss_increase": expanded_risk - base_risk,
        "base_certificate": base_certificate,
        "expanded_certificate": expanded_certificate,
        "certificate_transition": (
            "INVALIDATED_BY_MODEL_EXPANSION"
            if base_certificate and not expanded_certificate
            else "REMAINS_VALID"
            if base_certificate and expanded_certificate
            else "REMAINS_NOT_CERTIFIABLE"
        ),
        "base_optimal_policy": {
            "second_action_by_first_history": base_solution["action_map"],
            "terminal_decision_by_history": base_solution["decision_map"],
            "worst_branch_cost": base_solution["worst_branch_cost"],
        },
        "expanded_optimal_policy": {
            "second_action_by_first_history": expanded_solution["action_map"],
            "terminal_decision_by_history": expanded_solution["decision_map"],
            "worst_branch_cost": expanded_solution["worst_branch_cost"],
        },
        "model_ancestry_overlaps": [
            {
                "model_vertex_a": left["model_vertex_id"],
                "model_vertex_b": right["model_vertex_id"],
                "shared_group_ids": sorted(
                    set(left["dependence_group_ids"]) & set(right["dependence_group_ids"])
                ),
            }
            for index, left in enumerate(sorted(expanded_models, key=lambda row: row["model_vertex_id"]))
            for right in sorted(expanded_models, key=lambda row: row["model_vertex_id"])[index + 1:]
            if set(left["dependence_group_ids"]) & set(right["dependence_group_ids"])
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-input", type=Path, required=True)
    parser.add_argument("--expanded-input", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--sequential-oracle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    try:
        base_raw = args.base_input.read_bytes()
        expanded_raw = args.expanded_input.read_bytes()
        binding_raw = args.binding.read_bytes()
        oracle_module, oracle_raw = load_sequential_oracle(args.sequential_oracle)
        base = load_json(base_raw, "base input")
        expanded = load_json(expanded_raw, "expanded input")
        binding = load_json(binding_raw, "challenge binding")

        require(binding["schema_version"] == 1, "unsupported challenge binding schema version")
        require(re.fullmatch(r"[0-9a-f]{40}", binding["base_input_git_blob_sha1"]) is not None, "base Git blob SHA malformed")
        require(re.fullmatch(r"[0-9a-f]{40}", binding["expanded_input_git_blob_sha1"]) is not None, "expanded Git blob SHA malformed")
        require(re.fullmatch(r"[0-9a-f]{40}", binding["sequential_oracle_git_blob_sha1"]) is not None, "sequential oracle Git blob SHA malformed")
        require(git_blob_sha1(base_raw) == binding["base_input_git_blob_sha1"], "base input bytes do not match pinned Git blob")
        require(git_blob_sha1(expanded_raw) == binding["expanded_input_git_blob_sha1"], "expanded input bytes do not match pinned Git blob")
        require(git_blob_sha1(oracle_raw) == binding["sequential_oracle_git_blob_sha1"], "sequential oracle source differs from pinned Git blob")

        semantic = validate_pair(base, expanded, binding, oracle_module)
        base_sha = sha256(base_raw)
        expanded_sha = sha256(expanded_raw)
        challenge_material = (
            "connectome-model-ambiguity-set-expansion-v1\0"
            + binding["challenge_id"] + "\0"
            + base_sha + "\0"
            + expanded_sha + "\0"
            + binding["added_model_vertex_id"] + "\0"
            + binding["added_model_sha256"]
        ).encode("utf-8")
        receipt = {
            "schema_version": 1,
            "tool": "verify-connectome-model-ambiguity-expansion",
            "status": "APPEND_ONLY_MODEL_AMBIGUITY_MONOTONICITY_PASSED",
            "challenge_id": binding["challenge_id"],
            "base_input_raw_sha256": base_sha,
            "expanded_input_raw_sha256": expanded_sha,
            "base_input_git_blob_sha1": binding["base_input_git_blob_sha1"],
            "expanded_input_git_blob_sha1": binding["expanded_input_git_blob_sha1"],
            "sequential_oracle_git_blob_sha1": binding["sequential_oracle_git_blob_sha1"],
            "sequential_oracle_raw_sha256": sha256(oracle_raw),
            "challenge_digest_sha256": sha256(challenge_material),
            "frozen_query_world_action_policy_space": "IDENTICAL",
            "base_model_vertex_ids": binding["base_model_vertex_ids"],
            "expanded_model_vertex_ids": binding["expanded_model_vertex_ids"],
            "monotonicity_passed": True,
            "policy_space_mutation_detected": False,
            "independence_disposition": "NOT_ESTABLISHED_BY_ANCESTRY_LABELS",
            "biological_plausibility": "NOT_ESTABLISHED_BY_THIS_AUDIT",
            "ambiguity_set_completeness": "NOT_ESTABLISHED_BY_THIS_AUDIT",
            **semantic,
            "scope_note": "Exact monotonicity is conditional on the frozen finite policy class and declared model vertices. This audit does not prove that the appended model is biologically plausible or that the ambiguity set is complete."
        }
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        args.report.write_text(rendered, encoding="utf-8")
        print(
            "model ambiguity expansion oracle: PASS; "
            f"base_loss={receipt['base_static_robust_loss']}; "
            f"expanded_loss={receipt['expanded_static_robust_loss']}; "
            f"added={receipt['added_model_vertex_id']}"
        )
        print(f"receipt_sha256={sha256(rendered.encode('utf-8'))}")
        return 0
    except Exception as exc:
        print(f"model ambiguity expansion audit: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
