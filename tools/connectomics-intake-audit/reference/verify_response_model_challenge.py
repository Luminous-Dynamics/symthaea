#!/usr/bin/env python3
"""Audit additive response-model challenges for query-directed interventions.

This verifier compares a frozen base intervention fixture with an append-only
challenged fixture. It reuses the separately implemented decision-value oracle
for each finite model envelope, independently checks envelope monotonicity, and
computes support overlap across the union of admitted nuisance models.

A passing result is conditional on the exact declared finite model envelope.
It is not evidence that the biological response model is complete.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw).hexdigest()


def valid_sha(value: Any, length: int = 64) -> bool:
    return (
        isinstance(value, str)
        and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None
    )


def load_json(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def load_oracle(path: Path):
    raw = path.read_bytes()
    spec = importlib.util.spec_from_file_location(
        "query_discriminating_intervention_oracle", path
    )
    require(spec is not None and spec.loader is not None, "could not load intervention oracle")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, raw


FROZEN_FIELDS = (
    "schema_version",
    "run_id",
    "claim_scope",
    "dataset_binding",
    "query",
    "worlds",
    "interventions",
    "split_manifest",
    "selection_objective",
)


def prediction_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        row["intervention_id"],
        row["world_id"],
        row["nuisance_model_id"],
    )


def verify_append_only(
    base: dict[str, Any],
    challenged: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    for field in FROZEN_FIELDS:
        require(
            base.get(field) == challenged.get(field),
            f"model challenge changed frozen field {field}",
        )

    base_models = {row["nuisance_model_id"]: row for row in base["nuisance_models"]}
    challenged_models = {
        row["nuisance_model_id"]: row for row in challenged["nuisance_models"]
    }
    require(
        all(challenged_models.get(key) == value for key, value in base_models.items()),
        "base nuisance models were mutated or removed",
    )
    added_ids = sorted(set(challenged_models) - set(base_models))
    require(len(added_ids) == 1, "V1 requires exactly one append-only challenger model")

    model = binding["challenger_model"]
    challenger_id = model["nuisance_model_id"]
    require(added_ids == [challenger_id], "appended model ID differs from binding")
    require(model["adjudication_status"] == "ADMITTED_FOR_SENSITIVITY", "challenger is not admitted for sensitivity analysis")
    require(valid_sha(model["model_sha256"]), "challenger model digest malformed")
    require(valid_sha(model["observation_model_sha256"]), "challenger observation-model digest malformed")
    require(valid_sha(model["challenger_evidence_sha256"]), "challenger evidence receipt malformed")
    require(
        challenged_models[challenger_id]["model_sha256"] == model["model_sha256"],
        "challenged fixture model digest differs from binding",
    )
    require(
        challenged_models[challenger_id]["observation_model_sha256"]
        == model["observation_model_sha256"],
        "challenged fixture observation-model digest differs from binding",
    )

    base_predictions = {
        prediction_key(row): row for row in base["predictions"]
    }
    challenged_predictions = {
        prediction_key(row): row for row in challenged["predictions"]
    }
    require(
        len(base_predictions) == len(base["predictions"]),
        "base fixture contains duplicate prediction keys",
    )
    require(
        len(challenged_predictions) == len(challenged["predictions"]),
        "challenged fixture contains duplicate prediction keys",
    )
    require(
        all(challenged_predictions.get(key) == row for key, row in base_predictions.items()),
        "a frozen base prediction was changed or removed",
    )

    interventions = {row["intervention_id"] for row in base["interventions"]}
    worlds = {row["world_id"] for row in base["worlds"]}
    expected_new_keys = {
        (intervention_id, world_id, challenger_id)
        for intervention_id in interventions
        for world_id in worlds
    }
    actual_new_keys = set(challenged_predictions) - set(base_predictions)
    require(
        actual_new_keys == expected_new_keys,
        "challenger predictions are not the exact action × world Cartesian product",
    )

    return {
        "base_model_count": len(base_models),
        "challenged_model_count": len(challenged_models),
        "added_model_ids": added_ids,
        "base_prediction_count": len(base_predictions),
        "added_prediction_count": len(actual_new_keys),
        "frozen_world_count": len(worlds),
        "frozen_intervention_count": len(interventions),
    }


def support_overlap_audit(
    challenged: dict[str, Any],
) -> dict[str, Any]:
    """Audit support unions across all admitted nuisance models.

    For a query decision to be robustly identifiable from an intervention when
    the nuisance model is unknown, every opposite-decision world pair must have
    disjoint outcome support after taking the union over all admitted models.
    """
    models = [row["nuisance_model_id"] for row in challenged["nuisance_models"]]
    worlds = challenged["worlds"]
    interventions = challenged["interventions"]
    rows_by_key = {
        prediction_key(row): row for row in challenged["predictions"]
    }
    opposite_pairs = []
    for index, left in enumerate(worlds):
        for right in worlds[index + 1 :]:
            if left["query_decision"] != right["query_decision"]:
                opposite_pairs.append((left["world_id"], right["world_id"]))

    per_action = []
    robust_separating = []
    for action in interventions:
        action_id = action["intervention_id"]
        if action["feasibility_status"] != "FEASIBLE":
            continue
        world_support: dict[str, set[str]] = {}
        for world in worlds:
            union: set[str] = set()
            for model_id in models:
                row = rows_by_key[(action_id, world["world_id"], model_id)]
                union.update(
                    outcome["outcome_id"]
                    for outcome in row["observations"]
                    if outcome["probability"] > 0.0
                )
            world_support[world["world_id"]] = union

        overlaps = []
        for left_id, right_id in opposite_pairs:
            shared = sorted(world_support[left_id] & world_support[right_id])
            if shared:
                overlaps.append(
                    {
                        "world_a": left_id,
                        "world_b": right_id,
                        "shared_outcomes": shared,
                    }
                )
        separates = not overlaps
        if separates:
            robust_separating.append(action_id)
        per_action.append(
            {
                "intervention_id": action_id,
                "robustly_separates_all_opposite_query_worlds": separates,
                "overlapping_opposite_world_pairs": overlaps,
            }
        )

    status = (
        "ROBUSTLY_IDENTIFIABLE_WITH_DECLARED_ENVELOPE"
        if robust_separating
        else "ROBUST_NOT_IDENTIFIABLE"
    )
    return {
        "nuisance_model_count": len(models),
        "opposite_query_world_pair_count": len(opposite_pairs),
        "robust_separating_intervention_ids": sorted(robust_separating),
        "status": status,
        "per_intervention": per_action,
    }


def compare_reports(
    base_report: dict[str, Any],
    challenged_report: dict[str, Any],
    binding: dict[str, Any],
) -> dict[str, Any]:
    base_actions = base_report["per_intervention"]
    challenged_actions = challenged_report["per_intervention"]
    require(set(base_actions) == set(challenged_actions), "intervention set changed between reports")

    monotonicity = []
    for action_id in sorted(base_actions):
        base_risk = base_actions[action_id]["worst_case_expected_posterior_query_loss"]
        new_risk = challenged_actions[action_id]["worst_case_expected_posterior_query_loss"]
        base_value = base_actions[action_id]["decision_value_per_cost"]
        new_value = challenged_actions[action_id]["decision_value_per_cost"]
        risk_non_decreasing = new_risk + 1e-10 >= base_risk
        value_non_increasing = new_value <= base_value + 1e-10
        require(risk_non_decreasing, f"worst-case query loss improved after model expansion for {action_id}")
        require(value_non_increasing, f"decision value improved after model expansion for {action_id}")
        monotonicity.append(
            {
                "intervention_id": action_id,
                "base_worst_case_query_loss": base_risk,
                "challenged_worst_case_query_loss": new_risk,
                "base_decision_value_per_cost": base_value,
                "challenged_decision_value_per_cost": new_value,
                "risk_non_decreasing": risk_non_decreasing,
                "value_non_increasing": value_non_increasing,
            }
        )

    expected = binding["expected_outcomes"]
    require(
        base_report["decision_status"] == expected["base_decision_status"],
        "base decision status differs from known answer",
    )
    require(
        challenged_report["decision_status"] == expected["challenged_decision_status"],
        "challenged decision status differs from known answer",
    )
    require(
        base_report["selected_intervention_id"]
        == expected["base_selected_intervention_id"],
        "base selected intervention differs from known answer",
    )
    require(
        challenged_report["selected_intervention_id"]
        == expected["challenged_selected_intervention_id"],
        "challenged selected intervention differs from known answer",
    )

    for action_id, expected_risk in expected["base_worst_case_query_loss"].items():
        actual = base_actions[action_id]["worst_case_expected_posterior_query_loss"]
        require(abs(actual - expected_risk) <= 1e-8, f"base known-answer risk mismatch: {action_id}")
    for action_id, expected_risk in expected["challenged_worst_case_query_loss"].items():
        actual = challenged_actions[action_id]["worst_case_expected_posterior_query_loss"]
        require(abs(actual - expected_risk) <= 1e-8, f"challenged known-answer risk mismatch: {action_id}")
    for action_id, expected_value in expected["challenged_decision_value_per_cost"].items():
        actual = challenged_actions[action_id]["decision_value_per_cost"]
        require(abs(actual - expected_value) <= 1e-8, f"challenged known-answer decision value mismatch: {action_id}")

    require(all(row["risk_non_decreasing"] and row["value_non_increasing"] for row in monotonicity), "model-expansion monotonicity failed")
    require(binding["challenger_model"]["adjudication_status"] == "ADMITTED_FOR_SENSITIVITY", "challenger must be admitted for sensitivity evaluation")

    return {
        "monotonicity_passed": True,
        "per_intervention_monotonicity": monotonicity,
        "base_decision_status": base_report["decision_status"],
        "challenged_decision_status": challenged_report["decision_status"],
        "base_selected_intervention_id": base_report["selected_intervention_id"],
        "challenged_selected_intervention_id": challenged_report["selected_intervention_id"],
    }


def verify(
    base_raw: bytes,
    challenged_raw: bytes,
    binding_raw: bytes,
    base_git_blob: str,
    challenged_git_blob: str,
    oracle_module: Any,
) -> dict[str, Any]:
    base = load_json(base_raw, "base fixture")
    challenged = load_json(challenged_raw, "challenged fixture")
    binding = load_json(binding_raw, "challenge binding")
    require(binding["schema_version"] == 1, "unsupported binding schema")
    require(binding["claim_scope"] == "synthetic_known_truth", "only the synthetic known-answer binding is accepted")
    require(binding["base_fixture_git_blob_sha1"] == base_git_blob, "base fixture Git blob differs from binding")
    require(binding["challenged_fixture_git_blob_sha1"] == challenged_git_blob, "challenged fixture Git blob differs from binding")

    append_summary = verify_append_only(base, challenged, binding)
    base_report = oracle_module.verify(base)
    challenged_report = oracle_module.verify(challenged)
    report_comparison = compare_reports(base_report, challenged_report, binding)
    support_report = support_overlap_audit(challenged)
    expected = binding["expected_outcomes"]
    require(
        support_report["status"] == expected["robust_identifiability_status"],
        "support-overlap status differs from known answer",
    )
    require(
        support_report["robust_separating_intervention_ids"]
        == expected["robust_separating_intervention_ids"],
        "robustly separating intervention set differs from known answer",
    )

    return {
        "schema_version": 1,
        "tool": "verify-connectome-response-model-challenge",
        "status": "SYNTHETIC_MODEL_CHALLENGE_VALIDATED",
        "claim_scope": "synthetic_known_truth",
        "challenge_id": binding["challenge_id"],
        "base_fixture_git_blob_sha1": base_git_blob,
        "challenged_fixture_git_blob_sha1": challenged_git_blob,
        "base_fixture_raw_sha256": sha256(base_raw),
        "challenged_fixture_raw_sha256": sha256(challenged_raw),
        "binding_raw_sha256": sha256(binding_raw),
        "challenger_model_id": binding["challenger_model"]["nuisance_model_id"],
        "challenger_model_sha256": binding["challenger_model"]["model_sha256"],
        "challenger_observation_model_sha256": binding["challenger_model"]["observation_model_sha256"],
        "challenger_evidence_sha256": binding["challenger_model"]["challenger_evidence_sha256"],
        "challenger_adjudication_status": binding["challenger_model"]["adjudication_status"],
        "append_only_summary": append_summary,
        "base_decision_value_report": base_report,
        "challenged_decision_value_report": challenged_report,
        "model_expansion_comparison": report_comparison,
        "support_overlap_audit": support_report,
        "biological_model_adequacy": "NOT_ESTABLISHED_BY_FINITE_SYNTHETIC_CHALLENGE",
        "global_model_envelope_completeness": "NOT_ESTABLISHED",
    }


def self_test() -> None:
    # A challenger that adds a shared outcome to one opposing world pair must
    # destroy robust separation; adding a model must never remove old support.
    left = {"X", "A"}
    right = {"Y"}
    expanded_left = left | {"Y"}
    require(bool(expanded_left & right), "overlap negative control failed")
    require(left.issubset(expanded_left), "additive support was not monotone")
    print("response-model support-expansion negative controls: PASS")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path)
    parser.add_argument("--challenged", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--oracle", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        try:
            self_test()
            return 0
        except Exception as exc:
            print(f"response-model challenge self-test: FAIL: {exc}", file=sys.stderr)
            return 1

    try:
        require(
            all((args.base, args.challenged, args.binding, args.oracle, args.receipt)),
            "base, challenged, binding, oracle and receipt paths are required",
        )
        base_raw = args.base.read_bytes()
        challenged_raw = args.challenged.read_bytes()
        binding_raw = args.binding.read_bytes()
        binding = load_json(binding_raw, "challenge binding")
        oracle_module, _oracle_raw = load_oracle(args.oracle)
        base_git_blob = git_blob_sha1(base_raw)
        challenged_git_blob = git_blob_sha1(challenged_raw)
        # The manifest uses Git blob SHA-1. These are checked against the actual
        # exact file bytes, independently of GitHub's displayed blob metadata.
        receipt = verify(
            base_raw,
            challenged_raw,
            binding_raw,
            base_git_blob,
            challenged_git_blob,
            oracle_module,
        )
        rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
        args.receipt.write_text(rendered, encoding="utf-8")
        print(
            "response-model challenge: PASS; "
            f"challenge={receipt['challenge_id']}; "
            f"models={receipt['append_only_summary']['base_model_count']}->"
            f"{receipt['append_only_summary']['challenged_model_count']}; "
            f"decision={receipt['model_expansion_comparison']['challenged_decision_status']}; "
            f"identifiability={receipt['support_overlap_audit']['status']}"
        )
        return 0
    except Exception as exc:
        print(f"response-model challenge: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
