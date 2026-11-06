#!/usr/bin/env python3
"""Byte-bound integration oracle for synthetic mechanism-model animal transport.

This tool composes, but does not replace, the #10739 mechanism oracle and
#10753 cohort-transport oracle. It reports predictive model-family transport
at the independent animal unit. It never promotes that result to mechanism truth.
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
    return hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()


def valid_sha(value: Any, length: int = 64) -> bool:
    return isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is not None


def load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, f"cannot load independent oracle: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path.read_bytes()


def load_object(raw: bytes, label: str) -> dict[str, Any]:
    value = json.loads(raw)
    require(isinstance(value, dict), f"{label} must be a JSON object")
    return value


def exact_sign_test_p(wins: int, non_tied: int) -> float:
    require(0 <= wins <= non_tied, "sign-test wins outside [0,n]")
    if non_tied == 0:
        return 1.0
    return sum(math.comb(non_tied, k) for k in range(wins, non_tied + 1)) / (2**non_tied)


def find_exactly_one(rows: list[dict[str, Any]], predicate, label: str) -> dict[str, Any]:
    matches = [row for row in rows if predicate(row)]
    require(len(matches) == 1, f"{label}: expected exactly one row, found {len(matches)}")
    return matches[0]


def verify(
    mechanism_raw: bytes,
    mechanism_report_raw: bytes,
    transport_raw: bytes,
    transport_report_raw: bytes,
    binding: dict[str, Any],
    mechanism_data: dict[str, Any],
    mechanism_report: dict[str, Any],
    transport_data: dict[str, Any],
    transport_report: dict[str, Any],
    mechanism_oracle: Any,
    transport_oracle: Any,
    mechanism_rust_source_raw: bytes,
    mechanism_oracle_raw: bytes,
    transport_oracle_raw: bytes,
    bridge_oracle_raw: bytes,
    binding_raw: bytes,
) -> dict[str, Any]:
    require(binding["schema_version"] == 1, "unsupported integration binding version")
    require(binding["claim_scope"] == "synthetic_known_truth", "V1 bridge accepts synthetic fixtures only")
    require(mechanism_data["claim_scope"] == "synthetic_known_truth", "mechanism source is not the synthetic V1 fixture")
    require(transport_data["claim_scope"] == "synthetic_known_truth", "transport source is not the synthetic fixture")

    # Recompute both source-owner reports from the exact raw inputs.
    expected_mechanism = mechanism_oracle.verify(mechanism_data, mechanism_raw)
    mechanism_oracle.compare_json(expected_mechanism, mechanism_report)
    expected_transport = transport_oracle.verify(transport_data)
    require(expected_transport == transport_report, "transport report differs from the independent #10753 oracle")

    mb = binding["mechanism_binding"]
    tb = binding["transport_binding"]
    require(mechanism_data["query"]["query_id"] == mb["expected_query_id"], "mechanism query ID mismatch")
    require(mechanism_data["query"]["query_spec_sha256"] == mb["expected_query_spec_sha256"], "mechanism query digest mismatch")
    require(mechanism_report["input_sha256"] == sha256(mechanism_raw), "mechanism report is not bound to exact raw input bytes")
    require(mechanism_report["selected_intervention_id"] == mb["expected_selected_intervention_id"], "selected mechanism intervention mismatch")

    action_report = mechanism_report["action_reports"].get(mb["expected_selected_intervention_id"])
    require(action_report is not None, "selected intervention is absent from mechanism report")
    require(action_report["query_identified"] is mb["expected_query_identified"], "synthetic query gate differs from binding")
    require(action_report["mechanism_identified"] is mb["expected_mechanism_identified"], "synthetic mechanism gate differs from binding")

    expected_pair = {
        (row["mechanism_a"], row["mechanism_b"]): row
        for row in action_report["pairwise_results"]
    }
    map_rows = binding["mechanism_model_mappings"]
    mechanism_ids = [row["mechanism_id"] for row in map_rows]
    model_ids = [row["model_id"] for row in map_rows]
    require(len(mechanism_ids) == len(set(mechanism_ids)), "duplicate mechanism-to-model mapping")
    require(len(model_ids) == len(set(model_ids)), "one transport model cannot stand in for two mechanism hypotheses in V1")

    mechanisms = {row["mechanism_id"]: row for row in mechanism_data["mechanisms"]}
    transport_models = {row["model_id"]: row for row in transport_data["models"]}
    require(set(mechanism_ids).issubset(mechanisms), "mechanism mapping references an unknown mechanism")
    require(set(model_ids).issubset(transport_models), "mechanism mapping references an unknown transport model")
    require(set(transport_models) == set(model_ids), "transport model inventory differs from the frozen two-model mapping")
    expected_model_kind = {
        "DIRECT_SYNAPTIC": "ANATOMY_ONLY",
        "EXTRASYNAPTIC_LATENT": "WIRED_PLUS_EXTRASYNAPTIC",
    }
    for row in map_rows:
        require(row["mechanism_class"] == mechanisms[row["mechanism_id"]]["mechanism_class"], "mechanism class mapping mismatch")
        require(transport_models[row["model_id"]]["model_kind"] == expected_model_kind.get(row["mechanism_class"]),
                "transport model kind does not match the frozen mechanism-family mapping")
    require(len(mechanism_ids) == 2, "V1 compares exactly one pre-registered mechanism pair")
    mechanism_pair = tuple(sorted(mechanism_ids))
    pair_report = next(
        (row for row in action_report["pairwise_results"]
         if {row["mechanism_a"], row["mechanism_b"]} == set(mechanism_ids)),
        None,
    )
    require(pair_report is not None, "mapped mechanism pair is absent from the selected action report")
    require(pair_report["different_mechanism_class"], "mapped mechanism pair does not differ in mechanism class")
    require(not pair_report["different_query_decision"], "V1 predictive mechanism comparison must hold query decision fixed")
    require(pair_report["pair_identified_at_budget"] is True, "selected synthetic action does not identify the mapped mechanism pair")

    dataset = transport_data["dataset_binding"]
    split = transport_data["split_manifest"]
    require(dataset["dataset_revision_sha256"] == tb["dataset_revision_sha256"], "transport dataset revision mismatch")
    require(split["split_sha256"] == tb["split_sha256"], "transport animal split digest mismatch")
    require(split["lockbox_animal_ids"] == tb["lockbox_animal_ids"], "lockbox animal identity mismatch")
    require(split["lockbox_frozen_before_prediction"] is True, "transport lockbox was not frozen before prediction")
    require(expected_transport["split_integrity"] == "PASS_ANIMAL_IDS_DISJOINT", "upstream transport oracle did not qualify animal split")

    actions = {row["intervention_id"]: row for row in mechanism_data["interventions"]}
    selected_action = actions[mb["expected_selected_intervention_id"]]
    require(selected_action["feasibility_status"] == "FEASIBLE", "selected intervention is not feasible")

    condition_bindings = binding["intervention_condition_bindings"]
    require(all(row["intervention_id"] == selected_action["intervention_id"] for row in condition_bindings),
            "condition binding references another intervention")
    scopes = [row["mapping_scope"] for row in condition_bindings]
    require(scopes.count("PRIMARY_PAIRED_CONTRAST") == 1, "exactly one primary condition is required")
    require(scopes.count("OOD_ABSTENTION_CONTROL") == 1, "exactly one OOD control condition is required")
    condition_ids = [row["condition_id"] for row in condition_bindings]
    require(len(condition_ids) == len(set(condition_ids)), "duplicate intervention-to-condition binding")
    for row in condition_bindings:
        require(valid_sha(row["intervention_protocol_sha256"]), "intervention protocol digest malformed")
        require(valid_sha(row["mapping_receipt_sha256"]), "intervention-condition mapping receipt malformed")

    primary = next(row for row in condition_bindings if row["mapping_scope"] == "PRIMARY_PAIRED_CONTRAST")
    ood = next(row for row in condition_bindings if row["mapping_scope"] == "OOD_ABSTENTION_CONTROL")
    animals = {row["animal_id"]: row for row in transport_data["animals"]}
    observations = transport_data["observations"]
    predictions = transport_data["predictions"]
    pairs = {row["pair_id"]: row for row in transport_data["pairs"]}
    target_pair_id = tb["target_pair_id"]
    require(target_pair_id in pairs, "target neuron pair is absent from transport data")
    require(pairs[target_pair_id]["direct_wiring_status"] == tb["target_pair_direct_wiring_status"],
            "target-pair wiring disposition differs from the frozen manifest")
    require(pairs[target_pair_id]["direct_wiring_status"] == "NO_DIRECT_EDGE",
            "the known-answer target must be structurally unconnected")

    model_a = binding["comparison"]["model_a_id"]
    model_b = binding["comparison"]["model_b_id"]
    mapping = {row["mechanism_id"]: row["model_id"] for row in map_rows}
    require(model_a in transport_models and model_b in transport_models, "comparison model is absent")
    require(model_a != model_b, "paired model comparison requires distinct model IDs")
    require(set(model_ids) == {model_a, model_b}, "comparison models do not match the mechanism mapping")
    require(binding["comparison"]["alpha"] == 0.05, "V1 sign-test alpha must be preregistered at 0.05")
    require(binding["comparison"]["familywise_comparison_count"] == 1, "V1 known answer requires one frozen comparison")

    model_support = {
        model_id: {row["condition_id"] for row in model["supported_conditions"]}
        for model_id, model in transport_models.items()
    }
    primary_condition = primary["condition_id"]
    ood_condition = ood["condition_id"]
    require(primary_condition in model_support[model_a] and primary_condition in model_support[model_b],
            "both models must support the primary condition")
    require(ood_condition in model_support[model_a], "reference model must support the OOD control condition")
    require(ood_condition not in model_support[model_b], "OOD model must not claim support for the control condition")

    lockbox_primary_animals = sorted(
        animal_id for animal_id, animal in animals.items()
        if animal["cohort_role"] == "LOCKBOX" and animal["condition_id"] == primary_condition
    )
    require(len(lockbox_primary_animals) == 5, "known-answer primary condition must contain exactly five WT lockbox animals")
    require(len(set(tb["lockbox_animal_ids"])) == len(tb["lockbox_animal_ids"]), "duplicate lockbox animal ID")
    require(set(lockbox_primary_animals).issubset(set(tb["lockbox_animal_ids"])), "primary animals are not in the frozen lockbox")

    predictions_by_key = {(row["model_id"], row["observation_id"]): row for row in predictions}
    require(len(predictions_by_key) == len(predictions), "duplicate model/observation prediction")
    brier_by_model_animal: dict[str, dict[str, list[float]]] = {model_a: {}, model_b: {}}
    for animal_id in lockbox_primary_animals:
        obs = find_exactly_one(
            observations,
            lambda row: row["animal_id"] == animal_id
            and row["pair_id"] == target_pair_id
            and row["condition_id"] == primary_condition,
            f"primary observation for {animal_id}",
        )
        require(obs["measurement_disposition"] == "OBSERVED", f"primary observation is not measured for {animal_id}")
        require(obs["functional_disposition"] == binding["comparison"]["observation_disposition"],
                f"primary observation disposition differs from frozen scoring rule for {animal_id}")
        truth = 1.0
        for model_id in (model_a, model_b):
            prediction = predictions_by_key.get((model_id, obs["observation_id"]))
            require(prediction is not None, f"missing paired prediction: {model_id}/{obs['observation_id']}")
            require(prediction["prediction_disposition"] == "PREDICTED",
                    f"paired primary prediction abstained: {model_id}/{obs['observation_id']}")
            probability = prediction["probability_functional"]
            require(isinstance(probability, (int, float)) and math.isfinite(probability) and 0.0 <= probability <= 1.0,
                    "paired Brier probability is invalid")
            brier_by_model_animal[model_id].setdefault(animal_id, []).append((float(probability) - truth) ** 2)

    animal_brier = {
        model_id: {
            animal_id: sum(values) / len(values)
            for animal_id, values in sorted(by_animal.items())
        }
        for model_id, by_animal in brier_by_model_animal.items()
    }
    paired_animals = sorted(set(animal_brier[model_a]) & set(animal_brier[model_b]))
    differences = {
        animal_id: animal_brier[model_b][animal_id] - animal_brier[model_a][animal_id]
        for animal_id in paired_animals
    }
    wins_b = sum(value < 0.0 for value in differences.values())
    wins_a = sum(value > 0.0 for value in differences.values())
    ties = sum(value == 0.0 for value in differences.values())
    non_tied = wins_a + wins_b
    p_value = exact_sign_test_p(wins_b, non_tied)
    alpha = binding["comparison"]["alpha"]
    minimum = binding["transport_binding"]["minimum_paired_animals"]
    require(minimum == 5, "V1 requires five paired animals for the sign-test known answer")
    if non_tied < minimum:
        predictive_status = "NOT_ESTIMABLE_INSUFFICIENT_ANIMAL_SUPPORT"
    elif wins_b > wins_a and p_value <= alpha:
        predictive_status = "MODEL_B_FAVORED"
    else:
        predictive_status = "NO_CLEAR_PREFERENCE"

    require(len(paired_animals) == 5, "paired model comparison must include five independent animals")
    require(wins_b == 5 and wins_a == 0 and ties == 0, "synthetic paired animal outcomes differ from known answer")
    require(abs(p_value - 0.03125) <= 1e-12, "exact sign-test p-value differs from 1/32")
    require(predictive_status == binding["known_answer"]["expected_predictive_transport_status"],
            "predictive transport disposition differs from known answer")

    # The mutant is a separate OOD control, not part of the paired WT sign test.
    mutant_observations = [
        row for row in observations
        if row["animal_id"] in tb["lockbox_animal_ids"]
        and row["condition_id"] == ood_condition
        and row["measurement_disposition"] == "OBSERVED"
    ]
    require(len(mutant_observations) == 2, "known-answer mutant control must have two measured observations")
    model_b_ood = 0
    for obs in mutant_observations:
        prediction = predictions_by_key.get((model_b, obs["observation_id"]))
        require(prediction is not None, "missing OOD-control prediction record")
        require(prediction["prediction_disposition"] == "ABSTAINED_OUT_OF_DOMAIN",
                "model B must abstain on unsupported mutant observations")
        require(valid_sha(prediction["out_of_domain_receipt_sha256"]), "OOD abstention receipt malformed")
        model_b_ood += 1
    require(model_b_ood == binding["known_answer"]["expected_model_b_ood_abstentions_on_control_condition"],
            "model B OOD abstention count differs from known answer")

    require(binding["known_answer"]["biological_mechanism_status"] == "NOT_ESTABLISHED_BY_PREDICTIVE_TRANSPORT",
            "V1 may not promote predictive preference to biological mechanism truth")
    require(mapping.get("direct") == model_a and mapping.get("extra") == model_b,
            "mechanism-to-model mapping differs from the frozen known answer")

    upstream = binding["upstream_sources"]
    require(valid_sha(upstream["mechanism_commit_sha"], 40), "mechanism source commit SHA malformed")
    require(valid_sha(upstream["transport_commit_sha"], 40), "transport source commit SHA malformed")
    for field in ("mechanism_rust_git_blob_sha1", "mechanism_oracle_git_blob_sha1", "transport_oracle_git_blob_sha1"):
        require(valid_sha(upstream[field], 40), f"upstream Git blob identity malformed: {field}")

    return {
        "schema_version": 1,
        "tool": "verify-connectome-mechanism-model-animal-transport",
        "run_id": binding["run_id"],
        "claim_scope": "synthetic_known_truth",
        "mechanism_input_sha256": sha256(mechanism_raw),
        "mechanism_report_sha256": sha256(mechanism_report_raw),
        "mechanism_report_input_sha256": mechanism_report["input_sha256"],
        "transport_input_sha256": sha256(transport_raw),
        "transport_report_sha256": sha256(transport_report_raw),
        "binding_raw_sha256": sha256(binding_raw),
        "transport_dataset_revision_sha256": dataset["dataset_revision_sha256"],
        "transport_split_sha256": split["split_sha256"],
        "query_id": mechanism_data["query"]["query_id"],
        "query_spec_sha256": mechanism_data["query"]["query_spec_sha256"],
        "selected_intervention_id": mb["expected_selected_intervention_id"],
        "synthetic_query_gate": "IDENTIFIED" if action_report["query_identified"] else "UNIDENTIFIED",
        "synthetic_mechanism_gate": "IDENTIFIED" if action_report["mechanism_identified"] else "UNIDENTIFIED",
        "mechanism_pair": list(mechanism_pair),
        "model_pair": [model_a, model_b],
        "mechanism_source_origin_commit_sha": upstream["mechanism_commit_sha"],
        "transport_source_origin_commit_sha": upstream["transport_commit_sha"],
        "target_pair_id": target_pair_id,
        "target_pair_direct_wiring_status": pairs[target_pair_id]["direct_wiring_status"],
        "primary_condition_id": primary_condition,
        "paired_evaluable_animal_ids": paired_animals,
        "paired_animal_count": len(paired_animals),
        "model_a_animal_brier": animal_brier[model_a],
        "model_b_animal_brier": animal_brier[model_b],
        "paired_loss_difference_model_b_minus_model_a": differences,
        "model_b_wins": wins_b,
        "model_a_wins": wins_a,
        "ties": ties,
        "exact_sign_test_non_tied_n": non_tied,
        "exact_one_sided_sign_test_p": p_value,
        "alpha": alpha,
        "predictive_transport_status": predictive_status,
        "ood_control_condition_id": ood_condition,
        "model_b_ood_abstention_count": model_b_ood,
        "observation_scoring": "SHARED_DETERMINATE_OBSERVATIONS_WITHIN_ANIMAL",
        "independence_unit": "ANIMAL",
        "biological_mechanism_status": "NOT_ESTABLISHED_BY_PREDICTIVE_TRANSPORT",
        "status": "SYNTHETIC_MECHANISM_MODEL_TRANSPORT_CONTRACT_VALIDATED",
        "scope_note": "This receipt validates two synthetic owner contracts and animal-level predictive scoring only. It does not prove a causal mechanism, direct synapse, population transportability, or biological calibration."
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-input", type=Path, required=True)
    parser.add_argument("--mechanism-report", type=Path, required=True)
    parser.add_argument("--mechanism-oracle", type=Path, required=True)
    parser.add_argument("--mechanism-rust-source", type=Path, required=True)
    parser.add_argument("--transport-input", type=Path, required=True)
    parser.add_argument("--transport-report", type=Path, required=True)
    parser.add_argument("--transport-oracle", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    try:
        mechanism_raw = args.mechanism_input.read_bytes()
        mechanism_report_raw = args.mechanism_report.read_bytes()
        transport_raw = args.transport_input.read_bytes()
        transport_report_raw = args.transport_report.read_bytes()
        binding_raw = args.binding.read_bytes()
        mechanism_data = load_object(mechanism_raw, "mechanism input")
        mechanism_report = load_object(mechanism_report_raw, "mechanism report")
        transport_data = load_object(transport_raw, "transport input")
        transport_report = load_object(transport_report_raw, "transport report")
        binding = load_object(binding_raw, "integration binding")
        mechanism_oracle, mechanism_oracle_raw = load_module(args.mechanism_oracle, "mechanism_oracle")
        transport_oracle, transport_oracle_raw = load_module(args.transport_oracle, "transport_oracle")
        mechanism_rust_source_raw = args.mechanism_rust_source.read_bytes()
        bridge_oracle_raw = Path(__file__).read_bytes()

        upstream = binding["upstream_sources"]
        require(git_blob_sha1(mechanism_rust_source_raw) == upstream["mechanism_rust_git_blob_sha1"],
                "mechanism Rust source Git blob does not match pinned upstream source")
        require(git_blob_sha1(mechanism_oracle_raw) == upstream["mechanism_oracle_git_blob_sha1"],
                "mechanism oracle Git blob does not match pinned upstream source")
        require(git_blob_sha1(transport_oracle_raw) == upstream["transport_oracle_git_blob_sha1"],
                "transport oracle Git blob does not match pinned upstream source")

        receipt = verify(
            mechanism_raw, mechanism_report_raw, transport_raw, transport_report_raw,
            binding, mechanism_data, mechanism_report, transport_data, transport_report,
            mechanism_oracle, transport_oracle, mechanism_rust_source_raw,
            mechanism_oracle_raw, transport_oracle_raw, bridge_oracle_raw, binding_raw
        )
        receipt["mechanism_rust_git_blob_sha1"] = git_blob_sha1(mechanism_rust_source_raw)
        receipt["mechanism_oracle_git_blob_sha1"] = git_blob_sha1(mechanism_oracle_raw)
        receipt["transport_oracle_git_blob_sha1"] = git_blob_sha1(transport_oracle_raw)
        receipt["bridge_oracle_sha256"] = sha256(bridge_oracle_raw)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            "mechanism-model animal transport: PASS; "
            f"paired_animals={receipt['paired_animal_count']}; "
            f"model_b_wins={receipt['model_b_wins']}; "
            f"p={receipt['exact_one_sided_sign_test_p']}; "
            f"biology={receipt['biological_mechanism_status']}"
        )
        if args.self_test:
            require(exact_sign_test_p(5, 5) == 1.0 / 32.0, "sign-test exact enumeration self-test failed")
            require(abs(exact_sign_test_p(4, 5) - 6.0 / 32.0) <= 1e-12, "sign-test boundary self-test failed")
            require(exact_sign_test_p(0, 0) == 1.0, "empty sign-test self-test failed")
            print("animal-level paired sign-test self-checks: PASS")
        return 0
    except Exception as exc:
        print(f"mechanism-model transport bridge: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
