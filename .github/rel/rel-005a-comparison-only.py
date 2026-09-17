#!/usr/bin/env python3
"""REL-005A ComparisonOnly evaluator.

Consumes only the preregistered predicate contract, a sealed ExecutionOnly
observation, and its ObservationSeal. It cannot invoke production code.

A valid comparison with one or more failed predicates is still a successful
ComparisonOnly execution: predicate failure is adjudication data, not an
infrastructure/process failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import tempfile
from typing import Any

PREDICATE_CONTRACT_SCHEMA = "symthaea.rel.full-predicate-contract.v1"
PREDICATE_AUTHORITY = "PredicateContractOnly"
COMPARISON_CONTRACT_SCHEMA = "symthaea.rel.comparison-only-contract.v1"
COMPARISON_CONTRACT_AUTHORITY = "ComparisonOnlyContractOnly"
OBSERVATION_SCHEMA = "symthaea.rel.graft-frame-execution-observation.v3"
OBSERVATION_AUTHORITY = "ExecutionOnly"
SEAL_SCHEMA = "symthaea.rel.observation-seal.v3"
SEAL_AUTHORITY = "ObservationSeal"
OUTPUT_SCHEMA = "symthaea.rel.comparison-only.v1"
OUTPUT_AUTHORITY = "ComparisonOnly"

FALSE_UPSTREAM_CLAIMS = {
    "observation_sealed": False,
    "comparison_only_adjudicated": False,
    "rel_005a_qualified": False,
    "scientific_pass": False,
    "scientific_fail": False,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    require(isinstance(value, dict), f"{path}: top-level JSON must be an object")
    return value


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    return sha256_bytes(path.read_bytes())


def git_blob_sha1(path: pathlib.Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def is_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(float(value))


def compare_one(
    predicate: dict[str, Any],
    observed: Any,
    thresholds: dict[str, Any],
) -> tuple[bool, Any]:
    operator = predicate["operator"]

    if operator == "eq":
        expected = predicate["value"]
        if isinstance(expected, bool):
            require(isinstance(observed, bool), f"{predicate['id']}: expected boolean observation")
            return observed == expected, expected
        if type(expected) in (int, float):
            require(is_number(observed), f"{predicate['id']}: expected finite numeric observation")
            require(is_number(expected), f"{predicate['id']}: invalid numeric literal")
            return float(observed) == float(expected), expected
        require(type(observed) is type(expected), f"{predicate['id']}: observation type mismatch")
        return observed == expected, expected

    if operator in {"le", "gt"}:
        expected = predicate["value"]
        require(is_number(observed), f"{predicate['id']}: expected finite numeric observation")
        require(is_number(expected), f"{predicate['id']}: invalid numeric literal")
        if operator == "le":
            return float(observed) <= float(expected), expected
        return float(observed) > float(expected), expected

    if operator in {"le_threshold", "gt_threshold"}:
        threshold_name = predicate["threshold"]
        require(threshold_name in thresholds, f"{predicate['id']}: unknown threshold")
        expected = thresholds[threshold_name]
        require(is_number(observed), f"{predicate['id']}: expected finite numeric observation")
        require(is_number(expected), f"{predicate['id']}: invalid threshold value")
        if operator == "le_threshold":
            return float(observed) <= float(expected), {
                "threshold": threshold_name,
                "value": expected,
            }
        return float(observed) > float(expected), {
            "threshold": threshold_name,
            "value": expected,
        }

    raise ValueError(f"{predicate['id']}: unsupported operator {operator!r}")


def expected_chain_commitment(seal: dict[str, Any]) -> str:
    payload = {
        "artifact_manifest_sha256": seal["artifact_manifest_sha256"],
        "execution_receipt_sha256": seal["execution_receipt_sha256"],
        "execution_subject_head": seal["execution_subject_head"],
        "observation_sha256": seal["observation_sha256"],
    }
    return sha256_bytes(canonical_json_bytes(payload))


def verify_inputs(
    predicate_contract_path: pathlib.Path,
    comparison_contract_path: pathlib.Path,
    observation_path: pathlib.Path,
    seal_path: pathlib.Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    predicate_contract = load_json(predicate_contract_path)
    comparison_contract = load_json(comparison_contract_path)
    observation = load_json(observation_path)
    seal = load_json(seal_path)

    require(predicate_contract["schema"] == PREDICATE_CONTRACT_SCHEMA, "predicate schema mismatch")
    require(predicate_contract["authority"] == PREDICATE_AUTHORITY, "predicate authority mismatch")
    require(comparison_contract["schema"] == COMPARISON_CONTRACT_SCHEMA, "comparison contract schema mismatch")
    require(
        comparison_contract["authority"] == COMPARISON_CONTRACT_AUTHORITY,
        "comparison contract authority mismatch",
    )
    require(
        git_blob_sha1(predicate_contract_path) == comparison_contract["predicate_contract_blob"],
        "predicate contract Git blob mismatch",
    )

    require(observation["schema"] == OBSERVATION_SCHEMA, "observation schema mismatch")
    require(observation["authority"] == OBSERVATION_AUTHORITY, "observation authority mismatch")
    require(
        observation["frozen_scientific_subject"]
        == comparison_contract["frozen_scientific_subject"],
        "observation frozen subject mismatch",
    )
    require(
        observation["frozen_test_blob"] == comparison_contract["frozen_test_blob"],
        "observation frozen test blob mismatch",
    )
    require(
        observation["predicate_contract_head"]
        == comparison_contract["predicate_contract_head"],
        "observation predicate contract head mismatch",
    )
    require(observation.get("claims") == FALSE_UPSTREAM_CLAIMS, "observation claim boundary mismatch")

    require(seal["schema"] == SEAL_SCHEMA, "seal schema mismatch")
    require(seal["authority"] == SEAL_AUTHORITY, "seal authority mismatch")
    require(seal["observation_status"] == "sealed", "observation is not sealed")
    require(seal["adjudication"] == "not_run", "seal already carries adjudication")
    require(seal["scientific_result"] == "not_run", "seal already carries scientific result")
    require(
        seal["execution_subject_head"] == comparison_contract["execution_subject_head"],
        "seal execution subject mismatch",
    )
    require(
        seal["predicate_contract_head"] == comparison_contract["predicate_contract_head"],
        "seal predicate contract head mismatch",
    )
    require(
        seal["frozen_scientific_subject"]
        == comparison_contract["frozen_scientific_subject"],
        "seal frozen subject mismatch",
    )
    require(
        seal["frozen_test_blob"] == comparison_contract["frozen_test_blob"],
        "seal frozen test blob mismatch",
    )
    require(
        seal["observation_sha256"] == sha256_file(observation_path),
        "sealed observation SHA-256 mismatch",
    )
    require(
        seal["chain_commitment_sha256"] == expected_chain_commitment(seal),
        "seal chain commitment mismatch",
    )
    seal_claims = seal.get("claims")
    require(isinstance(seal_claims, dict), "seal claims missing")
    for key in (
        "comparison_only_adjudicated",
        "rel_005a_qualified",
        "scientific_pass",
        "scientific_fail",
    ):
        require(seal_claims.get(key) is False, f"seal claim {key} must be false")

    predicates = predicate_contract["predicates"]
    require(isinstance(predicates, list) and len(predicates) == 41, "predicate census mismatch")
    ids = [item["id"] for item in predicates]
    require(len(ids) == len(set(ids)), "duplicate predicate ids")

    return predicate_contract, comparison_contract, observation, seal


def adjudicate(
    predicate_contract_path: pathlib.Path,
    comparison_contract_path: pathlib.Path,
    observation_path: pathlib.Path,
    seal_path: pathlib.Path,
) -> dict[str, Any]:
    predicate_contract, comparison_contract, observation, seal = verify_inputs(
        predicate_contract_path,
        comparison_contract_path,
        observation_path,
        seal_path,
    )

    thresholds = predicate_contract["thresholds"]
    results: list[dict[str, Any]] = []
    failed_ids: list[str] = []

    for predicate in predicate_contract["predicates"]:
        key = predicate["observation"]
        require(key in observation, f"{predicate['id']}: missing observation {key}")
        observed = observation[key]
        require(observed is not None, f"{predicate['id']}: null observation {key}")
        passed, expected = compare_one(predicate, observed, thresholds)
        if not passed:
            failed_ids.append(predicate["id"])
        results.append(
            {
                "id": predicate["id"],
                "category": predicate["category"],
                "observation": key,
                "operator": predicate["operator"],
                "observed": observed,
                "expected": expected,
                "passed": passed,
            }
        )

    comparison_result = (
        "ALL_PREDICATES_PASS" if not failed_ids else "PREDICATE_FAILURES"
    )

    return {
        "schema": OUTPUT_SCHEMA,
        "authority": OUTPUT_AUTHORITY,
        "predicate_contract_head": comparison_contract["predicate_contract_head"],
        "execution_subject_head": comparison_contract["execution_subject_head"],
        "frozen_scientific_subject": comparison_contract["frozen_scientific_subject"],
        "frozen_test_blob": comparison_contract["frozen_test_blob"],
        "predicate_contract_sha256": sha256_file(predicate_contract_path),
        "comparison_contract_sha256": sha256_file(comparison_contract_path),
        "observation_sha256": sha256_file(observation_path),
        "seal_sha256": sha256_file(seal_path),
        "seal_chain_commitment_sha256": seal["chain_commitment_sha256"],
        "predicate_count": len(results),
        "passed_count": sum(1 for item in results if item["passed"]),
        "failed_count": len(failed_ids),
        "failed_predicate_ids": failed_ids,
        "comparison_result": comparison_result,
        "predicates": results,
        "claims": {
            "comparison_only_adjudicated": True,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }


def synthetic_pass_value(predicate: dict[str, Any], thresholds: dict[str, Any]) -> Any:
    op = predicate["operator"]
    if op == "eq":
        return predicate["value"]
    if op == "le":
        return predicate["value"]
    if op == "gt":
        return float(predicate["value"]) + 1.0
    if op == "le_threshold":
        return thresholds[predicate["threshold"]]
    if op == "gt_threshold":
        return float(thresholds[predicate["threshold"]]) + 1.0
    raise ValueError(op)


def synthetic_fail_value(predicate: dict[str, Any], thresholds: dict[str, Any]) -> Any:
    op = predicate["operator"]
    if op == "eq":
        expected = predicate["value"]
        return (not expected) if isinstance(expected, bool) else float(expected) + 1.0
    if op == "le":
        return float(predicate["value"]) + 1.0
    if op == "gt":
        return predicate["value"]
    if op == "le_threshold":
        return float(thresholds[predicate["threshold"]]) + 1.0
    if op == "gt_threshold":
        return thresholds[predicate["threshold"]]
    raise ValueError(op)


def make_synthetic_seal(
    observation_path: pathlib.Path,
    comparison_contract: dict[str, Any],
) -> dict[str, Any]:
    seal = {
        "schema": SEAL_SCHEMA,
        "authority": SEAL_AUTHORITY,
        "observation_status": "sealed",
        "adjudication": "not_run",
        "scientific_result": "not_run",
        "execution_subject_head": comparison_contract["execution_subject_head"],
        "predicate_contract_head": comparison_contract["predicate_contract_head"],
        "frozen_scientific_subject": comparison_contract["frozen_scientific_subject"],
        "frozen_test_blob": comparison_contract["frozen_test_blob"],
        "execution_receipt_sha256": "11" * 32,
        "artifact_manifest_sha256": "22" * 32,
        "observation_sha256": sha256_file(observation_path),
        "claims": {
            "comparison_only_adjudicated": False,
            "rel_005a_qualified": False,
            "scientific_pass": False,
            "scientific_fail": False,
        },
    }
    seal["chain_commitment_sha256"] = expected_chain_commitment(seal)
    return seal


def self_test(
    predicate_contract_path: pathlib.Path,
    comparison_contract_path: pathlib.Path,
) -> None:
    predicate_contract = load_json(predicate_contract_path)
    comparison_contract = load_json(comparison_contract_path)
    thresholds = predicate_contract["thresholds"]

    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        observation_path = root / "observation.json"
        seal_path = root / "seal.json"

        observation = {
            "schema": OBSERVATION_SCHEMA,
            "authority": OBSERVATION_AUTHORITY,
            "frozen_scientific_subject": comparison_contract["frozen_scientific_subject"],
            "frozen_test_blob": comparison_contract["frozen_test_blob"],
            "predicate_contract_head": comparison_contract["predicate_contract_head"],
            "claims": dict(FALSE_UPSTREAM_CLAIMS),
        }
        for predicate in predicate_contract["predicates"]:
            observation[predicate["observation"]] = synthetic_pass_value(
                predicate, thresholds
            )

        observation_path.write_text(json.dumps(observation, sort_keys=True) + "\n")
        seal = make_synthetic_seal(observation_path, comparison_contract)
        seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n")

        result = adjudicate(
            predicate_contract_path,
            comparison_contract_path,
            observation_path,
            seal_path,
        )
        require(result["comparison_result"] == "ALL_PREDICATES_PASS", "pass self-test failed")
        require(result["passed_count"] == 41 and result["failed_count"] == 0, "pass count self-test failed")

        first = predicate_contract["predicates"][0]
        observation[first["observation"]] = synthetic_fail_value(first, thresholds)
        observation_path.write_text(json.dumps(observation, sort_keys=True) + "\n")
        seal = make_synthetic_seal(observation_path, comparison_contract)
        seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n")
        result = adjudicate(
            predicate_contract_path,
            comparison_contract_path,
            observation_path,
            seal_path,
        )
        require(result["comparison_result"] == "PREDICATE_FAILURES", "failure self-test did not adjudicate")
        require(result["failed_predicate_ids"] == [first["id"]], "wrong failed predicate self-test")

        observation[first["observation"]] = synthetic_pass_value(first, thresholds)
        observation_path.write_text(json.dumps(observation, sort_keys=True) + "\n")
        seal = make_synthetic_seal(observation_path, comparison_contract)
        seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n")
        observation["tamper_marker"] = True
        observation_path.write_text(json.dumps(observation, sort_keys=True) + "\n")
        try:
            adjudicate(
                predicate_contract_path,
                comparison_contract_path,
                observation_path,
                seal_path,
            )
        except ValueError as exc:
            require("SHA-256 mismatch" in str(exc), "tamper self-test failed for wrong reason")
        else:
            raise ValueError("tamper self-test was accepted")

        observation.pop("tamper_marker")
        observation.pop(first["observation"])
        observation_path.write_text(json.dumps(observation, sort_keys=True) + "\n")
        seal = make_synthetic_seal(observation_path, comparison_contract)
        seal_path.write_text(json.dumps(seal, sort_keys=True) + "\n")
        try:
            adjudicate(
                predicate_contract_path,
                comparison_contract_path,
                observation_path,
                seal_path,
            )
        except ValueError as exc:
            require("missing observation" in str(exc), "missing-key self-test failed for wrong reason")
        else:
            raise ValueError("missing-key self-test was accepted")

    print(
        json.dumps(
            {
                "schema": "symthaea.rel.comparison-only-self-test.v1",
                "authority": "ComparisonOnlyContractOnly",
                "predicate_count": 41,
                "all_pass_case": "PASS",
                "predicate_failure_case": "PASS",
                "tamper_rejection_case": "PASS",
                "missing_observation_rejection_case": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predicate-contract", type=pathlib.Path, required=True)
    parser.add_argument("--comparison-contract", type=pathlib.Path, required=True)
    parser.add_argument("--observation", type=pathlib.Path)
    parser.add_argument("--seal", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test(args.predicate_contract, args.comparison_contract)
        return

    require(args.observation is not None, "--observation is required")
    require(args.seal is not None, "--seal is required")
    require(args.output is not None, "--output is required")

    result = adjudicate(
        args.predicate_contract,
        args.comparison_contract,
        args.observation,
        args.seal,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
